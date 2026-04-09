from __future__ import annotations
import json, time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Optional
import numpy as np

CADENCE_DB_PATH = Path('config/alert_cadence.json')
COOLDOWN_MIN  = 8
COOLDOWN_MAX  = 300
RESPONSE_WINDOW = 8
LATENCY_TREND_THRESHOLD = 3.0
COOLDOWN_SHORTEN_FACTOR = 0.75
COOLDOWN_LENGTHEN_FACTOR = 1.20
URGENCY_LEVELS = ['gentle', 'medium', 'urgent', 'emergency']


class ResponseRecord:
    __slots__ = ('alert_id', 'alert_type', 'sent_ts', 'ack_ts', 'latency_s', 'urgency')

    def __init__(self, alert_id, alert_type, urgency, sent_ts):
        self.alert_id   = alert_id
        self.alert_type = alert_type
        self.urgency    = urgency
        self.sent_ts    = sent_ts
        self.ack_ts:    Optional[float] = None
        self.latency_s: Optional[float] = None

    def acknowledge(self, ack_ts=None):
        self.ack_ts    = ack_ts or time.time()
        self.latency_s = self.ack_ts - self.sent_ts

    def to_dict(self):
        return {k: getattr(self, k) for k in self.__slots__}


class AdaptiveAlertCadence:
    def __init__(self, base_cooldown=30):
        self._base_cooldown = base_cooldown
        self._db            = self._load_db()
        self._pending: dict[str, ResponseRecord] = {}
        self._response_history: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=RESPONSE_WINDOW * 2))
        self._adapt_counter: dict[str, int] = defaultdict(int)
        self._cooldowns: dict[str, float] = {
            k: v for k, v in self._db.get('cooldowns', {}).items()}
        self._urgency_bumps: dict[str, int] = {
            k: v for k, v in self._db.get('urgency_bumps', {}).items()}

    def get_params(self, alert_type):
        cooldown  = self._cooldowns.get(alert_type, float(self._base_cooldown))
        cooldown  = float(np.clip(cooldown, COOLDOWN_MIN, COOLDOWN_MAX))
        bump      = self._urgency_bumps.get(alert_type, 0)
        final_idx = min(len(URGENCY_LEVELS) - 1, 0 + bump)
        urgency   = URGENCY_LEVELS[final_idx]
        stats     = self._get_type_stats(alert_type)
        meta = {
            'cooldown_secs':    cooldown,
            'urgency':          urgency,
            'urgency_bump':     bump,
            'avg_latency_s':    stats['avg_latency'],
            'trend':            stats['trend'],
            'samples':          stats['n'],
            'fatigue_detected': stats['fatigue'],
        }
        return cooldown, urgency, meta

    def record_sent(self, alert_id, alert_type, urgency):
        rec = ResponseRecord(alert_id=alert_id, alert_type=alert_type,
                             urgency=urgency, sent_ts=time.time())
        self._pending[alert_id] = rec
        print(f"[Cadence] Sent {alert_type} / {urgency} → "
              f"cooldown={self._cooldowns.get(alert_type, self._base_cooldown):.0f}s")

    def record_ack(self, alert_id):
        rec = self._pending.pop(alert_id, None)
        if rec is None:
            return None
        rec.acknowledge()
        self._response_history[rec.alert_type].append(rec.latency_s)
        self._adapt_counter[rec.alert_type] += 1
        self._db.setdefault('response_log', []).append(rec.to_dict())
        if len(self._db['response_log']) > 2000:
            self._db['response_log'] = self._db['response_log'][-1000:]
        if self._adapt_counter[rec.alert_type] % max(1, RESPONSE_WINDOW // 2) == 0:
            self._adapt_type(rec.alert_type)
        self._save_db()
        print(f"[Cadence] Ack {alert_id}: latency={rec.latency_s:.1f}s")
        return rec.latency_s

    def adapt(self):
        for alert_type in list(self._response_history.keys()):
            self._adapt_type(alert_type)

    def expire_unacknowledged(self, timeout_s=120.0):
        now = time.time()
        expired = [aid for aid, rec in self._pending.items()
                   if now - rec.sent_ts > timeout_s]
        for aid in expired:
            rec = self._pending.pop(aid)
            synthetic_latency = timeout_s * 2
            self._response_history[rec.alert_type].append(synthetic_latency)
            self._adapt_counter[rec.alert_type] += 1
            print(f"[Cadence] Alert {aid} expired unacknowledged — "
                  f"synthetic latency={synthetic_latency:.0f}s")

    def get_full_report(self):
        report = {}
        for atype in set(list(self._cooldowns.keys()) +
                         list(self._response_history.keys())):
            cooldown, urgency, meta = self.get_params(atype)
            report[atype] = meta
        return report

    def _adapt_type(self, alert_type):
        stats = self._get_type_stats(alert_type)
        if stats['n'] < 3:
            return
        trend   = stats['trend']
        fatigue = stats['fatigue']
        current_cd = self._cooldowns.get(alert_type, float(self._base_cooldown))

        if fatigue or trend > LATENCY_TREND_THRESHOLD:
            new_cd = max(COOLDOWN_MIN, current_cd * COOLDOWN_SHORTEN_FACTOR)
            self._cooldowns[alert_type] = new_cd
            bump = self._urgency_bumps.get(alert_type, 0)
            self._urgency_bumps[alert_type] = min(len(URGENCY_LEVELS)-1, bump+1)
            action = "TIGHTENED (fatigue detected)"
        elif trend < -LATENCY_TREND_THRESHOLD and stats['avg_latency'] < 30:
            effective_start = max(current_cd, float(self._base_cooldown))
            new_cd = min(COOLDOWN_MAX, effective_start * COOLDOWN_LENGTHEN_FACTOR)
            self._cooldowns[alert_type] = new_cd
            bump = self._urgency_bumps.get(alert_type, 0)
            self._urgency_bumps[alert_type] = max(0, bump - 1)
            action = "RELAXED (responsive caregiver)"
        else:
            action = "unchanged"

        self._save_db()
        print(f"[Cadence] Adapt '{alert_type}': {action} → "
              f"cooldown={self._cooldowns.get(alert_type, self._base_cooldown):.1f}s, "
              f"urgency_bump={self._urgency_bumps.get(alert_type, 0)}")

    def _get_type_stats(self, alert_type):
        samples = list(self._response_history.get(alert_type, []))
        if not samples:
            return {'n': 0, 'avg_latency': 0.0, 'trend': 0.0, 'fatigue': False}
        n   = len(samples)
        avg = float(np.mean(samples))
        slope = 0.0
        if n >= 3:
            x     = np.arange(n, dtype=float)
            slope = float(np.polyfit(x, samples, 1)[0])

        fatigue = avg >= 45.0 and slope > 0

        return {'n': n, 'avg_latency': round(avg, 2),
                'trend': round(slope, 3), 'fatigue': fatigue}

    def _load_db(self):
        if CADENCE_DB_PATH.exists():
            try:
                with open(CADENCE_DB_PATH, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_db(self):
        CADENCE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._db['cooldowns']     = dict(self._cooldowns)
        self._db['urgency_bumps'] = dict(self._urgency_bumps)
        self._db['updated_at']    = datetime.now().isoformat()
        with open(CADENCE_DB_PATH, 'w') as f:
            json.dump(self._db, f, indent=2)