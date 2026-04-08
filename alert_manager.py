"""
Alert Management System  (v2 — Patent Edition)
===============================================
New in v2:
  • AdaptiveAlertCadence integration (Patent: self-tuning cooldown based on
    caregiver response latency / alert fatigue detection).
  • acknowledge_alert() now feeds response latency back to the cadence engine.
  • create_alert() uses cadence-derived urgency when risk assessment is 'gentle'.

Pool detection removed.  Fire is the only tracked hazard type.
"""

import time
from datetime import datetime, timedelta
from collections import deque
import json
from pathlib import Path
import yaml

try:
    from adaptive_alert_cadence import AdaptiveAlertCadence
    CADENCE_AVAILABLE = True
except ImportError:
    CADENCE_AVAILABLE = False
    print("[AlertManager] adaptive_alert_cadence not available — using static cooldowns")


class AlertLevel:
    SILENT    = 0
    GENTLE    = 1
    MEDIUM    = 2
    URGENT    = 3
    EMERGENCY = 4

    @staticmethod
    def get_name(level):
        names = {0: "Silent", 1: "Gentle", 2: "Medium", 3: "Urgent", 4: "Emergency"}
        return names.get(level, "Unknown")

    @staticmethod
    def from_urgency(urgency_str: str) -> int:
        return {'emergency': 4, 'urgent': 3,
                'medium': 2, 'gentle': 1}.get(urgency_str, 1)


class AlertManager:
    def __init__(self, config_path='config/config.yaml'):
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            self.config = {}

        alerts_cfg = self.config.get('alerts', {})
        self._static_cooldown = alerts_cfg.get('cooldown_period', 30)
        self.escalation_time  = alerts_cfg.get('escalation_time', 120)

        self.alert_history  = deque(maxlen=100)
        self.last_alert_time: dict[str, float] = {}
        self.active_alerts:  dict[str, dict]   = {}

        self.notification_config = self.config.get('notification', {
            'mobile_app':        True,
            'push_notification': True,
            'email':             False,
            'sms':               False,
        })

        self.log_dir = Path('logs/alerts')
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # ----------------------------------------------------------------
        # Patent module: Adaptive alert cadence
        # ----------------------------------------------------------------
        self.cadence: AdaptiveAlertCadence | None = (
            AdaptiveAlertCadence(base_cooldown=self._static_cooldown)
            if CADENCE_AVAILABLE else None
        )

    # ------------------------------------------------------------------
    # Cooldown / should-send logic
    # ------------------------------------------------------------------

    def should_send_alert(self, alert_type: str, current_level: int):
        """
        Decide whether to send an alert, consulting the adaptive cadence
        engine for the effective cooldown period.
        """
        if current_level == AlertLevel.EMERGENCY:
            return True, "Emergency — no cooldown"

        # ---- Patent: use adaptive cooldown ----
        if self.cadence:
            effective_cooldown, _, meta = self.cadence.get_params(alert_type)
        else:
            effective_cooldown = self._static_cooldown
            meta = {}

        if alert_type in self.last_alert_time:
            elapsed = time.time() - self.last_alert_time[alert_type]
            if elapsed < effective_cooldown:
                remaining = int(effective_cooldown - elapsed)
                fatigue_tag = " [fatigue-mode]" if meta.get('fatigue_detected') else ""
                return False, f"Cooldown active ({remaining}s){fatigue_tag}"

        if alert_type in self.active_alerts:
            if current_level > self.active_alerts[alert_type]['level']:
                return True, "Alert escalation detected"

        return True, "Normal alert conditions met"

    # ------------------------------------------------------------------
    # Alert creation
    # ------------------------------------------------------------------

    def create_alert(self, assessment: dict, detections: dict) -> dict:
        base_urgency = assessment.get('alert_urgency', 'gentle')
        alert_type   = self._determine_alert_type(detections)

        # ---- Patent: cadence may bump urgency based on response history ----
        if self.cadence:
            _, cadence_urgency, meta = self.cadence.get_params(alert_type)
            # Use the higher urgency of assessment vs. cadence recommendation
            urg_order = ['gentle', 'medium', 'urgent', 'emergency']
            final_urgency = (cadence_urgency
                             if urg_order.index(cadence_urgency) >
                                urg_order.index(base_urgency)
                             else base_urgency)
            cadence_meta = meta
        else:
            final_urgency = base_urgency
            cadence_meta  = {}

        alert_level = AlertLevel.from_urgency(final_urgency)

        alert = {
            'id':                  self._generate_alert_id(),
            'timestamp':           datetime.now().isoformat(),
            'type':                alert_type,
            'level':               alert_level,
            'level_name':          AlertLevel.get_name(alert_level),
            'risk_score':          assessment.get('risk_score', 0.0),
            'risk_level':          assessment.get('risk_level_name', 'UNKNOWN'),
            'explanation':         assessment.get('explanation', {}),
            'recommended_actions': [],
            'detections':          detections,
            'acknowledged':        False,
            'resolved':            False,
            # Patent-novel fields
            'attention_state':     assessment.get('attention_state'),
            'developmental_stage': assessment.get('developmental_stage'),
            'habituation_active':  assessment.get('habituation_active', False),
            'cadence_meta':        cadence_meta,
        }
        return alert

    def _determine_alert_type(self, detections) -> str:
        if detections.get('fire'):
            return 'fire_hazard'
        return 'general_hazard'

    def _generate_alert_id(self) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"ALERT_{ts}_{len(self.alert_history)}"

    # ------------------------------------------------------------------
    # Sending
    # ------------------------------------------------------------------

    def send_alert(self, alert: dict) -> dict:
        alert_type  = alert['type']
        alert_level = alert['level']

        should_send, reason = self.should_send_alert(alert_type, alert_level)
        if not should_send and alert_level < AlertLevel.EMERGENCY:
            print(f"Alert suppressed: {reason}")
            return {'suppressed': True, 'reason': reason}

        delivery_status: dict = {}

        if self.notification_config.get('mobile_app', False):
            delivery_status['mobile_app'] = self._send_mobile_notification(alert)

        if self.notification_config.get('push_notification', False):
            delivery_status['push'] = self._send_push_notification(alert)

        if (self.notification_config.get('email', False)
                and alert_level >= AlertLevel.MEDIUM):
            delivery_status['email'] = self._send_email_notification(alert)

        if (self.notification_config.get('sms', False)
                and alert_level >= AlertLevel.URGENT):
            delivery_status['sms'] = self._send_sms_notification(alert)

        self.last_alert_time[alert_type] = time.time()
        self.active_alerts[alert_type]   = alert
        self._log_alert(alert, delivery_status)
        self.alert_history.append(alert)

        # ---- Patent: record send event in cadence engine ----
        if self.cadence:
            self.cadence.record_sent(
                alert['id'], alert_type,
                AlertLevel.get_name(alert_level).lower()
            )

        return delivery_status

    # ------------------------------------------------------------------
    # Notifications
    # ------------------------------------------------------------------

    def _send_mobile_notification(self, alert: dict) -> dict:
        # Build contextual subtitle from patent-novel fields
        subtitle_parts = []
        if alert.get('attention_state') == 'distracted':
            subtitle_parts.append("Caregiver distracted")
        if alert.get('attention_state') == 'absent':
            subtitle_parts.append("No caregiver detected")
        if alert.get('developmental_stage') in ('walking', 'running'):
            subtitle_parts.append(f"Child: {alert['developmental_stage']}")
        if alert.get('habituation_active'):
            subtitle_parts.append("Repeat approach detected")

        notification = {
            'title':    (f"{alert['level_name']} Alert: "
                         f"{alert['type'].replace('_',' ').title()}"),
            'body':     self._format_alert_message(alert),
            'subtitle': " | ".join(subtitle_parts) if subtitle_parts else "",
            'priority': 'high' if alert['level'] >= AlertLevel.URGENT else 'normal',
            'data':     {'alert_id': alert['id'], 'type': alert['type'],
                         'risk_score': alert['risk_score']},
        }
        print(f"\nMobile Notification: {notification['title']}")
        if notification['subtitle']:
            print(f"  Context: {notification['subtitle']}")
        print(f"  {notification['body'][:120]}")
        return {'success': True, 'notification': notification}

    def _send_push_notification(self, alert: dict) -> dict:
        print(f"Push Notification sent for alert {alert['id']}")
        return {'success': True}

    def _send_email_notification(self, alert: dict) -> dict:
        subject = f"[{alert['level_name']}] Baby Safety Alert: {alert['type']}"
        print(f"Email sent: {subject}")
        return {'success': True}

    def _send_sms_notification(self, alert: dict) -> dict:
        msg = (f"URGENT: {alert['level_name']} baby safety alert. "
               f"{self._format_alert_message(alert)}")
        print(f"SMS sent: {msg[:60]}...")
        return {'success': True}

    def _format_alert_message(self, alert: dict) -> str:
        explanation = alert.get('explanation', {})
        factors     = explanation.get('primary_factors', [])
        msg = f"Risk Level: {alert['risk_level']} (Score: {alert['risk_score']:.2f})"
        if factors:
            msg += "\nKey Factors: " + "; ".join(factors[:3])
        cadence = alert.get('cadence_meta', {})
        if cadence.get('fatigue_detected'):
            msg += f"\n[Alert cadence tightened — avg response {cadence.get('avg_latency_s',0):.0f}s]"
        return msg

    # ------------------------------------------------------------------
    # Acknowledgement — feeds back into cadence engine
    # ------------------------------------------------------------------

    def acknowledge_alert(self, alert_id: str) -> bool:
        for alert in self.alert_history:
            if alert['id'] == alert_id:
                alert['acknowledged']    = True
                alert['acknowledged_at'] = datetime.now().isoformat()

                # ---- Patent: feed response latency to cadence engine ----
                if self.cadence:
                    latency = self.cadence.record_ack(alert_id)
                    if latency is not None:
                        alert['response_latency_s'] = round(latency, 2)
                        print(f"[AlertManager] Response latency for {alert_id}: "
                              f"{latency:.1f}s → cadence updated")

                print(f"✓ Alert {alert_id} acknowledged")
                return True
        return False

    # ------------------------------------------------------------------
    # Resolve / escalate
    # ------------------------------------------------------------------

    def resolve_alert(self, alert_id: str, resolution_note=None) -> bool:
        for alert in self.alert_history:
            if alert['id'] == alert_id:
                alert['resolved']    = True
                alert['resolved_at'] = datetime.now().isoformat()
                if resolution_note:
                    alert['resolution_note'] = resolution_note
                self.active_alerts.pop(alert['type'], None)
                print(f"✓ Alert {alert_id} resolved")
                return True
        return False

    def check_escalation(self) -> list:
        current_time = time.time()
        escalated    = []

        for alert_type, alert in list(self.active_alerts.items()):
            if alert.get('acknowledged'):
                continue
            alert_ts = datetime.fromisoformat(alert['timestamp']).timestamp()
            if current_time - alert_ts >= self.escalation_time:
                alert['level']          = min(AlertLevel.EMERGENCY, alert['level'] + 1)
                alert['level_name']     = AlertLevel.get_name(alert['level'])
                alert['escalated']      = True
                alert['escalation_time'] = datetime.now().isoformat()
                print(f"\nALERT ESCALATION: {alert['id']} → {alert['level_name']}")

                # Also expire in cadence engine (counts as very slow response)
                if self.cadence:
                    self.cadence.expire_unacknowledged(timeout_s=self.escalation_time)

                self.send_alert(alert)
                escalated.append(alert)

        return escalated

    # ------------------------------------------------------------------
    # Logging / statistics
    # ------------------------------------------------------------------

    def _log_alert(self, alert: dict, delivery_status: dict):
        log_entry = {
            'alert':           alert,
            'delivery_status': delivery_status,
            'logged_at':       datetime.now().isoformat(),
        }
        log_file = self.log_dir / f"alerts_{datetime.now().strftime('%Y%m%d')}.json"
        logs: list = []
        if log_file.exists():
            try:
                with open(log_file, 'r') as f:
                    logs = json.load(f)
            except Exception:
                logs = []
        logs.append(log_entry)
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)

    def get_active_alerts(self) -> list:
        return list(self.active_alerts.values())

    def get_alert_statistics(self) -> dict:
        total = len(self.alert_history)
        if total == 0:
            return {'total_alerts': 0, 'by_level': {}, 'by_type': {},
                    'acknowledged_rate': 0.0, 'resolution_rate': 0.0,
                    'active_alerts': 0, 'cadence_report': {}}

        by_level: dict = {}
        by_type:  dict = {}
        for a in self.alert_history:
            by_level[a['level_name']] = by_level.get(a['level_name'], 0) + 1
            by_type[a['type']]        = by_type.get(a['type'],         0) + 1

        acknowledged = sum(1 for a in self.alert_history if a.get('acknowledged'))
        resolved     = sum(1 for a in self.alert_history if a.get('resolved'))

        avg_latency = None
        latencies   = [a['response_latency_s'] for a in self.alert_history
                       if 'response_latency_s' in a]
        if latencies:
            import numpy as np
            avg_latency = round(float(np.mean(latencies)), 2)

        cadence_report = self.cadence.get_full_report() if self.cadence else {}

        return {
            'total_alerts':        total,
            'by_level':            by_level,
            'by_type':             by_type,
            'acknowledged_rate':   acknowledged / total,
            'resolution_rate':     resolved / total,
            'active_alerts':       len(self.active_alerts),
            'avg_response_latency_s': avg_latency,
            'cadence_report':      cadence_report,   # Patent-novel field
        }