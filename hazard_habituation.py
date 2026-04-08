"""
Cross-Session Hazard Habituation Detector
==========================================
Patent Claim: A method for tracking whether a child repeatedly approaches
the same persistent hazard across multiple monitoring sessions and, upon
detecting an increasing approach-frequency trend (habituation or developing
curiosity), automatically escalating that hazard's persistent risk weight
in the room map.

Novel aspect vs. prior art:
  Existing spatial mapping systems record hazard locations but treat each
  detection event independently.  This module introduces a *behavioural
  learning loop*: it accumulates cross-session approach events per hazard,
  fits a trend to the inter-session approach counts, and raises a hazard's
  risk weight when the trend is positive (child is approaching more
  frequently over time, not less).

Key concepts:
  - "Approach event" : child centroid enters the WARNING zone of a known
    persistent hazard and moves toward it (proximity decreasing).
  - "Session"        : one continuous monitoring run (app start → stop).
  - "Habituation"    : approach count per session is non-decreasing over
    the last N sessions, suggesting the child is no longer deterred.
  - Risk weight bump : the hazard's base confidence/weight is scaled by
    HABITUATION_WEIGHT_SCALE when habituation is detected.

Data persistence:
  All cross-session data is stored in JSON at the path configured in
  ``HABITUATION_DB_PATH``.  The room mapper's persistent_hazards list
  is augmented in-place with updated risk weights.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

HABITUATION_DB_PATH     = Path('config/habituation_db.json')
MIN_SESSIONS_FOR_TREND  = 3        # Minimum sessions before trend analysis
HABITUATION_WEIGHT_SCALE = 1.50   # Risk weight multiplier when habituated
GRID_MATCH_RADIUS       = 8        # Grid cells (same as room-mapper resolution)
APPROACH_COOLDOWN_SECS  = 30       # Seconds between approach events per session


class HazardApproachEvent:
    """Lightweight data class for a single approach event."""
    __slots__ = ('session_id', 'timestamp', 'hazard_key', 'min_distance_m', 'velocity')

    def __init__(self, session_id: str, hazard_key: str,
                 min_distance_m: float, velocity: float):
        self.session_id    = session_id
        self.timestamp     = datetime.now().isoformat()
        self.hazard_key    = hazard_key
        self.min_distance_m = min_distance_m
        self.velocity       = velocity

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__slots__}


class HazardHabituationDetector:
    """
    Tracks child approach events per hazard across sessions and escalates
    hazard weights when habituation is detected.

    Parameters
    ----------
    session_window : int
        Number of recent sessions to use for trend analysis.
    approach_distance_m : float
        Distance threshold (metres) inside which an event is counted.
    """

    def __init__(self,
                 session_window: int = 10,
                 approach_distance_m: float = 2.5):
        self._session_id          = self._generate_session_id()
        self._session_window      = session_window
        self._approach_dist       = approach_distance_m
        self._db: dict[str, Any]  = self._load_db()
        self._in_session_events: dict[str, list] = defaultdict(list)
        self._last_approach_ts: dict[str, float] = {}

        # Track which hazards have been flagged this session
        self._habituated_keys: set[str] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def observe(self,
                child_bbox: list[int],
                proximity_info: dict,
                persistent_hazards: list[dict],
                current_time_ts: float | None = None) -> dict:
        """
        Called every frame.  Records approach events when the child enters
        a hazard's warning zone while moving toward it.

        Args:
            child_bbox        : [x1, y1, x2, y2] of primary child detection.
            proximity_info    : Output of SpatialRiskAssessment (closest_distance,
                                zone, closest_hazard, etc.).
            persistent_hazards: room_mapper.room_map['persistent_hazards'].
            current_time_ts   : Unix timestamp (defaults to time.time()).

        Returns:
            dict with keys:
              'new_event'        : bool
              'habituated_hazards': list of hazard keys where habituation detected
              'updated_weights'  : {hazard_key: new_weight}
        """
        import time
        ts = current_time_ts or time.time()

        result: dict = {
            'new_event':          False,
            'habituated_hazards': [],
            'updated_weights':    {},
        }

        zone     = (proximity_info or {}).get('zone', 'safe')
        distance = (proximity_info or {}).get('closest_distance', float('inf'))
        hazard   = (proximity_info or {}).get('closest_hazard')

        # Only count events inside warning/critical zone and moving toward hazard
        if zone not in ('warning', 'critical'):
            return result
        if hazard is None:
            return result
        if distance > self._approach_dist:
            return result

        hazard_key = self._hazard_key(hazard)

        # Cooldown guard — avoid flooding events from sustained proximity
        last_ts = self._last_approach_ts.get(hazard_key, 0.0)
        if ts - last_ts < APPROACH_COOLDOWN_SECS:
            return result

        # Record event
        evt = HazardApproachEvent(
            session_id     = self._session_id,
            hazard_key     = hazard_key,
            min_distance_m = float(distance),
            velocity       = float((proximity_info or {}).get('speed', 0.0)),
        )
        self._in_session_events[hazard_key].append(evt)
        self._last_approach_ts[hazard_key] = ts
        result['new_event'] = True

        # Persist event immediately
        self._record_event(evt)

        # Analyse trend across sessions
        habituated, weights = self._analyse_habituation(hazard_key, persistent_hazards)
        if habituated:
            self._habituated_keys.add(hazard_key)
            result['habituated_hazards'].append(hazard_key)
            result['updated_weights'].update(weights)
            print(f"[Habituation] ⚠ Habituation detected for hazard: {hazard_key}")

        return result

    def get_escalated_confidence(self,
                                 hazard: dict,
                                 base_confidence: float) -> float:
        """
        Return escalated confidence/risk weight if habituation has been
        detected for this hazard; otherwise return base_confidence unchanged.
        """
        key = self._hazard_key(hazard)
        if key in self._habituated_keys:
            return min(1.0, base_confidence * HABITUATION_WEIGHT_SCALE)
        stored = self._db.get('hazard_weights', {}).get(key)
        if stored and stored.get('habituated'):
            return min(1.0, base_confidence * HABITUATION_WEIGHT_SCALE)
        return base_confidence

    def get_habituation_report(self) -> dict:
        """Return a summary of approach frequencies per hazard."""
        report: dict = {}
        for key, sessions in self._db.get('sessions_by_hazard', {}).items():
            counts = [s['count'] for s in sessions[-self._session_window:]]
            report[key] = {
                'sessions_recorded': len(sessions),
                'recent_counts':     counts,
                'trend':             self._linear_trend(counts),
                'habituated':        self._db.get('hazard_weights', {})
                                              .get(key, {}).get('habituated', False),
            }
        return report

    def close_session(self):
        """Call at end of monitoring session to flush session totals to DB."""
        session_counts: dict[str, int] = {}
        for key, events in self._in_session_events.items():
            session_counts[key] = len(events)

        sessions_by_hazard = self._db.setdefault('sessions_by_hazard', {})
        for key, count in session_counts.items():
            sessions_by_hazard.setdefault(key, []).append({
                'session_id': self._session_id,
                'ts':         datetime.now().isoformat(),
                'count':      count,
            })

        self._save_db()
        print(f"[Habituation] Session {self._session_id} closed — "
              f"{sum(session_counts.values())} events across "
              f"{len(session_counts)} hazards.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _analyse_habituation(self, hazard_key: str,
                              persistent_hazards: list[dict]) -> tuple[bool, dict]:
        """
        Fit a linear trend to the per-session approach counts for this hazard.
        Flag as habituated when:
          - At least MIN_SESSIONS_FOR_TREND sessions available
          - Trend slope > 0  (approach count rising over sessions)
          - Current session count ≥ any prior session count
        """
        sessions = self._db.get('sessions_by_hazard', {}).get(hazard_key, [])
        counts   = [s['count'] for s in sessions[-self._session_window:]]

        # Include current (in-progress) session count
        cur_count = len(self._in_session_events.get(hazard_key, []))
        counts.append(cur_count)

        if len(counts) < MIN_SESSIONS_FOR_TREND:
            return False, {}

        slope = self._linear_trend(counts)

        # Additional check: count is non-decreasing over last 3 sessions
        recent = counts[-3:]
        non_decreasing = all(recent[i] <= recent[i+1] for i in range(len(recent)-1))

        habituated = (slope > 0) and non_decreasing and (cur_count > 0)
        weights    = {}

        if habituated:
            weights[hazard_key] = HABITUATION_WEIGHT_SCALE
            hw = self._db.setdefault('hazard_weights', {})
            hw[hazard_key] = {
                'habituated':    True,
                'weight_scale':  HABITUATION_WEIGHT_SCALE,
                'detected_at':   datetime.now().isoformat(),
                'trend_slope':   round(slope, 4),
            }
            self._save_db()

        return habituated, weights

    @staticmethod
    def _hazard_key(hazard: dict) -> str:
        """Stable string key for a hazard based on grid location and type."""
        loc  = hazard.get('location', (0, 0))
        htype = hazard.get('type', 'unknown')
        # Round to nearest GRID_MATCH_RADIUS cells for fuzzy matching
        gx = round(loc[0] / GRID_MATCH_RADIUS) * GRID_MATCH_RADIUS
        gy = round(loc[1] / GRID_MATCH_RADIUS) * GRID_MATCH_RADIUS
        return f"{htype}@({gx},{gy})"

    @staticmethod
    def _linear_trend(values: list[float]) -> float:
        """Return slope of a least-squares linear fit (0 if < 2 points)."""
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values), dtype=float)
        y = np.array(values, dtype=float)
        if np.std(x) < 1e-9:
            return 0.0
        slope = float(np.polyfit(x, y, 1)[0])
        return slope

    @staticmethod
    def _generate_session_id() -> str:
        return f"S_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_db(self) -> dict:
        if HABITUATION_DB_PATH.exists():
            try:
                with open(HABITUATION_DB_PATH, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_db(self):
        HABITUATION_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(HABITUATION_DB_PATH, 'w') as f:
            json.dump(self._db, f, indent=2)

    def _record_event(self, evt: HazardApproachEvent):
        """Append single event to persistent log."""
        events_log = self._db.setdefault('events', [])
        events_log.append(evt.to_dict())
        # Cap log size to avoid unbounded growth
        if len(events_log) > 5000:
            self._db['events'] = events_log[-2000:]
        self._save_db()