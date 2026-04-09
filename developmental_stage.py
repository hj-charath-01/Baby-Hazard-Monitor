from __future__ import annotations
import json
from collections import deque
from datetime import datetime
from enum import Enum
from pathlib import Path
import numpy as np


class DevelopmentalStage(Enum):
    UNKNOWN  = "unknown"
    LYING    = "lying"
    CRAWLING = "crawling"
    CRUISING = "cruising"
    WALKING  = "walking"
    RUNNING  = "running"


STAGE_ZONE_SCALES: dict[DevelopmentalStage, tuple[float, float, float]] = {
    DevelopmentalStage.UNKNOWN:  (1.00, 1.00, 1.00),
    DevelopmentalStage.LYING:    (0.60, 0.60, 0.70),
    DevelopmentalStage.CRAWLING: (0.75, 0.80, 0.85),
    DevelopmentalStage.CRUISING: (0.90, 0.95, 1.00),
    DevelopmentalStage.WALKING:  (1.00, 1.00, 1.00),
    DevelopmentalStage.RUNNING:  (1.30, 1.25, 1.15),
}

STAGE_RISK_NOTE: dict[DevelopmentalStage, str] = {
    DevelopmentalStage.UNKNOWN:  "Stage unknown — using default thresholds",
    DevelopmentalStage.LYING:    "Child appears to be lying — zones contracted",
    DevelopmentalStage.CRAWLING: "Child crawling — zones moderately contracted",
    DevelopmentalStage.CRUISING: "Child cruising furniture — near-baseline zones",
    DevelopmentalStage.WALKING:  "Child walking — baseline zones",
    DevelopmentalStage.RUNNING:  "Child running — zones expanded for speed",
}


class DevelopmentalStageEstimator:
    _AR_LYING    = 0.80
    _AR_CRAWLING = 1.60
    _AR_CRUISING = 2.20

    _VEL_SLOW   = 0.015
    _VEL_MEDIUM = 0.035
    _VEL_FAST   = 0.075

    def __init__(self, history_frames=45, frame_wh=(1280, 720)):
        self._history_len    = history_frames
        self._diag           = float(np.hypot(*frame_wh))
        self._ar_history:    deque[float] = deque(maxlen=history_frames)
        self._vel_history:   deque[float] = deque(maxlen=history_frames)
        self._stage_history: deque[DevelopmentalStage] = deque(maxlen=history_frames)
        self._prev_center: tuple[float, float] | None = None
        self._session_log: list[dict] = []

    def update(self, child_bbox):
        x1, y1, x2, y2 = child_bbox
        w  = max(1, x2 - x1)
        h  = max(1, y2 - y1)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        aspect_ratio = h / w
        self._ar_history.append(aspect_ratio)

        if self._prev_center is not None:
            dx  = (cx - self._prev_center[0]) / self._diag
            dy  = (cy - self._prev_center[1]) / self._diag
            vel = float(np.hypot(dx, dy))
        else:
            vel = 0.0
        self._vel_history.append(vel)
        self._prev_center = (cx, cy)

        stage        = self._classify_stage()
        self._stage_history.append(stage)
        stable_stage = self._stable_stage()
        scales       = STAGE_ZONE_SCALES[stable_stage]

        result = {
            'stage':               stable_stage,
            'stage_name':          stable_stage.value,
            'instant_stage':       stage.value,
            'aspect_ratio':        round(aspect_ratio, 3),
            'velocity_norm':       round(vel, 4),
            'zone_scale_critical': scales[0],
            'zone_scale_warning':  scales[1],
            'zone_scale_safe':     scales[2],
            'note':                STAGE_RISK_NOTE[stable_stage],
        }

        self._session_log.append({
            'ts':    datetime.now().isoformat(),
            'stage': stable_stage.value,
            'ar':    round(aspect_ratio, 3),
            'vel':   round(vel, 4),
        })
        if len(self._session_log) > 1000:
            self._session_log = self._session_log[-500:]
        return result

    def get_adapted_zones(self, base_zones):
        stage  = self._stable_stage()
        scales = STAGE_ZONE_SCALES[stage]
        return {
            'critical': base_zones.get('critical', 1.0) * scales[0],
            'warning':  base_zones.get('warning',  2.5) * scales[1],
            'safe':     base_zones.get('safe',     5.0) * scales[2],
        }

    def _classify_stage(self):
        if not self._ar_history:
            return DevelopmentalStage.UNKNOWN
        ar  = (float(np.median(list(self._ar_history)[-10:]))
               if len(self._ar_history) >= 10 else self._ar_history[-1])
        vel = (float(np.mean(list(self._vel_history)[-10:]))
               if len(self._vel_history) >= 10
               else (self._vel_history[-1] if self._vel_history else 0.0))

        if ar < self._AR_LYING and vel < self._VEL_SLOW:
            return DevelopmentalStage.LYING
        if ar < self._AR_CRAWLING:
            return DevelopmentalStage.CRAWLING
        if ar < self._AR_CRUISING:
            return DevelopmentalStage.CRUISING
        if vel >= self._VEL_FAST:
            return DevelopmentalStage.RUNNING
        if vel >= self._VEL_MEDIUM:
            return DevelopmentalStage.WALKING
        return DevelopmentalStage.CRUISING

    def _stable_stage(self):
        if not self._stage_history:
            return DevelopmentalStage.UNKNOWN
        from collections import Counter
        counts = Counter(self._stage_history)
        return counts.most_common(1)[0][0]

    def get_statistics(self):
        if not self._session_log:
            return {}
        from collections import Counter
        stages = [e['stage'] for e in self._session_log]
        counts = Counter(stages)
        total  = len(stages)
        return {
            'total_frames':       total,
            'stage_distribution': {k: round(v/total, 3) for k, v in counts.items()},
            'dominant_stage':     counts.most_common(1)[0][0],
        }

    def save_log(self, path='logs/developmental_stage.json'):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump({'log': self._session_log[-200:],
                       'stats': self.get_statistics()}, f, indent=2)

    def reset(self):
        self._ar_history.clear()
        self._vel_history.clear()
        self._stage_history.clear()
        self._prev_center = None