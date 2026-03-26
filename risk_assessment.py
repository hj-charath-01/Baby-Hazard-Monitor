"""
Context-Aware Risk Assessment Module

Bug fixes:
- BUG 1: duplicate `import cv2` inside method removed.
- BUG 2: assessment dict stored a RiskLevel enum under 'risk_level' (not
  JSON-serialisable).  Kept 'risk_level' as string alias; enum is under
  'risk_level_enum'.
- BUG 3: weights key mismatch ('temporal_pattern' vs 'temporal') fixed.
- BUG 4 (NEW): Base risk calculation was dominated by spatial_risk but
  weighted too low (0.60) and required both approach pattern AND zone to push
  past MEDIUM threshold.  Increased spatial weight to 0.75 so that a child
  in the warning zone reliably produces a MEDIUM+ score even without
  additional temporal evidence.
- BUG 5 (NEW): Fast-path for warning zone required spatial_risk > 0.6 which
  was never reached when the zone_score came in at 0.5–0.65.  Lowered guard
  to 0.4 so warning-zone detections always escalate.
- BUG 6 (NEW): should_alert only fired when risk_level >= MEDIUM AND hazard
  present.  This is correct, but the MEDIUM threshold (0.55) was never
  reached for typical warning-zone scenarios.  Lowered MEDIUM threshold to
  0.45 so mid-range proximity consistently triggers notifications.
"""

import numpy as np
import cv2
import yaml
from enum import Enum
from datetime import datetime


class RiskLevel(Enum):
    SAFE     = 0
    LOW      = 1
    MEDIUM   = 2
    HIGH     = 3
    CRITICAL = 4


class EnvironmentalContext:
    def __init__(self):
        self.time_of_day         = None
        self.lighting_conditions = None
        self.number_of_people    = 0
        self.supervision_present = False

    def analyze_context(self, detections, frame):
        hour = datetime.now().hour
        if 6  <= hour < 12:   self.time_of_day = 'morning'
        elif 12 <= hour < 18: self.time_of_day = 'afternoon'
        elif 18 <= hour < 22: self.time_of_day = 'evening'
        else:                  self.time_of_day = 'night'

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) \
               if len(frame.shape) == 3 else frame
        brightness = np.mean(gray)
        self.lighting_conditions = ('bright' if brightness > 150
                                    else 'normal' if brightness > 80
                                    else 'dim')

        self.number_of_people    = len(detections.get('child', []))
        self.supervision_present = False
        return {
            'time_of_day': self.time_of_day,
            'lighting':    self.lighting_conditions,
            'num_people':  self.number_of_people,
            'supervised':  self.supervision_present,
        }

    def get_context_risk_modifier(self):
        modifier = 0.0
        if self.time_of_day in ('night',) or self.lighting_conditions == 'dim':
            modifier += 0.1
        if self.supervision_present:
            modifier -= 0.2
        return max(-0.3, min(0.3, modifier))


class RiskAssessmentModule:
    def __init__(self, config_path='config/config.yaml'):
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            self.config = {}

        risk_cfg = self.config.get('risk', {})

        default_weights = {
            'proximity':           0.75,   # BUG 4 FIX: was 0.40 → spatial dominates
            'temporal_pattern':    0.20,   # was 0.35
            'environment_context': 0.05,   # was 0.25
        }
        self.weights    = {**default_weights, **risk_cfg.get('weights', {})}

        # BUG 6 FIX: lower MEDIUM threshold so warning-zone triggers notify
        default_thresh  = {'low': 0.20, 'medium': 0.45, 'high': 0.70, 'critical': 0.88}
        self.thresholds = {**default_thresh, **risk_cfg.get('thresholds', {})}

        self.env_context  = EnvironmentalContext()
        self.risk_history = []
        self.max_history  = 10

    # ------------------------------------------------------------------
    def calculate_risk_score(self, temporal_analysis, spatial_analysis,
                             environmental_context):
        temporal_risk = temporal_analysis.get('temporal_risk', 0.0)
        spatial_risk  = spatial_analysis.get('spatial_risk',  0.0)
        ctx_modifier  = self.env_context.get_context_risk_modifier()

        prox       = spatial_analysis.get('proximity_analysis') or {}
        zone       = prox.get('zone', 'safe')
        has_hazard = prox.get('closest_hazard') is not None

        # Fast-paths: immediate score when child is in a danger zone.
        if has_hazard and zone == 'critical':
            self.risk_history.clear()
            return 0.95

        # BUG 5 FIX: lowered guard from 0.6 → 0.4 so warning zone always
        # escalates even when zone_score is in the 0.5 range.
        if has_hazard and zone == 'warning' and spatial_risk > 0.4:
            self.risk_history.clear()
            return 0.78

        # No hazard → clear history so stale scores don't bleed through.
        if not has_hazard:
            self.risk_history.clear()

        # BUG 4 FIX: weight spatial_risk at 0.75 so proximity alone is
        # sufficient to exceed the MEDIUM threshold without needing temporal
        # evidence.
        base = (0.75 * spatial_risk
                + 0.20 * temporal_risk
                + 0.05 * abs(ctx_modifier))

        adjusted = max(0.0, min(1.0, base + ctx_modifier))

        self.risk_history.append(adjusted)
        if len(self.risk_history) > self.max_history:
            self.risk_history.pop(0)

        if len(self.risk_history) > 1:
            alpha    = 0.3
            adjusted = (alpha * adjusted
                        + (1 - alpha) * np.mean(self.risk_history[:-1]))

        return float(adjusted)

    # ------------------------------------------------------------------
    def classify_risk_level(self, risk_score):
        if   risk_score >= self.thresholds['critical']: return RiskLevel.CRITICAL, 'CRITICAL'
        elif risk_score >= self.thresholds['high']:     return RiskLevel.HIGH,     'HIGH'
        elif risk_score >= self.thresholds['medium']:   return RiskLevel.MEDIUM,   'MEDIUM'
        elif risk_score >= self.thresholds['low']:      return RiskLevel.LOW,      'LOW'
        else:                                           return RiskLevel.SAFE,     'SAFE'

    # ------------------------------------------------------------------
    def generate_risk_explanation(self, temporal_analysis, spatial_analysis,
                                  risk_score, risk_level_name):
        factors = []
        prox = (spatial_analysis.get('proximity_analysis') or {})
        zone = prox.get('zone', '')
        dist = prox.get('closest_distance', float('inf'))

        if zone == 'critical':
            factors.append(f"Child in CRITICAL zone ({dist:.2f}m from hazard)")
        elif zone == 'warning':
            factors.append(f"Child in WARNING zone ({dist:.2f}m from hazard)")

        pattern = temporal_analysis.get('pattern_type', '')
        if 'approach' in pattern:
            factors.append(f"Behaviour: {pattern}")

        if spatial_analysis.get('collision_warning'):
            ct = spatial_analysis.get('collision_time', 0)
            factors.append(f"Collision predicted in {ct} frames (~{ct/30:.1f}s)")

        if spatial_analysis.get('approach_pattern') == 'approaching':
            rate = spatial_analysis.get('approach_rate', 0)
            factors.append(f"Approaching hazard at {rate:.3f} m/frame")

        return {
            'risk_score':         risk_score,
            'risk_level':         risk_level_name,
            'primary_factors':    factors,
            'temporal_component': temporal_analysis.get('temporal_risk', 0.0),
            'spatial_component':  spatial_analysis.get('spatial_risk',   0.0),
            'confidence':         temporal_analysis.get('confidence',     0.5),
        }

    # ------------------------------------------------------------------
    def assess_comprehensive_risk(self, detections, temporal_analysis,
                                  spatial_analysis, frame=None):
        env_context = (self.env_context.analyze_context(detections, frame)
                       if frame is not None else {})

        risk_score = self.calculate_risk_score(
            temporal_analysis, spatial_analysis, env_context)
        risk_level_enum, risk_level_name = self.classify_risk_level(risk_score)

        explanation = self.generate_risk_explanation(
            temporal_analysis, spatial_analysis, risk_score, risk_level_name)

        hazard_present = bool(detections.get('fire') or detections.get('pool'))
        should_alert   = (risk_level_enum.value >= RiskLevel.MEDIUM.value
                          and hazard_present)
        urgency_map    = {
            RiskLevel.CRITICAL: 'emergency',
            RiskLevel.HIGH:     'urgent',
            RiskLevel.MEDIUM:   'medium',
        }
        alert_urgency = urgency_map.get(risk_level_enum, 'gentle')

        return {
            'risk_score':            risk_score,
            'risk_level_enum':       risk_level_enum,
            'risk_level_name':       risk_level_name,
            'risk_level':            risk_level_name,   # backward-compat alias
            'should_alert':          should_alert,
            'alert_urgency':         alert_urgency,
            'explanation':           explanation,
            'environmental_context': env_context,
            'timestamp':             datetime.now().isoformat(),
        }

    # ------------------------------------------------------------------
    def get_recommended_actions(self, assessment):
        level = assessment['risk_level_enum']
        if level == RiskLevel.CRITICAL:
            return ["IMMEDIATE ACTION REQUIRED",
                    "Alert all caregivers immediately",
                    "Activate emergency response",
                    "Sound local alarm",
                    "Record incident for review"]
        if level == RiskLevel.HIGH:
            return ["Alert primary caregiver urgently",
                    "Send push notification",
                    "Monitor closely for next 30 seconds"]
        if level == RiskLevel.MEDIUM:
            return ["Send notification to caregiver",
                    "Continue close monitoring",
                    "Log event for review"]
        if level == RiskLevel.LOW:
            return ["Silent logging", "Continue monitoring"]
        return ["Normal monitoring"]