"""
Context-Aware Risk Assessment Module  (v2 — Patent Edition)
============================================================
New in v2:
  • Caregiver attention multiplier (Patent: supervised-vs-distracted modifier)
  • Developmental-stage-adaptive zone scaling (Patent: mobility-stage thresholds)
  • Hazard habituation weight escalation (Patent: cross-session behavioural loop)

Pool detection removed.  Fire is the only tracked hazard.
"""

import numpy as np
import cv2
import yaml
from enum import Enum
from datetime import datetime

# ---------------------------------------------------------------------------
# Patent modules — imported with graceful fallbacks
# ---------------------------------------------------------------------------
try:
    from caregiver_attention import CaregiverAttentionEstimator, AttentionState
    ATTENTION_AVAILABLE = True
except ImportError:
    ATTENTION_AVAILABLE = False
    print("[RiskAssessment] caregiver_attention not available — skipping")

try:
    from developmental_stage import DevelopmentalStageEstimator, DevelopmentalStage
    DEVSTAGE_AVAILABLE = True
except ImportError:
    DEVSTAGE_AVAILABLE = False
    print("[RiskAssessment] developmental_stage not available — skipping")

try:
    from hazard_habituation import HazardHabituationDetector
    HABITUATION_AVAILABLE = True
except ImportError:
    HABITUATION_AVAILABLE = False
    print("[RiskAssessment] hazard_habituation not available — skipping")


# ---------------------------------------------------------------------------
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
        if   6 <= hour < 12:  self.time_of_day = 'morning'
        elif 12 <= hour < 18: self.time_of_day = 'afternoon'
        elif 18 <= hour < 22: self.time_of_day = 'evening'
        else:                 self.time_of_day = 'night'

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
        if self.time_of_day == 'night' or self.lighting_conditions == 'dim':
            modifier += 0.10
        if self.supervision_present:
            modifier -= 0.20
        return max(-0.30, min(0.30, modifier))


# ---------------------------------------------------------------------------
class RiskAssessmentModule:
    def __init__(self, config_path='config/config.yaml'):
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            self.config = {}

        risk_cfg = self.config.get('risk', {})

        default_weights = {
            'proximity':           0.75,
            'temporal_pattern':    0.20,
            'environment_context': 0.05,
        }
        self.weights    = {**default_weights, **risk_cfg.get('weights', {})}

        default_thresh  = {'low': 0.20, 'medium': 0.45,
                           'high': 0.70, 'critical': 0.88}
        self.thresholds = {**default_thresh, **risk_cfg.get('thresholds', {})}

        self.env_context  = EnvironmentalContext()
        self.risk_history = []
        self.max_history  = 10

        # ----------------------------------------------------------------
        # Patent modules
        # ----------------------------------------------------------------
        self.attention_estimator = (CaregiverAttentionEstimator()
                                    if ATTENTION_AVAILABLE else None)

        self.dev_stage_estimator = (DevelopmentalStageEstimator()
                                    if DEVSTAGE_AVAILABLE else None)

        self.habituation_detector = (HazardHabituationDetector()
                                     if HABITUATION_AVAILABLE else None)

    # ------------------------------------------------------------------
    def calculate_risk_score(self, temporal_analysis, spatial_analysis,
                             environmental_context,
                             attention_result=None,
                             dev_stage_result=None,
                             habituation_result=None):

        temporal_risk = temporal_analysis.get('temporal_risk', 0.0)
        spatial_risk  = spatial_analysis.get('spatial_risk',  0.0)
        ctx_modifier  = self.env_context.get_context_risk_modifier()

        prox       = spatial_analysis.get('proximity_analysis') or {}
        zone       = prox.get('zone', 'safe')
        has_hazard = prox.get('closest_hazard') is not None

        if has_hazard and zone == 'critical':
            self.risk_history.clear()
            base = 0.95
        elif has_hazard and zone == 'warning' and spatial_risk > 0.4:
            self.risk_history.clear()
            base = 0.78
        else:
            if not has_hazard:
                self.risk_history.clear()
            base = (0.75 * spatial_risk
                    + 0.20 * temporal_risk
                    + 0.05 * abs(ctx_modifier))

        adjusted = max(0.0, min(1.0, base + ctx_modifier))

        # ----------------------------------------------------------------
        # Patent modifier 1: Caregiver attention multiplier
        # ----------------------------------------------------------------
        attention_note = ""
        if attention_result:
            multiplier  = attention_result.get('smoothed_multiplier', 1.0)
            state_name  = attention_result.get('state_name', 'unknown')
            # Only apply multiplier when there is real risk on the table
            if adjusted > 0.10:
                adjusted = min(1.0, adjusted * multiplier)
            attention_note = f"Caregiver:{state_name}(×{multiplier:.2f})"

        # ----------------------------------------------------------------
        # Patent modifier 2: Developmental-stage zone scaling already applied
        # by SpatialRiskAssessment; record it here for the explanation only.
        # ----------------------------------------------------------------
        stage_note = ""
        if dev_stage_result:
            stage_note = f"Stage:{dev_stage_result.get('stage_name','?')}"

        # ----------------------------------------------------------------
        # Patent modifier 3: Habituation escalation
        # ----------------------------------------------------------------
        habituation_note = ""
        if habituation_result and habituation_result.get('habituated_hazards'):
            # Escalate score when habituation is active
            adjusted = min(1.0, adjusted + 0.12)
            habituation_note = "Habituation:detected"

        # Temporal smoothing
        self.risk_history.append(adjusted)
        if len(self.risk_history) > self.max_history:
            self.risk_history.pop(0)

        if len(self.risk_history) > 1:
            alpha    = 0.3
            adjusted = (alpha * adjusted
                        + (1 - alpha) * np.mean(self.risk_history[:-1]))

        return float(adjusted), attention_note, stage_note, habituation_note

    # ------------------------------------------------------------------
    def classify_risk_level(self, risk_score):
        if   risk_score >= self.thresholds['critical']: return RiskLevel.CRITICAL, 'CRITICAL'
        elif risk_score >= self.thresholds['high']:     return RiskLevel.HIGH,     'HIGH'
        elif risk_score >= self.thresholds['medium']:   return RiskLevel.MEDIUM,   'MEDIUM'
        elif risk_score >= self.thresholds['low']:      return RiskLevel.LOW,      'LOW'
        else:                                           return RiskLevel.SAFE,     'SAFE'

    # ------------------------------------------------------------------
    def generate_risk_explanation(self, temporal_analysis, spatial_analysis,
                                  risk_score, risk_level_name,
                                  attention_note='', stage_note='',
                                  habituation_note=''):
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

        if attention_note:
            factors.append(attention_note)
        if stage_note:
            factors.append(stage_note)
        if habituation_note:
            factors.append(habituation_note)

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
                                  spatial_analysis, frame=None,
                                  persistent_hazards=None):
        """
        Full pipeline including all patent-novel modifiers.

        Args:
            persistent_hazards : list from room_mapper.room_map['persistent_hazards']
                                  — needed for habituation detector.
        """
        env_context = (self.env_context.analyze_context(detections, frame)
                       if frame is not None else {})

        # ----------------------------------------------------------------
        # Patent module 1: Caregiver attention
        # ----------------------------------------------------------------
        attention_result = None
        if self.attention_estimator and frame is not None:
            person_dets  = detections.get('person', []) + detections.get('adult', [])
            child_bbox   = (detections['child'][0]['bbox']
                            if detections.get('child') else None)
            attention_result = self.attention_estimator.estimate(
                frame, person_dets, child_bbox)

        # ----------------------------------------------------------------
        # Patent module 2: Developmental stage
        # ----------------------------------------------------------------
        dev_stage_result = None
        if self.dev_stage_estimator and detections.get('child'):
            child_bbox = detections['child'][0]['bbox']
            dev_stage_result = self.dev_stage_estimator.update(child_bbox)

        # ----------------------------------------------------------------
        # Patent module 3: Hazard habituation
        # ----------------------------------------------------------------
        habituation_result = None
        if (self.habituation_detector and
                detections.get('child') and
                persistent_hazards is not None):
            child_bbox = detections['child'][0]['bbox']
            habituation_result = self.habituation_detector.observe(
                child_bbox      = child_bbox,
                proximity_info  = spatial_analysis.get('proximity_analysis'),
                persistent_hazards = persistent_hazards,
            )

        # ----------------------------------------------------------------
        # Core risk score
        # ----------------------------------------------------------------
        risk_score, attn_note, stage_note, hab_note = self.calculate_risk_score(
            temporal_analysis   = temporal_analysis,
            spatial_analysis    = spatial_analysis,
            environmental_context = env_context,
            attention_result    = attention_result,
            dev_stage_result    = dev_stage_result,
            habituation_result  = habituation_result,
        )

        risk_level_enum, risk_level_name = self.classify_risk_level(risk_score)

        explanation = self.generate_risk_explanation(
            temporal_analysis, spatial_analysis,
            risk_score, risk_level_name,
            attention_note   = attn_note,
            stage_note       = stage_note,
            habituation_note = hab_note,
        )

        hazard_present = bool(detections.get('fire'))
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
            'risk_level':            risk_level_name,
            'should_alert':          should_alert,
            'alert_urgency':         alert_urgency,
            'explanation':           explanation,
            'environmental_context': env_context,
            'attention_state':       (attention_result or {}).get('state_name'),
            'caregiver_multiplier':  (attention_result or {}).get('smoothed_multiplier', 1.0),
            'developmental_stage':   (dev_stage_result  or {}).get('stage_name'),
            'habituation_active':    bool((habituation_result or {}).get('habituated_hazards')),
            'timestamp':             datetime.now().isoformat(),
        }

    # ------------------------------------------------------------------
    def get_recommended_actions(self, assessment):
        level = assessment['risk_level_enum']
        actions = []

        if level == RiskLevel.CRITICAL:
            actions = [
                "IMMEDIATE ACTION REQUIRED",
                "Alert all caregivers immediately",
                "Activate emergency response",
                "Sound local alarm",
                "Record incident for review",
            ]
        elif level == RiskLevel.HIGH:
            actions = [
                "Alert primary caregiver urgently",
                "Send push notification",
                "Monitor closely for next 30 seconds",
            ]
        elif level == RiskLevel.MEDIUM:
            actions = [
                "Send notification to caregiver",
                "Continue close monitoring",
                "Log event for review",
            ]
        elif level == RiskLevel.LOW:
            actions = ["Silent logging", "Continue monitoring"]
        else:
            actions = ["Normal monitoring"]

        # Append contextual actions from patent modules
        if assessment.get('attention_state') == 'distracted':
            actions.insert(0, "Caregiver appears distracted — escalate alert priority")
        if assessment.get('attention_state') == 'absent':
            actions.insert(0, "No caregiver detected — escalate immediately")
        if assessment.get('habituation_active'):
            actions.append("Habituation detected — consider physical barrier")
        if assessment.get('developmental_stage') in ('walking', 'running'):
            actions.append("Mobile child — increase zone monitoring frequency")

        return actions

    # ------------------------------------------------------------------
    def close_session(self):
        """Call at end of session to flush habituation data."""
        if self.habituation_detector:
            self.habituation_detector.close_session()
        if self.attention_estimator:
            self.attention_estimator.save_log()
        if self.dev_stage_estimator:
            self.dev_stage_estimator.save_log()