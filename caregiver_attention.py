import cv2
import numpy as np
from collections import deque
from datetime import datetime
from enum import Enum
from pathlib import Path
import json


class AttentionState(Enum):
    WATCHING    = "watching"     # Facing camera / child area, yaw < 30°
    PERIPHERAL  = "peripheral"   # Partial attention, yaw 30–60°
    DISTRACTED  = "distracted"   # Looking away, yaw > 60°
    ABSENT      = "absent"       # No caregiver detected


# Risk multipliers indexed by AttentionState
ATTENTION_RISK_MULTIPLIER = {
    AttentionState.WATCHING:   0.60,   # Supervised → reduce risk
    AttentionState.PERIPHERAL: 1.00,   # Neutral
    AttentionState.DISTRACTED: 1.45,   # Unsupervised → escalate
    AttentionState.ABSENT:     1.60,   # No adult present → highest escalation
}


class HeadPoseEstimator:
    """
    Lightweight head-pose estimator using OpenCV face detection + landmark
    geometry.  Falls back gracefully when no face is detected.

    The yaw angle is approximated from the ratio of left/right facial
    half-widths (asymmetry heuristic), which is robust enough for the
    binary WATCHING / DISTRACTED classification required here.
    """

    # 3-D model points of a generic face (mm), canonical orientation
    _MODEL_POINTS = np.array([
        (  0.0,   0.0,    0.0),   # Nose tip
        (  0.0, -63.6,  -12.5),   # Chin
        (-43.3,  32.7,  -26.0),   # Left eye outer
        ( 43.3,  32.7,  -26.0),   # Right eye outer
        (-28.9, -28.9,  -24.1),   # Left mouth corner
        ( 28.9, -28.9,  -24.1),   # Right mouth corner
    ], dtype=np.float64)

    def __init__(self):
        face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        eye_cascade_path  = cv2.data.haarcascades + 'haarcascade_eye.xml'
        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
        self.eye_cascade  = cv2.CascadeClassifier(eye_cascade_path)

        self._mp_face = None
        try:
            import mediapipe as mp
            solutions = getattr(mp, 'solutions', None)
            if solutions is None:
                raise AttributeError("mediapipe.solutions not available in this version")
            self._mp_face = solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=3,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        except (ImportError, AttributeError, Exception):
            # Fall back to OpenCV-only head-pose estimation
            self._mp_face = None

    def estimate_yaw(self, frame, face_bbox):
        """
        Estimate head yaw angle (°) for a detected face region.

        Returns:
            yaw_deg  : float, signed yaw in degrees (positive = right turn)
            confidence : float [0,1]
        """
        x, y, w, h = face_bbox
        face_roi = frame[y:y+h, x:x+w]
        if face_roi.size == 0:
            return 0.0, 0.0

        # MediaPipe path 
        if self._mp_face is not None:
            try:
                rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                res = self._mp_face.process(rgb)
                if res.multi_face_landmarks:
                    lm = res.multi_face_landmarks[0].landmark
                    yaw_deg = self._yaw_from_landmarks_mp(lm, face_roi.shape)
                    return yaw_deg, 0.85
            except Exception:
                pass

        # OpenCV fallback: eye-symmetry heuristic 
        gray     = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        eyes     = self.eye_cascade.detectMultiScale(gray, 1.1, 5, minSize=(15, 15))
        if len(eyes) >= 2:
            eyes_sorted = sorted(eyes, key=lambda e: e[0])
            lx = eyes_sorted[0][0] + eyes_sorted[0][2] // 2
            rx = eyes_sorted[1][0] + eyes_sorted[1][2] // 2
            face_cx  = w // 2
            left_gap  = lx
            right_gap = w - rx
            # Asymmetry → approximate yaw
            asymmetry = (left_gap - right_gap) / max(w, 1)
            yaw_deg   = asymmetry * 90.0   # empirical scale
            return float(yaw_deg), 0.60

        # No eyes detected → assume turned away
        return 75.0, 0.30

    @staticmethod
    def _yaw_from_landmarks_mp(landmarks, shape):
        """Approximate yaw from MediaPipe face mesh nose/cheek landmarks."""
        h, w = shape[:2]
        # Nose tip (1), left cheek (234), right cheek (454)
        nose  = np.array([landmarks[1].x * w,  landmarks[1].y  * h])
        l_chk = np.array([landmarks[234].x * w, landmarks[234].y * h])
        r_chk = np.array([landmarks[454].x * w, landmarks[454].y * h])
        face_width  = np.linalg.norm(r_chk - l_chk)
        nose_offset = nose[0] - (l_chk[0] + r_chk[0]) / 2
        if face_width < 1:
            return 0.0
        yaw_deg = (nose_offset / face_width) * 90.0
        return float(yaw_deg)


class CaregiverAttentionEstimator:
    """
    Main caregiver attention estimator.

    Usage:
        estimator = CaregiverAttentionEstimator()
        result = estimator.estimate(frame, person_detections, child_bbox)
        risk_multiplier = result['risk_multiplier']
    """

    # Minimum bounding-box area (px²) to be considered an adult, not a child.
    # Adults typically occupy > 15 000 px² in a 1280×720 frame.
    ADULT_MIN_AREA = 12_000

    # Yaw thresholds (degrees, absolute value)
    YAW_WATCHING   = 30.0
    YAW_PERIPHERAL = 65.0

    def __init__(self, history_len: int = 30):
        self.pose_estimator = HeadPoseEstimator()
        self._state_history: deque = deque(maxlen=history_len)
        self._attention_log: list  = []

    # ------------------------------------------------------------------
    def estimate(self, frame, person_detections: list, child_bbox=None) -> dict:
        """
        Estimate caregiver attention and return a risk multiplier.

        Args:
            frame            : BGR video frame.
            person_detections: List of detection dicts with keys
                               {'bbox', 'confidence', 'class_name'}.
                               Both 'person' and 'adult' class names accepted.
            child_bbox       : [x1,y1,x2,y2] of the primary child, used to
                               exclude child-sized bounding boxes.

        Returns:
            {
              'state'           : AttentionState,
              'state_name'      : str,
              'yaw_degrees'     : float,
              'confidence'      : float,
              'risk_multiplier' : float,
              'smoothed_multiplier': float,
              'num_caregivers'  : int,
            }
        """
        adults = self._filter_adults(person_detections, child_bbox)

        if not adults:
            state, yaw, conf = AttentionState.ABSENT, 0.0, 1.0
        else:
            # Use the largest (closest) adult
            primary = max(adults, key=lambda d: (d['bbox'][2]-d['bbox'][0]) *
                                                (d['bbox'][3]-d['bbox'][1]))
            bbox    = primary['bbox']
            x1,y1,x2,y2 = bbox
            face_bbox = self._locate_head_region(x1, y1, x2, y2)

            yaw, conf = self.pose_estimator.estimate_yaw(frame, face_bbox)
            state     = self._classify_state(abs(yaw))

        raw_multiplier  = ATTENTION_RISK_MULTIPLIER[state]
        self._state_history.append(raw_multiplier)

        # Exponential smoothing so a single glance doesn't spike risk
        alpha   = 0.25
        history = list(self._state_history)
        smoothed = history[-1]
        for v in reversed(history[:-1]):
            smoothed = alpha * smoothed + (1 - alpha) * v

        result = {
            'state':               state,
            'state_name':          state.value,
            'yaw_degrees':         float(yaw),
            'confidence':          float(conf),
            'risk_multiplier':     float(raw_multiplier),
            'smoothed_multiplier': float(smoothed),
            'num_caregivers':      len(adults),
        }

        self._attention_log.append({
            'ts':    datetime.now().isoformat(),
            'state': state.value,
            'yaw':   round(float(yaw), 1),
        })
        if len(self._attention_log) > 500:
            self._attention_log = self._attention_log[-200:]

        return result

    # ------------------------------------------------------------------
    def _filter_adults(self, detections: list, child_bbox) -> list:
        """Return detections that look like adults (by size and class)."""
        accepted_classes = {'person', 'adult', 'human'}
        adults = []
        for d in detections:
            if d.get('class_name', '').lower() not in accepted_classes:
                continue
            x1,y1,x2,y2 = d['bbox']
            area = (x2-x1) * (y2-y1)
            if area < self.ADULT_MIN_AREA:
                continue
            # Skip if IoU > 0.3 with child bbox (same person)
            if child_bbox and self._iou(d['bbox'], child_bbox) > 0.30:
                continue
            adults.append(d)
        return adults

    @staticmethod
    def _locate_head_region(x1, y1, x2, y2):
        """Approximate head region = top 25% of person bounding box."""
        w, h = x2-x1, y2-y1
        head_h = max(30, int(h * 0.25))
        return (x1, y1, w, head_h)

    def _classify_state(self, yaw_abs: float) -> AttentionState:
        if   yaw_abs <= self.YAW_WATCHING:   return AttentionState.WATCHING
        elif yaw_abs <= self.YAW_PERIPHERAL: return AttentionState.PERIPHERAL
        else:                                return AttentionState.DISTRACTED

    @staticmethod
    def _iou(a, b) -> float:
        ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        if not inter: return 0.0
        ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / ua if ua else 0.0

    # ------------------------------------------------------------------
    def get_attention_statistics(self) -> dict:
        if not self._attention_log:
            return {}
        from collections import Counter
        states = [e['state'] for e in self._attention_log]
        counts = Counter(states)
        total  = len(states)
        return {
            'total_samples':     total,
            'state_distribution': {k: v/total for k, v in counts.items()},
            'distraction_rate':  counts.get('distracted', 0) / total,
            'absence_rate':      counts.get('absent',     0) / total,
        }

    def save_log(self, path='logs/caregiver_attention.json'):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump({'log': self._attention_log,
                       'stats': self.get_attention_statistics()}, f, indent=2)
        print(f"[CaregiverAttention] Log saved → {path}")