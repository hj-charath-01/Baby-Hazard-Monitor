"""
Live Video Detection — Baby Hazard Monitoring
==========================================
Standalone script for real-time hazard detection from webcam or video file.

Usage
-----
  python live_detection.py                          # webcam
  python live_detection.py --camera 1               # second camera
  python live_detection.py --video path/to/file.mp4 # video file
  python live_detection.py --video in.mp4 --output out.mp4

Keyboard controls
-----------------
  q  — quit
  p  — pause / resume
  s  — save screenshot
  r  — reset temporal / spatial history
  +  — raise confidence threshold by 0.05
  -  — lower confidence threshold by 0.05

FIXES
-----
- Demo child + fire positions updated so they move toward each other,
  triggering genuine CRITICAL/HIGH alerts.
- HUD completely redesigned: clean dark panel, coloured risk bar, crisp
  typography, no Unicode symbols that break OpenCV on some builds.
- draw_proximity_zones draws child-centred expanding rectangles instead of
  fixed circles; critical ring pulses red.
- Suppressed noisy per-frame print output; status printed every 60 frames.
- pixel_to_meter_ratio aligned with spatial_analysis.py (40 px/m).
"""

import argparse
import sys
import time
import threading
import queue
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARNING] ultralytics not installed — DEMO mode active.")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent))

try:
    from temporal_reasoning import TemporalPatternAnalyzer
    TEMPORAL_AVAILABLE = True
except Exception as e:
    TEMPORAL_AVAILABLE = False

try:
    from spatial_analysis import SpatialRiskAssessment
    SPATIAL_AVAILABLE = True
except Exception as e:
    SPATIAL_AVAILABLE = False

try:
    from risk_assessment import RiskAssessmentModule, RiskLevel
    RISK_AVAILABLE = True
except Exception as e:
    RISK_AVAILABLE = False

try:
    from alert_manager import AlertManager, AlertLevel
    ALERT_AVAILABLE = True
except Exception as e:
    ALERT_AVAILABLE = False

try:
    from adaptive_room_mapper import AdaptiveRoomMapper
    MAPPER_AVAILABLE = True
except Exception as e:
    MAPPER_AVAILABLE = False
    print(f"[WARNING] adaptive_room_mapper unavailable: {e}")


# ===========================================================================
# Colour palette  (BGR)
# ===========================================================================
_C = {
    'child':    (50,  220,  50),
    'fire':     (30,  80,  255),
    'pool':     (220, 100,   0),
    'smoke':    (170, 170, 170),
    'water':    (200, 160,   0),
    'default':  (200, 200, 200),
}

_RISK_BGR = {
    'SAFE':     (60,  200,  60),
    'LOW':      (60,  230, 230),
    'MEDIUM':   (30,  165, 255),
    'HIGH':     (20,   80, 255),
    'CRITICAL': (20,   20, 255),
}

_RISK_ACCENT = {          # brighter version for text
    'SAFE':     (100, 255, 100),
    'LOW':      (100, 255, 255),
    'MEDIUM':   (80,  200, 255),
    'HIGH':     (80,  120, 255),
    'CRITICAL': (80,   80, 255),
}

FONT      = cv2.FONT_HERSHEY_SIMPLEX
FONT_MONO = cv2.FONT_HERSHEY_DUPLEX


# ===========================================================================
# Detector wrapper
# ===========================================================================
class HazardDetector:
    CHILD_CLASSES = {'child', 'person', 'baby', 'toddler'}
    FIRE_CLASSES  = {'fire', 'flame', 'smoke'}
    POOL_CLASSES  = {'pool', 'water', 'drowning', 'swimming'}

    INFER_WIDTH  = 416
    HAZARD_CONF  = 0.65

    def __init__(self, model_dir='models', conf=0.45, iou=0.45):
        self.conf       = conf
        self.iou        = iou
        self.models     = {}
        self._demo_t    = 0
        self._use_demo  = False

        if YOLO_AVAILABLE:
            self._load_models(Path(model_dir))
        else:
            self._use_demo = True

    def _load_models(self, model_dir):
        candidates = {
            'child': ['child_detector.pt', 'child.pt'],
            'fire':  ['fire_detector.pt',  'fire.pt'],
            'pool':  ['pool_detector.pt',  'pool.pt'],
        }
        loaded = False
        for task, names in candidates.items():
            for name in names:
                p = model_dir / name
                if p.exists():
                    try:
                        self.models[task] = YOLO(str(p))
                        print(f"[INFO] Loaded {task}: {p}")
                        loaded = True
                        break
                    except Exception as e:
                        print(f"[WARN] {p}: {e}")
        if not loaded:
            try:
                self.models['general'] = YOLO('yolov8n.pt')
                loaded = True
            except Exception:
                pass
        if not loaded:
            self._use_demo = True

    def detect(self, frame):
        """Returns (detections_dict, frame).
        In demo mode the frame is replaced with a synthetic background."""
        if self._use_demo:
            result = self._demo_detections(frame)   # modifies frame in-place
            return result, frame
        result = self._detect_real(frame)
        return result, frame

    def _detect_real(self, frame):
        result = {'child': [], 'fire': [], 'pool': []}
        h, w   = frame.shape[:2]
        scale  = self.INFER_WIDTH / w
        ih     = int(h * scale)
        small  = cv2.resize(frame, (self.INFER_WIDTH, ih),
                            interpolation=cv2.INTER_LINEAR)

        BUCKETS = {'child': 'child', 'fire': 'fire', 'pool': 'pool', 'general': None}
        ALLOWED = {'child': self.CHILD_CLASSES,
                   'fire':  self.FIRE_CLASSES,
                   'pool':  self.POOL_CLASSES}

        for task, model in self.models.items():
            bucket   = BUCKETS.get(task)
            min_conf = self.conf if bucket == 'child' else self.HAZARD_CONF
            try:
                preds = model.predict(small, conf=min_conf, iou=self.iou,
                                      verbose=False)[0]
            except Exception as e:
                continue
            if preds.boxes is None:
                continue

            boxes   = preds.boxes.xyxy.cpu().numpy()
            confs   = preds.boxes.conf.cpu().numpy()
            cls_ids = preds.boxes.cls.cpu().numpy().astype(int)
            names   = preds.names

            for box, conf, cls_id in zip(boxes, confs, cls_ids):
                cn = names.get(cls_id, 'unknown').lower()
                x1 = max(0,     int(box[0] / scale))
                y1 = max(0,     int(box[1] / scale))
                x2 = min(w - 1, int(box[2] / scale))
                y2 = min(h - 1, int(box[3] / scale))
                det = dict(bbox=[x1,y1,x2,y2], confidence=float(conf),
                           class_name=cn, class_id=int(cls_id),
                           center=((x1+x2)//2,(y1+y2)//2),
                           area=(x2-x1)*(y2-y1))
                if bucket:
                    if cn in ALLOWED.get(bucket, set()):
                        result[bucket].append(det)
                else:
                    if cn in self.CHILD_CLASSES:  result['child'].append(det)
                    elif cn in self.FIRE_CLASSES:  result['fire'].append(det)
                    elif cn in self.POOL_CLASSES:  result['pool'].append(det)

        # Person-misclassification filter — only removes hazard boxes that
        # substantially OVERLAP a person bbox (IoU > 0.35).  The previous
        # 50 px gap / center-inside checks were too aggressive: they removed
        # real fire detections the moment a child was anywhere nearby, which
        # is exactly when alerts matter most.
        def _iou(a, b):
            ix1,iy1 = max(a[0],b[0]),max(a[1],b[1])
            ix2,iy2 = min(a[2],b[2]),min(a[3],b[3])
            inter = max(0,ix2-ix1)*max(0,iy2-iy1)
            if not inter: return 0.0
            u = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
            return inter/u if u else 0.0
        def _person_fp(hbbox, persons):
            for p in persons:
                if _iou(hbbox, p['bbox']) > 0.35:
                    return True
            return False
        for key in ('fire','pool'):
            result[key] = [h for h in result[key]
                           if not _person_fp(h['bbox'], result['child'])]
        return result

    # ------------------------------------------------------------------
    def _make_demo_frame(self, w, h, t):
        """
        Synthetic camera background for demo mode.
        Dark scene with a subtle grid, scanline overlay, and a dim
        'room' gradient — so the window always shows something useful
        even when no physical camera is attached.
        """
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        # Soft room-tone gradient (warm dark grey, brighter at centre)
        cx, cy = w // 2, h // 2
        Y, X   = np.ogrid[:h, :w]
        dist   = np.sqrt((X - cx)**2 + (Y - cy)**2).astype(np.float32)
        dist  /= max(dist.max(), 1)
        glow   = (1.0 - dist) * 38
        frame[:, :, 0] = glow.astype(np.uint8)        # B
        frame[:, :, 1] = (glow * 0.88).astype(np.uint8)  # G
        frame[:, :, 2] = (glow * 0.72).astype(np.uint8)  # R  → warm tint

        # Grid lines
        grid = 60
        for x in range(0, w, grid):
            cv2.line(frame, (x, 0), (x, h), (22, 24, 28), 1)
        for y in range(0, h, grid):
            cv2.line(frame, (0, y), (w, y), (22, 24, 28), 1)

        # Scanline texture (every other row dimmed slightly)
        frame[::2, :] = (frame[::2, :] * 0.88).astype(np.uint8)

        # "DEMO" watermark
        cv2.putText(frame, "DEMO MODE", (10, h - 14),
                    FONT, 0.42, (55, 62, 80), 1, cv2.LINE_AA)

        return frame

    def _demo_detections(self, frame):
        """
        Generate synthetic detections AND replace the frame with a
        synthetic background so the window is never blank/black.
        Child sweeps across; fire is stationary near the centre.
        """
        self._demo_t += 1
        h, w = frame.shape[:2]
        t    = self._demo_t

        # Replace frame with synthetic background
        bg = self._make_demo_frame(w, h, t)
        frame[:] = bg[:]

        result = {'child': [], 'fire': [], 'pool': []}

        # Fire fixed at ~55 % from left
        fx, fy = int(w * 0.55), int(h * 0.45)
        fw, fh = 70, 90

        # Child sweeps across, passing through fire (~8 s period at 30 fps)
        phase = (t % 240) / 240.0
        cx    = int(w * 0.10 + w * 0.80 * phase)
        cy    = int(h * 0.42 + h * 0.06 * np.sin(t * 0.08))
        cbw, cbh = 80, 130

        cx1 = max(0, cx - cbw // 2)
        cy1 = max(0, cy - cbh // 2)
        cx2 = min(w, cx + cbw // 2)
        cy2 = min(h, cy + cbh // 2)

        result['child'].append(dict(
            bbox=[cx1, cy1, cx2, cy2],
            confidence=0.93, class_name='child', class_id=0,
            center=(cx, cy), area=(cx2 - cx1) * (cy2 - cy1)))

        if t % 240 > 15:
            result['fire'].append(dict(
                bbox=[fx - fw // 2, fy - fh // 2, fx + fw // 2, fy + fh // 2],
                confidence=0.88, class_name='fire', class_id=1,
                center=(fx, fy), area=fw * fh))

        return result


# ===========================================================================
# Drawing helpers
# ===========================================================================

def draw_detections(frame, detections):
    vis = frame.copy()
    for key, dets in detections.items():
        colour = _C.get(key, _C['default'])
        for d in dets:
            x1, y1, x2, y2 = d['bbox']
            cv2.rectangle(vis, (x1,y1), (x2,y2), colour, 2)
            label = f"{d['class_name'].upper()}  {d['confidence']:.0%}"
            (lw, lh), _ = cv2.getTextSize(label, FONT, 0.48, 1)
            cv2.rectangle(vis, (x1, y1-lh-10), (x1+lw+8, y1), colour, -1)
            cv2.putText(vis, label, (x1+4, y1-4),
                        FONT, 0.48, (10, 10, 10), 1, cv2.LINE_AA)
    return vis


def draw_proximity_zones(frame, hazard_center, child_bbox=None,
                         pixel_per_metre=40, pulse=0):
    """
    FIX: Draw expanding child-centred rectangles showing proximity zones.
    Critical ring animates (pulsing red) when child is dangerously close.
    Falls back to circles around the hazard when no child bbox available.
    """
    vis = frame.copy()
    h, w = vis.shape[:2]

    zones = [
        (5.0, (0,160,0),   'SAFE'),
        (2.5, (0,140,255), 'WARN'),
        (1.0, (0,0,230),   'CRIT'),
    ]

    if child_bbox is not None:
        cx1, cy1, cx2, cy2 = child_bbox
        for metres, colour, label in zones:
            pad = int(metres * pixel_per_metre)
            rx1 = max(0,     cx1 - pad)
            ry1 = max(0,     cy1 - pad)
            rx2 = min(w-1,   cx2 + pad)
            ry2 = min(h-1,   cy2 + pad)
            if label == 'CRIT':
                # Pulse: brighten colour with sin wave
                p = int(abs(np.sin(pulse * 0.15)) * 80)
                colour = (min(255,colour[0]+p), colour[1], min(255,colour[2]+p))
                thickness = 3
            else:
                thickness = 1
            cv2.rectangle(vis, (rx1,ry1), (rx2,ry2), colour, thickness, cv2.LINE_AA)
            cv2.putText(vis, label, (rx1+4, ry1+14),
                        FONT, 0.38, colour, 1, cv2.LINE_AA)
    else:
        cx, cy = int(hazard_center[0]), int(hazard_center[1])
        for metres, colour, label in zones:
            r = int(metres * pixel_per_metre)
            cv2.circle(vis, (cx,cy), r, colour, 1, cv2.LINE_AA)
    return vis


def draw_trajectory(frame, trajectory, frame_w, frame_h):
    if trajectory is None or len(trajectory) < 2:
        return frame
    vis = frame.copy()
    pts = np.column_stack([
        np.clip(trajectory[:,0]*frame_w, 0, frame_w-1).astype(int),
        np.clip(trajectory[:,1]*frame_h, 0, frame_h-1).astype(int),
    ])
    n = len(pts)
    for i in range(n-1):
        alpha  = 1.0 - i/n
        colour = (int(180*alpha), int(60*(1-alpha)), int(180*(1-alpha)))
        cv2.line(vis, tuple(pts[i].tolist()), tuple(pts[i+1].tolist()),
                 colour, 2, cv2.LINE_AA)
    return vis


def draw_hud(frame, risk_level, risk_score, pattern, fps,
             conf_thresh, paused, frame_count, proximity_info,
             alert_count=0, pulse=0, map_state=None):
    """
    Polished HUD drawn as a right-side panel.

    ROOT-CAUSE FIX for the "black strip / black frame" bug:
      The previous version called
          cv2.addWeighted(overlay, 0.82, vis, 0.18, 0, vis)
      which (a) blended the dark rectangle over the ENTIRE frame (making the
      left / video half dark on many OpenCV builds) and (b) used `vis` as
      both src2 and dst, which is undefined behaviour in OpenCV.

      The fix:
        1. Panel width is adaptive: min(300, w // 3) so it never swallows
           the entire frame at low resolutions.
        2. Blending touches ONLY the panel ROI — the rest of the frame is
           never read from or written to by addWeighted.
        3. src2 and dst are always different arrays.
    """
    vis  = frame.copy()
    h, w = vis.shape[:2]

    # --- Adaptive panel width: at most 300 px, at most 1/3 of frame width ---
    pw = min(300, max(200, w // 3))
    px = w - pw          # left edge of panel

    rc = _RISK_BGR.get(risk_level,   (160, 160, 160))
    ra = _RISK_ACCENT.get(risk_level,(220, 220, 220))

    # ------------------------------------------------------------------ #
    # Blend ONLY the panel ROI — never touch the video region             #
    # ------------------------------------------------------------------ #
    panel_roi   = vis[0:h, px:w]               # view into vis (video region untouched)
    dark_panel  = np.full_like(panel_roi, (14, 16, 20))
    blended_roi = cv2.addWeighted(dark_panel, 0.82, panel_roi, 0.18, 0)
    vis[0:h, px:w] = blended_roi               # write back only the panel area

    # Thin separator line
    cv2.line(vis, (px, 0), (px, h), (50, 55, 65), 1)

    yw = pw - 20    # usable width inside panel
    y  = 18

    # Helper: draw text relative to panel origin
    def txt(s, yp, scale=0.44, col=(200, 205, 215), bold=False):
        cv2.putText(vis, s, (px + 10, yp),
                    FONT, scale, col, 2 if bold else 1, cv2.LINE_AA)

    def sep(yp, label=None):
        """Draw a horizontal rule with optional small label above it.
        Returns y-position ready for the first content line below."""
        lh = 12   # label font height in px
        gap = 6   # gap between label bottom and rule
        rule_y = yp + (lh + gap if label else 0)
        if label:
            cv2.putText(vis, label, (px + 10, yp + lh),
                        FONT, 0.32, (75, 88, 112), 1, cv2.LINE_AA)
        cv2.line(vis, (px + 6, rule_y), (px + pw - 6, rule_y), (38, 44, 56), 1)
        return rule_y + 14   # 14 px clear space below rule for first content

    # ---- Title ----
    cv2.putText(vis, "GUARD", (px + 10, y + 22),
                FONT_MONO, 0.78, (235, 240, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, "AI", (px + 10 + 72, y + 22),
                FONT_MONO, 0.78, rc, 2, cv2.LINE_AA)
    cv2.putText(vis, "BABY SAFETY MONITOR", (px + 10, y + 37),
                FONT, 0.32, (70, 82, 105), 1, cv2.LINE_AA)
    y += 52

    # ---- Risk bar ----
    bar_h  = 14
    filled = int(yw * max(0.0, min(1.0, risk_score)))
    cv2.rectangle(vis, (px + 10, y), (px + 10 + yw, y + bar_h), (30, 34, 44), -1)
    if filled > 0:
        cv2.rectangle(vis, (px + 10, y), (px + 10 + filled, y + bar_h), rc, -1)
    cv2.rectangle(vis, (px + 10, y), (px + 10 + yw, y + bar_h), (50, 56, 70), 1)
    y += bar_h + 5

    cv2.putText(vis, risk_level, (px + 10, y + 14),
                FONT_MONO, 0.58, ra, 2, cv2.LINE_AA)
    cv2.putText(vis, f"{risk_score:.3f}", (px + pw - 58, y + 14),
                FONT, 0.48, (130, 138, 155), 1, cv2.LINE_AA)
    y += 26

    # ---- Proximity ----
    y = sep(y + 6, "PROXIMITY")
    if proximity_info:
        dist = proximity_info.get('closest_distance', float('inf'))
        zone = proximity_info.get('zone', 'safe').upper()
        zc   = {'CRITICAL': (80, 80, 255),
                'WARNING':  (80, 165, 255),
                'SAFE':     (80, 200, 80)}.get(zone, (140, 140, 140))
        line = "No hazard" if dist == float('inf') else f"{dist:.2f}m  [{zone}]"
        txt(line, y, col=zc, bold=(dist != float('inf')))
    else:
        txt("No hazard in frame", y, col=(72, 82, 105))
    y += 20

    # ---- Behaviour ----
    y = sep(y + 6, "BEHAVIOUR")
    txt(pattern.replace('_', ' ').title()[:22], y, col=(170, 178, 198))
    y += 20

    # ---- System stats ----
    y = sep(y + 6, "SYSTEM")
    txt(f"FPS    {fps:5.1f}",        y); y += 17
    txt(f"Conf   {conf_thresh:.2f}", y); y += 17
    txt(f"Frame  {frame_count}",     y); y += 17
    ac = (80, 120, 255) if alert_count else (120, 128, 148)
    txt(f"Alerts {alert_count}", y, col=ac)
    y += 22

    # ---- Alert banner (only when HIGH/CRITICAL) ----
    if risk_level in ('CRITICAL', 'HIGH'):
        blink = int(abs(np.sin(pulse * 0.18)) * 200)
        fill  = (0, 0, max(0, blink - 40))
        cv2.rectangle(vis, (px + 6, y),     (px + pw - 6, y + 32), fill, -1)
        cv2.rectangle(vis, (px + 6, y),     (px + pw - 6, y + 32), rc, 1)
        msg = "!! CRITICAL ALERT" if risk_level == 'CRITICAL' else "!! HIGH RISK"
        cv2.putText(vis, msg, (px + 12, y + 21),
                    FONT_MONO, 0.46, ra, 2, cv2.LINE_AA)
        y += 40

    # ---- PAUSED ----
    if paused:
        cv2.putText(vis, "[ PAUSED ]", (px + 10, h - 44),
                    FONT_MONO, 0.56, (0, 210, 255), 2, cv2.LINE_AA)

    # ---- Room map status — anchored above the key legend ----
    legend_h = 5 * 14 + 16   # 5 keys × 14px + padding
    map_y    = h - legend_h - 36   # fixed slot above legends

    if map_state and map_y > y:   # only draw if it doesn't collide with flow
        if map_state.get('learning_mode'):
            prog     = map_state.get('learning_progress', 0)
            bar_w2   = yw
            filled2  = int(bar_w2 * prog / 100)
            # Label
            cv2.putText(vis, "ROOM MAPPING", (px + 10, map_y),
                        FONT, 0.32, (75, 88, 112), 1, cv2.LINE_AA)
            map_y += 14
            # Progress bar
            cv2.rectangle(vis, (px+10, map_y),
                          (px+10+bar_w2, map_y+10), (25, 30, 40), -1)
            cv2.rectangle(vis, (px+10, map_y),
                          (px+10+filled2, map_y+10), (50, 200, 80), -1)
            cv2.rectangle(vis, (px+10, map_y),
                          (px+10+bar_w2, map_y+10), (45, 55, 68), 1)
            map_y += 14
            cv2.putText(vis, f"Learning  {prog:.0f}%",
                        (px + 10, map_y),
                        FONT, 0.40, (60, 210, 90), 1, cv2.LINE_AA)
        else:
            ph = map_state.get('persistent_hazards', 0)
            hz = map_state.get('high_risk_zones', 0)
            cv2.putText(vis, "ROOM MAP  READY", (px + 10, map_y),
                        FONT, 0.32, (75, 88, 112), 1, cv2.LINE_AA)
            map_y += 14
            cv2.putText(vis, f"{ph} hazards  {hz} zones",
                        (px + 10, map_y),
                        FONT, 0.40, (100, 170, 230), 1, cv2.LINE_AA)

    # ---- Key legend (always at the very bottom) ----
    for i, (k, act) in enumerate(reversed(
            [("Q", "quit"), ("P", "pause"), ("S", "screenshot"),
             ("+/-", "conf"), ("R", "reset")])):
        cv2.putText(vis, f"{k}:{act}", (px + 10, h - 10 - i * 14),
                    FONT, 0.30, (52, 60, 78), 1, cv2.LINE_AA)

    return vis


# ===========================================================================
# Main monitoring loop
# ===========================================================================
class LiveDetector:
    def __init__(self, args):
        self.args        = args
        self.conf        = args.conf
        self.detector    = HazardDetector(args.models, conf=self.conf)
        self.paused      = False
        self.frame_count = 0
        self.fps_history = []
        self._pulse      = 0
        self._alert_count = 0

        config_path = args.config

        self.temporal  = TemporalPatternAnalyzer(config_path) if TEMPORAL_AVAILABLE else None
        self.spatial   = SpatialRiskAssessment(config_path)   if SPATIAL_AVAILABLE  else None
        self.risk_mod  = RiskAssessmentModule(config_path)    if RISK_AVAILABLE     else None
        self.alert_mgr = AlertManager(config_path)            if ALERT_AVAILABLE    else None

        # Room mapper — loads existing map if present, else starts learning
        if MAPPER_AVAILABLE:
            self.room_mapper = AdaptiveRoomMapper()
            self.room_mapper.load_room_map()   # no-op if map doesn't exist yet
        else:
            self.room_mapper = None

        self.out_dir = Path('outputs/live_sessions')
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def _run_analysis(self, frame, detections):
        temporal = {'temporal_risk':0.0,'pattern_type':'n/a',
                    'confidence':0.5,'trajectory':None}
        spatial  = {'spatial_risk':0.0,'proximity_analysis':None,
                    'collision_warning':False}
        risk     = {'risk_score':0.0,'risk_level_name':'SAFE',
                    'risk_level':'SAFE','risk_level_enum':None,
                    'should_alert':False,'alert_urgency':'gentle',
                    'explanation':{'primary_factors':[]}}

        if self.temporal:
            try:    temporal = self.temporal.analyze(detections)
            except Exception as e: pass

        if self.spatial:
            try:    spatial = self.spatial.assess_risk(detections)
            except Exception as e: pass

        if self.risk_mod:
            try:
                risk = self.risk_mod.assess_comprehensive_risk(
                    detections, temporal, spatial, frame)
            except Exception as e: pass

        # Update room map every frame (learning or monitoring)
        if self.room_mapper:
            try:
                # Flatten detections list for mapper (expects list of dicts)
                flat = [d for dets in detections.values() for d in dets]
                self.room_mapper.process_frame_for_mapping(frame, flat)
            except Exception:
                pass

        return temporal, spatial, risk

    # ------------------------------------------------------------------
    def _handle_alert(self, risk, detections):
        if not self.alert_mgr or not risk.get('should_alert'):
            return
        try:
            alert = self.alert_mgr.create_alert(risk, detections)
            alert['recommended_actions'] = (
                self.risk_mod.get_recommended_actions(risk) if self.risk_mod else [])
            status = self.alert_mgr.send_alert(alert)
            if not status.get('suppressed'):
                self._alert_count += 1
        except Exception as e:
            pass

    # ------------------------------------------------------------------
    def _build_display(self, frame, detections, temporal, spatial, risk, fps):
        # Apply room-map overlay first (heatmap + persistent hazard boxes +
        # "Learning Room: X%" banner) so detection boxes draw on top of it.
        if self.room_mapper:
            try:
                frame = self.room_mapper.visualize_room_map(frame)
            except Exception:
                pass

        vis = draw_detections(frame, detections)
        self._pulse += 1

        child_bbox_vis = None
        if spatial.get('proximity_analysis'):
            child_bbox_vis = spatial['proximity_analysis'].get('child_bbox')
        if child_bbox_vis is None and detections.get('child'):
            child_bbox_vis = detections['child'][0]['bbox']

        for key in ('fire', 'pool'):
            if detections.get(key):
                vis = draw_proximity_zones(
                    vis, detections[key][0]['center'],
                    child_bbox=child_bbox_vis,
                    pulse=self._pulse)
                break

        if temporal.get('trajectory') is not None:
            h, w = vis.shape[:2]
            vis = draw_trajectory(vis, temporal['trajectory'], w, h)

        map_state = None
        if self.room_mapper:
            try:
                map_state = self.room_mapper.get_current_map_state()
            except Exception:
                pass

        vis = draw_hud(
            vis,
            risk_level     = risk.get('risk_level_name', 'SAFE'),
            risk_score     = risk.get('risk_score', 0.0),
            pattern        = temporal.get('pattern_type', 'n/a'),
            fps            = fps,
            conf_thresh    = self.conf,
            paused         = self.paused,
            frame_count    = self.frame_count,
            proximity_info = spatial.get('proximity_analysis'),
            alert_count    = self._alert_count,
            pulse          = self._pulse,
            map_state      = map_state,
        )

        if spatial.get('collision_warning'):
            ct = spatial.get('collision_time', 0)
            h, w = vis.shape[:2]
            cv2.putText(vis,
                        f"COLLISION WARNING - {ct} frames (~{ct/30:.1f}s)",
                        (10, h - 16),
                        FONT_MONO, 0.62, (0, 0, 255), 2, cv2.LINE_AA)

        return vis

    # ------------------------------------------------------------------
    def run(self):
        source = self.args.video if self.args.video else self.args.camera
        cap    = cv2.VideoCapture(source)
        if not cap.isOpened():
            sys.exit(f"[ERROR] Cannot open: {source}")

        fps_src = cap.get(cv2.CAP_PROP_FPS) or 30
        w_src   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 1280
        h_src   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"\n{'='*60}")
        print("  GuardAI — Baby Hazard Monitor")
        print(f"{'='*60}")
        print(f"  Source     : {source}")
        print(f"  Resolution : {w_src}x{h_src}  @{fps_src:.0f}fps")
        print(f"  Frames     : {total if total>0 else 'live'}")
        print(f"  Mode       : {'DEMO' if self.detector._use_demo else 'LIVE'}")
        print(f"{'='*60}\n")

        writer = None
        if self.args.output:
            out_p = Path(self.args.output)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(out_p),
                                     cv2.VideoWriter_fourcc(*'mp4v'),
                                     fps_src, (w_src, h_src))

        is_file   = self.args.video is not None
        frame_del = (1.0/fps_src) if is_file else 0.0
        _STOP     = object()
        frame_q   = queue.Queue(maxsize=0 if is_file else 1)
        result_q  = queue.Queue(maxsize=2 if is_file else 1)
        stop_evt  = threading.Event()

        def capture_thread():
            t_next = time.time()
            while not stop_evt.is_set():
                if self.paused:
                    time.sleep(0.05)
                    continue
                if is_file and frame_del > 0:
                    now = time.time()
                    if now < t_next:
                        time.sleep(t_next - now)
                    t_next = time.time() + frame_del
                ret, frame = cap.read()
                if not ret:
                    frame_q.put(_STOP)
                    return
                if not is_file:
                    try: frame_q.get_nowait()
                    except queue.Empty: pass
                frame_q.put(frame)

        def inference_thread():
            while not stop_evt.is_set():
                try:   frame = frame_q.get(timeout=0.5)
                except queue.Empty: continue
                if frame is _STOP:
                    result_q.put(_STOP)
                    return
                det, frame   = self.detector.detect(frame)  # unpack (dets, frame)
                tmp, spa, rsk = self._run_analysis(frame, det)
                self._handle_alert(rsk, det)
                if not is_file:
                    try: result_q.get_nowait()
                    except queue.Empty: pass
                result_q.put((frame, det, tmp, spa, rsk))

        t_cap = threading.Thread(target=capture_thread,  daemon=True)
        t_inf = threading.Thread(target=inference_thread, daemon=True)
        t_cap.start()
        t_inf.start()

        WIN_TITLE = 'GuardAI - Baby Hazard Monitor'

        if not self.args.no_display:
            cv2.namedWindow(WIN_TITLE, cv2.WINDOW_NORMAL)
            # Sensible default: show at least 1024 wide, preserve aspect ratio
            disp_w = max(w_src, 1024)
            disp_h = int(disp_w * h_src / max(w_src, 1)) if w_src else 576
            cv2.resizeWindow(WIN_TITLE, disp_w, disp_h)

        t_prev   = time.time()
        last_res = None

        try:
            while True:
                if self.paused:
                    key = cv2.waitKey(100) & 0xFF
                    if self._handle_key(key, None):
                        break
                    continue

                if is_file:
                    try:   res = result_q.get(timeout=1.0)
                    except queue.Empty: res = None
                else:
                    try:   res = result_q.get_nowait()
                    except queue.Empty: res = None

                if res is None and last_res is None:
                    time.sleep(0.001)
                    continue
                if res is None:
                    frame, det, tmp, spa, rsk = last_res
                elif res is _STOP:
                    print("[INFO] End of stream.")
                    break
                else:
                    frame, det, tmp, spa, rsk = res
                    last_res = res
                    self.frame_count += 1

                now    = time.time()
                fps    = 1.0 / max(now - t_prev, 1e-6)
                t_prev = now
                self.fps_history.append(fps)
                if len(self.fps_history) > 30:
                    self.fps_history.pop(0)
                avg_fps = float(np.mean(self.fps_history))

                vis = self._build_display(frame, det, tmp, spa, rsk, avg_fps)

                if writer:
                    writer.write(vis)
                if not self.args.no_display:
                    cv2.imshow(WIN_TITLE, vis)
                    if self._handle_key(cv2.waitKey(1) & 0xFF, frame):
                        break

                if self.frame_count % 60 == 0:
                    rl  = rsk.get('risk_level_name','SAFE')
                    rs  = rsk.get('risk_score', 0.0)
                    nc  = sum(len(v) for v in det.values())
                    print(f"  [{self.frame_count:5d}] fps={avg_fps:.1f}  "
                          f"risk={rl}({rs:.3f})  dets={nc}  alerts={self._alert_count}")

        except KeyboardInterrupt:
            print("\n[INFO] Interrupted.")
        finally:
            stop_evt.set()
            t_cap.join(timeout=2)
            t_inf.join(timeout=2)
            cap.release()
            if writer: writer.release()
            cv2.destroyAllWindows()
            self._print_summary()

    # ------------------------------------------------------------------
    def _handle_key(self, key, frame):
        if key == ord('q'): return True
        if key == ord('p'):
            self.paused = not self.paused
            print(f"[INFO] {'Paused' if self.paused else 'Resumed'}")
        elif key == ord('s') and frame is not None:
            p = self.out_dir / f"shot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(str(p), frame)
            print(f"[INFO] Screenshot: {p}")
        elif key == ord('r'):
            if self.temporal: self.temporal.reset()
            if self.spatial:  self.spatial.proximity_history.clear()
            print("[INFO] History reset.")
        elif key == ord('+'):
            self.conf = min(0.95, self.conf+0.05)
            self.detector.conf = self.conf
            print(f"[INFO] Conf -> {self.conf:.2f}")
        elif key == ord('-'):
            self.conf = max(0.05, self.conf-0.05)
            self.detector.conf = self.conf
            print(f"[INFO] Conf -> {self.conf:.2f}")
        return False

    def _print_summary(self):
        print(f"\n{'='*60}")
        print("  SESSION SUMMARY")
        print(f"{'='*60}")
        print(f"  Frames   : {self.frame_count}")
        avg = np.mean(self.fps_history) if self.fps_history else 0
        print(f"  Avg FPS  : {avg:.1f}")
        print(f"  Alerts   : {self._alert_count}")
        if self.alert_mgr:
            stats = self.alert_mgr.get_alert_statistics()
            for lvl, cnt in stats.get('by_level',{}).items():
                print(f"    {lvl:12s}: {cnt}")
        print(f"{'='*60}\n")


# ===========================================================================
def parse_args():
    p = argparse.ArgumentParser(description='GuardAI — Live Baby Hazard Detection')
    src = p.add_mutually_exclusive_group()
    src.add_argument('--camera', type=int,  default=0)
    src.add_argument('--video',  type=str)
    p.add_argument('--output',     type=str)
    p.add_argument('--models',     type=str,   default='models')
    p.add_argument('--config',     type=str,   default='config/config.yaml')
    p.add_argument('--conf',       type=float, default=0.45)
    p.add_argument('--no-display', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    if args.video and not Path(args.video).exists():
        sys.exit(f"[ERROR] Video not found: {args.video}")
    LiveDetector(args).run()


if __name__ == '__main__':
    main()