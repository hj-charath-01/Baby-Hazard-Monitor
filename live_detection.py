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
"""

import argparse
import sys
import time
import threading
import queue
from datetime import datetime
from pathlib import Path
from typing import Optional          # BUG 2 FIX: use typing.Optional instead of X|Y

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Graceful import of optional heavy dependencies
# ---------------------------------------------------------------------------
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARNING] ultralytics not installed — running in DEMO (synthetic) mode.")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Local module imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

try:
    from temporal_reasoning import TemporalPatternAnalyzer
    TEMPORAL_AVAILABLE = True
except Exception as e:
    TEMPORAL_AVAILABLE = False
    print(f"[WARNING] temporal_reasoning unavailable: {e}")

try:
    from spatial_analysis import SpatialRiskAssessment
    SPATIAL_AVAILABLE = True
except Exception as e:
    SPATIAL_AVAILABLE = False
    print(f"[WARNING] spatial_analysis unavailable: {e}")

try:
    from risk_assessment import RiskAssessmentModule, RiskLevel
    RISK_AVAILABLE = True
except Exception as e:
    RISK_AVAILABLE = False
    print(f"[WARNING] risk_assessment unavailable: {e}")

try:
    from alert_manager import AlertManager, AlertLevel
    ALERT_AVAILABLE = True
except Exception as e:
    ALERT_AVAILABLE = False
    print(f"[WARNING] alert_manager unavailable: {e}")


# ===========================================================================
# Colour palette  (BGR)
# ===========================================================================
COLOURS = {
    'child':   (50,  205, 50),
    'fire':    (0,   60,  255),
    'pool':    (220, 90,  0),
    'smoke':   (160, 160, 160),
    'water':   (200, 150, 0),
    'default': (200, 200, 200),
}

RISK_COLOURS = {
    'SAFE':     (0,   200, 0),
    'LOW':      (0,   220, 220),
    'MEDIUM':   (0,   165, 255),
    'HIGH':     (0,   60,  255),
    'CRITICAL': (0,   0,   255),
}


# ===========================================================================
# Detector wrapper
# ===========================================================================
class HazardDetector:
    """
    Wraps up to three YOLO models (child / fire / pool).
    Falls back to synthetic demo mode if models are absent.
    """

    CHILD_CLASSES = {'child', 'person', 'baby', 'toddler'}

    # BUG 1 FIX: split into two separate sets so 'smoke' and 'flame' go to
    # fire bucket, and 'pool'/'water'/'drowning'/'swimming' go to pool bucket.
    # Original code used one HAZARD_CLASSES set with `'fire' in cls_name` check,
    # which silently routed 'smoke' → pool instead of fire.
    FIRE_CLASSES = {'fire', 'flame', 'smoke'}
    POOL_CLASSES = {'pool', 'water', 'drowning', 'swimming'}

    def __init__(self, model_dir: str = 'models',
                 conf: float = 0.45, iou: float = 0.45):
        self.conf      = conf
        self.iou       = iou
        self.models    = {}
        self._demo_t   = 0
        self._use_demo = False

        if YOLO_AVAILABLE:
            self._load_models(Path(model_dir))
        else:
            self._use_demo = True
            print("[INFO] DEMO mode: synthetic detections will be generated.")

    # ------------------------------------------------------------------
    def _load_models(self, model_dir: Path):
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
                        print(f"[INFO] Loaded {task} model: {p}")
                        loaded = True
                        break
                    except Exception as e:
                        print(f"[WARNING] Could not load {p}: {e}")

        if not loaded:
            try:
                self.models['general'] = YOLO('yolov8n.pt')
                print("[INFO] No task-specific models found — using yolov8n.pt")
                loaded = True
            except Exception:
                pass

        if not loaded:
            self._use_demo = True
            print("[INFO] No models available — switching to DEMO mode.")

    # ------------------------------------------------------------------
    # Inference width — run YOLO at this width for speed, map coords back.
    # At 416px vs 1280px: ~9x fewer pixels → ~2x faster inference on CPU.
    INFER_WIDTH = 416

    def detect(self, frame: np.ndarray) -> dict:
        if self._use_demo:
            return self._demo_detections(frame)

        result = {'child': [], 'fire': [], 'pool': []}
        h, w   = frame.shape[:2]

        # Downscale for faster inference, keep aspect ratio
        scale   = self.INFER_WIDTH / w
        infer_h = int(h * scale)
        small   = cv2.resize(frame, (self.INFER_WIDTH, infer_h), interpolation=cv2.INTER_LINEAR)
        TASK_BUCKETS = {
            'child':   'child',
            'fire':    'fire',
            'pool':    'pool',
            'general': None,
        }
        BUCKET_CLASSES = {
            'child': self.CHILD_CLASSES,
            'fire':  self.FIRE_CLASSES,
            'pool':  self.POOL_CLASSES,
        }
        # Hazard models require higher confidence than child detection.
        # 0.25 is too permissive — humans 3m away regularly score 0.3+ on
        # fire models trained on limited data, causing false emergencies.
        HAZARD_CONF = max(self.conf, 0.65)

        for task, model in self.models.items():
            forced_bucket = TASK_BUCKETS.get(task)
            min_conf  = self.conf if forced_bucket == 'child' else HAZARD_CONF

            try:
                preds = model.predict(small, conf=min_conf, iou=self.iou, verbose=False)[0]
            except Exception as e:
                print(f"[WARNING] Inference error ({task}): {e}")
                continue

            if preds.boxes is None:
                continue

            boxes = preds.boxes.xyxy.cpu().numpy()
            confs = preds.boxes.conf.cpu().numpy()
            cls_ids = preds.boxes.cls.cpu().numpy().astype(int)
            names = preds.names

            for box, conf, cls_id in zip(boxes, confs, cls_ids):
                cls_name = names.get(cls_id, 'unknown').lower()
                x1 = max(0,     int(box[0] / scale))
                y1 = max(0,     int(box[1] / scale))
                x2 = min(w - 1, int(box[2] / scale))
                y2 = min(h - 1, int(box[3] / scale))

                det = {
                    'bbox':       [x1, y1, x2, y2],
                    'confidence': float(conf),
                    'class_name': cls_name,
                    'class_id':   int(cls_id),
                    'center':     ((x1 + x2) // 2, (y1 + y2) // 2),
                    'area':       (x2 - x1) * (y2 - y1),
                }

                if forced_bucket:
                    allowed = BUCKET_CLASSES.get(forced_bucket, set())
                    if cls_name in allowed:
                        result[forced_bucket].append(det)
                else:
                    if cls_name in self.CHILD_CLASSES:
                        result['child'].append(det)
                    elif cls_name in self.FIRE_CLASSES:
                        result['fire'].append(det)
                    elif cls_name in self.POOL_CLASSES:
                        result['pool'].append(det)

        # ----------------------------------------------------------------
        # Person-aware hazard filtering — three checks, any one rejects:
        #
        # 1. IoU > 0.25   — hazard bbox overlaps the same region as a person
        # 2. Center-inside — hazard centroid falls inside a person bbox
        #                    (catches partial-overlap torso/head crops)
        # 3. Proximity < MARGIN px — hazard bbox is within 50px of a person edge
        #                    (catches adjacent misclassifications on clothing,
        #                     window glare, or background objects near people)
        #
        # All three are needed because:
        #   - IoU misses adjacent detections (shirt/jacket next to person)
        #   - Center-inside misses small torso crops whose center is outside
        #   - Proximity margin catches everything touching or near a person
        # ----------------------------------------------------------------
        PERSON_MARGIN = 50   # px — exclusion halo around each person bbox

        def _iou(a, b):
            ax1,ay1,ax2,ay2 = a
            bx1,by1,bx2,by2 = b
            ix1,iy1 = max(ax1,bx1), max(ay1,by1)
            ix2,iy2 = min(ax2,bx2), min(ay2,by2)
            inter   = max(0, ix2-ix1) * max(0, iy2-iy1)
            if inter == 0:
                return 0.0
            union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
            return inter / union if union > 0 else 0.0

        def _center_inside(haz, person):
            cx = (haz[0] + haz[2]) // 2
            cy = (haz[1] + haz[3]) // 2
            return person[0] <= cx <= person[2] and person[1] <= cy <= person[3]

        def _gap(a, b):
            """Minimum pixel gap between bbox edges (0 if touching/overlapping)."""
            ax1,ay1,ax2,ay2 = a
            bx1,by1,bx2,by2 = b
            dx = max(0, max(ax1, bx1) - min(ax2, bx2))
            dy = max(0, max(ay1, by1) - min(ay2, by2))
            return (dx*dx + dy*dy) ** 0.5

        def _is_person_misclassification(haz_bbox, persons):
            for p in persons:
                pb = p['bbox']
                if _iou(haz_bbox, pb) > 0.25:
                    return True
                if _center_inside(haz_bbox, pb):
                    return True
                if _gap(haz_bbox, pb) < PERSON_MARGIN:
                    return True
            return False

        children = result['child']
        for key in ('fire', 'pool'):
            result[key] = [
                haz for haz in result[key]
                if not _is_person_misclassification(haz['bbox'], children)
            ]

        return result

    # ------------------------------------------------------------------
    def _demo_detections(self, frame: np.ndarray) -> dict:
        """Generate plausible synthetic detections for demo purposes."""
        self._demo_t += 1
        h, w = frame.shape[:2]
        t    = self._demo_t

        result = {'child': [], 'fire': [], 'pool': []}

        # Child oscillates across frame
        cx = int(w * 0.2 + (w * 0.6) * (0.5 + 0.5 * np.sin(t * 0.04)))
        cy = int(h * 0.3 + h * 0.1  * np.sin(t * 0.06))
        bw, bh = 80, 120

        # BUG — demo bbox could go negative; clamp it
        x1 = max(0, cx - bw // 2)
        y1 = max(0, cy - bh // 2)
        x2 = min(w, cx + bw // 2)
        y2 = min(h, cy + bh // 2)

        result['child'].append({
            'bbox':       [x1, y1, x2, y2],
            'confidence': 0.92,
            'class_name': 'child',
            'class_id':   0,
            'center':     (cx, cy),
            'area':       (x2 - x1) * (y2 - y1),
        })

        # Static fire hazard (briefly disappears to test dynamic detection)
        if t % 120 > 20:
            fx, fy = int(w * 0.75), int(h * 0.45)
            fw, fh = 60, 80
            result['fire'].append({
                'bbox':       [fx - fw//2, fy - fh//2,
                               fx + fw//2, fy + fh//2],
                'confidence': 0.87,
                'class_name': 'fire',
                'class_id':   1,
                'center':     (fx, fy),
                'area':       fw * fh,
            })

        return result


# ===========================================================================
# Visualisation helpers
# ===========================================================================
def draw_detections(frame: np.ndarray, detections: dict) -> np.ndarray:
    vis = frame.copy()
    for key, dets in detections.items():
        colour = COLOURS.get(key, COLOURS['default'])
        for d in dets:
            x1, y1, x2, y2 = d['bbox']
            cv2.rectangle(vis, (x1, y1), (x2, y2), colour, 2)
            label = f"{d['class_name']} {d['confidence']:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                          0.5, 1)
            cv2.rectangle(vis, (x1, y1 - lh - 8), (x1 + lw + 4, y1),
                          colour, -1)
            cv2.putText(vis, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return vis


def draw_proximity_zones(frame: np.ndarray, hazard_center,
                         pixel_per_metre: float = 100,
                         child_bbox=None) -> np.ndarray:
    """
    BUG FIX: previously drew fixed-radius circles around the hazard center.
    A child bbox of 80x120px and a fire bbox 90px away (just outside the
    50px critical circle) showed no alert even though they were nearly
    touching.

    Now draws expanding rectangles around the CHILD bounding box — each ring
    shows exactly how far the child's edges are from the zone boundary.
    Falls back to circles around the hazard center when no child bbox is
    available (e.g. no child detected this frame).
    """
    vis = frame.copy()
    h, w = vis.shape[:2]

    if child_bbox is not None:
        cx1, cy1, cx2, cy2 = child_bbox
        for metres, colour, label in [
            (3.0, (0, 255, 0),   'safe'),
            (1.5, (0, 165, 255), 'warning'),
            (0.5, (0, 0, 255),   'critical'),
        ]:
            pad = int(metres * pixel_per_metre)
            rx1 = max(0,     cx1 - pad)
            ry1 = max(0,     cy1 - pad)
            rx2 = min(w - 1, cx2 + pad)
            ry2 = min(h - 1, cy2 + pad)
            cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), colour, 2)
            cv2.putText(vis, label, (rx1 + 4, ry1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)
    else:
        # fallback: circles around hazard center
        cx, cy = int(hazard_center[0]), int(hazard_center[1])
        for metres, colour, label in [
            (3.0, (0, 255, 0),   'safe'),
            (1.5, (0, 165, 255), 'warning'),
            (0.5, (0, 0, 255),   'critical'),
        ]:
            r = int(metres * pixel_per_metre)
            cv2.circle(vis, (cx, cy), r, colour, 2)
            cv2.putText(vis, label, (cx - 25, cy + r + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)
    return vis


def draw_trajectory(frame: np.ndarray, trajectory: np.ndarray,
                    frame_w: int, frame_h: int) -> np.ndarray:
    if trajectory is None or len(trajectory) < 2:
        return frame
    vis = frame.copy()
    pts = np.column_stack([
        np.clip(trajectory[:, 0] * frame_w, 0, frame_w - 1).astype(int),
        np.clip(trajectory[:, 1] * frame_h, 0, frame_h - 1).astype(int),
    ])
    for i in range(len(pts) - 1):
        alpha  = 1.0 - i / len(pts)
        colour = (int(200 * alpha), 0, int(200 * (1 - alpha)))
        # BUG 3 FIX: tuple(pts[i]) produces numpy int64 values which crash
        # OpenCV on some builds.  .tolist() converts to native Python ints.
        cv2.line(vis, tuple(pts[i].tolist()), tuple(pts[i + 1].tolist()),
                 colour, 2)
    return vis


# BUG 2 FIX: `dict | None` union syntax requires Python 3.10+.
# Use Optional[dict] from typing instead so it works on 3.8/3.9 too.
def draw_hud(frame: np.ndarray, risk_level: str, risk_score: float,
             pattern: str, fps: float, conf_thresh: float,
             paused: bool, frame_count: int,
             proximity_info: Optional[dict]) -> np.ndarray:
    """Draw the heads-up display panel."""
    vis     = frame.copy()
    h, w    = vis.shape[:2]
    panel_w = 340

    # BUG 5 FIX: cv2.addWeighted with vis as both source and destination is
    # unreliable.  Blend into a fresh copy, then assign back.
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, h), (20, 20, 20), -1)
    blended = np.empty_like(vis)
    cv2.addWeighted(overlay, 0.55, vis, 0.45, 0, blended)
    vis = blended

    risk_colour = RISK_COLOURS.get(risk_level, (200, 200, 200))
    y = 28

    def txt(text, pos_y, scale=0.55, colour=(230, 230, 230), bold=False):
        cv2.putText(vis, text, (10, pos_y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, colour,
                    2 if bold else 1)

    txt("HAZARD MONITOR", y, 0.65, (255, 255, 255), bold=True)
    y += 30

    # Risk bar
    bar_w = int((panel_w - 20) * max(0.0, min(1.0, risk_score)))
    cv2.rectangle(vis, (10, y), (panel_w - 10, y + 14), (60, 60, 60), -1)
    cv2.rectangle(vis, (10, y), (10 + bar_w,   y + 14), risk_colour, -1)
    y += 22

    txt(f"RISK: {risk_level}  ({risk_score:.3f})", y, 0.6,
        risk_colour, bold=True)
    y += 26

    txt(f"Pattern : {pattern[:28]}", y);   y += 22
    txt(f"FPS     : {fps:5.1f}",     y);   y += 22
    txt(f"Conf    : {conf_thresh:.2f}", y); y += 22
    txt(f"Frame   : {frame_count}",   y);   y += 28

    if proximity_info:
        dist = proximity_info.get('closest_distance', float('inf'))
        zone = proximity_info.get('zone', 'unknown')
        txt(f"Distance: {dist:.2f} m", y, colour=(100, 220, 255)); y += 22
        txt(f"Zone    : {zone}",       y, colour=(100, 220, 255)); y += 22

    if paused:
        txt("[ PAUSED ]", h - 40, 0.8, (0, 200, 255), bold=True)

    # Controls legend (bottom-right)
    for i, c in enumerate(reversed(
            ["q:quit", "p:pause", "s:screenshot", "r:reset", "+/-:conf"])):
        cv2.putText(vis, c, (w - 120, h - 12 - i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1)

    return vis


# ===========================================================================
# Main monitoring loop
# ===========================================================================
class LiveDetector:
    """Orchestrates real-time detection, analysis, and display."""

    def __init__(self, args):
        self.args        = args
        self.conf        = args.conf
        self.detector    = HazardDetector(args.models, conf=self.conf)
        self.paused      = False
        self.frame_count = 0
        self.fps_history = []

        config_path = args.config

        self.temporal  = TemporalPatternAnalyzer(config_path) \
                         if TEMPORAL_AVAILABLE else None
        self.spatial   = SpatialRiskAssessment(config_path) \
                         if SPATIAL_AVAILABLE else None
        self.risk_mod  = RiskAssessmentModule(config_path) \
                         if RISK_AVAILABLE else None
        self.alert_mgr = AlertManager(config_path) \
                         if ALERT_AVAILABLE else None

        self.out_dir = Path('outputs/live_sessions')
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def _run_analysis(self, frame, detections):
        temporal = {'temporal_risk': 0.0, 'pattern_type': 'n/a',
                    'confidence': 0.5, 'trajectory': None}
        spatial  = {'spatial_risk': 0.0, 'proximity_analysis': None,
                    'collision_warning': False}
        risk     = {'risk_score': 0.0, 'risk_level_name': 'SAFE',
                    'risk_level': 'SAFE', 'risk_level_enum': None,
                    'should_alert': False, 'alert_urgency': 'gentle',
                    'explanation': {'primary_factors': []}}

        if self.temporal:
            try:
                temporal = self.temporal.analyze(detections)
            except Exception as e:
                print(f"[WARNING] Temporal analysis error: {e}")

        if self.spatial:
            try:
                spatial = self.spatial.assess_risk(detections)
            except Exception as e:
                print(f"[WARNING] Spatial analysis error: {e}")

        if self.risk_mod:
            try:
                risk = self.risk_mod.assess_comprehensive_risk(
                    detections, temporal, spatial, frame)
            except Exception as e:
                print(f"[WARNING] Risk assessment error: {e}")

        return temporal, spatial, risk

    # ------------------------------------------------------------------
    def _handle_alert(self, risk, detections):
        if not self.alert_mgr or not risk.get('should_alert'):
            return
        try:
            alert = self.alert_mgr.create_alert(risk, detections)
            alert['recommended_actions'] = (
                self.risk_mod.get_recommended_actions(risk)
                if self.risk_mod else [])
            self.alert_mgr.send_alert(alert)
        except Exception as e:
            print(f"[WARNING] Alert error: {e}")

    # ------------------------------------------------------------------
    def _build_display(self, frame, detections, temporal, spatial, risk, fps):
        vis = draw_detections(frame, detections)

        for key in ('fire', 'pool'):
            if detections.get(key):
                # Pass child_bbox from spatial proximity so zones expand from
                # the child's edges rather than drawing circles at hazard center
                child_bbox_vis = (
                    (spatial.get('proximity_analysis') or {}).get('child_bbox')
                    or (detections['child'][0]['bbox']
                        if detections.get('child') else None)
                )
                vis = draw_proximity_zones(
                    vis, detections[key][0]['center'],
                    child_bbox=child_bbox_vis)
                break

        if temporal.get('trajectory') is not None:
            h, w = vis.shape[:2]
            vis = draw_trajectory(vis, temporal['trajectory'], w, h)

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
        )

        # BUG 4 FIX: cv2.putText does not support Unicode (e.g. ⚠ or —).
        # Replaced with ASCII-only text.
        if spatial.get('collision_warning'):
            ct = spatial.get('collision_time', 0)
            cv2.putText(vis,
                        f"!! COLLISION WARNING - {ct} frames",
                        (10, vis.shape[0] - 16),
                        cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 255), 2)

        return vis

    # ------------------------------------------------------------------
    def run(self):
        source = self.args.video if self.args.video else self.args.camera
        cap    = cv2.VideoCapture(source)

        if not cap.isOpened():
            sys.exit(f"[ERROR] Cannot open source: {source}")

        fps_src = cap.get(cv2.CAP_PROP_FPS) or 30
        w_src   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_src   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"\n{'='*60}")
        print(" LIVE HAZARD DETECTION")
        print(f"{'='*60}")
        print(f"  Source      : {source}")
        print(f"  Resolution  : {w_src}x{h_src}  FPS: {fps_src:.0f}")
        print(f"  Total frames: {total if total > 0 else 'live'}")
        print(f"  Conf thresh : {self.conf}")
        print(f"{'='*60}\n")

        writer = None
        if self.args.output:
            out_p  = Path(self.args.output)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(out_p), fourcc, fps_src,
                                     (w_src, h_src))
            print(f"[INFO] Writing output to: {out_p}")

        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        # Threaded pipeline — two modes:
#
        # LIVE (webcam):
#   frame_q  maxsize=1, drop-on-full so inference always gets the newest frame.
#   result_q maxsize=1, drop-on-full so display always shows the newest result.
#
        # VIDEO FILE:
#   frame_q  unbounded — every frame is queued, none dropped.
#   Capture is rate-limited to source FPS so the queue stays small.
#   result_q maxsize=2, display blocks so it stays in sync with inference.
        # ------------------------------------------------------------------
        is_file     = self.args.video is not None
        frame_delay = (1.0 / fps_src) if is_file else 0.0
        _STOP    = object()
        frame_q  = queue.Queue(maxsize=0 if is_file else 1)
        result_q = queue.Queue(maxsize=2 if is_file else 1)
        stop_evt = threading.Event()

        def capture_thread():
            t_next = time.time()
            while not stop_evt.is_set():
                if self.paused:
                    time.sleep(0.05)
                    continue
                if is_file and frame_delay > 0:
                    now = time.time()
                    if now < t_next:
                        time.sleep(t_next - now)
                    t_next = time.time() + frame_delay
                ret, frame = cap.read()
                if not ret:
                    frame_q.put(_STOP)
                    return
                if not is_file:
                    try:
                        frame_q.get_nowait()
                    except queue.Empty:
                        pass
                frame_q.put(frame)

        def inference_thread():
            while not stop_evt.is_set():
                try:
                    frame = frame_q.get(timeout=0.5)
                except queue.Empty:
                    continue
                if frame is _STOP:
                    result_q.put(_STOP)
                    return
                detections              = self.detector.detect(frame)
                temporal, spatial, risk = self._run_analysis(frame, detections)
                self._handle_alert(risk, detections)
                if not is_file:
                    try:
                        result_q.get_nowait()
                    except queue.Empty:
                        pass
                result_q.put((frame, detections, temporal, spatial, risk))

        t_cap  = threading.Thread(target=capture_thread,  daemon=True)
        t_inf  = threading.Thread(target=inference_thread, daemon=True)
        t_cap.start()
        t_inf.start()

        t_prev   = time.time()
        last_res = None   # most recent inference result, reused for display

        try:
            while True:
                if self.paused:
                    key = cv2.waitKey(100) & 0xFF
                    if self._handle_key(key, None):
                        break
                    continue

                # Video: block until inference delivers the next result
                # (every frame must be processed — no drops).
                # Live: poll so display stays responsive when inference lags.
                if is_file:
                    try:
                        res = result_q.get(timeout=1.0)
                    except queue.Empty:
                        res = None
                else:
                    try:
                        res = result_q.get_nowait()
                    except queue.Empty:
                        res = None

                if res is None and last_res is None:
                    time.sleep(0.001)
                    continue

                if res is None:
                    # Live only: reuse last result
                    frame, detections, temporal, spatial, risk = last_res
                elif res is _STOP:
                    # Sentinel from inference thread — stream ended
                    print("[INFO] End of stream.")
                    break
                else:
                    frame, detections, temporal, spatial, risk = res
                    last_res = res
                    self.frame_count += 1

                now    = time.time()
                fps    = 1.0 / max(now - t_prev, 1e-6)
                t_prev = now
                self.fps_history.append(fps)
                if len(self.fps_history) > 30:
                    self.fps_history.pop(0)
                avg_fps = float(np.mean(self.fps_history))

                vis = self._build_display(
                    frame, detections, temporal, spatial, risk, avg_fps)

                if writer:
                    writer.write(vis)

                if not self.args.no_display:
                    cv2.imshow('Baby Hazard Monitor - Live', vis)
                    if self._handle_key(cv2.waitKey(1) & 0xFF, frame):
                        break

                if self.frame_count % 30 == 0:
                    rl = risk.get('risk_level_name', 'SAFE')
                    rs = risk.get('risk_score', 0.0)
                    nc = sum(len(v) for v in detections.values())
                    print(f"  frame {self.frame_count:5d} | "
                          f"fps {avg_fps:5.1f} | "
                          f"risk {rl:8s} ({rs:.3f}) | "
                          f"detections {nc}")

        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user.")
        finally:
            stop_evt.set()
            t_cap.join(timeout=2)
            t_inf.join(timeout=2)
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            self._print_summary()

    # ------------------------------------------------------------------
    def _handle_key(self, key: int, frame) -> bool:
        """Return True to signal quit."""
        if key == ord('q'):
            return True
        if key == ord('p'):
            self.paused = not self.paused
            print(f"[INFO] {'Paused' if self.paused else 'Resumed'}")
        elif key == ord('s') and frame is not None:
            path = self.out_dir / f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(str(path), frame)
            print(f"[INFO] Screenshot saved: {path}")
        elif key == ord('r'):
            if self.temporal:
                self.temporal.reset()
            if self.spatial:
                self.spatial.proximity_history.clear()
            print("[INFO] History reset.")
        elif key == ord('+'):
            self.conf = min(0.95, self.conf + 0.05)
            self.detector.conf = self.conf
            print(f"[INFO] Confidence -> {self.conf:.2f}")
        elif key == ord('-'):
            self.conf = max(0.05, self.conf - 0.05)
            self.detector.conf = self.conf
            print(f"[INFO] Confidence -> {self.conf:.2f}")
        return False

    # ------------------------------------------------------------------
    def _print_summary(self):
        print(f"\n{'='*60}")
        print(" SESSION SUMMARY")
        print(f"{'='*60}")
        print(f"  Frames processed : {self.frame_count}")
        avg = np.mean(self.fps_history) if self.fps_history else 0
        print(f"  Average FPS      : {avg:.1f}")
        if self.alert_mgr:
            stats = self.alert_mgr.get_alert_statistics()
            print(f"  Total alerts     : {stats['total_alerts']}")
            for lvl, cnt in stats.get('by_level', {}).items():
                print(f"    {lvl:12s}: {cnt}")
        print(f"{'='*60}\n")


# ===========================================================================
# Entry point
# ===========================================================================
def parse_args():
    p = argparse.ArgumentParser(description='Live Baby Hazard Detection')
    src = p.add_mutually_exclusive_group()
    src.add_argument('--camera', type=int, default=0,
                     help='Webcam index (default: 0)')
    src.add_argument('--video', type=str,
                     help='Path to a video file')
    p.add_argument('--output',     type=str,   help='Save annotated output video')
    p.add_argument('--models',     type=str,   default='models')
    p.add_argument('--config',     type=str,   default='config/config.yaml')
    p.add_argument('--conf',       type=float, default=0.45)
    p.add_argument('--no-display', action='store_true',
                   help='Run headless (no window)')
    return p.parse_args()


def main():
    args = parse_args()
    if args.video and not Path(args.video).exists():
        sys.exit(f"[ERROR] Video file not found: {args.video}")
    LiveDetector(args).run()


if __name__ == '__main__':
    main()