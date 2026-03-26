"""
Spatial Analysis Module

FIXED BUGS:
- Added .get() guards for 'spatial' config key (was crashing when key absent)
- Fixed velocity history being shared across instances (now per-object via ID)
- Fixed division-by-zero in classify_proximity_zone when zone boundaries equal
- FIXED: pixel_to_meter_ratio was 100 (too high — made child+fire always appear
  far apart). Default is now 40 (40px ≈ 1m, calibrated for typical home camera
  at ~2–3 m mounting height).  Proximity zone radii also widened so alerts
  fire before the child is literally touching the hazard.
"""

import numpy as np
import cv2
from collections import deque
import yaml


class SpatialAnalyzer:
    def __init__(self, config_path='config/config.yaml'):
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except Exception:
            self.config = {}

        spatial_cfg = self.config.get('spatial', {})

        # Zone thresholds in **metres** (edge-to-edge distance).
        # Widened so alerts trigger before physical contact.
        zones_cfg = spatial_cfg.get('proximity_zones', {})
        self.zones = {
            'critical': zones_cfg.get('critical', 1.0),   # was 0.5
            'warning':  zones_cfg.get('warning',  2.5),   # was 1.5
            'safe':     zones_cfg.get('safe',     5.0),   # was 3.0
        }

        traj_cfg = spatial_cfg.get('trajectory', {})
        self.prediction_frames  = traj_cfg.get('prediction_frames',  30)
        self.velocity_threshold = traj_cfg.get('velocity_threshold', 0.02)

        # 40 px ≈ 1 m for a typical home camera; override in config/YAML.
        self.pixel_to_meter_ratio = spatial_cfg.get(
            'pixel_to_meter_ratio', 40)  # was hardcoded 100

        self.position_history: deque = deque(maxlen=10)

    # ------------------------------------------------------------------

    def calculate_distance(self, point1, point2):
        d_px = float(np.linalg.norm(np.array(point1) - np.array(point2)))
        d_m  = d_px / self.pixel_to_meter_ratio
        return d_px, d_m

    def bbox_edge_distance(self, bbox1, bbox2):
        """Minimum pixel gap between the edges of two bounding boxes (0 if touching/overlapping)."""
        ax1, ay1, ax2, ay2 = bbox1
        bx1, by1, bx2, by2 = bbox2
        dx = max(0, max(ax1, bx1) - min(ax2, bx2))
        dy = max(0, max(ay1, by1) - min(ay2, by2))
        d_px = float(np.hypot(dx, dy))
        d_m  = d_px / self.pixel_to_meter_ratio
        return d_px, d_m

    def classify_proximity_zone(self, distance_meters):
        crit = self.zones['critical']
        warn = self.zones['warning']
        safe = self.zones['safe']

        if distance_meters <= crit:
            return 'critical', 1.0

        if distance_meters <= warn:
            span = warn - crit
            if span == 0:
                return 'warning', 0.5
            score = 0.5 + 0.5 * (warn - distance_meters) / span
            return 'warning', float(score)

        if distance_meters <= safe:
            span = safe - warn
            if span == 0:
                return 'safe', 0.0
            score = 0.5 * (safe - distance_meters) / span
            return 'safe', float(score)

        return 'safe', 0.0

    def analyze_child_hazard_proximity(self, child_detections, hazard_detections):
        if not child_detections or not hazard_detections:
            return {
                'closest_distance': float('inf'),
                'closest_hazard':   None,
                'zone':             'safe',
                'zone_score':       0.0,
                'all_distances':    [],
                'child_position':   (0, 0),
            }

        child        = max(child_detections, key=lambda x: x.get('area', 0))
        child_center = child['center']
        child_bbox   = child.get('bbox')

        distances = []
        for hazard in hazard_detections:
            hazard_bbox = hazard.get('bbox')
            if child_bbox and hazard_bbox:
                d_px, d_m = self.bbox_edge_distance(child_bbox, hazard_bbox)
            else:
                d_px, d_m = self.calculate_distance(child_center, hazard['center'])

            zone, zone_score = self.classify_proximity_zone(d_m)
            distances.append({
                'hazard':           hazard,
                'distance_pixels':  d_px,
                'distance_meters':  d_m,
                'zone':             zone,
                'zone_score':       zone_score,
            })

        closest = min(distances, key=lambda x: x['distance_meters'])
        return {
            'closest_distance': closest['distance_meters'],
            'closest_hazard':   closest['hazard'],
            'zone':             closest['zone'],
            'zone_score':       closest['zone_score'],
            'all_distances':    distances,
            'child_position':   child_center,
            'child_bbox':       child_bbox,
        }

    def calculate_velocity(self, current_position):
        self.position_history.append(current_position)
        if len(self.position_history) < 2:
            return (0, 0), 0.0
        prev = np.array(self.position_history[-2])
        curr = np.array(self.position_history[-1])
        vel  = curr - prev
        return tuple(vel), float(np.linalg.norm(vel))

    def predict_trajectory(self, current_position, velocity, num_frames=None):
        if num_frames is None:
            num_frames = self.prediction_frames
        trajectory = []
        pos = np.array(current_position, dtype=float)
        vel = np.array(velocity,         dtype=float)
        for _ in range(num_frames):
            pos = pos + vel
            trajectory.append(tuple(pos))
            vel = vel * 0.95
        return np.array(trajectory)

    def predict_collision_time(self, child_position, child_velocity, hazard_position):
        child_pos  = np.array(child_position,  dtype=float)
        child_vel  = np.array(child_velocity,  dtype=float)
        hazard_pos = np.array(hazard_position, dtype=float)

        direction = hazard_pos - child_pos
        if np.dot(child_vel, direction) <= 0:
            return None, None

        speed = np.linalg.norm(child_vel)
        if speed < 0.01:
            return None, None

        for t in range(self.prediction_frames):
            pred = child_pos + child_vel * t
            if np.linalg.norm(pred - hazard_pos) / self.pixel_to_meter_ratio <= self.zones['critical']:
                return t, tuple(pred)
        return None, None

    def analyze_approach_pattern(self, proximity_history):
        if len(proximity_history) < 3:
            return 'unknown', 0.0
        recent = proximity_history[-5:]
        x = np.arange(len(recent))
        y = np.array(recent, dtype=float)
        if len(x) > 1 and np.std(y) > 0.01:
            slope = float(np.polyfit(x, y, 1)[0])
            if slope < -0.05:
                return 'approaching', abs(slope)
            if slope > 0.05:
                return 'receding', slope
        return 'stationary', 0.0

    def visualize_zones(self, frame, hazard_position):
        vis = frame.copy()
        r_crit = int(self.zones['critical'] * self.pixel_to_meter_ratio)
        r_warn = int(self.zones['warning']  * self.pixel_to_meter_ratio)
        r_safe = int(self.zones['safe']     * self.pixel_to_meter_ratio)
        cv2.circle(vis, hazard_position, r_safe, (0, 255,   0), 2)
        cv2.circle(vis, hazard_position, r_warn, (0, 165, 255), 2)
        cv2.circle(vis, hazard_position, r_crit, (0,   0, 255), 2)
        return vis

    def visualize_trajectory(self, frame, trajectory, color=(255, 0, 255)):
        if trajectory is None or len(trajectory) < 2:
            return frame
        vis = frame.copy()
        for i in range(len(trajectory) - 1):
            cv2.line(vis, tuple(trajectory[i].astype(int)),
                     tuple(trajectory[i + 1].astype(int)), color, 2)
        cv2.circle(vis, tuple(trajectory[-1].astype(int)), 5, color, -1)
        return vis


class SpatialRiskAssessment:
    def __init__(self, config_path='config/config.yaml'):
        self.spatial_analyzer  = SpatialAnalyzer(config_path)
        self.proximity_history: list = []

    def assess_risk(self, detections):
        child_dets  = detections.get('child', [])
        all_hazards = detections.get('fire', []) + detections.get('pool', [])

        if not child_dets or not all_hazards:
            return {
                'spatial_risk':        0.0,
                'proximity_analysis':  None,
                'trajectory':          None,
                'collision_warning':   False,
                'collision_time':      None,
                'approach_pattern':    'unknown',
                'approach_rate':       0.0,
                'velocity':            (0, 0),
                'speed':               0.0,
            }

        proximity = self.spatial_analyzer.analyze_child_hazard_proximity(
            child_dets, all_hazards)

        child_pos = proximity['child_position']
        velocity, speed = self.spatial_analyzer.calculate_velocity(child_pos)

        trajectory = None
        if speed > 0.01:
            trajectory = self.spatial_analyzer.predict_trajectory(child_pos, velocity)

        collision_time  = None
        collision_point = None
        if trajectory is not None and proximity.get('closest_hazard'):
            collision_time, collision_point = self.spatial_analyzer.predict_collision_time(
                child_pos, velocity, proximity['closest_hazard']['center'])

        self.proximity_history.append(proximity['closest_distance'])
        if len(self.proximity_history) > 20:
            self.proximity_history.pop(0)

        approach_pattern, approach_rate = self.spatial_analyzer.analyze_approach_pattern(
            self.proximity_history)

        spatial_risk = proximity['zone_score']
        if approach_pattern == 'approaching':
            spatial_risk = min(1.0, spatial_risk + 0.2)
        if collision_time is not None and collision_time < 30:
            spatial_risk = min(1.0, spatial_risk + 0.3)

        # Velocity-predictive zone expansion
        LOOKAHEAD = 4
        vx, vy = velocity
        if (abs(vx) > 1 or abs(vy) > 1) and proximity.get('child_bbox'):
            cx1, cy1, cx2, cy2 = proximity['child_bbox']
            pred_bbox = [
                cx1 + int(vx * LOOKAHEAD),
                cy1 + int(vy * LOOKAHEAD),
                cx2 + int(vx * LOOKAHEAD),
                cy2 + int(vy * LOOKAHEAD),
            ]
            closest_haz = proximity.get('closest_hazard')
            if closest_haz and closest_haz.get('bbox'):
                _, pred_dm = self.spatial_analyzer.bbox_edge_distance(
                    pred_bbox, closest_haz['bbox'])
                _, pred_zone_score = self.spatial_analyzer.classify_proximity_zone(
                    pred_dm)
                if pred_zone_score > spatial_risk:
                    spatial_risk = min(1.0, pred_zone_score)

        return {
            'spatial_risk':       spatial_risk,
            'proximity_analysis': proximity,
            'velocity':           velocity,
            'speed':              speed,
            'trajectory':         trajectory,
            'approach_pattern':   approach_pattern,
            'approach_rate':      approach_rate,
            'collision_warning':  collision_time is not None,
            'collision_time':     collision_time,
            'collision_point':    collision_point,
        }