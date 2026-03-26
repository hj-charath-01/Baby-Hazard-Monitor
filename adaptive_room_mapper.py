"""
Adaptive Room Mapping with Spatial Hazard Learning
Self-learning system that maps rooms and identifies persistent hazard locations

Patent Features:
- Automated room layout mapping
- Persistent hazard location identification
- Dynamic hazard detection (temporary hazards)
- 3D depth-based distance estimation
- Movement pattern analysis
- High-risk zone identification

FIXED BUGS:
- Removed hardcoded 1280/12.8 and 720/7.2 magic numbers in
  _update_hazard_locations and _update_movement_heatmap; frame dimensions
  are now read from the actual frame shape so the mapper works at any
  resolution.
- pixel_to_meter_ratio aligned with spatial_analysis.py (40 px/m).
"""

import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from collections import defaultdict, deque
import pickle


class AdaptiveRoomMapper:
    """
    Self-learning spatial hazard detection system.

    Patent Claim: Automated room layout mapping via camera movement with
    persistent hazard location identification.
    """

    def __init__(self, config_path='config/room_mapping_config.yaml'):
        self.config = self._load_config(config_path)

        self.room_map = {
            'layout':             None,
            'dimensions':         None,
            'persistent_hazards': [],
            'temporary_hazards':  [],
            'safe_zones':         [],
            'created_at':         None,
            'last_updated':       None,
        }

        self.hazard_history       = defaultdict(list)
        self.persistent_threshold = self.config.get('persistent_threshold', 15)

        self.movement_heatmap   = None
        self.heatmap_resolution = (100, 100)

        self.feature_points = []
        self.orb = cv2.ORB_create(nfeatures=500)   # was 1000 — halved for speed

        self.camera_matrix = None
        self.dist_coeffs   = None

        self.learning_mode            = True
        self.frames_processed         = 0
        self.learning_duration_frames = self.config.get('learning_frames', 900)

        # Cached frame size — filled on first frame; no hardcoded dimensions.
        self._frame_w = None
        self._frame_h = None

        print("Adaptive Room Mapper Initialized")
        print(f"Learning Mode: Enabled ({self.learning_duration_frames} frames)")
        print(f"Heatmap Resolution: {self.heatmap_resolution}")

    # ------------------------------------------------------------------
    def _load_config(self, config_path):
        default_config = {
            # --- SPEED-UP DEFAULTS ---
            # 900 frames  = ~30 s at 30 fps  (was 7200 / 4 min)
            'learning_frames':              900,
            # 15 sightings to lock a hazard  (was 100)
            'persistent_threshold':         15,
            # Early-finish: if a hazard is seen this many times we consider
            # the map "good enough" and don't wait for learning_frames.
            'early_finish_sightings':       40,
            'movement_decay':               0.92,
            'hazard_persistence_threshold': 0.6,
            'enable_depth_estimation':      True,
            'pixel_to_meter_ratio':         40,
            # Run ORB feature extraction only every N frames (was every frame)
            'orb_every_n_frames':           10,
        }
        try:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                return {**default_config, **config.get('room_mapping', {})}
        except Exception:
            return default_config

    # ------------------------------------------------------------------
    def _cache_frame_size(self, frame):
        if self._frame_h is None:
            self._frame_h, self._frame_w = frame.shape[:2]

    # ------------------------------------------------------------------
    def process_frame_for_mapping(self, frame, detections):
        """Process frame to build room map."""
        self._cache_frame_size(frame)
        self.frames_processed += 1

        if self.learning_mode:
            # ORB is slow — only run every N frames
            if self.frames_processed % self.config.get('orb_every_n_frames', 10) == 0:
                self._extract_spatial_features(frame)

            self._update_hazard_locations(frame, detections)
            self._update_movement_heatmap(frame, detections)

            # Early-finish: enough confident persistent sightings already
            early = self.config.get('early_finish_sightings', 40)
            max_sightings = max(
                (len(v) for v in self.hazard_history.values()), default=0)
            time_done  = self.frames_processed >= self.learning_duration_frames
            early_done = max_sightings >= early

            if time_done or early_done:
                reason = "early finish" if early_done else "time limit"
                print(f"[Mapper] Finalising ({reason}, "
                      f"{self.frames_processed} frames, "
                      f"max sightings={max_sightings})")
                self._finalize_room_map()
        else:
            self._detect_dynamic_hazards(detections)

        return self.get_current_map_state()

    # ------------------------------------------------------------------
    def _extract_spatial_features(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, _ = self.orb.detectAndCompute(gray, None)
        self.feature_points.append({
            'frame':     self.frames_processed,
            'keypoints': len(keypoints),
            'timestamp': datetime.now().isoformat(),
        })

    # ------------------------------------------------------------------
    def _update_hazard_locations(self, frame, detections):
        """Track hazard locations over time.

        BUG FIX: Grid-cell calculation now uses actual frame dimensions
        instead of hardcoded 1280/720 values.
        """
        h, w = frame.shape[:2]
        cell_w = w / self.heatmap_resolution[1]
        cell_h = h / self.heatmap_resolution[0]

        for det in detections:
            if det['class_name'] not in ['fire', 'pool', 'stove', 'sharp_object']:
                continue
            bbox = det['bbox']
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2

            grid_x = int(center_x / cell_w)
            grid_y = int(center_y / cell_h)
            location = (grid_x, grid_y)

            self.hazard_history[det['class_name']].append({
                'location':   location,
                'bbox':       bbox,
                'confidence': det['confidence'],
                'frame':      self.frames_processed,
            })

    # ------------------------------------------------------------------
    def _update_movement_heatmap(self, frame, detections):
        """Update child movement heatmap.

        BUG FIX: Grid conversion now uses actual frame dimensions instead
        of hardcoded 1280/720 values.
        """
        if self.movement_heatmap is None:
            self.movement_heatmap = np.zeros(self.heatmap_resolution, dtype=np.float32)

        self.movement_heatmap *= self.config['movement_decay']

        h, w = frame.shape[:2]
        updated = False
        for det in detections:
            if det['class_name'] != 'child':
                continue
            bbox = det['bbox']
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2

            grid_x = int((center_x / w) * self.heatmap_resolution[1])
            grid_y = int((center_y / h) * self.heatmap_resolution[0])

            grid_x = max(0, min(self.heatmap_resolution[1] - 1, grid_x))
            grid_y = max(0, min(self.heatmap_resolution[0] - 1, grid_y))

            self.movement_heatmap[grid_y, grid_x] += 1.0
            updated = True

        # Blur only every 5 frames to reduce CPU overhead
        if updated and self.frames_processed % 5 == 0:
            self.movement_heatmap = cv2.GaussianBlur(
                self.movement_heatmap, (5, 5), 0)

    # ------------------------------------------------------------------
    def _finalize_room_map(self):
        print("\nFinalizing Room Map...")
        self.learning_mode = False

        for hazard_type, history in self.hazard_history.items():
            if len(history) > self.persistent_threshold:
                clusters = self._cluster_hazard_locations(history)
                for cluster in clusters:
                    if cluster['count'] > self.persistent_threshold:
                        self.room_map['persistent_hazards'].append({
                            'type':       hazard_type,
                            'location':   cluster['center'],
                            'count':      cluster['count'],
                            'confidence': cluster['avg_confidence'],
                            'bbox':       cluster['bbox'],
                        })

        self._identify_high_risk_zones()
        self.room_map['created_at']   = datetime.now().isoformat()
        self.room_map['last_updated'] = datetime.now().isoformat()
        self._save_room_map()

        print(" Room map finalized")
        print(f"  Persistent Hazards : {len(self.room_map['persistent_hazards'])}")
        print(f"  High-Risk Zones    : {len(self.room_map.get('high_risk_zones', []))}")

    # ------------------------------------------------------------------
    def _cluster_hazard_locations(self, history):
        clusters = defaultdict(list)
        for detection in history:
            clusters[detection['location']].append(detection)

        result = []
        for location, dets in clusters.items():
            if len(dets) > 10:
                avg_bbox = np.mean([d['bbox'] for d in dets], axis=0)
                avg_conf = np.mean([d['confidence'] for d in dets])
                result.append({
                    'center':          location,
                    'count':           len(dets),
                    'bbox':            avg_bbox.tolist(),
                    'avg_confidence':  float(avg_conf),
                })
        return result

    # ------------------------------------------------------------------
    def _identify_high_risk_zones(self):
        if self.movement_heatmap is None:
            return

        high_risk_zones = []
        if self.movement_heatmap.max() > 0:
            normalized = self.movement_heatmap / self.movement_heatmap.max()
        else:
            normalized = self.movement_heatmap

        high_movement = normalized > 0.5

        for hazard in self.room_map['persistent_hazards']:
            gx, gy = hazard['location']
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    x, y = gx + dx, gy + dy
                    if (0 <= x < self.heatmap_resolution[1] and
                            0 <= y < self.heatmap_resolution[0]):
                        if high_movement[y, x]:
                            high_risk_zones.append({
                                'location':         (x, y),
                                'hazard_type':      hazard['type'],
                                'movement_intensity': float(normalized[y, x]),
                                'risk_score':        float(normalized[y, x] *
                                                           hazard['confidence']),
                            })

        self.room_map['high_risk_zones'] = high_risk_zones

    # ------------------------------------------------------------------
    def _detect_dynamic_hazards(self, detections):
        for det in detections:
            hazard_type = det['class_name']
            if hazard_type not in ['fire', 'pool', 'water', 'sharp_object']:
                continue

            bbox     = det['bbox']
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2

            is_persistent = False
            for p in self.room_map['persistent_hazards']:
                px = (p['bbox'][0] + p['bbox'][2]) / 2
                py = (p['bbox'][1] + p['bbox'][3]) / 2
                if np.sqrt((center_x - px)**2 + (center_y - py)**2) < 50:
                    is_persistent = True
                    break

            if not is_persistent:
                self.room_map['temporary_hazards'].append({
                    'type':        hazard_type,
                    'bbox':        bbox,
                    'detected_at': datetime.now().isoformat(),
                    'confidence':  det['confidence'],
                })
                print(f"Dynamic Hazard Detected: {hazard_type} at ({center_x:.0f}, {center_y:.0f})")

    # ------------------------------------------------------------------
    def estimate_distance_to_hazard(self, child_bbox, hazard_bbox):
        child_c  = [(child_bbox[0] + child_bbox[2]) / 2,
                    (child_bbox[1] + child_bbox[3]) / 2]
        hazard_c = [(hazard_bbox[0] + hazard_bbox[2]) / 2,
                    (hazard_bbox[1] + hazard_bbox[3]) / 2]

        pixel_dist  = np.sqrt((child_c[0] - hazard_c[0])**2 +
                              (child_c[1] - hazard_c[1])**2)
        meter_dist  = pixel_dist / self.config['pixel_to_meter_ratio']

        child_area  = ((child_bbox[2] - child_bbox[0]) *
                       (child_bbox[3] - child_bbox[1]))
        depth_factor = 1.0 / (child_area / 10000 + 1)
        estimated    = meter_dist * depth_factor

        return {
            'pixel_distance':  pixel_dist,
            'estimated_meters': estimated,
            'depth_adjusted':   True,
        }

    # ------------------------------------------------------------------
    def get_current_map_state(self):
        return {
            'learning_mode':      self.learning_mode,
            'frames_processed':   self.frames_processed,
            'persistent_hazards': len(self.room_map['persistent_hazards']),
            'temporary_hazards':  len(self.room_map['temporary_hazards']),
            'high_risk_zones':    len(self.room_map.get('high_risk_zones', [])),
            'learning_progress':  min(100, (self.frames_processed /
                                           self.learning_duration_frames) * 100),
        }

    # ------------------------------------------------------------------
    def visualize_room_map(self, frame):
        vis = frame.copy()

        if self.movement_heatmap is not None:
            hmap = cv2.resize(self.movement_heatmap,
                              (frame.shape[1], frame.shape[0]))
            if hmap.max() > 0:
                hmap = (hmap / hmap.max() * 255).astype(np.uint8)
            hmap_color = cv2.applyColorMap(hmap, cv2.COLORMAP_JET)
            vis = cv2.addWeighted(vis, 0.7, hmap_color, 0.3, 0)

        for hazard in self.room_map['persistent_hazards']:
            x1, y1, x2, y2 = [int(v) for v in hazard['bbox']]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(vis, f"{hazard['type']} (persistent)",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2)

        for hazard in self.room_map['temporary_hazards']:
            x1, y1, x2, y2 = [int(v) for v in hazard['bbox']]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.putText(vis, f"{hazard['type']} (new)",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 165, 255), 2)

        if self.learning_mode:
            progress = int(self.get_current_map_state()['learning_progress'])
            cv2.putText(vis, f"Learning Room: {progress}%",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)

        return vis

    # ------------------------------------------------------------------
    def _save_room_map(self):
        map_path = Path('config/room_map.json')
        map_path.parent.mkdir(exist_ok=True)
        with open(map_path, 'w') as f:
            json.dump(self.room_map, f, indent=2)
        if self.movement_heatmap is not None:
            np.save(Path('config/movement_heatmap.npy'), self.movement_heatmap)
        print(f"Room map saved to {map_path}")

    def load_room_map(self):
        map_path = Path('config/room_map.json')
        if not map_path.exists():
            print("No existing room map found")
            return False
        with open(map_path, 'r') as f:
            self.room_map = json.load(f)
        hmap_path = Path('config/movement_heatmap.npy')
        if hmap_path.exists():
            self.movement_heatmap = np.load(hmap_path)
        self.learning_mode = False
        print(f"Room map loaded ({len(self.room_map['persistent_hazards'])} hazards)")
        return True

    def get_hazard_proximity_alert(self, child_bbox):
        alerts = []
        for hazard in self.room_map['persistent_hazards']:
            info = self.estimate_distance_to_hazard(child_bbox, hazard['bbox'])
            if info['estimated_meters'] < 1.0:
                alerts.append({
                    'hazard_type':      hazard['type'],
                    'distance_meters':  info['estimated_meters'],
                    'severity':         'critical' if info['estimated_meters'] < 0.5
                                        else 'warning',
                    'hazard_location':  hazard['bbox'],
                })
        return alerts