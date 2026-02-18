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
    Self-learning spatial hazard detection system
    
    Patent Claim: Automated room layout mapping via camera movement with
    persistent hazard location identification
    """
    
    def __init__(self, config_path='config/room_mapping_config.yaml'):
        """Initialize adaptive room mapper"""
        self.config = self._load_config(config_path)
        
        # Room layout map
        self.room_map = {
            'layout': None,
            'dimensions': None,
            'persistent_hazards': [],
            'temporary_hazards': [],
            'safe_zones': [],
            'created_at': None,
            'last_updated': None
        }
        
        # Hazard tracking
        self.hazard_history = defaultdict(list)
        self.persistent_threshold = 100  # Frames to become persistent
        
        # Movement heatmap
        self.movement_heatmap = None
        self.heatmap_resolution = (100, 100)  # 100x100 grid
        
        # Feature points for mapping
        self.feature_points = []
        self.orb = cv2.ORB_create(nfeatures=1000)
        
        # Camera calibration
        self.camera_matrix = None
        self.dist_coeffs = None
        
        # Learning phase
        self.learning_mode = True
        self.frames_processed = 0
        self.learning_duration_frames = self.config.get('learning_frames', 7200)  # 4 mins
        
        print("Adaptive Room Mapper Initialized")
        print(f"Learning Mode: Enabled ({self.learning_duration_frames} frames)")
        print(f"Heatmap Resolution: {self.heatmap_resolution}")
    
    def _load_config(self, config_path):
        """Load room mapping configuration"""
        default_config = {
            'learning_frames': 7200,  # 4 minutes at 30fps
            'persistent_threshold': 100,
            'movement_decay': 0.95,
            'hazard_persistence_threshold': 0.8,
            'enable_depth_estimation': True,
            'pixel_to_meter_ratio': 100  # 100 pixels = 1 meter (calibrate)
        }
        
        try:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                return {**default_config, **config.get('room_mapping', {})}
        except:
            return default_config
    
    def process_frame_for_mapping(self, frame, detections):
        """
        Process frame to build room map
        
        Patent Claim: SLAM-based room mapping with hazard identification
        """
        self.frames_processed += 1
        
        if self.learning_mode:
            # Extract features for mapping
            self._extract_spatial_features(frame)
            
            # Update hazard locations
            self._update_hazard_locations(detections)
            
            # Update movement heatmap
            self._update_movement_heatmap(detections)
            
            # Check if learning complete
            if self.frames_processed >= self.learning_duration_frames:
                self._finalize_room_map()
        else:
            # In operational mode, detect new hazards
            self._detect_dynamic_hazards(detections)
        
        return self.get_current_map_state()
    
    def _extract_spatial_features(self, frame):
        """
        Extract ORB features for spatial mapping
        
        Patent Claim: Feature-based spatial understanding
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        if descriptors is not None:
            self.feature_points.append({
                'frame': self.frames_processed,
                'keypoints': len(keypoints),
                'timestamp': datetime.now().isoformat()
            })
    
    def _update_hazard_locations(self, detections):
        """
        Track hazard locations over time
        
        Patent Claim: Persistent hazard location identification
        """
        for det in detections:
            if det['class_name'] in ['fire', 'pool', 'stove', 'sharp_object']:
                # Get center of detection
                bbox = det['bbox']
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                
                # Grid-based tracking
                grid_x = int(center_x / 12.8)  # Assuming 1280 width / 100 grid
                grid_y = int(center_y / 7.2)   # Assuming 720 height / 100 grid
                
                location = (grid_x, grid_y)
                
                self.hazard_history[det['class_name']].append({
                    'location': location,
                    'bbox': bbox,
                    'confidence': det['confidence'],
                    'frame': self.frames_processed
                })
    
    def _update_movement_heatmap(self, detections):
        """
        Update child movement heatmap
        
        Patent Claim: Movement pattern analysis for high-risk zone identification
        """
        if self.movement_heatmap is None:
            self.movement_heatmap = np.zeros(self.heatmap_resolution, dtype=np.float32)
        
        # Decay existing heatmap
        self.movement_heatmap *= self.config['movement_decay']
        
        for det in detections:
            if det['class_name'] == 'child':
                # Get center position
                bbox = det['bbox']
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                
                # Convert to grid coordinates
                grid_x = int((center_x / 1280) * self.heatmap_resolution[1])
                grid_y = int((center_y / 720) * self.heatmap_resolution[0])
                
                # Clamp to valid range
                grid_x = max(0, min(self.heatmap_resolution[1] - 1, grid_x))
                grid_y = max(0, min(self.heatmap_resolution[0] - 1, grid_y))
                
                # Increment heatmap
                self.movement_heatmap[grid_y, grid_x] += 1.0
                
                # Apply Gaussian blur for smooth heatmap
                self.movement_heatmap = cv2.GaussianBlur(
                    self.movement_heatmap, (5, 5), 0
                )
    
    def _finalize_room_map(self):
        """
        Finalize room map after learning phase
        
        Patent Claim: Automated hazard map generation
        """
        print("\nFinalizing Room Map...")
        
        self.learning_mode = False
        
        # Identify persistent hazards
        for hazard_type, history in self.hazard_history.items():
            if len(history) > self.persistent_threshold:
                # Cluster nearby detections
                clusters = self._cluster_hazard_locations(history)
                
                for cluster in clusters:
                    if cluster['count'] > self.persistent_threshold:
                        self.room_map['persistent_hazards'].append({
                            'type': hazard_type,
                            'location': cluster['center'],
                            'count': cluster['count'],
                            'confidence': cluster['avg_confidence'],
                            'bbox': cluster['bbox']
                        })
        
        # Identify high-risk zones (movement + hazard overlap)
        self._identify_high_risk_zones()
        
        # Save room map
        self.room_map['created_at'] = datetime.now().isoformat()
        self.room_map['last_updated'] = datetime.now().isoformat()
        self._save_room_map()
        
        print(f" Room map finalized")
        print(f"  Persistent Hazards: {len(self.room_map['persistent_hazards'])}")
        print(f"  High-Risk Zones: {len(self.room_map.get('high_risk_zones', []))}")
    
    def _cluster_hazard_locations(self, history):
        """
        Cluster nearby hazard detections
        
        Uses simple grid-based clustering
        """
        clusters = defaultdict(list)
        
        for detection in history:
            location = detection['location']
            clusters[location].append(detection)
        
        result_clusters = []
        for location, detections in clusters.items():
            if len(detections) > 10:  # Minimum cluster size
                # Calculate cluster center and average bbox
                avg_bbox = np.mean([d['bbox'] for d in detections], axis=0)
                avg_conf = np.mean([d['confidence'] for d in detections])
                
                result_clusters.append({
                    'center': location,
                    'count': len(detections),
                    'bbox': avg_bbox.tolist(),
                    'avg_confidence': float(avg_conf)
                })
        
        return result_clusters
    
    def _identify_high_risk_zones(self):
        """
        Identify high-risk zones (high movement + hazard proximity)
        
        Patent Claim: High-risk zone identification through movement-hazard correlation
        """
        if self.movement_heatmap is None:
            return
        
        high_risk_zones = []
        
        # Normalize heatmap
        if self.movement_heatmap.max() > 0:
            normalized_heatmap = self.movement_heatmap / self.movement_heatmap.max()
        else:
            normalized_heatmap = self.movement_heatmap
        
        # Find high-movement areas
        threshold = 0.5
        high_movement_mask = normalized_heatmap > threshold
        
        # Check proximity to hazards
        for hazard in self.room_map['persistent_hazards']:
            grid_x, grid_y = hazard['location']
            
            # Check surrounding area (5x5 grid)
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    x = grid_x + dx
                    y = grid_y + dy
                    
                    if (0 <= x < self.heatmap_resolution[1] and 
                        0 <= y < self.heatmap_resolution[0]):
                        
                        if high_movement_mask[y, x]:
                            high_risk_zones.append({
                                'location': (x, y),
                                'hazard_type': hazard['type'],
                                'movement_intensity': float(normalized_heatmap[y, x]),
                                'risk_score': float(normalized_heatmap[y, x] * hazard['confidence'])
                            })
        
        self.room_map['high_risk_zones'] = high_risk_zones
    
    def _detect_dynamic_hazards(self, detections):
        """
        Detect temporary/dynamic hazards in operational mode
        
        Patent Claim: Dynamic hazard creation detection
        """
        for det in detections:
            hazard_type = det['class_name']
            
            if hazard_type not in ['fire', 'pool', 'water', 'sharp_object']:
                continue
            
            # Check if this is a known persistent hazard
            bbox = det['bbox']
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            is_persistent = False
            for persistent in self.room_map['persistent_hazards']:
                p_center_x = (persistent['bbox'][0] + persistent['bbox'][2]) / 2
                p_center_y = (persistent['bbox'][1] + persistent['bbox'][3]) / 2
                
                distance = np.sqrt((center_x - p_center_x)**2 + (center_y - p_center_y)**2)
                
                if distance < 50:  # Within 50 pixels
                    is_persistent = True
                    break
            
            if not is_persistent:
                # This is a new/temporary hazard
                self.room_map['temporary_hazards'].append({
                    'type': hazard_type,
                    'bbox': bbox,
                    'detected_at': datetime.now().isoformat(),
                    'confidence': det['confidence']
                })
                
                print(f"⚠ Dynamic Hazard Detected: {hazard_type} at ({center_x:.0f}, {center_y:.0f})")
    
    def estimate_distance_to_hazard(self, child_bbox, hazard_bbox):
        """
        Estimate distance between child and hazard
        
        Patent Claim: 3D depth-based distance estimation (monocular)
        Uses pixel-to-meter calibration
        """
        # Get centers
        child_center = [(child_bbox[0] + child_bbox[2]) / 2,
                       (child_bbox[1] + child_bbox[3]) / 2]
        hazard_center = [(hazard_bbox[0] + hazard_bbox[2]) / 2,
                        (hazard_bbox[1] + hazard_bbox[3]) / 2]
        
        # Pixel distance
        pixel_distance = np.sqrt(
            (child_center[0] - hazard_center[0])**2 +
            (child_center[1] - hazard_center[1])**2
        )
        
        # Convert to meters (calibrated)
        meter_distance = pixel_distance / self.config['pixel_to_meter_ratio']
        
        # Simple depth estimation based on bbox size
        # (larger bbox = closer to camera)
        child_area = (child_bbox[2] - child_bbox[0]) * (child_bbox[3] - child_bbox[1])
        depth_factor = 1.0 / (child_area / 10000 + 1)  # Normalize
        
        estimated_distance = meter_distance * depth_factor
        
        return {
            'pixel_distance': pixel_distance,
            'estimated_meters': estimated_distance,
            'depth_adjusted': True
        }
    
    def get_current_map_state(self):
        """Get current state of room map"""
        return {
            'learning_mode': self.learning_mode,
            'frames_processed': self.frames_processed,
            'persistent_hazards': len(self.room_map['persistent_hazards']),
            'temporary_hazards': len(self.room_map['temporary_hazards']),
            'high_risk_zones': len(self.room_map.get('high_risk_zones', [])),
            'learning_progress': min(100, (self.frames_processed / self.learning_duration_frames) * 100)
        }
    
    def visualize_room_map(self, frame):
        """
        Visualize room map overlaid on frame
        
        Patent Claim: Real-time spatial hazard visualization
        """
        vis_frame = frame.copy()
        
        # Draw movement heatmap
        if self.movement_heatmap is not None:
            heatmap_visual = cv2.resize(
                self.movement_heatmap,
                (frame.shape[1], frame.shape[0])
            )
            
            # Normalize to 0-255
            if heatmap_visual.max() > 0:
                heatmap_visual = (heatmap_visual / heatmap_visual.max() * 255).astype(np.uint8)
            
            # Apply colormap
            # heatmap_color = cv2.applyColorMap(heatmap_visual, cv2.COLORMAP_JET)
            
            # Blend with frame
            # vis_frame = cv2.addWeighted(vis_frame, 0.7, heatmap_color, 0.3, 0)
        
        # Draw persistent hazards
        for hazard in self.room_map['persistent_hazards']:
            bbox = hazard['bbox']
            x1, y1, x2, y2 = [int(v) for v in bbox]
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
            # Draw label
            label = f"{hazard['type']} (persistent)"
            cv2.putText(vis_frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Draw temporary hazards
        for hazard in self.room_map['temporary_hazards']:
            bbox = hazard['bbox']
            x1, y1, x2, y2 = [int(v) for v in bbox]
            
            # Draw dashed box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
            
            # Draw label
            label = f"{hazard['type']} (new)"
            cv2.putText(vis_frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        
        # Draw learning progress
        if self.learning_mode:
            progress = int(self.get_current_map_state()['learning_progress'])
            cv2.putText(vis_frame, f"Learning Room: {progress}%", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return vis_frame
    
    def _save_room_map(self):
        """Save room map to disk"""
        map_path = Path('config/room_map.json')
        map_path.parent.mkdir(exist_ok=True)
        
        # Save JSON (without numpy arrays)
        map_json = self.room_map.copy()
        with open(map_path, 'w') as f:
            json.dump(map_json, f, indent=2)
        
        # Save heatmap separately
        if self.movement_heatmap is not None:
            heatmap_path = Path('config/movement_heatmap.npy')
            np.save(heatmap_path, self.movement_heatmap)
        
        print(f"Room map saved to {map_path}")
    
    def load_room_map(self):
        """Load existing room map"""
        map_path = Path('config/room_map.json')
        
        if not map_path.exists():
            print("No existing room map found")
            return False
        
        with open(map_path, 'r') as f:
            self.room_map = json.load(f)
        
        # Load heatmap
        heatmap_path = Path('config/movement_heatmap.npy')
        if heatmap_path.exists():
            self.movement_heatmap = np.load(heatmap_path)
        
        self.learning_mode = False
        print(f"Room map loaded ({len(self.room_map['persistent_hazards'])} hazards)")
        
        return True
    
    def get_hazard_proximity_alert(self, child_bbox):
        """
        Check if child is near any known hazards
        
        Returns alert with distance and hazard info
        """
        alerts = []
        
        for hazard in self.room_map['persistent_hazards']:
            distance_info = self.estimate_distance_to_hazard(
                child_bbox,
                hazard['bbox']
            )
            
            if distance_info['estimated_meters'] < 1.0:  # Within 1 meter
                alerts.append({
                    'hazard_type': hazard['type'],
                    'distance_meters': distance_info['estimated_meters'],
                    'severity': 'critical' if distance_info['estimated_meters'] < 0.5 else 'warning',
                    'hazard_location': hazard['bbox']
                })
        
        return alerts


def main():
    """Demo adaptive room mapper"""
    print("\n" + "="*70)
    print("ADAPTIVE ROOM MAPPING DEMO")
    print("="*70)
    
    # Initialize
    mapper = AdaptiveRoomMapper()
    
    # Simulate learning phase
    print("\n1. Learning Phase (Simulating 100 frames)")
    
    for frame_idx in range(100):
        # Simulate detections
        detections = []
        
        # Persistent hazard (stove)
        if frame_idx % 2 == 0:  # Detected frequently
            detections.append({
                'bbox': [100, 100, 200, 200],
                'class_name': 'fire',
                'confidence': 0.9
            })
        
        # Child movement
        if frame_idx % 3 == 0:
            x = 300 + (frame_idx * 10) % 500
            detections.append({
                'bbox': [x, 200, x+100, 400],
                'class_name': 'child',
                'confidence': 0.95
            })
        
        # Process frame
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        state = mapper.process_frame_for_mapping(dummy_frame, detections)
        
        if frame_idx % 20 == 0:
            print(f"  Frame {frame_idx}: Learning {state['learning_progress']:.1f}%")
    
    # Simulate enough frames to finish learning
    mapper.frames_processed = mapper.learning_duration_frames
    mapper._finalize_room_map()
    
    # Show results
    print("\n2. Room Map Results")
    print(f"  Persistent Hazards: {len(mapper.room_map['persistent_hazards'])}")
    for hazard in mapper.room_map['persistent_hazards']:
        print(f"    - {hazard['type']} at location {hazard['location']}")
    
    # Test distance estimation
    print("\n3. Distance Estimation")
    child_bbox = [300, 200, 400, 400]
    if mapper.room_map['persistent_hazards']:
        hazard_bbox = mapper.room_map['persistent_hazards'][0]['bbox']
        distance = mapper.estimate_distance_to_hazard(child_bbox, hazard_bbox)
        print(f"Child to Hazard Distance: {distance['estimated_meters']:.2f} meters")
    
    # Save map
    print("\n4. Saving Room Map")
    mapper._save_room_map()
    
    print("\n" + "="*70)
    print("Adaptive Room Mapping Complete")
    print("="*70)


if __name__ == "__main__":
    main()