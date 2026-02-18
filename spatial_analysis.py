"""
Spatial Analysis Module
Implements proximity zone calculation, distance estimation,
and trajectory prediction for hazard assessment
"""

import numpy as np
import cv2
from scipy.spatial import distance
import yaml


class SpatialAnalyzer:
    """
    Spatial analysis for child-hazard proximity assessment
    Includes distance calculation, zone classification, and trajectory prediction
    """
    
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Proximity zones (in meters)
        self.zones = self.config['spatial']['proximity_zones']
        
        # Calibration parameters (pixels to meters)
        # These should be calibrated for each camera setup
        self.pixel_to_meter_ratio = 100  # pixels per meter (adjustable)
        
        # Trajectory settings
        self.prediction_frames = self.config['spatial']['trajectory']['prediction_frames']
        self.velocity_threshold = self.config['spatial']['trajectory']['velocity_threshold']
        
        # History for velocity calculation
        self.position_history = []
        self.max_history = 10
    
    def calculate_distance(self, point1, point2):
        """
        Calculate Euclidean distance between two points
        
        Args:
            point1: (x, y) coordinates
            point2: (x, y) coordinates
            
        Returns:
            distance_pixels: Distance in pixels
            distance_meters: Estimated distance in meters
        """
        p1 = np.array(point1)
        p2 = np.array(point2)
        
        distance_pixels = np.linalg.norm(p1 - p2)
        distance_meters = distance_pixels / self.pixel_to_meter_ratio
        
        return distance_pixels, distance_meters
    
    def classify_proximity_zone(self, distance_meters):
        """
        Classify distance into proximity zones
        
        Returns:
            zone: 'critical', 'warning', or 'safe'
            zone_score: Normalized score (0-1, higher is more dangerous)
        """
        if distance_meters <= self.zones['critical']:
            return 'critical', 1.0
        elif distance_meters <= self.zones['warning']:
            # Linear interpolation between critical and warning
            score = 0.5 + 0.5 * (self.zones['warning'] - distance_meters) / \
                    (self.zones['warning'] - self.zones['critical'])
            return 'warning', score
        elif distance_meters <= self.zones['safe']:
            # Linear interpolation between warning and safe
            score = 0.5 * (self.zones['safe'] - distance_meters) / \
                    (self.zones['safe'] - self.zones['warning'])
            return 'safe', score
        else:
            return 'safe', 0.0
    
    def analyze_child_hazard_proximity(self, child_detections, hazard_detections):
        """
        Analyze proximity between child and all detected hazards
        
        Args:
            child_detections: List of child detections
            hazard_detections: List of hazard detections (fire/pool)
            
        Returns:
            proximity_analysis: Dictionary with detailed proximity information
        """
        if not child_detections or not hazard_detections:
            return {
                'closest_distance': float('inf'),
                'closest_hazard': None,
                'zone': 'safe',
                'zone_score': 0.0,
                'all_distances': []
            }
        
        # Get primary child (largest bounding box)
        child = max(child_detections, key=lambda x: x['area'])
        child_center = child['center']
        
        # Calculate distances to all hazards
        distances = []
        for hazard in hazard_detections:
            hazard_center = hazard['center']
            dist_pixels, dist_meters = self.calculate_distance(
                child_center, hazard_center
            )
            
            zone, zone_score = self.classify_proximity_zone(dist_meters)
            
            distances.append({
                'hazard': hazard,
                'distance_pixels': dist_pixels,
                'distance_meters': dist_meters,
                'zone': zone,
                'zone_score': zone_score
            })
        
        # Find closest hazard
        closest = min(distances, key=lambda x: x['distance_meters'])
        
        proximity_analysis = {
            'closest_distance': closest['distance_meters'],
            'closest_hazard': closest['hazard'],
            'zone': closest['zone'],
            'zone_score': closest['zone_score'],
            'all_distances': distances,
            'child_position': child_center
        }
        
        return proximity_analysis
    
    def calculate_velocity(self, current_position):
        """
        Calculate velocity from position history
        
        Returns:
            velocity: (vx, vy) velocity vector in pixels/frame
            speed: Scalar speed in pixels/frame
        """
        self.position_history.append(current_position)
        
        # Maintain max history
        if len(self.position_history) > self.max_history:
            self.position_history.pop(0)
        
        if len(self.position_history) < 2:
            return (0, 0), 0
        
        # Calculate velocity from last two positions
        prev_pos = np.array(self.position_history[-2])
        curr_pos = np.array(self.position_history[-1])
        
        velocity = curr_pos - prev_pos
        speed = np.linalg.norm(velocity)
        
        return tuple(velocity), speed
    
    def predict_trajectory(self, current_position, velocity, num_frames=None):
        """
        Predict future trajectory based on current velocity
        
        Args:
            current_position: Current (x, y) position
            velocity: (vx, vy) velocity vector
            num_frames: Number of frames to predict (default from config)
            
        Returns:
            trajectory: Array of predicted positions
        """
        if num_frames is None:
            num_frames = self.prediction_frames
        
        trajectory = []
        pos = np.array(current_position)
        vel = np.array(velocity)
        
        for _ in range(num_frames):
            pos = pos + vel
            trajectory.append(tuple(pos))
            
            # Apply simple friction/deceleration
            vel = vel * 0.95
        
        return np.array(trajectory)
    
    def predict_collision_time(self, child_position, child_velocity, hazard_position):
        """
        Predict time until collision with hazard
        
        Returns:
            collision_time: Number of frames until collision (None if no collision)
            collision_point: Predicted collision point
        """
        child_pos = np.array(child_position, dtype=float)
        child_vel = np.array(child_velocity, dtype=float)
        hazard_pos = np.array(hazard_position, dtype=float)
        
        # Check if moving toward hazard
        direction_to_hazard = hazard_pos - child_pos
        if np.dot(child_vel, direction_to_hazard) <= 0:
            return None, None  # Moving away from hazard
        
        # Simple linear prediction
        speed = np.linalg.norm(child_vel)
        if speed < 0.01:
            return None, None
        
        # Project trajectory
        for t in range(self.prediction_frames):
            predicted_pos = child_pos + child_vel * t
            distance_to_hazard = np.linalg.norm(predicted_pos - hazard_pos)
            
            # Check if within critical zone
            if distance_to_hazard / self.pixel_to_meter_ratio <= self.zones['critical']:
                return t, tuple(predicted_pos)
        
        return None, None
    
    def analyze_approach_pattern(self, proximity_history):
        """
        Analyze whether child is approaching or moving away from hazard
        
        Args:
            proximity_history: List of recent distance measurements
            
        Returns:
            pattern: 'approaching', 'receding', or 'stationary'
            rate: Rate of approach/recession (meters/frame)
        """
        if len(proximity_history) < 3:
            return 'unknown', 0.0
        
        # Calculate trend
        recent = proximity_history[-5:]
        
        # Linear regression
        x = np.arange(len(recent))
        y = np.array(recent)
        
        if len(x) > 1 and np.std(y) > 0.01:
            slope = np.polyfit(x, y, 1)[0]
            
            if slope < -0.05:  # Approaching
                return 'approaching', abs(slope)
            elif slope > 0.05:  # Receding
                return 'receding', slope
        
        return 'stationary', 0.0
    
    def visualize_zones(self, frame, hazard_position):
        """
        Draw proximity zones around hazard
        
        Args:
            frame: Input image
            hazard_position: (x, y) position of hazard
            
        Returns:
            frame: Image with zones drawn
        """
        vis_frame = frame.copy()
        
        # Calculate radii in pixels
        critical_radius = int(self.zones['critical'] * self.pixel_to_meter_ratio)
        warning_radius = int(self.zones['warning'] * self.pixel_to_meter_ratio)
        safe_radius = int(self.zones['safe'] * self.pixel_to_meter_ratio)
        
        # Draw zones (outer to inner for proper layering)
        cv2.circle(vis_frame, hazard_position, safe_radius, (0, 255, 0), 2)      # Green
        cv2.circle(vis_frame, hazard_position, warning_radius, (0, 165, 255), 2)  # Orange
        cv2.circle(vis_frame, hazard_position, critical_radius, (0, 0, 255), 2)   # Red
        
        # Add labels
        cv2.putText(vis_frame, 'Critical', 
                   (hazard_position[0] - 30, hazard_position[1] + critical_radius + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(vis_frame, 'Warning', 
                   (hazard_position[0] - 30, hazard_position[1] + warning_radius + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        cv2.putText(vis_frame, 'Safe', 
                   (hazard_position[0] - 20, hazard_position[1] + safe_radius + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return vis_frame
    
    def visualize_trajectory(self, frame, trajectory, color=(255, 0, 255)):
        """
        Draw predicted trajectory on frame
        
        Args:
            frame: Input image
            trajectory: Array of (x, y) positions
            color: Line color (BGR)
        """
        if trajectory is None or len(trajectory) < 2:
            return frame
        
        vis_frame = frame.copy()
        
        # Draw trajectory line
        for i in range(len(trajectory) - 1):
            pt1 = tuple(trajectory[i].astype(int))
            pt2 = tuple(trajectory[i + 1].astype(int))
            cv2.line(vis_frame, pt1, pt2, color, 2)
        
        # Draw endpoint
        endpoint = tuple(trajectory[-1].astype(int))
        cv2.circle(vis_frame, endpoint, 5, color, -1)
        
        return vis_frame


class SpatialRiskAssessment:
    """High-level spatial risk assessment"""
    
    def __init__(self, config_path='config/config.yaml'):
        self.spatial_analyzer = SpatialAnalyzer(config_path)
        self.proximity_history = []
        
    def assess_risk(self, detections):
        """
        Comprehensive spatial risk assessment
        
        Returns:
            risk_assessment: Dictionary with spatial risk analysis
        """
        child_detections = detections.get('child', [])
        fire_detections = detections.get('fire', [])
        pool_detections = detections.get('pool', [])
        
        # Combine all hazards
        all_hazards = fire_detections + pool_detections
        
        if not child_detections or not all_hazards:
            return {
                'spatial_risk': 0.0,
                'proximity_analysis': None,
                'trajectory_analysis': None,
                'collision_warning': False
            }
        
        # Proximity analysis
        proximity = self.spatial_analyzer.analyze_child_hazard_proximity(
            child_detections, all_hazards
        )
        
        # Velocity and trajectory
        child_pos = proximity['child_position']
        velocity, speed = self.spatial_analyzer.calculate_velocity(child_pos)
        
        trajectory = None
        if speed > 0.01:
            trajectory = self.spatial_analyzer.predict_trajectory(
                child_pos, velocity
            )
        
        # Collision prediction
        collision_time = None
        collision_point = None
        if trajectory is not None and proximity['closest_hazard']:
            collision_time, collision_point = self.spatial_analyzer.predict_collision_time(
                child_pos, velocity, proximity['closest_hazard']['center']
            )
        
        # Update history
        self.proximity_history.append(proximity['closest_distance'])
        if len(self.proximity_history) > 20:
            self.proximity_history.pop(0)
        
        # Approach pattern
        approach_pattern, approach_rate = self.spatial_analyzer.analyze_approach_pattern(
            self.proximity_history
        )
        
        # Calculate overall spatial risk
        spatial_risk = proximity['zone_score']
        
        # Increase risk if approaching
        if approach_pattern == 'approaching':
            spatial_risk = min(1.0, spatial_risk + 0.2)
        
        # Increase risk if collision predicted
        if collision_time is not None and collision_time < 30:  # Within 1 second
            spatial_risk = min(1.0, spatial_risk + 0.3)
        
        risk_assessment = {
            'spatial_risk': spatial_risk,
            'proximity_analysis': proximity,
            'velocity': velocity,
            'speed': speed,
            'trajectory': trajectory,
            'approach_pattern': approach_pattern,
            'approach_rate': approach_rate,
            'collision_warning': collision_time is not None,
            'collision_time': collision_time,
            'collision_point': collision_point
        }
        
        return risk_assessment


def main():
    """Test spatial analysis module"""
    print("\n" + "="*60)
    print("SPATIAL ANALYSIS MODULE TEST")
    print("="*60)
    
    assessor = SpatialRiskAssessment()
    
    # Simulate scenario: child approaching fire
    print("\nSimulating child approaching fire...")
    
    for i in range(20):
        # Child moving toward fire
        detections = {
            'child': [{'center': (400 + i*10, 360), 'area': 5000}],
            'fire': [{'center': (600, 360)}],
            'pool': []
        }
        
        risk = assessor.assess_risk(detections)
        
        if i % 5 == 0:
            print(f"\nFrame {i}:")
            print(f"  Spatial Risk: {risk['spatial_risk']:.3f}")
            print(f"  Zone: {risk['proximity_analysis']['zone']}")
            print(f"  Distance: {risk['proximity_analysis']['closest_distance']:.2f}m")
            print(f"  Pattern: {risk['approach_pattern']}")
            print(f"  Collision Warning: {risk['collision_warning']}")
    
    print("\n✓ Spatial analysis module test complete")


if __name__ == "__main__":
    main()