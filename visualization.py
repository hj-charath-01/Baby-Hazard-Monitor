"""
Visualization Utilities
Tools for visualizing detection results, statistics, and system performance
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime


class ResultVisualizer:
    """Visualize detection and monitoring results"""
    
    def __init__(self):
        self.colors = {
            'child': (0, 255, 0),    # Green
            'fire': (0, 0, 255),     # Red
            'pool': (255, 0, 0),     # Blue
            'warning': (0, 165, 255), # Orange
            'critical': (0, 0, 255)   # Red
        }
    
    def draw_detections(self, image, detections, show_labels=True):
        """
        Draw bounding boxes for all detections
        
        Args:
            image: Input image
            detections: Dictionary with detection results
            show_labels: Whether to show class labels
        
        Returns:
            vis_image: Image with drawn detections
        """
        vis_image = image.copy()
        
        for det_type in ['child', 'fire', 'pool']:
            if det_type in detections:
                color = self.colors[det_type]
                
                for det in detections[det_type]:
                    x1, y1, x2, y2 = det['bbox']
                    conf = det['confidence']
                    
                    # Draw bounding box
                    cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
                    
                    if show_labels:
                        # Draw label background
                        label = f"{det_type}: {conf:.2f}"
                        label_size, _ = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                        )
                        
                        cv2.rectangle(
                            vis_image,
                            (x1, y1 - label_size[1] - 10),
                            (x1 + label_size[0], y1),
                            color, -1
                        )
                        
                        # Draw label text
                        cv2.putText(
                            vis_image, label,
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1
                        )
        
        return vis_image
    
    def draw_proximity_zones(self, image, hazard_center, zone_radii):
        """
        Draw proximity zones around hazard
        
        Args:
            image: Input image
            hazard_center: (x, y) center of hazard
            zone_radii: Dict with zone radii in pixels
        
        Returns:
            vis_image: Image with zones drawn
        """
        vis_image = image.copy()
        
        # Create semi-transparent overlay
        overlay = vis_image.copy()
        
        # Draw zones from outer to inner
        if 'safe' in zone_radii:
            cv2.circle(overlay, hazard_center, zone_radii['safe'], 
                      (0, 255, 0), 2)
        
        if 'warning' in zone_radii:
            cv2.circle(overlay, hazard_center, zone_radii['warning'], 
                      (0, 165, 255), 2)
        
        if 'critical' in zone_radii:
            cv2.circle(overlay, hazard_center, zone_radii['critical'], 
                      (0, 0, 255), 2)
        
        # Blend overlay
        cv2.addWeighted(overlay, 0.7, vis_image, 0.3, 0, vis_image)
        
        return vis_image
    
    def draw_trajectory(self, image, trajectory_points, color=(255, 0, 255)):
        """
        Draw predicted trajectory
        
        Args:
            image: Input image
            trajectory_points: List of (x, y) points
            color: Line color
        
        Returns:
            vis_image: Image with trajectory drawn
        """
        if trajectory_points is None or len(trajectory_points) < 2:
            return image
        
        vis_image = image.copy()
        
        # Draw trajectory line
        points = np.array(trajectory_points, dtype=np.int32)
        cv2.polylines(vis_image, [points], False, color, 2)
        
        # Draw points
        for i, point in enumerate(points):
            # Fade points along trajectory
            alpha = 1.0 - (i / len(points)) * 0.5
            radius = 3 if i == len(points) - 1 else 2
            cv2.circle(vis_image, tuple(point), radius, color, -1)
        
        return vis_image
    
    def create_risk_gauge(self, risk_score, risk_level, size=(300, 150)):
        """
        Create a risk level gauge visualization
        
        Args:
            risk_score: Risk score (0-1)
            risk_level: Risk level name
            size: Gauge size (width, height)
        
        Returns:
            gauge_image: Gauge visualization
        """
        w, h = size
        gauge = np.ones((h, w, 3), dtype=np.uint8) * 255
        
        # Draw arc background
        center = (w // 2, h - 10)
        radius = min(w, h) - 30
        
        # Draw risk zones on arc
        angles = [(-180, -144), (-144, -108), (-108, -72), (-72, -36), (-36, 0)]
        colors = [(0, 255, 0), (100, 255, 0), (0, 165, 255), (0, 100, 255), (0, 0, 255)]
        
        for (start, end), color in zip(angles, colors):
            cv2.ellipse(gauge, center, (radius, radius), 0, start, end, color, 20)
        
        # Draw needle
        angle = -180 + (risk_score * 180)
        needle_len = radius - 10
        needle_x = int(center[0] + needle_len * np.cos(np.radians(angle)))
        needle_y = int(center[1] + needle_len * np.sin(np.radians(angle)))
        cv2.line(gauge, center, (needle_x, needle_y), (0, 0, 0), 3)
        cv2.circle(gauge, center, 8, (0, 0, 0), -1)
        
        # Draw text
        text = f"{risk_level}: {risk_score:.2f}"
        cv2.putText(gauge, text, (w//2 - 80, 30),
                   cv2.FONT_HERSHEY_BOLD, 0.7, (0, 0, 0), 2)
        
        return gauge
    
    def create_dashboard(self, frame, detections, risk_assessment, 
                        temporal_analysis, spatial_analysis):
        """
        Create comprehensive monitoring dashboard
        
        Args:
            frame: Current video frame
            detections: Detection results
            risk_assessment: Risk assessment results
            temporal_analysis: Temporal analysis results
            spatial_analysis: Spatial analysis results
        
        Returns:
            dashboard: Complete dashboard image
        """
        h, w = frame.shape[:2]
        
        # Create dashboard layout (frame + side panel)
        panel_width = 400
        dashboard = np.ones((h, w + panel_width, 3), dtype=np.uint8) * 255
        
        # Place main frame
        dashboard[:h, :w] = frame
        
        # Side panel
        panel = dashboard[:, w:]
        y_pos = 20
        
        # Title
        cv2.putText(panel, "MONITORING DASHBOARD", (20, y_pos),
                   cv2.FONT_HERSHEY_BOLD, 0.7, (0, 0, 0), 2)
        y_pos += 40
        
        # Risk gauge
        gauge = self.create_risk_gauge(
            risk_assessment['risk_score'],
            risk_assessment['risk_level_name'],
            size=(360, 120)
        )
        panel[y_pos:y_pos+120, 20:380] = gauge
        y_pos += 140
        
        # Detection counts
        cv2.putText(panel, "DETECTIONS:", (20, y_pos),
                   cv2.FONT_HERSHEY_BOLD, 0.6, (0, 0, 0), 2)
        y_pos += 25
        
        for det_type in ['child', 'fire', 'pool']:
            count = len(detections.get(det_type, []))
            color = self.colors[det_type]
            cv2.putText(panel, f"  {det_type.capitalize()}: {count}", (30, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_pos += 22
        
        y_pos += 10
        
        # Temporal info
        cv2.putText(panel, "TEMPORAL ANALYSIS:", (20, y_pos),
                   cv2.FONT_HERSHEY_BOLD, 0.6, (0, 0, 0), 2)
        y_pos += 25
        
        pattern = temporal_analysis.get('pattern_type', 'unknown')[:30]
        cv2.putText(panel, f"  Pattern: {pattern}", (30, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        y_pos += 20
        
        temp_risk = temporal_analysis.get('temporal_risk', 0)
        cv2.putText(panel, f"  Risk: {temp_risk:.3f}", (30, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        y_pos += 30
        
        # Spatial info
        cv2.putText(panel, "SPATIAL ANALYSIS:", (20, y_pos),
                   cv2.FONT_HERSHEY_BOLD, 0.6, (0, 0, 0), 2)
        y_pos += 25
        
        if spatial_analysis.get('proximity_analysis'):
            prox = spatial_analysis['proximity_analysis']
            zone = prox.get('zone', 'unknown')
            dist = prox.get('closest_distance', 0)
            
            cv2.putText(panel, f"  Zone: {zone}", (30, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
            y_pos += 20
            cv2.putText(panel, f"  Distance: {dist:.2f}m", (30, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
            y_pos += 20
        
        if spatial_analysis.get('collision_warning'):
            cv2.putText(panel, "  ⚠ COLLISION WARNING", (30, y_pos),
                       cv2.FONT_HERSHEY_BOLD, 0.5, (0, 0, 255), 2)
        
        # Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(panel, timestamp, (20, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        
        return dashboard


class StatisticsVisualizer:
    """Visualize system statistics and analytics"""
    
    @staticmethod
    def plot_detection_timeline(log_file, save_path='detection_timeline.png'):
        """Plot detection frequency over time"""
        # Load log data
        with open(log_file, 'r') as f:
            logs = json.load(f)
        
        # Extract timestamps and counts
        timestamps = []
        child_counts = []
        fire_counts = []
        pool_counts = []
        
        for log in logs:
            timestamps.append(log['timestamp'])
            detections = log.get('detections', {})
            child_counts.append(len(detections.get('child', [])))
            fire_counts.append(len(detections.get('fire', [])))
            pool_counts.append(len(detections.get('pool', [])))
        
        # Plot
        plt.figure(figsize=(15, 6))
        plt.plot(timestamps, child_counts, label='Child', color='green')
        plt.plot(timestamps, fire_counts, label='Fire', color='red')
        plt.plot(timestamps, pool_counts, label='Pool', color='blue')
        plt.xlabel('Time')
        plt.ylabel('Detection Count')
        plt.title('Detection Frequency Timeline')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
    
    @staticmethod
    def plot_risk_distribution(log_file, save_path='risk_distribution.png'):
        """Plot risk score distribution"""
        with open(log_file, 'r') as f:
            logs = json.load(f)
        
        risk_scores = [log['risk_assessment']['risk_score'] for log in logs]
        
        plt.figure(figsize=(10, 6))
        plt.hist(risk_scores, bins=50, color='blue', alpha=0.7, edgecolor='black')
        plt.axvline(x=0.3, color='green', linestyle='--', label='Low threshold')
        plt.axvline(x=0.6, color='orange', linestyle='--', label='Medium threshold')
        plt.axvline(x=0.8, color='red', linestyle='--', label='High threshold')
        plt.xlabel('Risk Score')
        plt.ylabel('Frequency')
        plt.title('Risk Score Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
    
    @staticmethod
    def create_summary_report(stats, save_path='summary_report.png'):
        """Create visual summary report"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Detection counts
        ax = axes[0, 0]
        categories = ['Child', 'Fire', 'Pool']
        counts = [stats['detections'][cat.lower()] for cat in categories]
        colors = ['green', 'red', 'blue']
        ax.bar(categories, counts, color=colors, alpha=0.7)
        ax.set_title('Total Detections by Type')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)
        
        # Alert distribution
        ax = axes[0, 1]
        alert_levels = ['Silent', 'Gentle', 'Medium', 'Urgent', 'Emergency']
        alert_counts = [20, 15, 10, 7, 3]  # Example data
        ax.pie(alert_counts, labels=alert_levels, autopct='%1.1f%%',
               colors=['green', 'yellow', 'orange', 'red', 'darkred'])
        ax.set_title('Alert Level Distribution')
        
        # Performance metrics
        ax = axes[1, 0]
        metrics = ['FPS', 'Latency\n(ms)', 'Frames\nProcessed']
        values = [
            stats.get('average_fps', 0),
            stats.get('average_latency', 0) * 1000,
            stats.get('frames_processed', 0)
        ]
        ax.bar(metrics, values, color='purple', alpha=0.7)
        ax.set_title('Performance Metrics')
        ax.grid(True, alpha=0.3)
        
        # System info
        ax = axes[1, 1]
        ax.axis('off')
        info_text = f"""
        System Performance Summary
        
        Total Frames Processed: {stats.get('frames_processed', 0)}
        Average FPS: {stats.get('average_fps', 0):.2f}
        Average Latency: {stats.get('average_latency', 0)*1000:.2f}ms
        
        Total Alerts Sent: {stats.get('alerts_sent', 0)}
        
        Detection Counts:
          • Child: {stats['detections'].get('child', 0)}
          • Fire: {stats['detections'].get('fire', 0)}
          • Pool: {stats['detections'].get('pool', 0)}
        """
        ax.text(0.1, 0.5, info_text, fontsize=12, family='monospace',
               verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Summary report saved to {save_path}")


def main():
    """Test visualization utilities"""
    print("\n" + "="*70)
    print("VISUALIZATION UTILITIES")
    print("="*70)
    
    print("\nThis module provides:")
    print("  • Detection visualization with bounding boxes")
    print("  • Proximity zone visualization")
    print("  • Trajectory visualization")
    print("  • Risk gauge and dashboard creation")
    print("  • Statistics and analytics plots")
    
    print("\nUsage:")
    print("  from visualization import ResultVisualizer, StatisticsVisualizer")
    print("  visualizer = ResultVisualizer()")
    print("  vis_image = visualizer.draw_detections(image, detections)")
    
    # Demo visualization
    print("\n" + "-"*70)
    print("Demo: Creating sample visualization")
    print("-"*70)
    
    # Create sample image
    image = np.ones((480, 640, 3), dtype=np.uint8) * 200
    
    # Sample detections
    detections = {
        'child': [
            {'bbox': [100, 100, 200, 300], 'confidence': 0.95, 'class_name': 'child'}
        ],
        'fire': [
            {'bbox': [400, 150, 500, 300], 'confidence': 0.88, 'class_name': 'fire'}
        ]
    }
    
    visualizer = ResultVisualizer()
    vis_image = visualizer.draw_detections(image, detections)
    
    print("✓ Sample detections drawn")
    
    # Create risk gauge
    gauge = visualizer.create_risk_gauge(0.75, "HIGH")
    print("✓ Risk gauge created")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()