"""
Real-Time Baby Hazard Monitoring System
Integrates all modules for complete end-to-end monitoring
"""

import cv2
import time
import yaml
import sys
from pathlib import Path
import numpy as np
from datetime import datetime
import threading
import queue

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from models.multi_task_detector import MultiTaskDetector
from models.temporal_reasoning import TemporalPatternAnalyzer
from models.spatial_analysis import SpatialRiskAssessment
from models.risk_assessment import RiskAssessmentModule
from models.alert_manager import AlertManager


class BabyHazardMonitor:
    """
    Main monitoring system that orchestrates all components
    """
    
    def __init__(self, config_path='config/config.yaml'):
        """Initialize monitoring system"""
        print("\n" + "="*60)
        print("SMART BABY HAZARD MONITORING SYSTEM")
        print("Initializing...")
        print("="*60)
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize all modules
        print("\n1. Loading detection models...")
        self.detector = MultiTaskDetector(config_path)
        self.detector.load_models()
        
        print("\n2. Initializing temporal reasoning...")
        self.temporal_analyzer = TemporalPatternAnalyzer(config_path)
        
        print("\n3. Initializing spatial analysis...")
        self.spatial_analyzer = SpatialRiskAssessment(config_path)
        
        print("\n4. Initializing risk assessment...")
        self.risk_assessor = RiskAssessmentModule(config_path)
        
        print("\n5. Initializing alert management...")
        self.alert_manager = AlertManager(config_path)
        
        # Video settings
        self.target_fps = self.config['video']['fps']
        self.frame_time = 1.0 / self.target_fps
        
        # Processing flags
        self.running = False
        self.paused = False
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'detections': {'child': 0, 'fire': 0, 'pool': 0},
            'alerts_sent': 0,
            'average_fps': 0,
            'average_latency': 0
        }
        
        # Performance tracking
        self.processing_times = []
        self.max_time_samples = 100
        
        # Output directory
        self.output_dir = Path('outputs/monitoring_sessions')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n✓ System initialization complete!")
        print(f"{'='*60}\n")
    
    def process_frame(self, frame):
        """
        Process a single frame through the complete pipeline
        
        Args:
            frame: Input video frame (numpy array)
            
        Returns:
            results: Dictionary with all processing results
        """
        start_time = time.time()
        
        # Step 1: Multi-task detection
        detections = self.detector.detect(frame)
        
        # Update detection statistics
        for det_type in ['child', 'fire', 'pool']:
            if len(detections.get(det_type, [])) > 0:
                self.stats['detections'][det_type] += 1
        
        # Step 2: Temporal analysis
        temporal_analysis = self.temporal_analyzer.analyze(detections)
        
        # Step 3: Spatial analysis
        spatial_analysis = self.spatial_analyzer.assess_risk(detections)
        
        # Step 4: Risk assessment
        risk_assessment = self.risk_assessor.assess_comprehensive_risk(
            detections,
            temporal_analysis,
            spatial_analysis,
            frame
        )
        
        # Step 5: Alert management
        alert_sent = None
        if risk_assessment['should_alert']:
            alert = self.alert_manager.create_alert(risk_assessment, detections)
            alert['recommended_actions'] = self.risk_assessor.get_recommended_actions(risk_assessment)
            delivery_status = self.alert_manager.send_alert(alert)
            
            if not delivery_status.get('suppressed'):
                alert_sent = alert
                self.stats['alerts_sent'] += 1
        
        # Calculate processing time
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        if len(self.processing_times) > self.max_time_samples:
            self.processing_times.pop(0)
        
        # Compile results
        results = {
            'detections': detections,
            'temporal_analysis': temporal_analysis,
            'spatial_analysis': spatial_analysis,
            'risk_assessment': risk_assessment,
            'alert_sent': alert_sent,
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat()
        }
        
        # Update statistics
        self.stats['frames_processed'] += 1
        self.stats['average_latency'] = np.mean(self.processing_times)
        
        return results
    
    def visualize_results(self, frame, results):
        """
        Create visualization of detection and analysis results
        
        Args:
            frame: Original frame
            results: Processing results
            
        Returns:
            vis_frame: Annotated frame
        """
        vis_frame = frame.copy()
        
        # Draw detections
        vis_frame = self.detector.visualize_detections(vis_frame, results['detections'])
        
        # Draw spatial analysis (proximity zones and trajectory)
        spatial_analysis = results['spatial_analysis']
        if spatial_analysis['proximity_analysis'] and spatial_analysis['proximity_analysis']['closest_hazard']:
            hazard_pos = tuple(map(int, spatial_analysis['proximity_analysis']['closest_hazard']['center']))
            vis_frame = self.spatial_analyzer.spatial_analyzer.visualize_zones(vis_frame, hazard_pos)
        
        if spatial_analysis.get('trajectory') is not None:
            vis_frame = self.spatial_analyzer.spatial_analyzer.visualize_trajectory(
                vis_frame, spatial_analysis['trajectory']
            )
        
        # Add information overlay
        vis_frame = self._add_info_overlay(vis_frame, results)
        
        return vis_frame
    
    def _add_info_overlay(self, frame, results):
        """Add information overlay to frame"""
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay panel
        overlay = frame.copy()
        panel_height = 200
        cv2.rectangle(overlay, (0, 0), (400, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Risk assessment info
        risk = results['risk_assessment']
        y_pos = 25
        
        # Risk level with color
        risk_level = risk['risk_level_name']
        risk_colors = {
            'SAFE': (0, 255, 0),
            'LOW': (0, 255, 255),
            'MEDIUM': (0, 165, 255),
            'HIGH': (0, 100, 255),
            'CRITICAL': (0, 0, 255)
        }
        color = risk_colors.get(risk_level, (255, 255, 255))
        
        cv2.putText(frame, f"RISK: {risk_level}", (10, y_pos),
                   cv2.FONT_HERSHEY_BOLD, 0.7, color, 2)
        y_pos += 30
        
        # Risk score
        cv2.putText(frame, f"Score: {risk['risk_score']:.3f}", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += 25
        
        # Temporal pattern
        temporal = results['temporal_analysis']
        cv2.putText(frame, f"Pattern: {temporal['pattern_type'][:20]}", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        y_pos += 25
        
        # Spatial info
        spatial = results['spatial_analysis']
        if spatial['proximity_analysis']:
            prox = spatial['proximity_analysis']
            cv2.putText(frame, f"Zone: {prox['zone']}", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            y_pos += 20
            cv2.putText(frame, f"Dist: {prox['closest_distance']:.2f}m", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        y_pos += 30
        
        # Alert status
        if results['alert_sent']:
            alert = results['alert_sent']
            cv2.putText(frame, f"ALERT: {alert['level_name']}", (10, y_pos),
                       cv2.FONT_HERSHEY_BOLD, 0.6, (0, 0, 255), 2)
        y_pos += 25
        
        # Performance stats
        fps = 1.0 / results['processing_time'] if results['processing_time'] > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
        y_pos += 20
        cv2.putText(frame, f"Latency: {results['processing_time']*1000:.1f}ms", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
        
        return frame
    
    def monitor_video_file(self, video_path, display=True, save_output=True):
        """
        Monitor a video file
        
        Args:
            video_path: Path to video file
            display: Whether to display output
            save_output: Whether to save annotated video
        """
        print(f"\n{'='*60}")
        print(f"MONITORING VIDEO: {video_path}")
        print(f"{'='*60}\n")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"✗ Error: Cannot open video file: {video_path}")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Total frames: {total_frames}")
        print(f"  Duration: {total_frames/fps:.1f}s\n")
        
        # Video writer for output
        video_writer = None
        if save_output:
            output_path = self.output_dir / f"monitored_{Path(video_path).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            print(f"Saving output to: {output_path}\n")
        
        # Processing loop
        self.running = True
        frame_count = 0
        
        try:
            while self.running:
                if self.paused:
                    time.sleep(0.1)
                    continue
                
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Process frame
                results = self.process_frame(frame)
                
                # Visualize
                vis_frame = self.visualize_results(frame, results)
                
                # Save to video
                if video_writer:
                    video_writer.write(vis_frame)
                
                # Display
                if display:
                    cv2.imshow('Baby Hazard Monitor', vis_frame)
                    
                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('p'):
                        self.paused = not self.paused
                        print("Paused" if self.paused else "Resumed")
                    elif key == ord('s'):
                        # Save screenshot
                        screenshot_path = self.output_dir / f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                        cv2.imwrite(str(screenshot_path), vis_frame)
                        print(f"Screenshot saved: {screenshot_path}")
                
                # Progress update
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% | Frames: {frame_count}/{total_frames} | "
                          f"FPS: {1.0/results['processing_time']:.1f} | "
                          f"Risk: {results['risk_assessment']['risk_level_name']}")
        
        except KeyboardInterrupt:
            print("\n\nMonitoring interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            if video_writer:
                video_writer.release()
            if display:
                cv2.destroyAllWindows()
            
            self.running = False
        
        # Print final statistics
        self._print_statistics()
    
    def monitor_camera(self, camera_index=0, display=True):
        """
        Monitor live camera feed
        
        Args:
            camera_index: Camera device index (0 for default)
            display: Whether to display output
        """
        print(f"\n{'='*60}")
        print(f"MONITORING LIVE CAMERA (Index: {camera_index})")
        print(f"{'='*60}\n")
        
        # Open camera
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"✗ Error: Cannot open camera {camera_index}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['video']['resolution']['width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['video']['resolution']['height'])
        cap.set(cv2.CAP_PROP_FPS, self.config['video']['fps'])
        
        print("Camera opened successfully")
        print("Press 'q' to quit, 'p' to pause/resume, 's' for screenshot\n")
        
        # Processing loop
        self.running = True
        
        try:
            while self.running:
                if self.paused:
                    time.sleep(0.1)
                    continue
                
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Process frame
                results = self.process_frame(frame)
                
                # Visualize
                vis_frame = self.visualize_results(frame, results)
                
                # Display
                if display:
                    cv2.imshow('Baby Hazard Monitor - Live', vis_frame)
                    
                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('p'):
                        self.paused = not self.paused
                        print("Paused" if self.paused else "Resumed")
                    elif key == ord('s'):
                        screenshot_path = self.output_dir / f"live_screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                        cv2.imwrite(str(screenshot_path), vis_frame)
                        print(f"Screenshot saved: {screenshot_path}")
                
                # Maintain target FPS
                time.sleep(max(0, self.frame_time - results['processing_time']))
        
        except KeyboardInterrupt:
            print("\n\nMonitoring interrupted by user")
        
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()
            
            self.running = False
        
        # Print final statistics
        self._print_statistics()
    
    def _print_statistics(self):
        """Print monitoring session statistics"""
        print(f"\n{'='*60}")
        print("MONITORING SESSION STATISTICS")
        print(f"{'='*60}")
        print(f"Total Frames Processed: {self.stats['frames_processed']}")
        print(f"\nDetections:")
        print(f"  Child: {self.stats['detections']['child']}")
        print(f"  Fire: {self.stats['detections']['fire']}")
        print(f"  Pool: {self.stats['detections']['pool']}")
        print(f"\nAlerts Sent: {self.stats['alerts_sent']}")
        print(f"Average Latency: {self.stats['average_latency']*1000:.2f}ms")
        
        if self.stats['frames_processed'] > 0:
            avg_fps = self.stats['frames_processed'] / sum(self.processing_times)
            print(f"Average FPS: {avg_fps:.2f}")
        
        # Alert statistics
        alert_stats = self.alert_manager.get_alert_statistics()
        if alert_stats['total_alerts'] > 0:
            print(f"\nAlert Statistics:")
            print(f"  Total: {alert_stats['total_alerts']}")
            print(f"  Acknowledged: {alert_stats['acknowledged_rate']*100:.1f}%")
            print(f"  Resolved: {alert_stats['resolution_rate']*100:.1f}%")
            print(f"  By Level:")
            for level, count in alert_stats['by_level'].items():
                print(f"    {level}: {count}")
        
        print(f"{'='*60}\n")


def main():
    """Main entry point"""
    print("\n" + "#"*60)
    print("# SMART BABY HAZARD MONITORING SYSTEM")
    print("# Real-Time Monitoring Interface")
    print("#"*60)
    
    # Initialize system
    monitor = BabyHazardMonitor()
    
    # Demo mode - process sample video
    print("\n" + "="*60)
    print("DEMO MODE")
    print("="*60)
    print("\nOptions:")
    print("1. Monitor video file")
    print("2. Monitor live camera")
    print("3. Run test simulation")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == '1':
        video_path = input("Enter video file path: ").strip()
        if Path(video_path).exists():
            monitor.monitor_video_file(video_path, display=True, save_output=True)
        else:
            print(f"Video file not found: {video_path}")
    
    elif choice == '2':
        camera_index = input("Enter camera index (default 0): ").strip()
        camera_index = int(camera_index) if camera_index else 0
        monitor.monitor_camera(camera_index, display=True)
    
    else:
        print("\nRunning test simulation...")
        print("Note: For actual monitoring, provide a video file or camera.")


if __name__ == "__main__":
    main()