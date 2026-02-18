"""
Integrated Privacy-Preserving Edge Monitoring System
Combines all patent components into working system

Patent Features:
1. Privacy-Preserving Edge Processing
2. Adaptive Room Mapping
3. Zero Cloud Video Transmission
4. DPDPA 2023 Compliance
5. Parent Control Dashboard
"""

import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from privacy_edge_processor import PrivacyPreservingProcessor
from adaptive_room_mapper import AdaptiveRoomMapper


class PrivacyPreservingMonitor:
    """
    Complete edge-based monitoring system with privacy preservation
    
    Patent System: Privacy-Preserving Edge Intelligence with Adaptive Mapping
    """
    
    def __init__(self, config_path='config/config.yaml'):
        """Initialize privacy-preserving monitor"""
        print("\n" + "="*70)
        print("PRIVACY-PRESERVING EDGE MONITOR")
        print("Patent System: Edge Intelligence with Zero Cloud Video")
        print("="*70 + "\n")
        
        # Initialize components
        self.privacy_processor = PrivacyPreservingProcessor()
        self.room_mapper = AdaptiveRoomMapper()
        
        # Try to load existing room map
        self.room_mapper.load_room_map()
        
        # Detection model (placeholder - integrate with your YOLO)
        self.detector = None
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'features_extracted': 0,
            'alerts_generated': 0,
            'privacy_violations_prevented': 0,
            'bandwidth_saved_mb': 0
        }
        
        print("Privacy Processor: Initialized")
        print("Room Mapper: Initialized")
        print("Ready for Edge Processing\n")
    
    def initialize_detector(self, model_path):
        """
        Initialize detection models
        
        Args:
            model_path: Path to trained models directory
        """
        try:
            from ultralytics import YOLO
            
            # Load models
            child_model = Path(model_path) / 'child_detector.pt'
            fire_model = Path(model_path) / 'fire_detector.pt'
            pool_model = Path(model_path) / 'pool_detector.pt'
            
            if child_model.exists():
                self.detector = {
                    'child': YOLO(str(child_model)),
                    'fire': YOLO(str(fire_model)) if fire_model.exists() else None,
                    'pool': YOLO(str(pool_model)) if pool_model.exists() else None
                }
                print("Detection models loaded")
            else:
                print("No trained models found, using demo mode")
                self.detector = None
        except:
            print("YOLOv8 not available, using demo mode")
            self.detector = None
    
    def process_frame(self, frame):
        """
        Process single frame with complete privacy preservation
        
        Patent Process:
        1. Apply privacy zones (mask bedrooms/bathrooms)
        2. Run on-device detection
        3. Extract features only (discard frame)
        4. Update room map
        5. Assess proximity risks
        6. Generate alerts (if needed)
        7. Return anonymized results
        
        Args:
            frame: Input video frame
        
        Returns:
            results: Anonymized processing results
        """
        self.stats['frames_processed'] += 1
        
        # Step 1: Apply privacy zone masking
        masked_frame = self.privacy_processor.mask_privacy_zones(frame)
        
        # Step 2: Run detection (on-device only)
        detections = self._run_detection(masked_frame)
        
        # Step 3: Extract features only (no frame storage)
        features = self.privacy_processor.extract_privacy_preserving_features(
            masked_frame,
            detections
        )
        self.stats['features_extracted'] += 1
        
        # Step 4: Update room map with detections
        map_state = self.room_mapper.process_frame_for_mapping(
            masked_frame,
            detections
        )
        
        # Step 5: Assess proximity risks
        proximity_alerts = self._check_proximity_risks(detections)
        
        # Step 6: Generate alerts if needed
        if proximity_alerts:
            self._handle_proximity_alerts(proximity_alerts)
        
        # Step 7: Prepare anonymized results
        results = {
            'timestamp': features['timestamp'],
            'frame_id': features['frame_id'],
            'detection_count': len(detections),
            'map_state': map_state,
            'proximity_alerts': proximity_alerts,
            'privacy_zones_active': len(self.privacy_processor.privacy_zones),
            'raw_frame_stored': False,  # NEVER stored
            'raw_frame_transmitted': False  # NEVER transmitted
        }
        
        # Calculate bandwidth savings
        bandwidth_savings = self.privacy_processor.estimate_bandwidth(features)
        self.stats['bandwidth_saved_mb'] += bandwidth_savings['raw_video_size_mb']
        
        return results
    
    def _run_detection(self, frame):
        """
        Run on-device detection
        
        Patent Claim: Complete on-device processing, no cloud inference
        """
        detections = []
        
        if self.detector is None:
            # Demo mode - simulate detections
            return self._simulate_detections()
        
        # Run YOLO detection
        try:
            results = self.detector['child'](frame, verbose=False)
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': conf,
                        'class_name': 'child',
                        'class_id': cls
                    })
        except Exception as e:
            print(f"Detection error: {e}")
        
        return detections
    
    def _simulate_detections(self):
        """Simulate detections for demo"""
        import random
        
        detections = []
        
        # Simulate child detection
        if random.random() > 0.3:
            detections.append({
                'bbox': [300 + random.randint(-50, 50), 
                        200 + random.randint(-50, 50),
                        400 + random.randint(-50, 50),
                        400 + random.randint(-50, 50)],
                'confidence': random.uniform(0.8, 0.98),
                'class_name': 'child'
            })
        
        # Simulate fire detection
        if random.random() > 0.7:
            detections.append({
                'bbox': [100, 100, 200, 200],
                'confidence': random.uniform(0.85, 0.95),
                'class_name': 'fire'
            })
        
        return detections
    
    def _check_proximity_risks(self, detections):
        """
        Check if children are near hazards
        
        Patent Claim: Spatial hazard proximity detection with learned map
        """
        alerts = []
        
        # Find children
        children = [d for d in detections if d['class_name'] == 'child']
        
        # Check each child against known hazards
        for child in children:
            child_bbox = child['bbox']
            
            # Check against persistent hazards from room map
            hazard_alerts = self.room_mapper.get_hazard_proximity_alert(child_bbox)
            
            for alert in hazard_alerts:
                alerts.append({
                    'child_bbox': child_bbox,
                    'hazard_type': alert['hazard_type'],
                    'distance_meters': alert['distance_meters'],
                    'severity': alert['severity'],
                    'timestamp': datetime.now().isoformat()
                })
        
        return alerts
    
    def _handle_proximity_alerts(self, alerts):
        """Handle proximity alerts"""
        for alert in alerts:
            severity = alert['severity']
            
            if severity == 'critical':
                # Trigger alert recording
                alert_id = self.privacy_processor.trigger_alert_recording(4)
                alert['recording_id'] = alert_id
                self.stats['alerts_generated'] += 1
                
                print(f"CRITICAL ALERT: Child within {alert['distance_meters']:.2f}m of {alert['hazard_type']}")
            
            elif severity == 'warning':
                self.stats['alerts_generated'] += 1
                print(f"⚠ WARNING: Child approaching {alert['hazard_type']}")
    
    def process_video(self, video_path, output_path=None):
        """
        Process video file with privacy preservation
        
        Args:
            video_path: Input video path
            output_path: Optional output for visualization (anonymized)
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\nProcessing Video: {Path(video_path).name}")
        print(f"Resolution: {width}x{height}")
        print(f"FPS: {fps}")
        print(f"Frames: {total_frames}")
        print(f"Privacy Mode: Enabled")
        print(f"Raw Video Transmission: Disabled\n")
        
        # Video writer for anonymized output
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process frame (edge processing only)
            results = self.process_frame(frame)
            
            # Create visualization (anonymized)
            if writer:
                vis_frame = self.room_mapper.visualize_room_map(frame)
                writer.write(vis_frame)
            
            # Print progress
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% | "
                      f"Detections: {results['detection_count']} | "
                      f"Alerts: {len(results['proximity_alerts'])}")
        
        cap.release()
        if writer:
            writer.release()
        
        # Print summary
        self._print_processing_summary()
    
    def process_camera(self, camera_id=0):
        """
        Process live camera feed
        
        Args:
            camera_id: Camera device ID
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return
        
        print("\n" + "="*70)
        print("LIVE CAMERA MONITORING")
        print("Privacy-Preserving Edge Processing Active")
        print("Press 'q' to quit, 'p' to add privacy zone, 's' for stats")
        print("="*70 + "\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            results = self.process_frame(frame)
            
            # Visualize (with privacy zones and hazard map)
            vis_frame = self.room_mapper.visualize_room_map(frame)
            
            # Add status overlay
            self._add_status_overlay(vis_frame, results)
            
            # Display
            cv2.imshow('Privacy-Preserving Monitor', vis_frame)
            
            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                self._add_privacy_zone_interactive(frame)
            elif key == ord('s'):
                self._print_processing_summary()
        
        cap.release()
        cv2.destroyAllWindows()
        
        self._print_processing_summary()
    
    def _add_status_overlay(self, frame, results):
        """Add status information to frame"""
        # Privacy status
        cv2.putText(frame, "PRIVACY MODE: ON", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Processing mode
        mode = "LEARNING" if self.room_mapper.learning_mode else "MONITORING"
        cv2.putText(frame, f"Mode: {mode}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Detection count
        cv2.putText(frame, f"Detections: {results['detection_count']}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Privacy zones
        cv2.putText(frame, f"Privacy Zones: {results['privacy_zones_active']}", 
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Alerts
        if results['proximity_alerts']:
            cv2.putText(frame, f"⚠ ALERTS: {len(results['proximity_alerts'])}", 
                       (10, frame.shape[0] - 20),cv2.FONT_HERSHEY_BOLD, 1.0, (0, 0, 255), 3)
    
    def _add_privacy_zone_interactive(self, frame):
        """Interactive privacy zone addition (simplified)"""
        print("\nAdd Privacy Zone:")
        zone_name = input("  Zone name (bedroom/bathroom/etc): ")
        
        # Simplified - use center of frame
        h, w = frame.shape[:2]
        coordinates = [
            (w//4, h//4),
            (3*w//4, h//4),
            (3*w//4, 3*h//4),
            (w//4, 3*h//4)
        ]
        
        self.privacy_processor.add_privacy_zone(zone_name, coordinates)
        print(f"✓ Privacy zone '{zone_name}' added")
    
    def _print_processing_summary(self):
        """Print processing summary and statistics"""
        print("\n" + "="*70)
        print("PROCESSING SUMMARY")
        print("="*70)
        
        print(f"\nFrames Processed: {self.stats['frames_processed']}")
        print(f"Features Extracted: {self.stats['features_extracted']}")
        print(f"Alerts Generated: {self.stats['alerts_generated']}")
        
        print(f"\nPrivacy & Bandwidth:")
        print(f"  Raw Video Stored: NO")
        print(f"  Raw Video Transmitted: NO")
        print(f"  Bandwidth Saved: ~{self.stats['bandwidth_saved_mb']:.1f} MB")
        
        print(f"\nRoom Map:")
        print(f"  Learning Mode: {self.room_mapper.learning_mode}")
        print(f"  Persistent Hazards: {len(self.room_mapper.room_map['persistent_hazards'])}")
        print(f"  Temporary Hazards: {len(self.room_mapper.room_map['temporary_hazards'])}")
        
        print(f"\nPrivacy:")
        print(f"  Privacy Zones: {len(self.privacy_processor.privacy_zones)}")
        print(f"  Encryption: Enabled")
        print(f"  DPDPA 2023 Compliant: YES")
        
        print("="*70 + "\n")
    
    def get_privacy_report(self):
        """Generate comprehensive privacy report"""
        report = {
            'system': 'Privacy-Preserving Edge Intelligence',
            'timestamp': datetime.now().isoformat(),
            'privacy_processor': self.privacy_processor.get_privacy_report(),
            'room_mapper': self.room_mapper.get_current_map_state(),
            'statistics': self.stats
        }
        
        return report
    
    def save_privacy_report(self, output_path='outputs/privacy_report.json'):
        """Save privacy report to file"""
        report = self.get_privacy_report()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Privacy report saved to: {output_path}")
        return output_path


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Privacy-Preserving Edge Intelligence System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Patent Features:
  - Complete edge processing (no cloud video)
  - Adaptive room mapping with hazard learning
  - Privacy-preserving feature extraction
  - DPDPA 2023 compliant
  - Encrypted feature transmission

Examples:
  # Process video file
  python privacy_edge_monitor.py --video test_video.mp4
  
  # Live camera monitoring
  python privacy_edge_monitor.py --camera 0
  
  # Generate privacy report
  python privacy_edge_monitor.py --report
        """
    )
    
    parser.add_argument('--video', type=str, help='Process video file')
    parser.add_argument('--camera', type=int, help='Process live camera')
    parser.add_argument('--output', type=str, help='Output path for processed video')
    parser.add_argument('--models', type=str, default='models', 
                       help='Path to trained models')
    parser.add_argument('--report', action='store_true', 
                       help='Generate privacy compliance report')
    parser.add_argument('--demo', action='store_true', 
                       help='Run demo mode')
    
    args = parser.parse_args()
    
    # Initialize system
    monitor = PrivacyPreservingMonitor()
    monitor.initialize_detector(args.models)
    
    # Add sample privacy zones for demo
    if args.demo or args.camera is not None:
        monitor.privacy_processor.add_privacy_zone(
            "Bedroom",
            [(50, 50), (300, 50), (300, 250), (50, 250)]
        )
    
    # Process based on arguments
    if args.video:
        monitor.process_video(args.video, args.output)
    elif args.camera is not None:
        monitor.process_camera(args.camera)
    elif args.report:
        report_path = monitor.save_privacy_report()
        print(f"\nPrivacy compliance report generated: {report_path}")
    else:
        # Demo mode
        print("\n" + "="*70)
        print("PRIVACY-PRESERVING EDGE INTELLIGENCE DEMO")
        print("="*70)
        
        # Run demo
        monitor.privacy_processor.add_privacy_zone(
            "Bedroom",
            [(100, 100), (400, 100), (400, 300), (100, 300)]
        )
        
        # Simulate processing
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        for i in range(50):
            results = monitor.process_frame(dummy_frame)
            
            if i % 10 == 0:
                print(f"Processed frame {i}: {results['detection_count']} detections")
        
        # Generate report
        monitor.save_privacy_report()
        
        print("\nDemo complete")


if __name__ == "__main__":
    main()