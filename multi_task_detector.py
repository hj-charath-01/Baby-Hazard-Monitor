"""
Multi-Task Detection Module
Implements child, fire, and pool detection using YOLOv8
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import yaml

class MultiTaskDetector:
    """
    Multi-task detector for child, fire, and pool detection
    Uses YOLOv8 with shared backbone and task-specific heads
    """
    
    def __init__(self, config_path='config/config.yaml'):
        """Initialize multi-task detector"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device(
            self.config['hardware']['device'] 
            if torch.cuda.is_available() else 'cpu'
        )
        
        # Initialize separate models for each task
        self.child_detector = None
        self.fire_detector = None
        self.pool_detector = None
        
        # Detection parameters
        self.conf_threshold = self.config['models']['confidence_threshold']
        self.iou_threshold = self.config['models']['iou_threshold']
        self.input_size = tuple(self.config['models']['input_size'])
        
    def load_models(self):
        """Load pre-trained or trained models for each task"""
        print("Loading detection models...")
        
        # Child Detection Model
        child_model_path = self.config['models']['child_detection']['model_path']
        if Path(child_model_path).exists():
            self.child_detector = YOLO(child_model_path)
            print(f"✓ Loaded child detection model from {child_model_path}")
        else:
            # Load pre-trained YOLOv8 for training
            self.child_detector = YOLO('yolov8n.pt')
            print("✓ Loaded pre-trained YOLOv8n for child detection")
        
        # Fire Detection Model
        fire_model_path = self.config['models']['fire_detection']['model_path']
        if Path(fire_model_path).exists():
            self.fire_detector = YOLO(fire_model_path)
            print(f"✓ Loaded fire detection model from {fire_model_path}")
        else:
            self.fire_detector = YOLO('yolov8n.pt')
            print("✓ Loaded pre-trained YOLOv8n for fire detection")
        
        # Pool Detection Model
        pool_model_path = self.config['models']['pool_detection']['model_path']
        if Path(pool_model_path).exists():
            self.pool_detector = YOLO(pool_model_path)
            print(f"✓ Loaded pool detection model from {pool_model_path}")
        else:
            self.pool_detector = YOLO('yolov8n.pt')
            print("✓ Loaded pre-trained YOLOv8n for pool detection")
        
        # Move models to device
        for model in [self.child_detector, self.fire_detector, self.pool_detector]:
            if model:
                model.to(self.device)
    
    def train_child_detector(self, data_yaml, epochs=100):
        """Train child detection model"""
        print("\n" + "="*60)
        print("TRAINING CHILD DETECTION MODEL")
        print("="*60)
        
        results = self.child_detector.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=self.input_size[0],
            batch=self.config['hardware']['batch_size'] * 4,
            device=self.device,
            project='models/runs/child',
            name='train',
            patience=15,
            save=True,
            plots=True,
            val=True
        )
        
        # Save best model
        self.child_detector.save('models/child_detector.pt')
        print("✓ Child detection model trained and saved")
        
        return results
    
    def train_fire_detector(self, data_yaml, epochs=100):
        """Train fire detection model"""
        print("\n" + "="*60)
        print("TRAINING FIRE DETECTION MODEL")
        print("="*60)
        
        results = self.fire_detector.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=self.input_size[0],
            batch=self.config['hardware']['batch_size'] * 4,
            device=self.device,
            project='models/runs/fire',
            name='train',
            patience=15,
            save=True,
            plots=True,
            val=True
        )
        
        self.fire_detector.save('models/fire_detector.pt')
        print("✓ Fire detection model trained and saved")
        
        return results
    
    def train_pool_detector(self, data_yaml, epochs=100):
        """Train pool detection model"""
        print("\n" + "="*60)
        print("TRAINING POOL DETECTION MODEL")
        print("="*60)
        
        results = self.pool_detector.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=self.input_size[0],
            batch=self.config['hardware']['batch_size'] * 4,
            device=self.device,
            project='models/runs/pool',
            name='train',
            patience=15,
            save=True,
            plots=True,
            val=True
        )
        
        self.pool_detector.save('models/pool_detector.pt')
        print("✓ Pool detection model trained and saved")
        
        return results
    
    def detect(self, frame):
        """
        Perform multi-task detection on a single frame
        
        Args:
            frame: Input image (numpy array)
            
        Returns:
            detections: Dictionary containing all detections
        """
        detections = {
            'child': [],
            'fire': [],
            'pool': [],
            'timestamp': None
        }
        
        # Resize frame to input size
        frame_resized = cv2.resize(frame, self.input_size)
        
        # Child Detection
        if self.child_detector:
            child_results = self.child_detector.predict(
                frame_resized,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False
            )[0]
            
            detections['child'] = self._parse_yolo_results(
                child_results, 
                frame.shape,
                self.config['models']['child_detection']['classes']
            )
        
        # Fire Detection
        if self.fire_detector:
            fire_results = self.fire_detector.predict(
                frame_resized,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False
            )[0]
            
            detections['fire'] = self._parse_yolo_results(
                fire_results,
                frame.shape,
                self.config['models']['fire_detection']['classes']
            )
        
        # Pool Detection
        if self.pool_detector:
            pool_results = self.pool_detector.predict(
                frame_resized,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False
            )[0]
            
            detections['pool'] = self._parse_yolo_results(
                pool_results,
                frame.shape,
                self.config['models']['pool_detection']['classes']
            )
        
        return detections
    
    def _parse_yolo_results(self, results, original_shape, class_names):
        """Parse YOLO detection results into standardized format"""
        detections = []
        
        if results.boxes is None or len(results.boxes) == 0:
            return detections
        
        boxes = results.boxes.xyxy.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        
        # Scale boxes back to original image size
        h_scale = original_shape[0] / self.input_size[1]
        w_scale = original_shape[1] / self.input_size[0]
        
        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = box
            
            # Scale coordinates
            x1 = int(x1 * w_scale)
            y1 = int(y1 * h_scale)
            x2 = int(x2 * w_scale)
            y2 = int(y2 * h_scale)
            
            detection = {
                'bbox': [x1, y1, x2, y2],
                'confidence': float(conf),
                'class_id': int(cls_id),
                'class_name': class_names[cls_id] if cls_id < len(class_names) else 'unknown',
                'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                'area': (x2 - x1) * (y2 - y1)
            }
            
            detections.append(detection)
        
        return detections
    
    def visualize_detections(self, frame, detections):
        """Draw bounding boxes and labels on frame"""
        vis_frame = frame.copy()
        
        # Color codes for different detection types
        colors = {
            'child': (0, 255, 0),    # Green
            'fire': (0, 0, 255),     # Red
            'pool': (255, 0, 0)      # Blue
        }
        
        for det_type, dets in detections.items():
            if det_type == 'timestamp':
                continue
                
            color = colors.get(det_type, (255, 255, 255))
            
            for det in dets:
                x1, y1, x2, y2 = det['bbox']
                conf = det['confidence']
                cls_name = det['class_name']
                
                # Draw bounding box
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{cls_name}: {conf:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                
                # Background for label
                cv2.rectangle(
                    vis_frame,
                    (x1, y1 - label_size[1] - 10),
                    (x1 + label_size[0], y1),
                    color,
                    -1
                )
                
                # Text
                cv2.putText(
                    vis_frame,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )
        
        return vis_frame


def main():
    """Test detection module"""
    print("\n" + "="*60)
    print("MULTI-TASK DETECTION MODULE TEST")
    print("="*60)
    
    detector = MultiTaskDetector()
    detector.load_models()
    
    print("\n✓ Detection module initialized successfully")
    print(f"Device: {detector.device}")
    print(f"Input size: {detector.input_size}")
    print(f"Confidence threshold: {detector.conf_threshold}")


if __name__ == "__main__":
    main()