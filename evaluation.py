"""
Model Evaluation and Testing Utilities
Comprehensive evaluation metrics and testing tools
"""

import torch
import numpy as np
from pathlib import Path
import cv2
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict


class DetectionEvaluator:
    """Evaluate object detection models"""
    
    def __init__(self, iou_threshold=0.5):
        """
        Initialize evaluator
        
        Args:
            iou_threshold: IoU threshold for matching predictions to ground truth
        """
        self.iou_threshold = iou_threshold
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.predictions = []
        self.ground_truths = []
        self.image_ids = []
    
    def add_predictions(self, pred_boxes, pred_scores, pred_classes, 
                       gt_boxes, gt_classes, image_id):
        """
        Add predictions and ground truth for one image
        
        Args:
            pred_boxes: Predicted bounding boxes [N, 4] (x1, y1, x2, y2)
            pred_scores: Prediction confidence scores [N]
            pred_classes: Predicted class labels [N]
            gt_boxes: Ground truth bounding boxes [M, 4]
            gt_classes: Ground truth class labels [M]
            image_id: Unique image identifier
        """
        self.predictions.append({
            'boxes': np.array(pred_boxes),
            'scores': np.array(pred_scores),
            'classes': np.array(pred_classes)
        })
        
        self.ground_truths.append({
            'boxes': np.array(gt_boxes),
            'classes': np.array(gt_classes)
        })
        
        self.image_ids.append(image_id)
    
    def calculate_iou(self, box1, box2):
        """
        Calculate Intersection over Union (IoU) between two boxes
        
        Args:
            box1, box2: Boxes in format [x1, y1, x2, y2]
        
        Returns:
            iou: Intersection over Union score
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        iou = intersection / union if union > 0 else 0
        return iou
    
    def calculate_ap(self, recalls, precisions):
        """
        Calculate Average Precision (AP)
        
        Args:
            recalls: Array of recall values
            precisions: Array of precision values
        
        Returns:
            ap: Average Precision score
        """
        # Add sentinel values at the start and end
        recalls = np.concatenate(([0.], recalls, [1.]))
        precisions = np.concatenate(([0.], precisions, [0.]))
        
        # Compute the precision envelope
        for i in range(len(precisions) - 1, 0, -1):
            precisions[i - 1] = max(precisions[i - 1], precisions[i])
        
        # Calculate area under curve
        indices = np.where(recalls[1:] != recalls[:-1])[0]
        ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
        
        return ap
    
    def evaluate_class(self, class_id):
        """
        Evaluate detection performance for a specific class
        
        Returns:
            metrics: Dictionary with precision, recall, AP, etc.
        """
        # Collect all predictions and ground truths for this class
        all_pred_boxes = []
        all_pred_scores = []
        all_gt_boxes = []
        
        for pred, gt in zip(self.predictions, self.ground_truths):
            # Filter by class
            pred_mask = pred['classes'] == class_id
            gt_mask = gt['classes'] == class_id
            
            if pred_mask.any():
                all_pred_boxes.extend(pred['boxes'][pred_mask])
                all_pred_scores.extend(pred['scores'][pred_mask])
            
            if gt_mask.any():
                all_gt_boxes.extend(gt['boxes'][gt_mask])
        
        if len(all_gt_boxes) == 0:
            return {
                'ap': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'num_gt': 0,
                'num_pred': len(all_pred_boxes)
            }
        
        # Sort predictions by score
        if len(all_pred_boxes) > 0:
            sorted_indices = np.argsort(all_pred_scores)[::-1]
            all_pred_boxes = np.array(all_pred_boxes)[sorted_indices]
            all_pred_scores = np.array(all_pred_scores)[sorted_indices]
        else:
            return {
                'ap': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'num_gt': len(all_gt_boxes),
                'num_pred': 0
            }
        
        # Calculate precision and recall at each prediction
        num_gt = len(all_gt_boxes)
        tp = np.zeros(len(all_pred_boxes))
        fp = np.zeros(len(all_pred_boxes))
        
        gt_matched = set()
        
        for i, pred_box in enumerate(all_pred_boxes):
            max_iou = 0
            max_idx = -1
            
            for j, gt_box in enumerate(all_gt_boxes):
                if j in gt_matched:
                    continue
                
                iou = self.calculate_iou(pred_box, gt_box)
                if iou > max_iou:
                    max_iou = iou
                    max_idx = j
            
            if max_iou >= self.iou_threshold:
                tp[i] = 1
                gt_matched.add(max_idx)
            else:
                fp[i] = 1
        
        # Compute precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / num_gt
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        # Calculate AP
        ap = self.calculate_ap(recalls, precisions)
        
        return {
            'ap': float(ap),
            'precision': float(precisions[-1]) if len(precisions) > 0 else 0.0,
            'recall': float(recalls[-1]) if len(recalls) > 0 else 0.0,
            'num_gt': num_gt,
            'num_pred': len(all_pred_boxes),
            'precisions': precisions.tolist(),
            'recalls': recalls.tolist()
        }
    
    def evaluate(self, class_names):
        """
        Evaluate all classes and calculate mAP
        
        Args:
            class_names: List of class names
        
        Returns:
            results: Dictionary with evaluation results
        """
        results = {}
        aps = []
        
        for class_id, class_name in enumerate(class_names):
            metrics = self.evaluate_class(class_id)
            results[class_name] = metrics
            aps.append(metrics['ap'])
        
        # Calculate mAP
        results['mAP'] = float(np.mean(aps))
        results['num_images'] = len(self.image_ids)
        
        return results
    
    def print_results(self, results):
        """Print evaluation results in formatted table"""
        print("\n" + "="*70)
        print("EVALUATION RESULTS")
        print("="*70)
        
        print(f"\nNumber of images: {results['num_images']}")
        print(f"Overall mAP: {results['mAP']:.4f}")
        
        print("\nPer-Class Results:")
        print("-" * 70)
        print(f"{'Class':<20} {'AP':<10} {'Precision':<12} {'Recall':<10} {'GT':<8} {'Pred':<8}")
        print("-" * 70)
        
        for class_name, metrics in results.items():
            if class_name not in ['mAP', 'num_images']:
                print(f"{class_name:<20} {metrics['ap']:<10.4f} "
                      f"{metrics['precision']:<12.4f} {metrics['recall']:<10.4f} "
                      f"{metrics['num_gt']:<8} {metrics['num_pred']:<8}")
        
        print("="*70 + "\n")
    
    def plot_precision_recall_curves(self, results, save_path='pr_curves.png'):
        """Plot precision-recall curves for all classes"""
        plt.figure(figsize=(12, 8))
        
        for class_name, metrics in results.items():
            if class_name not in ['mAP', 'num_images']:
                if 'recalls' in metrics and 'precisions' in metrics:
                    recalls = metrics['recalls']
                    precisions = metrics['precisions']
                    ap = metrics['ap']
                    
                    plt.plot(recalls, precisions, 
                            label=f"{class_name} (AP={ap:.3f})")
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend()
        plt.grid(True)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Precision-Recall curves saved to {save_path}")
        plt.close()


class ConfusionMatrix:
    """Confusion matrix for multi-class detection"""
    
    def __init__(self, num_classes, class_names):
        """
        Initialize confusion matrix
        
        Args:
            num_classes: Number of classes
            class_names: List of class names
        """
        self.num_classes = num_classes
        self.class_names = class_names
        self.matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int32)
        # +1 for background class
    
    def update(self, pred_classes, gt_classes):
        """Update confusion matrix with predictions"""
        for pred, gt in zip(pred_classes, gt_classes):
            self.matrix[gt, pred] += 1
    
    def plot(self, save_path='confusion_matrix.png', normalize=False):
        """Plot confusion matrix"""
        matrix = self.matrix.copy()
        
        if normalize:
            matrix = matrix.astype(np.float32)
            row_sums = matrix.sum(axis=1, keepdims=True)
            matrix = np.divide(matrix, row_sums, 
                             where=row_sums != 0, 
                             out=np.zeros_like(matrix))
        
        plt.figure(figsize=(10, 8))
        plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        tick_marks = np.arange(len(self.class_names) + 1)
        labels = self.class_names + ['background']
        plt.xticks(tick_marks, labels, rotation=45)
        plt.yticks(tick_marks, labels)
        
        # Add text annotations
        thresh = matrix.max() / 2
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                if normalize:
                    text = f'{value:.2f}'
                else:
                    text = f'{int(value)}'
                
                plt.text(j, i, text,
                        ha="center", va="center",
                        color="white" if value > thresh else "black")
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
        plt.close()


class ModelTester:
    """Test detection models on images and videos"""
    
    def __init__(self, model, class_names):
        """
        Initialize tester
        
        Args:
            model: Detection model
            class_names: List of class names
        """
        self.model = model
        self.class_names = class_names
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def test_image(self, image_path, conf_threshold=0.5):
        """
        Test model on a single image
        
        Args:
            image_path: Path to image
            conf_threshold: Confidence threshold
        
        Returns:
            detections: List of detected objects
            vis_image: Visualized image with detections
        """
        # Load image
        image = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Prepare input
        input_tensor = self._preprocess_image(image_rgb)
        
        # Inference
        with torch.no_grad():
            predictions = self.model(input_tensor)
        
        # Process predictions
        detections = self._process_predictions(
            predictions, 
            conf_threshold,
            image.shape
        )
        
        # Visualize
        vis_image = self._visualize_detections(image, detections)
        
        return detections, vis_image
    
    def test_directory(self, images_dir, output_dir, conf_threshold=0.5):
        """
        Test model on all images in a directory
        
        Args:
            images_dir: Directory containing images
            output_dir: Directory to save results
            conf_threshold: Confidence threshold
        """
        images_dir = Path(images_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all images
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
        
        print(f"\nTesting on {len(image_files)} images...")
        
        results = []
        
        for image_path in tqdm(image_files):
            detections, vis_image = self.test_image(image_path, conf_threshold)
            
            # Save visualized image
            output_path = output_dir / f"result_{image_path.name}"
            cv2.imwrite(str(output_path), vis_image)
            
            results.append({
                'image': image_path.name,
                'detections': len(detections),
                'objects': detections
            })
        
        # Save results JSON
        results_path = output_dir / 'test_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {output_dir}")
        print(f"  - Visualized images: {len(image_files)}")
        print(f"  - Results JSON: {results_path}")
    
    def _preprocess_image(self, image):
        """Preprocess image for model input"""
        # Resize and normalize
        image = cv2.resize(image, (640, 640))
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        return image.to(self.device)
    
    def _process_predictions(self, predictions, conf_threshold, original_shape):
        """Process model predictions"""
        # This is model-specific
        # Placeholder implementation
        detections = []
        return detections
    
    def _visualize_detections(self, image, detections):
        """Draw bounding boxes on image"""
        vis_image = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            class_name = det['class_name']
            
            # Draw box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(vis_image, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return vis_image


def main():
    """Test evaluation utilities"""
    print("\n" + "="*70)
    print("EVALUATION AND TESTING UTILITIES")
    print("="*70)
    
    print("\nThis module provides:")
    print("  • Detection evaluation with mAP calculation")
    print("  • Precision-Recall curve plotting")
    print("  • Confusion matrix visualization")
    print("  • Model testing on images and videos")
    
    print("\nUsage:")
    print("  from evaluation import DetectionEvaluator, ModelTester")
    print("  evaluator = DetectionEvaluator()")
    print("  results = evaluator.evaluate(class_names)")
    
    # Demo evaluation
    print("\n" + "-"*70)
    print("Demo: Creating sample evaluation")
    print("-"*70)
    
    evaluator = DetectionEvaluator()
    
    # Add some dummy predictions
    for i in range(10):
        pred_boxes = np.random.rand(3, 4) * 100
        pred_scores = np.random.rand(3)
        pred_classes = np.random.randint(0, 3, 3)
        
        gt_boxes = np.random.rand(2, 4) * 100
        gt_classes = np.random.randint(0, 3, 2)
        
        evaluator.add_predictions(
            pred_boxes, pred_scores, pred_classes,
            gt_boxes, gt_classes, f"image_{i}"
        )
    
    results = evaluator.evaluate(['child', 'fire', 'pool'])
    evaluator.print_results(results)
    
    print("="*70)


if __name__ == "__main__":
    main()