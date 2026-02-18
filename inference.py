"""
Inference and Testing Utilities
Comprehensive testing tools for trained models
"""

import cv2
import numpy as np
from pathlib import Path
import torch
from ultralytics import YOLO
import time
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from collections import defaultdict


class ModelInference:
    """Inference engine for trained models"""
    
    def __init__(self, model_path, conf_threshold=0.45, iou_threshold=0.45):
        """
        Initialize inference engine
        
        Args:
            model_path: Path to trained model
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
        """
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        print(f"\nLoading model: {model_path}")
        self.model = YOLO(str(model_path))
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"Device: {self.device}")
        print(f"Confidence threshold: {conf_threshold}")
        print(f"IoU threshold: {iou_threshold}\n")
    
    def predict_image(self, image_path, save_output=True, output_dir='inference_results'):
        """
        Run inference on single image
        
        Args:
            image_path: Path to image
            save_output: Whether to save annotated image
            output_dir: Directory to save results
        
        Returns:
            results: Detection results
        """
        image_path = Path(image_path)
        
        # Run inference
        results = self.model.predict(
            source=str(image_path),
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )[0]
        
        # Save annotated image if requested
        if save_output:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            annotated = results.plot()
            output_path = output_dir / f"result_{image_path.name}"
            cv2.imwrite(str(output_path), annotated)
        
        return results
    
    def predict_video(self, video_path, output_path=None, display=False):
        """
        Run inference on video
        
        Args:
            video_path: Path to video file
            output_path: Path to save output video
            display: Whether to display results
        
        Returns:
            stats: Video inference statistics
        """
        video_path = Path(video_path)
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"✗ Failed to open video: {video_path}")
            return None
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\nVideo: {video_path.name}")
        print(f"Resolution: {width}x{height}")
        print(f"FPS: {fps}")
        print(f"Frames: {total_frames}\n")
        
        # Video writer
        writer = None
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Processing stats
        stats = {
            'total_frames': 0,
            'detections_per_frame': [],
            'inference_times': [],
            'avg_confidence': []
        }
        
        # Process video
        pbar = tqdm(total=total_frames, desc="Processing")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Inference
            start_time = time.time()
            results = self.model.predict(
                source=frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False
            )[0]
            inference_time = time.time() - start_time
            
            # Get annotated frame
            annotated = results.plot()
            
            # Update stats
            stats['total_frames'] += 1
            stats['inference_times'].append(inference_time)
            
            num_detections = len(results.boxes) if results.boxes is not None else 0
            stats['detections_per_frame'].append(num_detections)
            
            if num_detections > 0:
                confidences = results.boxes.conf.cpu().numpy()
                stats['avg_confidence'].append(np.mean(confidences))
            
            # Write frame
            if writer:
                writer.write(annotated)
            
            # Display
            if display:
                cv2.imshow('Inference', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            pbar.update(1)
        
        pbar.close()
        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"VIDEO INFERENCE SUMMARY")
        print(f"{'='*70}")
        print(f"Total Frames: {stats['total_frames']}")
        print(f"Avg FPS: {1.0 / np.mean(stats['inference_times']):.2f}")
        print(f"Avg Inference Time: {np.mean(stats['inference_times'])*1000:.2f}ms")
        print(f"Total Detections: {sum(stats['detections_per_frame'])}")
        print(f"Avg Detections/Frame: {np.mean(stats['detections_per_frame']):.2f}")
        if stats['avg_confidence']:
            print(f"Avg Confidence: {np.mean(stats['avg_confidence']):.3f}")
        print(f"{'='*70}\n")
        
        if output_path:
            print(f"Output saved to: {output_path}")
        
        return stats
    
    def predict_directory(self, images_dir, output_dir='batch_results'):
        """
        Run inference on directory of images
        
        Args:
            images_dir: Directory containing images
            output_dir: Directory to save results
        
        Returns:
            results_summary: Summary of all detections
        """
        images_dir = Path(images_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all images
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(images_dir.glob(ext))
        
        print(f"\nFound {len(image_files)} images")
        
        results_summary = {
            'total_images': len(image_files),
            'images_with_detections': 0,
            'total_detections': 0,
            'class_counts': defaultdict(int),
            'inference_times': []
        }
        
        # Process all images
        for img_path in tqdm(image_files, desc="Processing"):
            start_time = time.time()
            results = self.predict_image(img_path, save_output=True, output_dir=output_dir)
            inference_time = time.time() - start_time
            
            results_summary['inference_times'].append(inference_time)
            
            if results.boxes is not None and len(results.boxes) > 0:
                results_summary['images_with_detections'] += 1
                results_summary['total_detections'] += len(results.boxes)
                
                # Count classes
                classes = results.boxes.cls.cpu().numpy().astype(int)
                for cls in classes:
                    results_summary['class_counts'][int(cls)] += 1
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"BATCH INFERENCE SUMMARY")
        print(f"{'='*70}")
        print(f"Total Images: {results_summary['total_images']}")
        print(f"Images with Detections: {results_summary['images_with_detections']}")
        print(f"Total Detections: {results_summary['total_detections']}")
        print(f"Avg Detections/Image: {results_summary['total_detections']/results_summary['total_images']:.2f}")
        print(f"Avg Inference Time: {np.mean(results_summary['inference_times'])*1000:.2f}ms")
        
        if results_summary['class_counts']:
            print(f"\nClass Distribution:")
            for cls, count in sorted(results_summary['class_counts'].items()):
                print(f"  Class {cls}: {count}")
        
        print(f"{'='*70}\n")
        print(f" Results saved to: {output_dir}")
        
        # Save summary JSON
        summary_path = output_dir / 'summary.json'
        with open(summary_path, 'w') as f:
            # Convert defaultdict to dict for JSON
            summary_to_save = dict(results_summary)
            summary_to_save['class_counts'] = dict(summary_to_save['class_counts'])
            json.dump(summary_to_save, f, indent=2)
        
        return results_summary


class PerformanceBenchmark:
    """Benchmark model performance"""
    
    def __init__(self, model_path):
        """
        Initialize benchmark
        
        Args:
            model_path: Path to model
        """
        self.model = YOLO(str(model_path))
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.results = {}
    
    def benchmark_speed(self, image_sizes=[320, 640, 1280], num_iterations=100):
        """
        Benchmark inference speed at different resolutions
        
        Args:
            image_sizes: List of image sizes to test
            num_iterations: Number of iterations per size
        
        Returns:
            speed_results: Dictionary with speed benchmarks
        """
        print(f"\n{'='*70}")
        print(f"SPEED BENCHMARK")
        print(f"{'='*70}\n")
        
        speed_results = {}
        
        for size in image_sizes:
            print(f"Testing {size}x{size}...")
            
            # Create dummy image
            dummy_img = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
            
            # Warmup
            for _ in range(10):
                _ = self.model.predict(dummy_img, verbose=False, device=self.device)
            
            # Benchmark
            times = []
            for _ in tqdm(range(num_iterations), desc=f"{size}x{size}"):
                start = time.time()
                _ = self.model.predict(dummy_img, verbose=False, device=self.device)
                times.append(time.time() - start)
            
            speed_results[size] = {
                'mean': np.mean(times) * 1000,  # ms
                'std': np.std(times) * 1000,
                'min': np.min(times) * 1000,
                'max': np.max(times) * 1000,
                'fps': 1.0 / np.mean(times)
            }
            
            print(f"  Mean: {speed_results[size]['mean']:.2f}ms")
            print(f"  FPS: {speed_results[size]['fps']:.2f}")
        
        self.results['speed'] = speed_results
        
        # Plot results
        self._plot_speed_benchmark(speed_results)
        
        return speed_results
    
    def benchmark_batch_sizes(self, batch_sizes=[1, 4, 8, 16, 32], image_size=640):
        """
        Benchmark different batch sizes
        
        Args:
            batch_sizes: List of batch sizes to test
            image_size: Image resolution
        
        Returns:
            batch_results: Dictionary with batch benchmarks
        """
        print(f"\n{'='*70}")
        print(f"BATCH SIZE BENCHMARK")
        print(f"{'='*70}\n")
        
        batch_results = {}
        
        for batch_size in batch_sizes:
            print(f"Testing batch size {batch_size}...")
            
            # Create dummy batch
            dummy_batch = [
                np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)
                for _ in range(batch_size)
            ]
            
            try:
                # Warmup
                for _ in range(5):
                    _ = self.model.predict(dummy_batch, verbose=False, device=self.device)
                
                # Benchmark
                times = []
                for _ in range(20):
                    start = time.time()
                    _ = self.model.predict(dummy_batch, verbose=False, device=self.device)
                    times.append(time.time() - start)
                
                batch_results[batch_size] = {
                    'total_time': np.mean(times) * 1000,
                    'per_image': np.mean(times) / batch_size * 1000,
                    'throughput': batch_size / np.mean(times)
                }
                
                print(f"  Time per image: {batch_results[batch_size]['per_image']:.2f}ms")
                print(f"  Throughput: {batch_results[batch_size]['throughput']:.2f} img/s")
                
            except Exception as e:
                print(f"  Failed: {str(e)}")
                continue
        
        self.results['batch'] = batch_results
        
        return batch_results
    
    def _plot_speed_benchmark(self, speed_results):
        """Plot speed benchmark results"""
        sizes = list(speed_results.keys())
        means = [speed_results[s]['mean'] for s in sizes]
        stds = [speed_results[s]['std'] for s in sizes]
        fps = [speed_results[s]['fps'] for s in sizes]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Inference time
        ax1.bar(range(len(sizes)), means, yerr=stds, capsize=5, alpha=0.7)
        ax1.set_xticks(range(len(sizes)))
        ax1.set_xticklabels([f"{s}x{s}" for s in sizes])
        ax1.set_ylabel('Inference Time (ms)')
        ax1.set_title('Inference Time vs Image Size')
        ax1.grid(axis='y', alpha=0.3)
        
        # FPS
        ax2.bar(range(len(sizes)), fps, alpha=0.7, color='green')
        ax2.set_xticks(range(len(sizes)))
        ax2.set_xticklabels([f"{s}x{s}" for s in sizes])
        ax2.set_ylabel('FPS')
        ax2.set_title('FPS vs Image Size')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('speed_benchmark.png', dpi=150, bbox_inches='tight')
        print(f"\n Speed benchmark plot saved to: speed_benchmark.png")
        plt.close()
    
    def save_results(self, output_path='benchmark_results.json'):
        """Save benchmark results"""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f" Benchmark results saved to: {output_path}")


def main():
    """Main inference testing"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Model Inference and Testing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test on single image
  python inference.py --model models/child_detector.pt --image test.jpg
  
  # Test on video
  python inference.py --model models/fire_detector.pt --video test.mp4 --output result.mp4
  
  # Test on directory
  python inference.py --model models/pool_detector.pt --dir test_images/
  
  # Benchmark speed
  python inference.py --model models/child_detector.pt --benchmark
        """
    )
    
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--image', type=str, help='Path to test image')
    parser.add_argument('--video', type=str, help='Path to test video')
    parser.add_argument('--dir', type=str, help='Directory of test images')
    parser.add_argument('--output', type=str, help='Output path for results')
    parser.add_argument('--conf', type=float, default=0.45, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IoU threshold')
    parser.add_argument('--display', action='store_true', help='Display results (for video)')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmark')
    
    args = parser.parse_args()
    
    # Benchmark mode
    if args.benchmark:
        benchmark = PerformanceBenchmark(args.model)
        benchmark.benchmark_speed()
        benchmark.benchmark_batch_sizes()
        benchmark.save_results()
        return
    
    # Create inference engine
    inference = ModelInference(args.model, args.conf, args.iou)
    
    # Single image
    if args.image:
        results = inference.predict_image(args.image)
        print(f" Processed: {args.image}")
    
    # Video
    elif args.video:
        stats = inference.predict_video(args.video, args.output, args.display)
    
    # Directory
    elif args.dir:
        summary = inference.predict_directory(args.dir, args.output or 'batch_results')
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()