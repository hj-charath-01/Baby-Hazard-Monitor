"""
Data Validation and Preparation Utilities
Validates datasets and prepares them for YOLO training
"""

import os
import cv2
import numpy as np
from pathlib import Path
import yaml
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
import shutil


class DatasetValidator:
    """Validate and prepare datasets for training"""
    
    def __init__(self, dataset_dir):
        """
        Initialize validator
        
        Args:
            dataset_dir: Root directory of dataset
        """
        self.dataset_dir = Path(dataset_dir)
        self.issues = []
        self.stats = defaultdict(lambda: defaultdict(int))
    
    def validate_structure(self):
        """Validate dataset directory structure"""
        print(f"\n{'='*70}")
        print(f"VALIDATING DATASET STRUCTURE: {self.dataset_dir.name}")
        print(f"{'='*70}\n")
        
        required_dirs = [
            'train/images',
            'train/labels',
            'val/images',
            'val/labels'
        ]
        
        optional_dirs = [
            'test/images',
            'test/labels'
        ]
        
        all_valid = True
        
        # Check required directories
        for dir_path in required_dirs:
            full_path = self.dataset_dir / dir_path
            if full_path.exists():
                print(f"✓ Found: {dir_path}")
            else:
                print(f"✗ Missing: {dir_path}")
                self.issues.append(f"Missing required directory: {dir_path}")
                all_valid = False
        
        # Check optional directories
        for dir_path in optional_dirs:
            full_path = self.dataset_dir / dir_path
            if full_path.exists():
                print(f"✓ Found: {dir_path} (optional)")
            else:
                print(f"⚠ Missing: {dir_path} (optional)")
        
        return all_valid
    
    def validate_images_and_labels(self, split='train'):
        """
        Validate image-label pairs
        
        Args:
            split: 'train', 'val', or 'test'
        
        Returns:
            stats: Dictionary with validation statistics
        """
        print(f"\n{'='*70}")
        print(f"VALIDATING {split.upper()} SET")
        print(f"{'='*70}\n")
        
        images_dir = self.dataset_dir / split / 'images'
        labels_dir = self.dataset_dir / split / 'labels'
        
        if not images_dir.exists():
            print(f"✗ Images directory not found: {images_dir}")
            return {}
        
        # Get all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(images_dir.glob(ext))
        
        print(f"Found {len(image_files)} images")
        
        stats = {
            'total_images': len(image_files),
            'valid_images': 0,
            'corrupted_images': 0,
            'missing_labels': 0,
            'empty_labels': 0,
            'valid_pairs': 0,
            'total_boxes': 0,
            'class_distribution': defaultdict(int),
            'image_sizes': [],
            'boxes_per_image': []
        }
        
        print("\nValidating image-label pairs...")
        
        for image_path in tqdm(image_files):
            # Check image
            try:
                img = cv2.imread(str(image_path))
                if img is None:
                    stats['corrupted_images'] += 1
                    self.issues.append(f"Corrupted image: {image_path.name}")
                    continue
                
                h, w = img.shape[:2]
                stats['image_sizes'].append((w, h))
                stats['valid_images'] += 1
                
            except Exception as e:
                stats['corrupted_images'] += 1
                self.issues.append(f"Error reading {image_path.name}: {str(e)}")
                continue
            
            # Check label
            label_path = labels_dir / f"{image_path.stem}.txt"
            
            if not label_path.exists():
                stats['missing_labels'] += 1
                self.issues.append(f"Missing label: {label_path.name}")
                continue
            
            # Validate label file
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                if len(lines) == 0:
                    stats['empty_labels'] += 1
                    self.issues.append(f"Empty label file: {label_path.name}")
                    continue
                
                num_boxes = 0
                for line_num, line in enumerate(lines, 1):
                    parts = line.strip().split()
                    
                    if len(parts) != 5:
                        self.issues.append(
                            f"Invalid label format in {label_path.name}, "
                            f"line {line_num}: expected 5 values, got {len(parts)}"
                        )
                        continue
                    
                    try:
                        class_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:])
                        
                        # Validate ranges
                        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and
                               0 <= width <= 1 and 0 <= height <= 1):
                            self.issues.append(
                                f"Invalid bbox coordinates in {label_path.name}, "
                                f"line {line_num}: values must be in [0, 1]"
                            )
                            continue
                        
                        stats['class_distribution'][class_id] += 1
                        num_boxes += 1
                        
                    except ValueError as e:
                        self.issues.append(
                            f"Invalid values in {label_path.name}, "
                            f"line {line_num}: {str(e)}"
                        )
                        continue
                
                stats['total_boxes'] += num_boxes
                stats['boxes_per_image'].append(num_boxes)
                stats['valid_pairs'] += 1
                
            except Exception as e:
                self.issues.append(f"Error reading {label_path.name}: {str(e)}")
                continue
        
        # Print statistics
        print(f"\n{'-'*70}")
        print(f"VALIDATION RESULTS")
        print(f"{'-'*70}")
        print(f"Total Images:      {stats['total_images']}")
        print(f"Valid Images:      {stats['valid_images']}")
        print(f"Corrupted Images:  {stats['corrupted_images']}")
        print(f"Missing Labels:    {stats['missing_labels']}")
        print(f"Empty Labels:      {stats['empty_labels']}")
        print(f"Valid Pairs:       {stats['valid_pairs']}")
        print(f"Total Boxes:       {stats['total_boxes']}")
        
        if stats['boxes_per_image']:
            print(f"Avg Boxes/Image:   {np.mean(stats['boxes_per_image']):.2f}")
        
        if stats['class_distribution']:
            print(f"\nClass Distribution:")
            for class_id, count in sorted(stats['class_distribution'].items()):
                print(f"  Class {class_id}: {count} boxes")
        
        print(f"{'-'*70}\n")
        
        return stats
    
    def create_split(self, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, shuffle=True):
        """
        Create train/val/test splits from a single directory
        
        Args:
            train_ratio: Ratio of training data
            val_ratio: Ratio of validation data
            test_ratio: Ratio of test data
            shuffle: Whether to shuffle before splitting
        """
        print(f"\n{'='*70}")
        print(f"CREATING DATA SPLITS")
        print(f"{'='*70}\n")
        
        # Assuming all images are in a single directory initially
        all_images_dir = self.dataset_dir / 'images'
        all_labels_dir = self.dataset_dir / 'labels'
        
        if not all_images_dir.exists():
            print(f"✗ Source images directory not found: {all_images_dir}")
            return
        
        # Get all images
        image_files = list(all_images_dir.glob('*.jpg')) + \
                     list(all_images_dir.glob('*.png'))
        
        print(f"Found {len(image_files)} images")
        
        # Shuffle if requested
        if shuffle:
            np.random.shuffle(image_files)
        
        # Calculate split indices
        n = len(image_files)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        splits = {
            'train': image_files[:train_end],
            'val': image_files[train_end:val_end],
            'test': image_files[val_end:]
        }
        
        # Create directories and move files
        for split_name, files in splits.items():
            print(f"\n{split_name.upper()}: {len(files)} images")
            
            # Create directories
            (self.dataset_dir / split_name / 'images').mkdir(parents=True, exist_ok=True)
            (self.dataset_dir / split_name / 'labels').mkdir(parents=True, exist_ok=True)
            
            # Copy files
            for img_path in tqdm(files, desc=f"Copying {split_name}"):
                # Copy image
                dst_img = self.dataset_dir / split_name / 'images' / img_path.name
                shutil.copy(img_path, dst_img)
                
                # Copy label
                label_path = all_labels_dir / f"{img_path.stem}.txt"
                if label_path.exists():
                    dst_label = self.dataset_dir / split_name / 'labels' / label_path.name
                    shutil.copy(label_path, dst_label)
        
        print(f"\n✓ Splits created successfully")
    
    def visualize_dataset(self, split='train', num_samples=9, save_path=None):
        """
        Visualize random samples from dataset
        
        Args:
            split: 'train', 'val', or 'test'
            num_samples: Number of samples to visualize
            save_path: Path to save visualization
        """
        images_dir = self.dataset_dir / split / 'images'
        labels_dir = self.dataset_dir / split / 'labels'
        
        if not images_dir.exists():
            print(f"✗ Images directory not found: {images_dir}")
            return
        
        # Get random images
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
        samples = np.random.choice(image_files, min(num_samples, len(image_files)), replace=False)
        
        # Create visualization
        grid_size = int(np.ceil(np.sqrt(num_samples)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
        axes = axes.flatten() if num_samples > 1 else [axes]
        
        for idx, img_path in enumerate(samples):
            # Read image
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            
            # Read labels
            label_path = labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:])
                            
                            # Convert to pixel coordinates
                            x1 = int((x_center - width/2) * w)
                            y1 = int((y_center - height/2) * h)
                            x2 = int((x_center + width/2) * w)
                            y2 = int((y_center + height/2) * h)
                            
                            # Draw box
                            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            cv2.putText(img, str(class_id), (x1, y1-5),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Plot
            axes[idx].imshow(img)
            axes[idx].set_title(img_path.name, fontsize=8)
            axes[idx].axis('off')
        
        # Hide unused subplots
        for idx in range(len(samples), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Visualization saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_report(self, save_path='dataset_report.json'):
        """Generate comprehensive dataset report"""
        report = {
            'dataset_dir': str(self.dataset_dir),
            'validation_time': str(np.datetime64('now')),
            'issues': self.issues,
            'splits': {}
        }
        
        # Validate each split
        for split in ['train', 'val', 'test']:
            split_dir = self.dataset_dir / split
            if split_dir.exists():
                stats = self.validate_images_and_labels(split)
                report['splits'][split] = {
                    'total_images': stats.get('total_images', 0),
                    'valid_pairs': stats.get('valid_pairs', 0),
                    'total_boxes': stats.get('total_boxes', 0),
                    'class_distribution': dict(stats.get('class_distribution', {}))
                }
        
        # Save report
        save_path = Path(save_path)
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✓ Report saved to: {save_path}")
        
        return report


def convert_to_yolo_format(source_dir, output_dir, annotation_format='coco'):
    """
    Convert annotations to YOLO format
    
    Args:
        source_dir: Directory with source annotations
        output_dir: Output directory for YOLO format
        annotation_format: Source format ('coco', 'voc', 'labelme')
    """
    print(f"\n{'='*70}")
    print(f"CONVERTING TO YOLO FORMAT")
    print(f"{'='*70}")
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"Format: {annotation_format}")
    print(f"{'='*70}\n")
    
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if annotation_format == 'coco':
        # COCO JSON to YOLO
        import json
        
        annotation_file = source_dir / 'annotations.json'
        if not annotation_file.exists():
            print(f"✗ COCO annotation file not found: {annotation_file}")
            return
        
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)
        
        # Create image_id to filename mapping
        images = {img['id']: img for img in coco_data['images']}
        
        # Group annotations by image
        annotations_by_image = defaultdict(list)
        for ann in coco_data['annotations']:
            annotations_by_image[ann['image_id']].append(ann)
        
        # Convert each image's annotations
        for img_id, annotations in tqdm(annotations_by_image.items()):
            image_info = images[img_id]
            img_width = image_info['width']
            img_height = image_info['height']
            
            # Create YOLO label file
            label_filename = Path(image_info['file_name']).stem + '.txt'
            label_path = output_dir / label_filename
            
            with open(label_path, 'w') as f:
                for ann in annotations:
                    category_id = ann['category_id']
                    bbox = ann['bbox']  # [x, y, width, height]
                    
                    # Convert to YOLO format (center_x, center_y, width, height) - normalized
                    x_center = (bbox[0] + bbox[2] / 2) / img_width
                    y_center = (bbox[1] + bbox[3] / 2) / img_height
                    width = bbox[2] / img_width
                    height = bbox[3] / img_height
                    
                    f.write(f"{category_id} {x_center} {y_center} {width} {height}\n")
        
        print(f"✓ Converted {len(annotations_by_image)} images")
    
    else:
        print(f"✗ Format '{annotation_format}' not yet implemented")


def main():
    """Test dataset validation"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Dataset Validation and Preparation')
    parser.add_argument('dataset_dir', type=str, help='Dataset directory')
    parser.add_argument('--validate', action='store_true', help='Validate dataset')
    parser.add_argument('--split', action='store_true', help='Create train/val/test splits')
    parser.add_argument('--visualize', action='store_true', help='Visualize samples')
    parser.add_argument('--report', action='store_true', help='Generate report')
    
    args = parser.parse_args()
    
    validator = DatasetValidator(args.dataset_dir)
    
    if args.validate:
        validator.validate_structure()
        validator.validate_images_and_labels('train')
        validator.validate_images_and_labels('val')
    
    if args.split:
        validator.create_split()
    
    if args.visualize:
        validator.visualize_dataset(save_path='dataset_samples.png')
    
    if args.report:
        validator.generate_report()


if __name__ == "__main__":
    main()