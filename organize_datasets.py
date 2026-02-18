"""
Dataset Reorganization Script
Organizes existing downloaded datasets into proper training structure
"""

import shutil
from pathlib import Path
import yaml
from tqdm import tqdm
import os


class DatasetOrganizer:
    """Reorganize downloaded datasets into proper structure"""
    
    def __init__(self, source_root, target_root='data'):
        """
        Initialize organizer
        
        Args:
            source_root: Root directory with downloaded datasets
            target_root: Target directory for organized datasets
        """
        self.source_root = Path(source_root)
        self.target_root = Path(target_root)
        self.target_root.mkdir(exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"DATASET ORGANIZER")
        print(f"{'='*70}")
        print(f"Source: {self.source_root}")
        print(f"Target: {self.target_root}")
        print(f"{'='*70}\n")
    
    def organize_child_dataset(self):
        """Organize child detection dataset"""
        print("\n" + "="*70)
        print("ORGANIZING CHILD DETECTION DATASET")
        print("="*70)
        
        # Source: child_kaggle/detect_child/
        source = self.source_root / 'child_kaggle' / 'detect_child'
        target = self.target_root / 'child_detection'
        
        if not source.exists():
            print(f"✗ Source not found: {source}")
            return False
        
        print(f"Source: {source}")
        print(f"Target: {target}")
        
        # Copy structure
        self._copy_dataset_structure(source, target, has_val=False)
        
        # Create data.yaml
        self._create_yaml(
            target,
            nc=1,
            names=['child']
        )
        
        print("✓ Child dataset organized")
        return True
    
    def organize_fire_datasets(self):
        """Organize and merge fire detection datasets"""
        print("\n" + "="*70)
        print("ORGANIZING FIRE DETECTION DATASETS")
        print("="*70)
        
        target = self.target_root / 'fire_detection'
        target.mkdir(exist_ok=True)
        
        # Source 1: fire_kaggle
        source1 = self.source_root / 'fire_kaggle'
        
        # Source 2: Indoor Fire Smoke
        source2 = self.source_root / 'Indoor Fire Smoke'
        
        if not source1.exists() and not source2.exists():
            print(f"✗ No fire datasets found")
            return False
        
        # Strategy: Use the dataset with more complete structure
        # Indoor Fire Smoke seems to have data.yaml and valid folder
        if source2.exists():
            print(f"\nUsing: Indoor Fire Smoke (more complete)")
            self._copy_dataset_structure(source2, target, has_val=True, val_name='valid')
            
            # If fire_kaggle exists, we could merge it, but for simplicity use one
            if source1.exists():
                print(f"Note: fire_kaggle also available, using Indoor Fire Smoke only")
        elif source1.exists():
            print(f"\nUsing: fire_kaggle")
            self._copy_dataset_structure(source1, target, has_val=True)
        
        # Create/update data.yaml
        self._create_yaml(
            target,
            nc=3,
            names=['fire', 'smoke', 'neutral']
        )
        
        print("✓ Fire dataset organized")
        return True
    
    def organize_pool_dataset(self):
        """Organize pool detection dataset"""
        print("\n" + "="*70)
        print("ORGANIZING POOL DETECTION DATASET")
        print("="*70)
        
        # Source: pool_and_drowning_kaggle
        source = self.source_root / 'pool_and_drowning_kaggle'
        target = self.target_root / 'pool_detection'
        
        if not source.exists():
            print(f"✗ Source not found: {source}")
            return False
        
        print(f"Source: {source}")
        print(f"Target: {target}")
        
        # Copy structure (has valid instead of val)
        self._copy_dataset_structure(source, target, has_val=True, val_name='valid')
        
        # Create data.yaml
        self._create_yaml(
            target,
            nc=2,
            names=['drowning', 'swimming']
        )
        
        print("✓ Pool dataset organized")
        return True
    
    def _copy_dataset_structure(self, source, target, has_val=True, val_name='val'):
        """
        Copy dataset structure from source to target
        
        Args:
            source: Source directory
            target: Target directory
            has_val: Whether dataset has validation set
            val_name: Name of validation folder ('val' or 'valid')
        """
        target.mkdir(exist_ok=True)
        
        # Define splits to copy
        splits = ['train', 'test']
        if has_val:
            splits.append(val_name)
        
        for split in splits:
            source_split = source / split
            
            if not source_split.exists():
                print(f"⚠ Warning: {split} folder not found in {source.name}")
                continue
            
            # Determine target split name (normalize 'valid' to 'val')
            target_split_name = 'val' if split == 'valid' else split
            target_split = target / target_split_name
            
            print(f"\nCopying {split}...")
            
            # Look for images and labels folders
            for folder in ['images', 'labels']:
                source_folder = source_split / folder
                target_folder = target_split / folder
                
                if source_folder.exists():
                    # Create target folder
                    target_folder.mkdir(parents=True, exist_ok=True)
                    
                    # Copy files
                    files = list(source_folder.glob('*'))
                    files = [f for f in files if f.is_file() and not f.name.startswith('.')]
                    
                    print(f"  Copying {len(files)} files from {folder}/")
                    
                    for file in tqdm(files, desc=f"  {split}/{folder}"):
                        target_file = target_folder / file.name
                        if not target_file.exists():
                            shutil.copy2(file, target_file)
                else:
                    print(f"  ⚠ {folder}/ not found in {split}")
        
        # Count files
        self._print_dataset_stats(target)
    
    def _create_yaml(self, dataset_dir, nc, names):
        """Create YOLO dataset YAML file"""
        yaml_content = {
            'path': str(dataset_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': nc,
            'names': names
        }
        
        yaml_path = dataset_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
        
        print(f"\n✓ Created: {yaml_path}")
        print(f"  Classes: {names}")
    
    def _print_dataset_stats(self, dataset_dir):
        """Print dataset statistics"""
        print(f"\nDataset Statistics:")
        
        for split in ['train', 'val', 'test']:
            split_dir = dataset_dir / split
            if not split_dir.exists():
                continue
            
            images_dir = split_dir / 'images'
            labels_dir = split_dir / 'labels'
            
            num_images = len(list(images_dir.glob('*'))) if images_dir.exists() else 0
            num_labels = len(list(labels_dir.glob('*.txt'))) if labels_dir.exists() else 0
            
            print(f"  {split:6s}: {num_images:5d} images, {num_labels:5d} labels")
    
    def organize_all(self):
        """Organize all datasets"""
        print("\n" + "#"*70)
        print("#" + " "*20 + "ORGANIZING ALL DATASETS" + " "*24 + "#")
        print("#"*70)
        
        results = {}
        
        # Organize each dataset
        results['child'] = self.organize_child_dataset()
        results['fire'] = self.organize_fire_datasets()
        results['pool'] = self.organize_pool_dataset()
        
        # Summary
        print("\n" + "="*70)
        print("ORGANIZATION SUMMARY")
        print("="*70)
        
        for dataset, success in results.items():
            status = "✓ Success" if success else "✗ Failed"
            print(f"{dataset.capitalize():10s}: {status}")
        
        print("="*70)
        
        # Next steps
        if all(results.values()):
            print("\n✓ All datasets organized successfully!")
            print("\nNext steps:")
            print("  1. Validate datasets:")
            print("     python data_validation.py data/child_detection --validate")
            print("     python data_validation.py data/fire_detection --validate")
            print("     python data_validation.py data/pool_detection --validate")
            print("\n  2. Start training:")
            print("     python proper_training.py --all --epochs 100")
        else:
            print("\n⚠ Some datasets failed to organize. Check the errors above.")
        
        print()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Organize Downloaded Datasets for Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  # If your datasets are in ~/Desktop/DA/TARP/
  python organize_datasets.py ~/Desktop/DA/TARP/

  # This will create organized datasets in ./data/
  
  # Then validate:
  python data_validation.py data/child_detection --validate
  
  # Then train:
  python proper_training.py --all --epochs 100
        """
    )
    
    parser.add_argument(
        'source_dir',
        type=str,
        nargs='?',
        default='.',
        help='Source directory containing downloaded datasets (default: current directory)'
    )
    
    parser.add_argument(
        '--target',
        type=str,
        default='data',
        help='Target directory for organized datasets (default: data)'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['child', 'fire', 'pool', 'all'],
        default='all',
        help='Which dataset to organize (default: all)'
    )
    
    args = parser.parse_args()
    
    # Create organizer
    organizer = DatasetOrganizer(args.source_dir, args.target)
    
    # Organize requested datasets
    if args.dataset == 'all':
        organizer.organize_all()
    elif args.dataset == 'child':
        organizer.organize_child_dataset()
    elif args.dataset == 'fire':
        organizer.organize_fire_datasets()
    elif args.dataset == 'pool':
        organizer.organize_pool_dataset()


if __name__ == "__main__":
    main()