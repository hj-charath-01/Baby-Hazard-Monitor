"""
Comprehensive YOLOv8 Training Script
Production-ready training for child, fire, and pool detection models
"""

import os
import sys
from pathlib import Path
import yaml
import torch
import numpy as np
from datetime import datetime
import json
import argparse
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns


class YOLOTrainer:
    """
    Production-ready YOLO trainer with comprehensive features
    """
    
    def __init__(self, model_name='yolov8n.pt', config_path='config/config.yaml'):
        """
        Initialize YOLO trainer
        
        Args:
            model_name: Base model to use (yolov8n, yolov8s, yolov8m, etc.)
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Training parameters
        self.img_size = self.config['models']['input_size'][0]
        self.batch_size = self.config['hardware'].get('batch_size', 16)
        
        # Results directory
        self.results_dir = Path('training_results')
        self.results_dir.mkdir(exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"YOLO TRAINER INITIALIZED")
        print(f"{'='*70}")
        print(f"Model: {model_name}")
        print(f"Device: {self.device}")
        print(f"Image Size: {self.img_size}")
        print(f"Batch Size: {self.batch_size}")
        print(f"{'='*70}\n")
    
    def create_dataset_yaml(self, dataset_type, data_dir):
        """
        Create YOLO-format dataset YAML file
        
        Args:
            dataset_type: 'child', 'fire', or 'pool'
            data_dir: Root directory of dataset
        
        Returns:
            yaml_path: Path to created YAML file
        """
        data_dir = Path(data_dir)
        
        # Class names based on dataset type
        class_configs = {
            'child': {
                'nc': 1,
                'names': ['child']
            },
            'fire': {
                'nc': 3,
                'names': ['fire', 'flame', 'smoke']
            },
            'pool': {
                'nc': 2,
                'names': ['pool', 'water']
            }
        }
        
        config = class_configs.get(dataset_type, {'nc': 1, 'names': [dataset_type]})
        
        # Create YAML content
        yaml_content = {
            'path': str(data_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': config['nc'],
            'names': config['names']
        }
        
        # Save YAML file
        yaml_path = data_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        
        print(f"Created dataset YAML: {yaml_path}")
        return yaml_path
    
    def train_model(self, data_yaml, output_name, epochs=100, 
                   patience=15, save_period=10, **kwargs):
        """
        Train YOLO model with comprehensive settings
        
        Args:
            data_yaml: Path to dataset YAML file
            output_name: Name for output model and results
            epochs: Number of training epochs
            patience: Early stopping patience
            save_period: Save checkpoint every N epochs
            **kwargs: Additional YOLO training arguments
        
        Returns:
            results: Training results object
        """
        print(f"\n{'='*70}")
        print(f"TRAINING: {output_name}")
        print(f"{'='*70}")
        print(f"Data YAML: {data_yaml}")
        print(f"Epochs: {epochs}")
        print(f"Patience: {patience}")
        print(f"{'='*70}\n")
        
        # Load model
        model = YOLO(self.model_name)
        
        # Training arguments
        train_args = {
            'data': str(data_yaml),
            'epochs': epochs,
            'imgsz': self.img_size,
            'batch': self.batch_size,
            'device': self.device,
            'project': 'training_results',
            'name': output_name,
            'patience': patience,
            'save': True,
            'save_period': save_period,
            'plots': True,
            'val': True,
            'verbose': True,
            
            # Optimization
            'optimizer': 'AdamW',
            'lr0': 0.001,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            
            # Augmentation
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.0,
            
            # Additional settings
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'label_smoothing': 0.0,
            'nbs': 64,
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0,
        }
        
        # Update with custom arguments
        train_args.update(kwargs)
        
        # Train model
        try:
            results = model.train(**train_args)
            
            print(f"\n{'='*70}")
            print(f"TRAINING COMPLETE: {output_name}")
            print(f"{'='*70}")
            
            # Save final model
            model_save_path = Path('models') / f"{output_name}.pt"
            model_save_path.parent.mkdir(exist_ok=True)
            
            # Export best model
            best_model_path = Path('training_results') / output_name / 'weights' / 'best.pt'
            if best_model_path.exists():
                import shutil
                shutil.copy(best_model_path, model_save_path)
                print(f"Best model saved to: {model_save_path}")
            
            return results
            
        except Exception as e:
            print(f"\nTraining failed: {str(e)}")
            raise
    
    def validate_model(self, model_path, data_yaml):
        """
        Validate trained model
        
        Args:
            model_path: Path to trained model
            data_yaml: Path to dataset YAML
        
        Returns:
            metrics: Validation metrics
        """
        print(f"\n{'='*70}")
        print(f"VALIDATING MODEL")
        print(f"{'='*70}")
        print(f"Model: {model_path}")
        print(f"Data: {data_yaml}")
        print(f"{'='*70}\n")
        
        # Load model
        model = YOLO(model_path)
        
        # Validate
        metrics = model.val(
            data=str(data_yaml),
            imgsz=self.img_size,
            batch=self.batch_size,
            device=self.device,
            plots=True,
            save_json=True
        )
        
        # Print results
        print(f"\n{'='*70}")
        print(f"VALIDATION RESULTS")
        print(f"{'='*70}")
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        print(f"Precision: {metrics.box.mp:.4f}")
        print(f"Recall: {metrics.box.mr:.4f}")
        print(f"{'='*70}\n")
        
        return metrics
    
    def export_model(self, model_path, format='onnx'):
        """
        Export model for deployment
        
        Args:
            model_path: Path to trained model
            format: Export format (onnx, tflite, tensorrt, etc.)
        
        Returns:
            export_path: Path to exported model
        """
        print(f"\n{'='*70}")
        print(f"EXPORTING MODEL")
        print(f"{'='*70}")
        print(f"Model: {model_path}")
        print(f"Format: {format}")
        print(f"{'='*70}\n")
        
        # Load model
        model = YOLO(model_path)
        
        # Export
        export_path = model.export(
            format=format,
            imgsz=self.img_size,
            device=self.device
        )
        
        print(f"Model exported to: {export_path}")
        return export_path


class TrainingOrchestrator:
    """
    Orchestrate training for all detection models
    """
    
    def __init__(self):
        self.trainer = YOLOTrainer()
        self.training_log = {
            'start_time': datetime.now().isoformat(),
            'models': {}
        }
    
    def train_all_models(self, epochs=100, base_data_dir='data'):
        """
        Train all three detection models
        
        Args:
            epochs: Number of epochs for each model
            base_data_dir: Base directory containing datasets
        """
        base_data_dir = Path(base_data_dir)
        
        models_to_train = [
            {
                'name': 'child_detector',
                'type': 'child',
                'data_dir': base_data_dir / 'child_detection',
                'description': 'Child Detection Model'
            },
            {
                'name': 'fire_detector',
                'type': 'fire',
                'data_dir': base_data_dir / 'fire_detection',
                'description': 'Fire Detection Model'
            },
            {
                'name': 'pool_detector',
                'type': 'pool',
                'data_dir': base_data_dir / 'pool_detection',
                'description': 'Pool Detection Model'
            }
        ]
        
        print(f"\n{'#'*70}")
        print(f"{'TRAINING ALL MODELS':^70}")
        print(f"{'#'*70}\n")
        
        for i, model_config in enumerate(models_to_train, 1):
            print(f"\n{'='*70}")
            print(f"MODEL {i}/3: {model_config['description']}")
            print(f"{'='*70}\n")
            
            try:
                # Check if data directory exists
                if not model_config['data_dir'].exists():
                    print(f"Data directory not found: {model_config['data_dir']}")
                    print(f"Skipping {model_config['name']}...")
                    continue
                
                # Create dataset YAML
                data_yaml = self.trainer.create_dataset_yaml(
                    model_config['type'],
                    model_config['data_dir']
                )
                
                # Train model
                start_time = datetime.now()
                results = self.trainer.train_model(
                    data_yaml,
                    model_config['name'],
                    epochs=epochs
                )
                end_time = datetime.now()
                
                # Log results
                self.training_log['models'][model_config['name']] = {
                    'type': model_config['type'],
                    'epochs': epochs,
                    'start_time': start_time.isoformat(),
                    'end_time': end_time.isoformat(),
                    'duration': str(end_time - start_time),
                    'status': 'completed'
                }
                
                # Validate
                model_path = Path('models') / f"{model_config['name']}.pt"
                if model_path.exists():
                    metrics = self.trainer.validate_model(model_path, data_yaml)
                    self.training_log['models'][model_config['name']]['metrics'] = {
                        'mAP50': float(metrics.box.map50),
                        'mAP50-95': float(metrics.box.map),
                        'precision': float(metrics.box.mp),
                        'recall': float(metrics.box.mr)
                    }
                
                print(f"\n {model_config['description']} training completed!")
                
            except Exception as e:
                print(f"\nFailed to train {model_config['name']}: {str(e)}")
                self.training_log['models'][model_config['name']] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        # Save training log
        self.training_log['end_time'] = datetime.now().isoformat()
        self._save_training_log()
        
        # Generate summary report
        self._generate_summary_report()
    
    def _save_training_log(self):
        """Save training log to JSON"""
        log_path = Path('training_results') / 'training_log.json'
        log_path.parent.mkdir(exist_ok=True)
        
        with open(log_path, 'w') as f:
            json.dump(self.training_log, f, indent=2)
        
        print(f"\nTraining log saved to: {log_path}")
    
    def _generate_summary_report(self):
        """Generate training summary report"""
        print(f"\n{'#'*70}")
        print(f"{'TRAINING SUMMARY REPORT':^70}")
        print(f"{'#'*70}\n")
        
        for model_name, info in self.training_log['models'].items():
            print(f"\n{model_name.upper().replace('_', ' ')}")
            print(f"{'-'*70}")
            print(f"Status: {info['status']}")
            
            if info['status'] == 'completed':
                print(f"Duration: {info.get('duration', 'N/A')}")
                
                if 'metrics' in info:
                    metrics = info['metrics']
                    print(f"\nMetrics:")
                    print(f"  mAP@0.5:      {metrics['mAP50']:.4f}")
                    print(f"  mAP@0.5:0.95: {metrics['mAP50-95']:.4f}")
                    print(f"  Precision:    {metrics['precision']:.4f}")
                    print(f"  Recall:       {metrics['recall']:.4f}")
            else:
                print(f"Error: {info.get('error', 'Unknown error')}")
        
        print(f"\n{'#'*70}\n")
        
        # Create visualization
        self._plot_training_summary()
    
    def _plot_training_summary(self):
        """Create visualization of training results"""
        completed_models = {
            name: info for name, info in self.training_log['models'].items()
            if info['status'] == 'completed' and 'metrics' in info
        }
        
        if not completed_models:
            print("No completed models to visualize")
            return
        
        # Extract metrics
        model_names = list(completed_models.keys())
        metrics_data = {
            'mAP@0.5': [completed_models[m]['metrics']['mAP50'] for m in model_names],
            'mAP@0.5:0.95': [completed_models[m]['metrics']['mAP50-95'] for m in model_names],
            'Precision': [completed_models[m]['metrics']['precision'] for m in model_names],
            'Recall': [completed_models[m]['metrics']['recall'] for m in model_names]
        }
        
        # Create plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Results Summary', fontsize=16, fontweight='bold')
        
        colors = ['#2ecc71', '#3498db', '#e74c3c']
        
        # mAP@0.5
        ax = axes[0, 0]
        bars = ax.bar(model_names, metrics_data['mAP@0.5'], color=colors[:len(model_names)])
        ax.set_ylabel('mAP@0.5')
        ax.set_title('Mean Average Precision @ IoU 0.5')
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}', ha='center', va='bottom')
        
        # mAP@0.5:0.95
        ax = axes[0, 1]
        bars = ax.bar(model_names, metrics_data['mAP@0.5:0.95'], color=colors[:len(model_names)])
        ax.set_ylabel('mAP@0.5:0.95')
        ax.set_title('Mean Average Precision @ IoU 0.5:0.95')
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}', ha='center', va='bottom')
        
        # Precision
        ax = axes[1, 0]
        bars = ax.bar(model_names, metrics_data['Precision'], color=colors[:len(model_names)])
        ax.set_ylabel('Precision')
        ax.set_title('Precision')
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom')
        
        # Recall
        ax = axes[1, 1]
        bars = ax.bar(model_names, metrics_data['Recall'], color=colors[:len(model_names)])
        ax.set_ylabel('Recall')
        ax.set_title('Recall')
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = Path('training_results') / 'training_summary.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f" Summary plot saved to: {plot_path}")
        plt.close()


def main():
    """Main training entry point"""
    parser = argparse.ArgumentParser(
        description='Train Baby Hazard Detection Models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all models with 100 epochs
  python training.py --all --epochs 100
  
  # Train specific model
  python training.py --model child --epochs 50
  
  # Train with custom batch size
  python training.py --all --epochs 100 --batch 32
  
  # Export trained model
  python training.py --export models/child_detector.pt --format onnx
        """
    )
    
    parser.add_argument('--all', action='store_true', help='Train all models (child, fire, pool)')
    parser.add_argument('--model', type=str, choices=['child', 'fire', 'pool'], help='Train specific model')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs (default: 100)')
    parser.add_argument('--batch', type=int, default=16, help='Batch size (default: 16)')
    parser.add_argument('--data-dir', type=str, default='data', help='Base data directory (default: data)')
    parser.add_argument('--model-size', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'], help='YOLO model size: n(ano), s(mall), m(edium), l(arge), x(large)')
    parser.add_argument('--validate', type=str, help='Validate model at given path')
    parser.add_argument('--export', type=str, help='Export model at given path')
    parser.add_argument('--format', type=str, default='onnx', choices=['onnx', 'tflite', 'tensorrt', 'coreml'], help='Export format (default: onnx)')
    
    args = parser.parse_args()
    
    # Validation mode
    if args.validate:
        trainer = YOLOTrainer(f'yolov8{args.model_size}.pt')
        # Determine data yaml from model name
        model_name = Path(args.validate).stem
        if 'child' in model_name:
            data_yaml = 'data/child_detection/data.yaml'
        elif 'fire' in model_name:
            data_yaml = 'data/fire_detection/data.yaml'
        else:
            data_yaml = 'data/pool_detection/data.yaml'
        
        trainer.validate_model(args.validate, data_yaml)
        return
    
    # Export mode
    if args.export:
        trainer = YOLOTrainer()
        trainer.export_model(args.export, args.format)
        return
    
    # Training mode
    if args.all:
        # Train all models
        orchestrator = TrainingOrchestrator()
        orchestrator.train_all_models(epochs=args.epochs, base_data_dir=args.data_dir)
    
    elif args.model:
        # Train specific model
        trainer = YOLOTrainer(f'yolov8{args.model_size}.pt')
        
        model_configs = {
            'child': ('child_detection', 'child_detector'),
            'fire': ('fire_detection', 'fire_detector'),
            'pool': ('pool_detection', 'pool_detector')
        }
        
        data_subdir, output_name = model_configs[args.model]
        data_dir = Path(args.data_dir) / data_subdir
        
        if not data_dir.exists():
            print(f" Data directory not found: {data_dir}")
            print(f"Please run data_preparation.py first")
            return
        
        # Create dataset YAML
        data_yaml = trainer.create_dataset_yaml(args.model, data_dir)
        
        # Train
        trainer.train_model(data_yaml, output_name, epochs=args.epochs, batch=args.batch)
        
        # Validate
        model_path = Path('models') / f"{output_name}.pt"
        if model_path.exists():
            trainer.validate_model(model_path, data_yaml)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()