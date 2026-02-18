#!/usr/bin/env python3
"""
Complete Setup Script for Baby Hazard Monitoring System
This script organizes all downloaded files into the correct structure
"""

import os
import shutil
from pathlib import Path
import sys

def print_header(text):
    print("\n" + "="*70)
    print(text.center(70))
    print("="*70 + "\n")

def create_directory_structure():
    """Create all necessary directories"""
    print_header("CREATING DIRECTORY STRUCTURE")
    
    directories = [
        'config',
        'models',
        'data',
        'data/child_detection',
        'data/fire_detection',
        'data/pool_detection',
        'logs',
        'logs/alerts',
        'logs/alerts/recordings',
        'outputs',
        'outputs/monitoring_sessions',
        'training_results',
        'test_results',
        'notebooks',
        'Documentation'
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {dir_path}/")
    
    # Create __init__.py in models (CRITICAL!)
    init_file = Path('models/__init__.py')
    init_file.touch()
    print(f"\n✓ Created: models/__init__.py (package marker)")

def organize_python_files():
    """Organize Python files into correct locations"""
    print_header("ORGANIZING PYTHON FILES")
    
    # Core scripts stay in root
    root_scripts = [
        'setup.py',
        'demo.py',
        'main_monitor.py',
        'data_preparation.py',
        'organize_datasets.py',
        'data_validation.py',
        'proper_training.py',
        'train_pipeline.py',
        'model_training.py',
        'inference.py',
        'evaluation.py',
        'visualization.py',
        'privacy_edge_processor.py',
        'adaptive_room_mapper.py',
        'privacy_edge_monitor.py'
    ]
    
    # Model module files go in models/
    model_files = [
        'multi_task_detector.py',
        'temporal_reasoning.py',
        'spatial_analysis.py',
        'risk_assessment.py',
        'alert_manager.py'
    ]
    
    print("Core scripts (should be in root):")
    for script in root_scripts:
        if Path(script).exists():
            print(f"  ✓ {script}")
        else:
            print(f"  ✗ {script} - NOT FOUND")
    
    print("\nModel modules (should be in models/):")
    for model_file in model_files:
        src = Path(model_file)
        dst = Path('models') / model_file
        
        if src.exists() and src != dst:
            shutil.copy(src, dst)
            print(f"  ✓ Moved {model_file} to models/")
        elif dst.exists():
            print(f"  ✓ {model_file} already in models/")
        else:
            print(f"  ✗ {model_file} - NOT FOUND")

def organize_documentation():
    """Organize documentation files"""
    print_header("ORGANIZING DOCUMENTATION")
    
    doc_files = [
        'README.md',
        'START_HERE.md',
        'PROJECT_SUMMARY.md',
        'QUICKSTART.md',
        'INSTALLATION_GUIDE.md',
        'TRAINING_GUIDE.md',
        'TRAINING_README.md',
        'QUICK_START_WITH_DATA.md',
        'PROJECT_STRUCTURE.md',
        'COMPLETE_FILE_LISTING.md',
        'PRIVACY_SYSTEM_README.md',
        'PATENT_DOCUMENTATION.md',
        'COMPLETE_FILE_STRUCTURE.md'
    ]
    
    # Keep main docs in root, optionally copy to Documentation/
    for doc in doc_files:
        if Path(doc).exists():
            print(f"  ✓ {doc}")
            # Also copy to Documentation folder
            shutil.copy(doc, Path('Documentation') / doc)
        else:
            print(f"  ✗ {doc} - NOT FOUND")

def organize_config():
    """Organize configuration files"""
    print_header("ORGANIZING CONFIGURATION")
    
    config_files = [
        'config.yaml'
    ]
    
    for config in config_files:
        src = Path(config)
        dst = Path('config') / config
        
        if src.exists() and src != dst:
            shutil.copy(src, dst)
            print(f"  ✓ Moved {config} to config/")
        elif dst.exists():
            print(f"  ✓ {config} already in config/")
        else:
            print(f"  ⚠ {config} - NOT FOUND (will create default)")
            create_default_config()

def create_default_config():
    """Create default config.yaml if missing"""
    config_content = """# Smart Baby Hazard Monitoring System - Configuration

# Video settings
video:
  resolution:
    width: 1280
    height: 720
  fps: 30

# Model settings
models:
  input_size: [640, 640]
  confidence_threshold: 0.45
  iou_threshold: 0.45

# Risk assessment
risk:
  weights:
    proximity: 0.4
    temporal: 0.3
    environmental: 0.3
  thresholds:
    low: 0.3
    medium: 0.6
    high: 0.8
    critical: 0.9

# Alert settings
alerts:
  cooldown_period: 30
  escalation_time: 120
  channels:
    - mobile_app
    - push_notification
    - email

# Hardware settings
hardware:
  device: cpu  # Change to 'cuda' if you have GPU
  batch_size: 16
  num_workers: 4

# Paths
paths:
  models: models
  data: data
  outputs: outputs
  logs: logs
"""
    
    with open('config/config.yaml', 'w') as f:
        f.write(config_content)
    print("  ✓ Created default config.yaml")

def check_requirements():
    """Check if requirements files exist"""
    print_header("CHECKING REQUIREMENTS")
    
    req_files = ['requirements.txt', 'requirements_privacy.txt']
    
    for req in req_files:
        if Path(req).exists():
            print(f"  ✓ {req}")
        else:
            print(f"  ⚠ {req} - NOT FOUND")
            if req == 'requirements.txt':
                create_requirements_file()

def create_requirements_file():
    """Create requirements.txt if missing"""
    requirements = """# Core Dependencies
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
opencv-python>=4.5.0
Pillow>=9.0.0
ultralytics>=8.0.0
PyYAML>=6.0
pandas>=1.3.0
scipy>=1.7.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
tqdm>=4.62.0
requests>=2.27.0
python-dateutil>=2.8.2
cryptography>=41.0.0
jupyter>=1.0.0
ipython>=8.0.0
"""
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements)
    print("  ✓ Created requirements.txt")

def generate_summary():
    """Generate setup summary"""
    print_header("SETUP SUMMARY")
    
    # Count files
    py_files = len(list(Path('.').glob('*.py')))
    md_files = len(list(Path('.').glob('*.md')))
    model_files = len(list(Path('models').glob('*.py')))
    
    print(f"Python scripts in root: {py_files}")
    print(f"Documentation files: {md_files}")
    print(f"Model modules: {model_files}")
    
    # Check critical files
    critical_files = [
        'proper_training.py',
        'organize_datasets.py',
        'privacy_edge_monitor.py',
        'models/__init__.py',
        'config/config.yaml',
        'requirements.txt'
    ]
    
    print("\nCritical files check:")
    all_present = True
    for file in critical_files:
        exists = Path(file).exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {file}")
        if not exists:
            all_present = False
    
    return all_present

def print_next_steps(all_present):
    """Print next steps"""
    print_header("NEXT STEPS")
    
    if all_present:
        print("✅ ALL CRITICAL FILES PRESENT!")
        print("\nYou can now:")
        print("\n1. Create virtual environment:")
        print("   python3 -m venv venv")
        print("   source venv/bin/activate  # On Linux/Mac")
        print("   # venv\\Scripts\\activate   # On Windows")
        print("\n2. Install dependencies:")
        print("   pip install -r requirements.txt")
        print("\n3. Organize your datasets:")
        print("   python organize_datasets.py .")
        print("\n4. Validate datasets:")
        print("   python data_validation.py data/child_detection --validate")
        print("\n5. Train models:")
        print("   python proper_training.py --all --epochs 100")
        print("\n6. Test privacy system:")
        print("   python privacy_edge_monitor.py --demo")
    else:
        print("⚠️ SOME CRITICAL FILES ARE MISSING!")
        print("\nPlease ensure all files are downloaded to this directory.")
        print("Missing files are marked with ✗ above.")

def main():
    print_header("BABY HAZARD MONITOR - SETUP SCRIPT")
    
    print("This script will organize all files into the correct structure.")
    print("Current directory:", Path.cwd())
    
    input("\nPress Enter to continue...")
    
    # Run setup steps
    create_directory_structure()
    organize_python_files()
    organize_documentation()
    organize_config()
    check_requirements()
    
    # Generate summary
    all_present = generate_summary()
    print_next_steps(all_present)
    
    print_header("SETUP COMPLETE!")

if __name__ == "__main__":
    main()