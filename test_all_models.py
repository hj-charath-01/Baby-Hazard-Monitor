from pathlib import Path
from ultralytics import YOLO

models = {
    'child': 'models/child_detector.pt',
    'fire': 'models/fire_detector.pt',
    'pool': 'models/pool_detector.pt'
}

test_images = {
    'child': 'data/child_detection/test/images',
    'fire': 'data/fire_detection/test/images',
    'pool': 'data/pool_detection/test/images'
}

print("\n" + "="*70)
print("TESTING ALL MODELS")
print("="*70 + "\n")

for name, model_path in models.items():
    print(f"\nTesting {name.upper()} detector...")
    
    # Load model
    model = YOLO(model_path)
    
    # Get first test image
    test_dir = Path(test_images[name])
    test_imgs = list(test_dir.glob('*.jpg'))[:5]  # First 5 images
    
    if not test_imgs:
        test_imgs = list(test_dir.glob('*.png'))[:5]
    
    if test_imgs:
        # Run inference
        results = model.predict(
            test_imgs[0],
            save=True,
            project='test_results',
            name=name
        )
        
        print(f" {name} detector: {len(results[0].boxes)} detections")
        print(f"  Results saved to: test_results/{name}/")
    else:
        print(f"✗ No test images found for {name}")

print("\n" + "="*70)
print(" ALL MODELS TESTED")
print("="*70)
print("\nView results in: test_results/")
