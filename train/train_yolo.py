# Script to fine-tune the YOLO Object Detector
# Data input: split_data/yolo_data
import os
import sys
import shutil

# NOTE: You need the 'ultralytics' package installed for this script.
# pip install ultralytics
try:
    from ultralytics import YOLO
except ImportError:
    print("FATAL: 'ultralytics' library not found. Please run 'pip install ultralytics'.")
    sys.exit(1)

# Define paths
TRAIN_ROOT = os.path.dirname(os.path.abspath(__file__))
YOLO_DATA_DIR = os.path.join(TRAIN_ROOT, 'split_data', 'yolo_data')
MODEL_OUTPUT_DIR = os.path.join(TRAIN_ROOT, '..', 'models', 'yolo_output')
YOLO_CONFIG_PATH = os.path.join(TRAIN_ROOT, 'yolo_config.yaml') # Create this file manually

def train_yolo_detector():
    """
    Fine-tunes a pre-trained YOLOv8 model for document object detection.
    """
    print("Starting YOLO Model Training...")
    
    if not os.path.exists(YOLO_DATA_DIR):
        print(f"FATAL: YOLO data directory not found: {YOLO_DATA_DIR}")
        print("Please run split_data.py and ensure data exists.")
        return

    # 1. Configuration Check
    if not os.path.exists(YOLO_CONFIG_PATH):
        print(f"FATAL: YOLO config file not found. Create '{YOLO_CONFIG_PATH}' first!")
        print("It should contain paths to your train/val image folders, nc (class count), and names (class labels).")
        return
        
    # 2. Load a pre-trained model (YOLOv8s is a good starting point)
    model = YOLO('yolov8s.pt')

    # 3. Start training
    results = model.train(
        data=YOLO_CONFIG_PATH,
        epochs=50,             # Number of epochs
        imgsz=640,             # Image size
        batch=16,              # Batch size
        name='docera_yolo_run',
        project=MODEL_OUTPUT_DIR
    )

    print("\n✅ YOLO Training Complete.")
    
    # 4. Copy the best weights for use in the main Docera pipeline
    final_weights_path = os.path.join(MODEL_OUTPUT_DIR, 'docera_yolo_run', 'weights', 'best.pt')
    
    if os.path.exists(final_weights_path):
        shutil.copy(final_weights_path, os.path.join(TRAIN_ROOT, '..', 'models', 'best.pt'))
        print("Copied best.pt to Docera/models/best.pt for pipeline use.")
    else:
        print("WARNING: Could not locate final trained weights.")

if __name__ == "__main__":
    train_yolo_detector()