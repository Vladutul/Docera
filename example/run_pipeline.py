import os
import sys
import cv2

# --- Set up Python Path ---
# This ensures the script can find the 'docera' package in the parent directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the main pipeline class
try:
    from docera.pipeline import HybridPipeline
except ImportError as e:
    print(f"ERROR: Could not import Docera pipeline. Did you run the setup script correctly?")
    print(f"Details: {e}")
    sys.exit(1)

def run_example():
    """Initializes the Docera pipeline and runs it on a sample image."""
    
    # 1. Define Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Placeholder for the actual YOLO model weights (as defined in pipeline.py)
    # NOTE: This path is relative to the *Docera root*
    yolo_model_path = os.path.abspath(os.path.join(base_dir, '..', 'models', 'best.pt'))
    
    # Placeholder for the input image
    image_path = os.path.join(base_dir, 'sample_invoice.png')

    if not os.path.exists(image_path):
        print(f"FATAL: Sample image not found at {image_path}")
        print("Please place a test image named 'sample_invoice.png' in the 'example/' folder.")
        return

    # 2. Initialize the Pipeline
    print(f"Starting Docera Hybrid Pipeline...")
    try:
        # Pass the path to the YOLO model to the constructor
        pipeline = HybridPipeline(yolo_model_path=yolo_model_path)
    except FileNotFoundError as e:
        print(f"FATAL: Model initialization failed. Ensure your 'best.pt' is in the 'models/' directory.")
        print(e)
        return
    except Exception as e:
        print(f"FATAL: An unexpected error occurred during pipeline initialization: {e}")
        return

    # 3. Process the Document
    print(f"Processing image: {image_path}")
    fields, annotated_img = pipeline.process(image_path)

    # 4. Display Results
    print("\n" + "="*50)
    print("✅ Extraction Results (JSON-like Structure):")
    print("="*50)
    
    # Pretty print the extracted fields
    import json
    print(json.dumps(fields, indent=4))
    
    # Save the annotated image
    output_path = os.path.join(base_dir, 'annotated_output.jpg')
    cv2.imwrite(output_path, annotated_img)
    print(f"\nAnnotated image saved to: {output_path}")

    # Optional: Display the annotated image (requires a working OpenCV display environment)
    # cv2.imshow('Annotated Document', annotated_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    run_example()