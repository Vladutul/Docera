import os
import random
import shutil
import sys

# Define the root directory (where this script is executed)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# --- Training Code Templates ---
# These are the contents for the Python training scripts.

TRAIN_CODE_CONTENTS = {
    
    "train/split_data.py": """
import os
import random
import shutil

# Define the root directory for easy path construction
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
RAW_DOCS = os.path.join(BASE_DIR, 'raw_documents')
SPLIT_OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'split_data')

def create_yolo_splits(image_source_dir=RAW_DOCS, output_root=os.path.join(SPLIT_OUTPUT, 'yolo_data'), split_ratio=0.8):
    \"\"\"
    Splits the document images and their corresponding YOLO-format labels (TXT files)
    into train and validation folders using the 80/20 rule.
    
    ASSUMPTION: Every image file (e.g., 'doc1.jpg') has a corresponding label file ('doc1.txt')
    in the same raw_documents directory.
    \"\"\"
    
    # 1. Prepare directories
    train_img_dir = os.path.join(output_root, 'train', 'images')
    val_img_dir = os.path.join(output_root, 'val', 'images')
    train_lbl_dir = os.path.join(output_root, 'train', 'labels')
    val_lbl_dir = os.path.join(output_root, 'val', 'labels')
    
    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        os.makedirs(d, exist_ok=True)

    # 2. Get all image file names
    image_files = [f for f in os.listdir(image_source_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    random.shuffle(image_files) # Crucial: Randomize the split!

    # 3. Calculate the split index
    train_count = int(len(image_files) * split_ratio)
    
    train_files = image_files[:train_count]
    val_files = image_files[train_count:]
    
    print(f"Total files: {len(image_files)}. Train: {len(train_files)}, Val: {len(val_files)}")

    # 4. Copy files
    for split_type, files, img_out, lbl_out in [('Train', train_files, train_img_dir, train_lbl_dir), 
                                                ('Validation', val_files, val_img_dir, val_lbl_dir)]:
        for filename in files:
            name, ext = os.path.splitext(filename)
            label_filename = name + '.txt'
            
            # Copy image
            shutil.copy(os.path.join(image_source_dir, filename), os.path.join(img_out, filename))
            
            # Copy label (Assumes labels are in the same folder, adjust if needed)
            label_source_path = os.path.join(image_source_dir, label_filename)
            if os.path.exists(label_source_path):
                shutil.copy(label_source_path, os.path.join(lbl_out, label_filename))
            else:
                print(f"WARNING: Missing label file for {filename}")

    print("YOLO data split and copy complete.")

if __name__ == "__main__":
    print("Running data splitting scripts...")
    # NOTE: You must populate train/data/raw_documents with images AND labels first.
    create_yolo_splits()
    # Add calls for LayoutLM and Classification data splitting here.
""",

    "train/train_yolo.py": """
import os
import sys

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
    \"\"\"
    Fine-tunes a pre-trained YOLOv8 model for document object detection.
    \"\"\"
    print("Starting YOLO Model Training...")
    
    if not os.path.exists(YOLO_DATA_DIR):
        print(f"FATAL: YOLO data directory not found: {YOLO_DATA_DIR}")
        print("Please run split_data.py and ensure data exists.")
        return

    # 1. Create a configuration file (YOLO_CONFIG_PATH) that defines:
    #    - train: {YOLO_DATA_DIR}/train/images
    #    - val: {YOLO_DATA_DIR}/val/images
    #    - nc: Number of classes (e.g., 5)
    #    - names: List of class names (e.g., ['total', 'date', 'vendor', ...])
    if not os.path.exists(YOLO_CONFIG_PATH):
        print(f"FATAL: YOLO config file not found. Create '{YOLO_CONFIG_PATH}' first!")
        return
        
    # 2. Load a pre-trained model (e.g., YOLOv8s for speed)
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
    # The best weights are typically saved to {MODEL_OUTPUT_DIR}/docera_yolo_run/weights/best.pt
    # You should copy this 'best.pt' file to your main 'models/best.pt'
    final_weights_path = os.path.join(MODEL_OUTPUT_DIR, 'docera_yolo_run', 'weights', 'best.pt')
    
    if os.path.exists(final_weights_path):
        print(f"Final model weights saved to: {final_weights_path}")
        # The line below is crucial for integrating into the main Docera pipeline
        shutil.copy(final_weights_path, os.path.join(TRAIN_ROOT, '..', 'models', 'best.pt'))
        print("Copied best.pt to Docera/models/best.pt for pipeline use.")
    else:
        print("WARNING: Could not locate final trained weights.")

if __name__ == "__main__":
    train_yolo_detector()
""",

    "train/train_layoutlm.py": """
import os
import json
import torch
from datasets import load_dataset, Dataset, Features, Sequence, Value, Array2D, Image as ImageFeature
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor, Trainer, TrainingArguments
from PIL import Image

# Define paths
TRAIN_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_FILES = {
    "train": os.path.join(TRAIN_ROOT, 'split_data', 'layoutlm_data', 'train.json'),
    "validation": os.path.join(TRAIN_ROOT, 'split_data', 'layoutlm_data', 'val.json')
}
MODEL_NAME = "microsoft/layoutlmv3-base"
MODEL_OUTPUT_DIR = os.path.join(TRAIN_ROOT, '..', 'models', 'layoutlm_finetuned')

# Placeholder for actual NER labels (Must match your annotation scheme)
# O (Outside), B- (Begin), I- (Inside)
LABEL_LIST = ["O", "B-TOTAL", "I-TOTAL", "B-DATE", "I-DATE", "B-VENDOR", "I-VENDOR"]
id2label = {i: label for i, label in enumerate(LABEL_LIST)}
label2id = {label: i for i, label in enumerate(LABEL_LIST)}

# 1. Load Processor
processor = LayoutLMv3Processor.from_pretrained(MODEL_NAME, apply_ocr=False)

def preprocess_data(examples):
    \"\"\"Tokenizes the text and aligns the labels with the tokens.\"\"\"
    # NOTE: This function requires that your dataset already contains:
    # 1. 'image' (as a file path or PIL object)
    # 2. 'words' (list of OCR tokens)
    # 3. 'bbox' (list of normalized 0-1000 bounding boxes)
    # 4. 'ner_tags' (list of numerical label IDs)
    
    images = [Image.open(img_path).convert("RGB") for img_path in examples['image_path']]
    
    # Process the data using the LayoutLMv3 processor
    encoded_inputs = processor(
        images, examples['words'], boxes=examples['bbox'], word_labels=examples['ner_tags'],
        truncation=True, padding="max_length", max_length=512
    )

    # Convert the actual label IDs to the appropriate tensor format
    encoded_inputs['labels'] = encoded_inputs.pop('labels')
    
    return encoded_inputs

def train_layoutlm_refiner():
    \"\"\"Fine-tunes the LayoutLMv3 model for document NER.\"\"\"
    print("Starting LayoutLMv3 Model Training...")
    
    # 1. Load Dataset from JSON files
    try:
        # Requires JSON files in the format: [{"image_path": "...", "words": [...], "bbox": [...], "ner_tags": [...]}]
        dataset = load_dataset('json', data_files=DATA_FILES)
    except Exception as e:
        print(f"FATAL: Error loading dataset. Ensure JSON files are correctly formatted and exist. {e}")
        return

    # 2. Apply Preprocessing
    tokenized_dataset = dataset.map(preprocess_data, batched=True, remove_columns=['words', 'bbox', 'ner_tags'])

    # 3. Load Model
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=len(LABEL_LIST), 
        id2label=id2label, 
        label2id=label2id
    )

    # 4. Define Training Arguments
    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        per_device_train_batch_size=4,
        num_train_epochs=10,
        evaluation_strategy="epoch",
        logging_steps=100,
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    # 5. Initialize and Run Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=processor.tokenizer,
    )

    trainer.train()

    # 6. Save Final Model
    trainer.save_model(MODEL_OUTPUT_DIR)
    processor.save_pretrained(MODEL_OUTPUT_DIR) 
    print(f"\n✅ LayoutLMv3 Training Complete. Model saved to {MODEL_OUTPUT_DIR}")

if __name__ == "__main__":
    # NOTE: You must install the 'datasets' library: pip install datasets
    train_layoutlm_refiner()
""",

    "train/train_classifier.py": """
import os
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import DistilBertForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

# Define paths
TRAIN_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_FILES = {
    "train": os.path.join(TRAIN_ROOT, 'split_data', 'classification_data', 'train.csv'),
    "validation": os.path.join(TRAIN_ROOT, 'split_data', 'classification_data', 'val.csv')
}
MODEL_NAME = "distilbert-base-uncased"
MODEL_OUTPUT_DIR = os.path.join(TRAIN_ROOT, '..', 'models', 'classifier_finetuned')

# Placeholder for actual Classification labels (e.g., Document Types)
LABEL_LIST = ["RECEIPT", "INVOICE", "CONTRACT", "OTHER"]
label2id = {label: i for i, label in enumerate(LABEL_LIST)}
id2label = {i: label for i, label in enumerate(LABEL_LIST)}

# 1. Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    \"\"\"Tokenizes the text column and converts labels to IDs.\"\"\"
    tokenized = tokenizer(examples["text"], truncation=True, padding="max_length")
    # Convert string label (e.g., "RECEIPT") to integer ID (e.g., 0)
    tokenized["labels"] = [label2id[l] for l in examples["label"]] 
    return tokenized

def train_distilbert_classifier():
    \"\"\"Fine-tunes the DistilBERT model for document type classification.\"\"\"
    print("Starting DistilBERT Classifier Training...")

    # 1. Load Dataset from CSV files
    try:
        train_df = pd.read_csv(DATA_FILES["train"])
        val_df = pd.read_csv(DATA_FILES["validation"])
        
        # Convert Pandas DataFrames to Hugging Face Dataset objects
        dataset = DatasetDict({
            "train": Dataset.from_pandas(train_df),
            "validation": Dataset.from_pandas(val_df)
        })
    except Exception as e:
        print(f"FATAL: Error loading CSV dataset. Ensure CSVs exist and have 'text' and 'label' columns. {e}")
        return

    # 2. Apply Tokenization
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # 3. Load Model
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=len(LABEL_LIST), 
        id2label=id2label, 
        label2id=label2id
    )

    # 4. Define Training Arguments
    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        evaluation_strategy="epoch",
        logging_steps=50,
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    # 5. Initialize and Run Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
    )

    trainer.train()

    # 6. Save Final Model
    trainer.save_model(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
    print(f"\n✅ DistilBERT Training Complete. Model saved to {MODEL_OUTPUT_DIR}")

if __name__ == "__main__":
    # NOTE: You must install the 'datasets' and 'pandas' libraries: pip install datasets pandas
    train_distilbert_classifier()
""",
    
    # Helper files for the folder structure
    "train/__init__.py": "",
    "train/data/raw_documents/.gitkeep": "# Placeholder for raw document images\n",
    "train/data/annotations/.gitkeep": "# Placeholder for annotation files\n",
    "train/split_data/yolo_data/train/.gitkeep": "# 80% YOLO training data\n",
    "train/split_data/yolo_data/val/.gitkeep": "# 20% YOLO validation data\n",
    "train/split_data/layoutlm_data/train.json": "[]", 
    "train/split_data/layoutlm_data/val.json": "[]",   
    "train/split_data/classification_data/train.csv": "text,label\n", 
    "train/split_data/classification_data/val.csv": "text,label\n",
}

# --- Execution Logic ---

def populate_train_code():
    """Creates all directories and writes training code content to files."""
    print("🚀 Starting creation of training scripts and folder structure...")
    
    # 1. Create all sub-directories and files
    for path, content in TRAIN_CODE_CONTENTS.items():
        full_path = os.path.join(PROJECT_ROOT, path)
        directory = os.path.dirname(full_path)

        # Create the parent directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory.replace(PROJECT_ROOT + os.sep, '')}")

        # Create the file and write content
        try:
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content.strip())
        except Exception as e:
            print(f"!!! ERROR writing file {path}: {e}")

    print("\n✅ Training scripts and folder structure successfully populated!")
    print("\n--- NEXT STEPS ---")
    print("1. Place your raw data into the 'train/data/' folder.")
    print("2. Run 'python train/split_data.py' to perform the 80/20 split.")
    print("3. Install necessary ML libraries (ultralytics, datasets, pandas).")
    print("4. Run your training scripts (e.g., 'python train/train_yolo.py').")


if __name__ == "__main__":
    # Ensure the old populate script is not run
    if "populate_train_folder.py" in sys.argv[0]:
        print("Please rename this script to 'populate_train_code.py' before running.")
        sys.exit(1)
        
    populate_train_code()