# Script to fine-tune the LayoutLMv3 NER Model
# Data input: split_data/layoutlm_data
import os
import torch
from datasets import load_dataset, DatasetDict
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
# The order matters! O (Outside) must be index 0.
LABEL_LIST = ["O", "B-TOTAL", "I-TOTAL", "B-DATE", "I-DATE", "B-VENDOR", "I-VENDOR", "B-ITEM", "I-ITEM"]
id2label = {i: label for i, label in enumerate(LABEL_LIST)}
label2id = {label: i for i, label in enumerate(LABEL_LIST)}

# 1. Load Processor
processor = LayoutLMv3Processor.from_pretrained(MODEL_NAME, apply_ocr=False)

def preprocess_data(examples):
    """Tokenizes the text and aligns the labels with the tokens for LayoutLMv3."""
    
    # Load images based on their paths in the dataset
    images = [Image.open(img_path).convert("RGB") for img_path in examples['image_path']]
    
    # Process the data using the LayoutLMv3 processor
    encoded_inputs = processor(
        images, examples['words'], boxes=examples['bbox'], word_labels=examples['ner_tags'],
        truncation=True, padding="max_length", max_length=512
    )

    # Ensure the 'labels' key is correctly present
    encoded_inputs['labels'] = encoded_inputs.pop('labels')
    
    return encoded_inputs

def train_layoutlm_refiner():
    """Fine-tunes the LayoutLMv3 model for document NER."""
    print("Starting LayoutLMv3 Model Training...")
    
    # 1. Load Dataset from JSON files
    try:
        # Assumes JSON files contain a list of objects with 'image_path', 'words', 'bbox', and 'ner_tags'
        dataset = load_dataset('json', data_files=DATA_FILES)
    except Exception as e:
        print(f"FATAL: Error loading dataset. Ensure JSON files are correctly formatted. {e}")
        return

    # 2. Apply Preprocessing
    tokenized_dataset = dataset.map(preprocess_data, batched=True, remove_columns=['words', 'bbox', 'ner_tags', 'image_path'])

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
    train_layoutlm_refiner()