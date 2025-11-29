# Script to fine-tune the DistilBERT Classifier
# Data input: split_data/classification_data
import os
import pandas as pd
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
    """Tokenizes the text column and converts labels to IDs."""
    tokenized = tokenizer(examples["text"], truncation=True, padding="max_length")
    # Convert string label (e.g., "RECEIPT") to integer ID (e.g., 0)
    tokenized["labels"] = [label2id[l] for l in examples["label"]] 
    return tokenized

def train_distilbert_classifier():
    """Fine-tunes the DistilBERT model for document type classification."""
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
    train_distilbert_classifier()