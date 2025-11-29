# Script to perform the 80/20 train/validation split
import os
import random
import shutil

def split_files(source_dir, train_dir, val_dir, split_ratio=0.8):
    """Splits files from source_dir into training and validation directories."""
    
    # 1. Get all file names
    all_files = [f for f in os.listdir(source_dir) if f.endswith(('.png', '.jpg', '.json'))]
    random.shuffle(all_files) # Crucial: Shuffle the list!

    # 2. Calculate the split index
    train_count = int(len(all_files) * split_ratio)
    
    # 3. Split the list
    train_files = all_files[:train_count]
    val_files = all_files[train_count:]
    
    # 4. Copy files to the new directories
    for filename in train_files:
        shutil.copy(os.path.join(source_dir, filename), os.path.join(train_dir, filename))
    for filename in val_files:
        shutil.copy(os.path.join(source_dir, filename), os.path.join(val_dir, filename))
        
    print(f"Split complete. Train: {len(train_files)} files, Val: {len(val_files)} files.")

# --- Usage Example ---
# raw_image_dir = "train/data/raw_documents"
# yolo_train_output = "train/split_data/yolo_data/train"
# yolo_val_output = "train/split_data/yolo_data/val"
# split_files(raw_image_dir, yolo_train_output, yolo_val_output)