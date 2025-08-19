import os
import shutil
import random

# Paths
RAW_DATASET_DIR = "raw_dataset"
DATASET_DIR = "dataset"

CATEGORIES = ["with_mask", "without_mask"]
SPLIT_RATIO = 0.8  # 80% train, 20% test

for category in CATEGORIES:
    src_dir = os.path.join(RAW_DATASET_DIR, category)
    images = os.listdir(src_dir)
    random.shuffle(images)

    split_index = int(len(images) * SPLIT_RATIO)
    train_images = images[:split_index]
    test_images = images[split_index:]

    # Train and test destination
    train_dest = os.path.join(DATASET_DIR, "train", category)
    test_dest = os.path.join(DATASET_DIR, "test", category)

    os.makedirs(train_dest, exist_ok=True)
    os.makedirs(test_dest, exist_ok=True)

    # Move/copy train images
    for img in train_images:
        shutil.copy(os.path.join(src_dir, img), os.path.join(train_dest, img))

    # Move/copy test images
    for img in test_images:
        shutil.copy(os.path.join(src_dir, img), os.path.join(test_dest, img))

print("✅ Dataset split into train and test successfully!")
