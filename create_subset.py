import os
import random
import shutil
from pathlib import Path
from tqdm import tqdm

# Define paths
original_dataset_path = Path("/group/nwc-group/tony/pmfeed_4_3_16_hand_labelled")
subset_path = Path("/user/work/yf20630/cow-dataset-project/datasets/subset_tiny")
train_subset_path = subset_path / "train"
test_subset_path = subset_path / "test"

# Create new folder structure
for i in range(1, 9):
    (train_subset_path / f"class{i}").mkdir(parents=True, exist_ok=True)
    (test_subset_path / f"class{i}").mkdir(parents=True, exist_ok=True)

# Define the percentage for sampling
train_percentage = 0.0004
test_percentage = 0.0004

# Split images by cow ID on-the-fly
def split_and_sample_images_by_cow(original_dataset_path, train_dest, test_dest, train_percentage, test_percentage):
    cow_dict = {i: [] for i in range(1, 9)}

    # Collect all images and split by cow ID
    all_images = list(original_dataset_path.glob("*.jpg"))
    for image in tqdm(all_images, desc="Processing images"):
        cow_id = int(image.stem.split('_')[-1])
        cow_dict[cow_id].append(image)
    
    # Sample and copy images for each cow ID
    for cow_id, images in cow_dict.items():
        random.shuffle(images)
        train_sample_size = int(len(images) * train_percentage)
        test_sample_size = int(len(images) * test_percentage)
        
        # Randomly sample train and test images
        train_images = random.sample(images, train_sample_size)
        remaining_images = [img for img in images if img not in train_images]
        test_images = random.sample(remaining_images, test_sample_size)
        
        for image in train_images:
            shutil.copy(image, train_dest / f"class{cow_id}" / image.name)
        for image in test_images:
            shutil.copy(image, test_dest / f"class{cow_id}" / image.name)

split_and_sample_images_by_cow(original_dataset_path, train_subset_path, test_subset_path, train_percentage, test_percentage)

print("Sampling and copying completed.")
