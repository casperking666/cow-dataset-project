import os
import glob
import shutil
from sklearn.model_selection import train_test_split

def split_dataset(image_dir, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    # Get all image paths
    image_paths = glob.glob(os.path.join(image_dir, '*.jpg'))
    
    # Debug: Print the number of images found
    print(f"Found {len(image_paths)} images in {image_dir}")
    
    if not image_paths:
        raise ValueError(f"No images found in directory {image_dir}")

    # Split the dataset
    train_paths, test_paths = train_test_split(image_paths, test_size=test_ratio, random_state=42)
    val_ratio_adjusted = val_ratio / (1 - test_ratio)  # Adjust val_ratio for remaining data
    train_paths, val_paths = train_test_split(train_paths, test_size=val_ratio_adjusted, random_state=42)

    return train_paths, val_paths, test_paths

def copy_files(file_paths, destination_folder):
    os.makedirs(destination_folder, exist_ok=True)
    for file_path in file_paths:
        shutil.copy(file_path, destination_folder)

image_dir = '/group/nwc-group/tony/pmfeed_4_3_16_hand_labelled'
output_dir = './images'

train_dir = os.path.join(output_dir, 'train')
val_dir = os.path.join(output_dir, 'val')
test_dir = os.path.join(output_dir, 'test')

train_paths, val_paths, test_paths = split_dataset(image_dir)

copy_files(train_paths, train_dir)
copy_files(val_paths, val_dir)
copy_files(test_paths, test_dir)
