import os
import torch
from torchvision.transforms import v2
from torchvision.io import read_image, write_png
from tqdm import tqdm

# Set up the augmentation transforms
augmentation_transforms = v2.Compose([
    # transforms.RandomResizedCrop(size=(224, 224), scale=(0.5, 1.0)),
    v2.ColorJitter(hue=0.015, saturation=0.3, brightness=0.3, contrast=0.3),
    v2.RandomAffine(degrees=5,translate=(0.05, 0.05),scale=(0.9, 1.1)),
    v2.RandomZoomOut(side_range=(1.0, 1.3)),
    v2.AutoAugment(v2.AutoAugmentPolicy.IMAGENET),
])

def augment_and_save_images(source_dir, target_dir):
    # Create the target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    # Iterate through all class directories
    for class_name in os.listdir(source_dir):
        class_dir = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        # Create corresponding class directory in the target folder
        target_class_dir = os.path.join(target_dir, class_name)
        os.makedirs(target_class_dir, exist_ok=True)

        # Process all images in the class directory
        for img_name in tqdm(os.listdir(class_dir), desc=f"Processing {class_name}"):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            # Load the image
            img_path = os.path.join(class_dir, img_name)
            img = read_image(img_path)

            # Save the original image
            original_target_path = os.path.join(target_class_dir, f"original_{img_name}")
            write_png(img, original_target_path)

            # Apply augmentation and save
            aug_img = augmentation_transforms(img)
            aug_target_path = os.path.join(target_class_dir, f"augmented_{img_name}")
            write_png(aug_img, aug_target_path)

# Set the paths
source_dir = "./datasets/subset_tiny/train"
target_dir = "./datasets/subset_tiny_torch_aug_best/train"

# Run the augmentation process
augment_and_save_images(source_dir, target_dir)

print("Dataset augmentation completed!")