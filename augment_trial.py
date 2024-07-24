import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt

def apply_augmentations(image):
    # Original image
    augmented_images = [image]
    titles = ['Original']

    # 1. Scaling
    scale_transform = transforms.Resize((int(image.height * 0.5), int(image.width * 0.5)))
    augmented_images.append(scale_transform(image))
    titles.append('Scaled (50%)')

    # 2. HSV changes
    hsv_transform = transforms.ColorJitter(hue=0.2, saturation=0.3, brightness=0.3)
    augmented_images.append(hsv_transform(image))
    titles.append('HSV Adjusted')

    # 3. Random Erasing
    erasing_transform = transforms.RandomErasing(p=1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0)
    erased_img = erasing_transform(TF.to_tensor(image))
    augmented_images.append(TF.to_pil_image(erased_img))
    titles.append('Random Erasing')

    # 4. Random Crop
    crop_fraction = 0.8
    crop_transform = transforms.RandomCrop((int(image.height * crop_fraction), 
                                            int(image.width * crop_fraction)))
    augmented_images.append(crop_transform(image))
    titles.append(f'Random Crop ({crop_fraction:.0%})')

    return augmented_images, titles

# Load the image
image_path = "images_sample/pmfeed_4_3_16_frame_36409_cow_1.jpg"  # Replace with your image path
original_img = Image.open(image_path)

# Apply augmentations
augmented_images, titles = apply_augmentations(original_img)

# Create a grid of images
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
fig.suptitle("Image Augmentation Visualization", fontsize=16)

for ax, img, title in zip(axes, augmented_images, titles):
    ax.imshow(img)
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()

# Save the figure
output_path = "augmentation_visualization.png"  # You can change the path and filename
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Visualization saved to {output_path}")

# If you want to display the plot as well (optional, might not work on HPC without X11 forwarding)
# plt.show()