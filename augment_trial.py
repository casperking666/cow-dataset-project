import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt

def show_image(img, title):
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Load the image
image_path = "images_sample/pmfeed_4_3_16_frame_36409_cow_1.jpg"  # Replace with your image path
original_img = Image.open(image_path)

# Show original image
show_image(original_img, "Original Image")

# 1. Scaling
scale_transform = transforms.Resize((int(original_img.height * 0.5), int(original_img.width * 0.5)))
scaled_img = scale_transform(original_img)
show_image(scaled_img, "Scaled Image (50%)")

# 2. HSV changes
hsv_transform = transforms.ColorJitter(hue=0.2, saturation=0.3, brightness=0.3)
hsv_img = hsv_transform(original_img)
show_image(hsv_img, "HSV Adjusted Image")

# 3. Random Erasing
erasing_transform = transforms.RandomErasing(p=1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0)
erased_img = erasing_transform(TF.to_tensor(original_img))
show_image(TF.to_pil_image(erased_img), "Random Erasing")

# 4. Random Crop
crop_fraction = 0.8
crop_transform = transforms.RandomCrop((int(original_img.height * crop_fraction), 
                                        int(original_img.width * crop_fraction)))
cropped_img = crop_transform(original_img)
show_image(cropped_img, f"Random Crop ({crop_fraction:.0%})")