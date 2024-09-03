import torch
import torchvision.transforms.v2 as transforms
from torchvision.io import read_image
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Load the image
img = Image.open("datasets/subset/test/class4/pmfeed_4_3_16_frame_6861_cow_4.jpg")

# Define transformations
transformations = [
    ("Original", transforms.Lambda(lambda x: x)),
    ("Resize", transforms.Resize(size=(int(img.height*0.5), int(img.width*0.5)))),
    ("ScaleJitter", transforms.RandomResizedCrop(size=img.size, scale=(0.8, 1.2))),
    ("RandomShortestSize", transforms.RandomResize(min_size=256, max_size=512)),
    ("RandomResize", transforms.RandomResize(min_size=256, max_size=512)),
    ("Pad", transforms.Pad(padding=50)),
    ("RandomZoomOut", transforms.RandomZoomOut(fill=0)),
    ("RandomRotation", transforms.RandomRotation(degrees=30)),
    ("RandomAffine", transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))),
    ("RandomPerspective", transforms.RandomPerspective(distortion_scale=0.5, p=1.0)),
    ("ElasticTransform", transforms.ElasticTransform(alpha=250.0, sigma=5.0)),
    ("ColorJitter", transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)),
    ("RandomChannelPermutation", transforms.Lambda(lambda x: x[torch.randperm(3)])),
    ("RandomPhotometricDistort", transforms.RandomPhotometricDistort()),
    ("Grayscale", transforms.Grayscale(3)),
    ("RGB", transforms.Lambda(lambda x: x)),  # No-op for RGB images
    ("RandomGrayscale", transforms.RandomGrayscale(p=0.5)),
    ("GaussianBlur", transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))),
    ("GaussianNoise", transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.1)),
    ("RandomInvert", transforms.RandomInvert(p=1.0)),
    ("RandomPosterize", transforms.RandomPosterize(bits=2)),
    ("RandomSolarize", transforms.RandomSolarize(threshold=128)),
    ("RandomAdjustSharpness", transforms.RandomAdjustSharpness(sharpness_factor=2)),
    ("RandomAutocontrast", transforms.RandomAutocontrast(p=1.0)),
    ("RandomEqualize", transforms.RandomEqualize(p=1.0))
]



# transformations = [
#     ("Resize", transforms.Resize(size=(int(img.height*0.5), int(img.width*0.5)))),
#     ("ScaleJitter", transforms.RandomResizedCrop(size=img.size, scale=(0.8, 1.2))),
#     ("RandomResize", transforms.RandomResize(min_size=256, max_size=512)),
#     ("RandomZoomOut", transforms.RandomZoomOut(fill=0)),
#     ("RandomRotation", transforms.RandomRotation(degrees=30)),
#     ("RandomAffine", transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))),
#     ("RandomPerspective", transforms.RandomPerspective(distortion_scale=0.5, p=1.0)),
#     ("ColorJitter", transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)),
#     ("RandomChannelPermutation", transforms.Lambda(lambda x: x[torch.randperm(3)])),
#     ("RandomPhotometricDistort", transforms.RandomPhotometricDistort()),
#     ("RandomGrayscale", transforms.RandomGrayscale(p=0.5)),
#     ("GaussianBlur", transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))),
#     ("GaussianNoise", transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.1)),
#     ("RandomPosterize", transforms.RandomPosterize(bits=2)),
#     ("RandomSolarize", transforms.RandomSolarize(threshold=128)),
#     ("RandomAdjustSharpness", transforms.RandomAdjustSharpness(sharpness_factor=2)),
#     ("RandomAutocontrast", transforms.RandomAutocontrast(p=1.0)),
#     ("RandomEqualize", transforms.RandomEqualize(p=1.0))
# ]


# Apply transformations and collect results
results = []
to_tensor = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True)
])

for name, transform in transformations:
    transformed_img = transform(to_tensor(img))
    results.append((name, transformed_img))

# Create a grid of images
rows, cols = 5, 5
fig, axs = plt.subplots(rows, cols, figsize=(30, 30))
# fig.suptitle("Data Augmentations on Cow Image", fontsize=42, y=0.98)  # Increased title font size

for i, (name, img_tensor) in enumerate(results):
    row, col = i // cols, i % cols
    ax = axs[row, col]
    # Convert to numpy and clip values
    img_np = img_tensor.permute(1, 2, 0).numpy()
    img_np = np.clip(img_np, 0, 1)
    ax.imshow(img_np)
    ax.set_title(name, pad=5, fontsize=29)  # Increased function name font size
    ax.axis('off')

# Remove spacing between subplots
plt.subplots_adjust(wspace=0, hspace=0.4)  # Increased vertical space for larger titles

# Adjust layout to prevent overlapping with the main title
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save the figure
plt.savefig('cow_augmentations.png', dpi=300, bbox_inches='tight')
plt.close(fig)

print("Image saved as 'cow_augmentations.png'")