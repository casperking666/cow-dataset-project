import os
import glob
import re
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CowCropsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = glob.glob(os.path.join(root_dir, '*.jpg'))
        self.pattern = re.compile(r'.*_(\d+)_cow_(\d+)\.jpg')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        # Extract the cow ID from the filename
        match = self.pattern.match(os.path.basename(img_path))
        if match:
            frame = int(match.group(1))
            cow_id = int(match.group(2))
        else:
            raise ValueError(f"Filename {img_path} does not match expected pattern.")
        
        if self.transform:
            image = self.transform(image)
        
        return image, cow_id - 1  # Subtract 1 to make cow_id range from 0 to 7

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),  # Convert images to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
])

# Create the dataset
dataset = CowCropsDataset(root_dir='/group/nwc-group/tony/pmfeed_4_3_16_hand_labelled', transform=transform)

# Create the dataloader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# Example usage
for images, labels in dataloader:
    print(images.shape)  # Should print torch.Size([32, 3, 224, 224])
    print(labels.shape)  # Should print torch.Size([32])
    break
