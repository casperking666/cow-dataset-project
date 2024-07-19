import os
import shutil

# Paths
source_dir = '/user/work/yf20630/cow-dataset-project/datasets/cow_crops/images'
destination_dir = '/user/work/yf20630/cow-dataset-project/datasets/cow_cls'

# Classes
classes = [f'class{i}' for i in range(1, 9)]

# Creating directories
for split in ['train', 'val', 'test']:
    for cls in classes:
        os.makedirs(os.path.join(destination_dir, split, cls), exist_ok=True)

# Move images to corresponding class directories
for split in ['train', 'val', 'test']:
    split_dir = os.path.join(source_dir, split)
    for img_name in os.listdir(split_dir):
        cow_id = img_name.split('_')[-1].split('.')[0]  # Extract cow ID from filename
        cls_dir = os.path.join(destination_dir, split, f'class{cow_id}')
        shutil.copy(os.path.join(split_dir, img_name), os.path.join(cls_dir, img_name))
