import os
import shutil

def create_labels(image_dir, label_dir):
    os.makedirs(label_dir, exist_ok=True)
    
    # Loop through each image file
    for image_filename in os.listdir(image_dir):
        if image_filename.endswith('.jpg'):
            # Extract cow ID from the filename
            cow_id = int(image_filename.split('_')[-1].split('.')[0]) - 1  # Assuming class index starts from 0
            
            # Create corresponding label filename
            label_filename = os.path.splitext(image_filename)[0] + '.txt'
            
            # Write the label file
            label_filepath = os.path.join(label_dir, label_filename)
            with open(label_filepath, 'w') as label_file:
                label_file.write(f"{cow_id}\n")

# Directories for train, val, test
image_base_dir = './datasets/cow-labelled-images/images'
label_base_dir = './datasets/cow-labelled-images/labels'

# Create labels for each split
create_labels(os.path.join(image_base_dir, 'train'), os.path.join(label_base_dir, 'train'))
create_labels(os.path.join(image_base_dir, 'val'), os.path.join(label_base_dir, 'val'))
create_labels(os.path.join(image_base_dir, 'test'), os.path.join(label_base_dir, 'test'))
