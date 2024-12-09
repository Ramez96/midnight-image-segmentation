import os
import random
import shutil

# Define paths for images and labels
image_dir = '../dataset/images'
label_dir = './dataset/regions'

# Define output directories for train, validation, and test
output_image_dirs = {
    'train': '../dataset/train/images',
    'val': '../dataset/val/images',
    'test': '../dataset/test/images'
}

output_label_dirs = {
    'train': '../dataset/train/labels',
    'val': '../dataset/val/labels',
    'test': '../dataset/test/labels'
}

# Ensure output directories exist
for dir_path in output_image_dirs.values():
    os.makedirs(dir_path, exist_ok=True)

for dir_path in output_label_dirs.values():
    os.makedirs(dir_path, exist_ok=True)

# Get list of all image files in the source directory
image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

# Shuffle the image files list
random.shuffle(image_files)

# Split the files into 70%, 15%, and 15% for training, validation, and testing
total_images = len(image_files)
train_split = int(total_images * 0.7)
val_split = int(total_images * 0.85)  # End of validation split (85% total, so 70%+15%)

train_files = image_files[:train_split]
val_files = image_files[train_split:val_split]
test_files = image_files[val_split:]

# Move images and corresponding labels to the respective directories
for file_list, split in zip([train_files, val_files, test_files], ['train', 'val', 'test']):
    for file in file_list:
        # Define paths for images and labels
        img_src_path = os.path.join(image_dir, file)
        label_src_path = os.path.join(label_dir, file.replace('.jpg', '.regions.txt'))  # Adjust label extension

        if os.path.exists(label_src_path):
            img_dest_path = os.path.join(output_image_dirs[split], file)
            label_dest_path = os.path.join(output_label_dirs[split], file.replace('.jpg', '.regions.txt'))  # Adjust label extension
            
            # Move image and label to respective directories
            shutil.move(img_src_path, img_dest_path)
            shutil.move(label_src_path, label_dest_path)
        else:
            print(f"Label for {file} not found. Skipping this file.")

print(f"Successfully split the dataset into 3 sets:")
print(f"- {len(train_files)} images for training")
print(f"- {len(val_files)} images for validation")
print(f"- {len(test_files)} images for testing")
