import os
import shutil

# Define source directories
source_train_images = 'Pedestrian_dataset/train_images'
source_train_masks = 'Pedestrian_dataset/train_masks'

# Define destination directories
dest_train_images = 'pede/train_images'
dest_train_masks = 'pede/train_masks'

# Ensure destination directories exist
os.makedirs(dest_train_images, exist_ok=True)
os.makedirs(dest_train_masks, exist_ok=True)

# Define the range of images to search for and copy
start = 0
end = 300

# Loop through the specified range
for i in range(start, end + 1):
    # Construct the filenames
    image_filename = f'image_{i}.png'
    
    # Define full paths for the train images
    source_image_path = os.path.join(source_train_images, image_filename)
    dest_image_path = os.path.join(dest_train_images, image_filename)
    
    # Define full paths for the train masks
    source_mask_path = os.path.join(source_train_masks, image_filename)
    dest_mask_path = os.path.join(dest_train_masks, image_filename)
    
    # Copy the image file if it exists
    if os.path.exists(source_image_path):
        shutil.copy2(source_image_path, dest_image_path)
        print(f'Copied {source_image_path} to {dest_image_path}')
    else:
        print(f'{source_image_path} does not exist, skipping.')

    # Copy the mask file if it exists
    if os.path.exists(source_mask_path):
        shutil.copy2(source_mask_path, dest_mask_path)
        print(f'Copied {source_mask_path} to {dest_mask_path}')
    else:
        print(f'{source_mask_path} does not exist, skipping.')

print("Finished copying images and masks.")
