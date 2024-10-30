import os
import random
from PIL import Image

# Define source folders for images and masks
src_train_images = './road_lines_dataset/train/images'
src_train_masks = './road_lines_dataset/train/masks'
src_valid_images = './road_lines_dataset/valid/images'
src_valid_masks = './road_lines_dataset/valid/masks'

# Define destination folders
output_train_images = "./road_lines_dataset/train_images"
output_train_masks = "./road_lines_dataset/train_masks"
output_val_images = "./road_lines_dataset/val_images"
output_val_masks = "./road_lines_dataset/val_masks"

# Create output directories if they don't exist
os.makedirs(output_train_images, exist_ok=True)
os.makedirs(output_train_masks, exist_ok=True)
os.makedirs(output_val_images, exist_ok=True)
os.makedirs(output_val_masks, exist_ok=True)

def save_images_and_masks(src_images_folder, src_masks_folder, dest_images_folder, dest_masks_folder, val_split=0.1):
    # Gather all image files (assuming .jpg or .png) in the source folder
    image_files = [f for f in os.listdir(src_images_folder) if f.endswith(('.jpg', '.png'))]
    
    # Perform a train-validation split (10% for validation)
    val_count = int(len(image_files) * val_split)
    val_files = random.sample(image_files, val_count)
    train_files = [f for f in image_files if f not in val_files]

    def process_files(files, image_dest, mask_dest):
        for filename in files:
            img_path = os.path.join(src_images_folder, filename)
            mask_path = os.path.join(src_masks_folder, os.path.splitext(filename)[0] + '.png')  # Masks are expected in .png format

            # Open image and convert it to RGB, then save as .png in the destination folder
            image = Image.open(img_path).convert('RGB')
            image_dest_path = os.path.join(image_dest, os.path.splitext(filename)[0] + '.png')
            image.save(image_dest_path)
            
            # Open mask and save it as .png in the destination folder
            mask = Image.open(mask_path).convert('RGB')
            mask_dest_path = os.path.join(mask_dest, os.path.splitext(filename)[0] + '.png')
            mask.save(mask_dest_path)

            print(f"Saved image and mask for: {filename}")

    # Process and save train and validation files
    process_files(train_files, dest_images_folder, dest_masks_folder)
    process_files(val_files, output_val_images, output_val_masks)

# Save train and validation images and masks with 10% for validation
save_images_and_masks(src_train_images, src_train_masks, output_train_images, output_train_masks, val_split=0.1)
save_images_and_masks(src_valid_images, src_valid_masks, output_val_images, output_val_masks, val_split=1.0)

print("All images and masks have been saved in the specified output folders.")
