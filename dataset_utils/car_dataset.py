import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# Define the input folders and their vehicle color mappings
folders = {
   # "./data2": [(0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 0, 230), (119, 11, 32), (0, 0, 90), (0, 0, 110)],
    #"./data3": [(0, 255, 102)],
    "./data4": [(64, 0, 128), (192, 0, 192), (192, 64, 128), (192, 128, 192)]
}

# Define output paths
output_train_images = "./vehicle_dataset/train_images"
output_train_masks = "./vehicle_dataset/train_masks"
output_val_images = "./vehicle_dataset/val_images"
output_val_masks = "./vehicle_dataset/val_masks"

# Create the output directories if they don't exist
os.makedirs(output_train_images, exist_ok=True)
os.makedirs(output_train_masks, exist_ok=True)
os.makedirs(output_val_images, exist_ok=True)
os.makedirs(output_val_masks, exist_ok=True)

def transform_image_colors(mask_image, vehicle_colors):
    img_np = np.array(mask_image)
    
    # Initialize a new image with the same shape, setting all to (0, 0, 0)
    vehicle_mask = np.zeros_like(img_np)
    
    # Apply transformations for each color in the vehicle color set
    for color in vehicle_colors:
        mask = np.all(img_np == color, axis=-1)
        vehicle_mask[mask] = (0, 255, 102)  # Set vehicle color
    
    return Image.fromarray(vehicle_mask)

def process_folder(input_folder, vehicle_colors):
    images_path = os.path.join(input_folder, "images")
    masks_path = os.path.join(input_folder, "masks")
    
    images = [f for f in os.listdir(images_path) if f.endswith(".png")]
    
    # Perform a train-validation split (90% train, 10% val)
    train_files, val_files = train_test_split(images, test_size=0.1, random_state=42)

    # Function to process and save the files
    def process_and_save(files, image_dest, mask_dest):
        count = 0
        for filename in files:
            image_path = os.path.join(images_path, filename)
            mask_path = os.path.join(masks_path, filename)
            
            # Open image and mask
            image = Image.open(image_path)
            mask = Image.open(mask_path).convert('RGB')
            
            # Transform the mask
            transformed_mask = transform_image_colors(mask, vehicle_colors)
            
            # Save both image and transformed mask
            image.save(os.path.join(image_dest, filename))
            transformed_mask.save(os.path.join(mask_dest, filename))
            
            # Update and print progress
            count += 1
            if count % 100 == 0:
                print(f"Processed {count} images so far...")

    # Process training files
    print(f"Processing training files in {input_folder}...")
    process_and_save(train_files, output_train_images, output_train_masks)
    
    # Process validation files
    print(f"Processing validation files in {input_folder}...")
    process_and_save(val_files, output_val_images, output_val_masks)

# Loop over each folder and process
for folder, vehicle_colors in folders.items():
    process_folder(folder, vehicle_colors)

print("Color transformation and file organization complete!")
