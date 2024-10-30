import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# Define input paths for the Penn-Fudan dataset
input_images_path = "./PennFudanPed/images"
input_masks_path = "./PennFudanPed/masks"

# Define output paths
output_train_images = "./pede/train_images"
output_train_masks = "./pede/train_masks"
output_val_images = "./pede/val_images"
output_val_masks = "./pede/val_masks"

# Create the output directories if they don't exist
os.makedirs(output_train_images, exist_ok=True)
os.makedirs(output_train_masks, exist_ok=True)
os.makedirs(output_val_images, exist_ok=True)
os.makedirs(output_val_masks, exist_ok=True)

def transform_mask_to_color(mask_image, pedestrian_color=(0, 255, 102)):
    # Convert the mask image to a NumPy array
    mask_np = np.array(mask_image)
    
    # Create a new RGB image for the transformed mask
    colored_mask = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8)
    
    # Transform the mask: set pixels labeled as pedestrians to the specified color
    pedestrian_mask = mask_np > 0  # assuming pedestrian pixels are non-zero
    colored_mask[pedestrian_mask] = pedestrian_color
    
    return Image.fromarray(colored_mask)

# Get the list of image files
image_files = [f for f in os.listdir(input_images_path) if f.endswith(".png")]

# Split the dataset into training and validation sets (90% train, 10% val)
train_files, val_files = train_test_split(image_files, test_size=0.1, random_state=42)

# Function to process and save files to their respective folders
def process_and_save(files, image_dest, mask_dest):
    for filename in files:
        # Paths to the image and mask
        image_path = os.path.join(input_images_path, filename)
        mask_path = os.path.join(input_masks_path, filename.replace(".png", "_mask.png"))
        
        # Load the image and mask
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        # Transform the mask to the specified pedestrian color
        colored_mask = transform_mask_to_color(mask)
        
        # Save the image and transformed mask to the destination folders
        image.save(os.path.join(image_dest, filename))
        colored_mask.save(os.path.join(mask_dest, filename))

# Process and save training files
process_and_save(train_files, output_train_images, output_train_masks)

# Process and save validation files
process_and_save(val_files, output_val_images, output_val_masks)

print("Penn-Fudan dataset transformation and split complete!")
