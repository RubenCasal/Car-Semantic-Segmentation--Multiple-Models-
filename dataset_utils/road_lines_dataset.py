import os
import numpy as np
from PIL import Image

# Define source folders for masks
src_train_masks = './road_lines_dataset/train/masks'
src_valid_masks = './road_lines_dataset/valid/masks'

# Define the color transformation map
transform_map = {
    (0, 0, 0): (0, 0, 0),
    (128, 0, 0): (0, 0, 0),
    (0, 128, 0): (0, 255, 102),
    (128, 128, 0): (0, 255, 102)
}

def transform_image_colors(mask_image):
    img_np = np.array(mask_image)
    # Initialize a blank mask of the same shape
    transformed_mask = np.zeros_like(img_np)

    # Apply transformations based on the color map
    for src_color, target_color in transform_map.items():
        # Create a mask for the current color
        match = np.all(img_np == src_color, axis=-1)
        transformed_mask[match] = target_color

    return Image.fromarray(transformed_mask)

def transform_masks_in_directory(src_masks_folder):
    # Ensure the output folder exists (overwriting in place here)
    os.makedirs(src_masks_folder, exist_ok=True)
    
    # Process each mask in the directory
    for filename in os.listdir(src_masks_folder):
        if filename.endswith('.png'):
            mask_path = os.path.join(src_masks_folder, filename)

            # Load the mask image
            mask = Image.open(mask_path).convert('RGB')
            
            # Transform the mask
            transformed_mask = transform_image_colors(mask)
            
            # Save the transformed mask in the same folder, overwriting the original
            transformed_mask.save(mask_path)
            print(f"Transformed and saved mask: {mask_path}")

# Transform masks in both train and validation mask directories
transform_masks_in_directory(src_train_masks)
transform_masks_in_directory(src_valid_masks)

print("All mask transformations complete!")
