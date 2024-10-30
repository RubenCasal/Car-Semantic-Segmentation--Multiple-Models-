import os
import numpy as np
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split
import shutil

# Paths for the new dataset
input_images_path = "./road_marks/train/images"
input_labels_path = "./road_marks/train/labels"

output_train_images = "./road_marks/train_images"
output_train_masks = "./road_marks/train_masks"
output_val_images = "./road_marks/val_images"
output_val_masks = "./road_marks/val_masks"

# Define the color for road marks
roadmark_color = (0, 255, 102)

# Create the output directories if they don't exist
os.makedirs(output_train_images, exist_ok=True)
os.makedirs(output_train_masks, exist_ok=True)
os.makedirs(output_val_images, exist_ok=True)
os.makedirs(output_val_masks, exist_ok=True)

def create_mask_from_txt(txt_path, image_size):
    width, height = image_size
    # Initialize an empty mask with black background
    mask = Image.new('RGB', (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(mask)
    
    # Read the .txt file and parse lines
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split()
        # Skip the first part (class label) and convert the rest to float pairs
        points = [(float(parts[i]) * width, float(parts[i + 1]) * height) for i in range(1, len(parts), 2)]
        
        # Draw the polygon with the roadmark color
        draw.polygon(points, fill=roadmark_color)
    
    return mask

# List all images and corresponding .txt files
image_files = [f for f in os.listdir(input_images_path) if f.endswith(".jpg")]
txt_files = [f.replace(".jpg", ".txt") for f in image_files]

# Split into training and validation sets (90% train, 10% val)
train_images, val_images, train_txts, val_txts = train_test_split(
    image_files, txt_files, test_size=0.1, random_state=42
)

# Function to process and save images and masks
def process_and_save(images, txts, image_dest, mask_dest):
    count = 0
    for img_file, txt_file in zip(images, txts):
        image_path = os.path.join(input_images_path, img_file)
        txt_path = os.path.join(input_labels_path, txt_file)
        
        # Check if the .txt file exists
        if not os.path.exists(txt_path):
            print(f"Warning: TXT file not found for {img_file}. Skipping...")
            continue
        
        # Open image and create mask
        image = Image.open(image_path)
        mask = create_mask_from_txt(txt_path, image.size)
        
        # Save the image as .png
        output_img_name = img_file.replace(".jpg", ".png")
        image.save(os.path.join(image_dest, output_img_name), format='PNG')
        mask.save(os.path.join(mask_dest, output_img_name), format='PNG')
        
        # Update and print progress
        count += 1
        if count % 100 == 0:
            print(f"Processed {count} images so far...")

# Process training files
process_and_save(train_images, train_txts, output_train_images, output_train_masks)
# Process validation files
process_and_save(val_images, val_txts, output_val_images, output_val_masks)

print("Mask creation from TXT and file organization complete!")
