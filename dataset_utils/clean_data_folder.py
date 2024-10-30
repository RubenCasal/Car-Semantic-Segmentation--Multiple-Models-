import os
import numpy as np
from PIL import Image

# Define the dataset folders and their subfolder structure
datasets = [
    "vehicle_dataset",
    "traffic_dataset",
    "Pedestrian_dataset",
    "road_dataset"
]

# Define the subfolders for images and masks
subfolders = ["train_images", "train_masks", "val_images", "val_masks"]

def is_completely_black(mask_path):
    # Load the mask and convert it to a numpy array
    mask = Image.open(mask_path).convert("RGB")
    mask_np = np.array(mask)
    
    # Check if all values are zero (black)
    return np.all(mask_np == 0)

def clean_dataset(dataset_folder):
    total_deleted = 0  # Counter for deleted images in this dataset
    
    # Process both train and validation folders
    for split in ["train", "val"]:
        images_folder = os.path.join(dataset_folder, f"{split}_images")
        masks_folder = os.path.join(dataset_folder, f"{split}_masks")
        
        # List all mask files
        for filename in os.listdir(masks_folder):
            mask_path = os.path.join(masks_folder, filename)
            image_path = os.path.join(images_folder, filename)  # Assume corresponding image has the same name
            
            # Check if the mask is completely black
            if is_completely_black(mask_path):
                # Remove both the mask and the corresponding image
                os.remove(mask_path)
                os.remove(image_path)
                print(f"Removed completely black mask and image: {filename} in {dataset_folder}/{split}")
                
                # Increment the counter
                total_deleted += 1
    
    # Print total deleted images for this dataset
    print(f"Total deleted images in {dataset_folder}: {total_deleted}")

# Loop through each dataset and clean it
for dataset in datasets:
    clean_dataset(dataset)

print("Dataset cleaning complete! All completely black masks and corresponding images have been removed.")
