import os

# Path to the folder containing mask images
mask_folder = "data3/masks"

# Loop through all files in the folder
for filename in os.listdir(mask_folder):
    if filename.startswith("mask_") and filename.endswith(".png"):
        # Replace 'mask_' with 'image_' in the filename
        new_filename = filename.replace("mask_", "image_", 1)
        
        # Full file paths
        old_file = os.path.join(mask_folder, filename)
        new_file = os.path.join(mask_folder, new_filename)
        
        # Rename the file
        os.rename(old_file, new_file)

print("Renaming complete!")
