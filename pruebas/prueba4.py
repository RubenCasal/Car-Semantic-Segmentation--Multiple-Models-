import os
from PIL import Image

def find_corrupted_images(folder):
    corrupted_images = []
    
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        
        try:
            # Open the image file
            img = Image.open(img_path)
            img.verify()  # Verify that it is a valid image
        except (IOError, SyntaxError, ValueError) as e:
            # If the image is corrupted, add it to the list and print the error
            print(f"Corrupted image found: {img_path}")
            corrupted_images.append(img_path)
    
    # Print a summary
    if corrupted_images:
        print("\nSummary: Corrupted images found:")
        for img in corrupted_images:
            print(img)
    else:
        print("No corrupted images found.")
    
    return corrupted_images

# Specify the directory with your images
image_directory = "./vehicle_dataset/train_images"
corrupted_images = find_corrupted_images(image_directory)
