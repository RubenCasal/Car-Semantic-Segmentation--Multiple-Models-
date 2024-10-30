import os
from PIL import Image
import numpy as np
'''
label_mapping = {
    (128, 64, 128):  (64, 32, 32),     # road
    (244, 35, 232): (128, 128, 96),    # sidewalk
    (70, 70, 70):  (128, 128, 96),      # building
    (102, 102, 156): (128, 128, 96),   # wall
    (190, 153, 153): (128, 128, 96),   # fence
    (153, 153, 153): (128, 128, 96),   # pole
    (250, 170, 30): (250, 170, 30),    # traffic light
    (220, 220, 0): (250, 170, 30),     # traffic sign
    (107, 142, 35): (128, 128, 96),    # vegetation
    (152, 251, 152): (128, 128, 96),   # terrain
    (70, 130, 180): (128, 128, 96),    # sky
    (220, 20, 60):  (220, 20, 60),      # person
    (255, 0, 0): (0, 255, 102),        # rider
    (0, 0, 142): (0, 255, 102),        # car
    (0, 0, 70): (0, 255, 102),         # truck
    (0, 60, 100): (0, 255, 102),       # bus
    (0, 80, 100): (0, 255, 102),       # train
    (0, 0, 230): (0, 255, 102),        # motorcycle
    (119, 11, 32): (0, 255, 102),      # bicycle
    (250, 170, 160): (128, 128, 96),   # parking
    (230, 150, 140): (128, 128, 96),   # rail track
    (111, 74, 0): (128, 128, 96),      # dynamic
    (81, 0, 81): (128, 128, 96),       # ground
    (180, 165, 180): (128, 128, 96),   # guard rail
    (150, 100, 100): (128, 128, 96),   # bridge
    (150, 120, 90): (128, 128, 96),    # tunnel
    (0, 0, 90): (0, 255, 102),        # caravan
    (0, 0, 110): (0, 255, 102),       # trailer
    (0, 0, 0): (128, 128, 96),              # unlabeled, ego vehicle, rectification border, out of roi, static (void class)
}

'''
label_mapping1 = {
    (128, 64, 128): (0, 0, 0),        # road
    (244, 35, 232): (0, 0, 0),        # sidewalk
    (70, 70, 70): (0, 0, 0),          # building
    (102, 102, 156): (0, 0, 0),       # wall
    (190, 153, 153): (0, 0, 0),       # fence
    (153, 153, 153): (0, 0, 0),       # pole
    (250, 170, 30): (0, 0, 0),        # traffic light
    (220, 220, 0): (0, 0, 0),         # traffic sign
    (107, 142, 35): (0, 0, 0),        # vegetation
    (152, 251, 152): (0, 0, 0),       # terrain
    (70, 130, 180): (0, 0, 0),        # sky
    (220, 20, 60): (0, 0, 0),         # person
    (255, 0, 0): (0, 0, 0),           # rider
    (0, 0, 142): (0, 255, 102),       # car
    (0, 0, 70): (0, 255, 102),        # truck
    (0, 60, 100): (0, 255, 102),      # bus
    (0, 80, 100): (0, 255, 102),      # train
    (0, 0, 230): (0, 255, 102),       # motorcycle
    (119, 11, 32): (0, 255, 102),     # bicycle
    (250, 170, 160): (0, 0, 0),       # parking
    (230, 150, 140): (0, 0, 0),       # rail track
    (111, 74, 0): (0, 0, 0),          # dynamic
    (81, 0, 81): (0, 0, 0),           # ground
    (180, 165, 180): (0, 0, 0),       # guard rail
    (150, 100, 100): (0, 0, 0),       # bridge
    (150, 120, 90): (0, 0, 0),        # tunnel
    (0, 0, 90): (0, 255, 102),        # caravan
    (0, 0, 110): (0, 255, 102),       # trailer
    (0, 0, 0): (0, 0, 0),             # unlabeled, ego vehicle, rectification border, out of roi, static (void class)
}
# Path to the mask images
input_folder = "data2/masks"
output_folder = "./vehicles_dataset/masks"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Map old RGB values to new RGB values
color_mapping = label_mapping1

def transform_image_colors(image_path, output_path):
    # Open the image and convert to numpy array
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)
    
    # Loop over the old color -> new color mapping
    for old_color, new_color in color_mapping.items():
        # Create a mask for pixels matching the old color
        mask = np.all(img_np == old_color, axis=-1)
        # Replace the old color with the new color
        img_np[mask] = new_color

    # Convert numpy array back to an image
    transformed_img = Image.fromarray(img_np)
    
    # Save the transformed image
    transformed_img.save(output_path)

# Loop through all images in the folder
for filename in os.listdir(input_folder):
    if filename.endswith(".png"):  # Assuming the mask images are in .png format
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        # Transform the image colors
        transform_image_colors(input_path, output_path)

print("Color transformation complete!")
