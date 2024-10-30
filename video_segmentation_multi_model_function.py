import os
import cv2
import torch
import numpy as np
from utils import convert_class_to_rgb, index_to_rgb
from model import UNET
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define colors for each model's prediction
ROAD_LINES_COLOR = (0, 255, 0)      # Green for road lines
ROAD_COLOR = (0, 0, 255)            # Red for road
VEHICLES_COLOR = (255, 0, 0)        # Blue for vehicles
PEDESTRIANS_COLOR = (255, 255, 0)   # Yellow for pedestrians

def load_model(checkpoint_path):
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

def generate_segmented_video(input_video_path, output_video_path, 
                             pedestrian_model_path, road_marks_model_path, 
                             road_model_path, vehicles_model_path,
                             image_height=360, image_width=480, transparency=0.3):
    
    # Check if output directory exists, create if it doesn't
    output_dir = os.path.dirname(output_video_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load each model with its corresponding checkpoint
    pedestrian_model = load_model(pedestrian_model_path)
    road_lines_model = load_model(road_marks_model_path)
    road_model = load_model(road_model_path)
    vehicles_model = load_model(vehicles_model_path)
    
    models = [pedestrian_model, road_lines_model, road_model, vehicles_model]
    colors = [PEDESTRIANS_COLOR, ROAD_LINES_COLOR, ROAD_COLOR, VEHICLES_COLOR]
    
    # Video processing
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Ensure VideoWriter opens properly
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    if not out.isOpened():
        print(f"Error: Could not write video to {output_video_path}")
        return

    transform = A.Compose([
        A.Resize(height=image_height, width=image_width),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2()
    ])

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            augmented = transform(image=frame_rgb)
            input_frame = augmented["image"].unsqueeze(0).to(DEVICE)

            combined_mask = np.zeros((image_height, image_width, 3), dtype=np.uint8)
            
            # Iterate over each model, predict, and combine with different colors
            for model, color in zip(models, colors):
                preds = model(input_frame).squeeze(1)
                preds = torch.sigmoid(preds)
                preds = (preds < 0.6).float().squeeze(0).cpu().numpy().astype(np.uint8)
                
                # Convert to color mask
                color_mask = np.zeros((image_height, image_width, 3), dtype=np.uint8)
                color_mask[preds == 1] = color
                combined_mask = cv2.addWeighted(combined_mask, 1, color_mask, 1, 0)
            
            seg_image_resized = cv2.resize(combined_mask, (frame_width, frame_height))
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            blended_frame = cv2.addWeighted(frame_bgr, 1 - transparency, seg_image_resized, transparency, 0)
            out.write(blended_frame)

    # Release resources
    cap.release()
    out.release()
    print(f"Video saved to {output_video_path}")

# Example usage:
# generate_segmented_video("input_video.mp4", "output_video.mp4",
#                          "pedestrian2_model.pth.tar", "road_marks2_model.pth.tar",
#                          "road_model.pth.tar", "vehicles2_model.pth.tar")
