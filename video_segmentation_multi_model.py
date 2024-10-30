import cv2
import torch
import numpy as np
from utils import convert_class_to_rgb, index_to_rgb
from model import UNET
import os
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define colors for each model's prediction
ROAD_LINES_COLOR = (0, 255, 0)  # Green for road lines
ROAD_COLOR = (0, 0, 255)        # Red for road
VEHICLES_COLOR = (255, 0, 0)    # Blue for vehicles
PEDESTRIANS_COLOR = (255, 255, 0)  # Yellow for pedestrians
def load_model(checkpoint_path):
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

def process_video(video_path, output_path, models, colors, device, image_height=360, image_width=480, transparency=0.3):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

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
            input_frame = augmented["image"].unsqueeze(0).to(device)

            combined_mask = np.zeros((image_height, image_width, 3), dtype=np.uint8)
            
            # Iterate over each model, predict, and combine with different colors
            for model, color in zip(models, colors):
                preds = model(input_frame).squeeze(1)
                preds = torch.sigmoid(preds)
                preds = (preds < 0.5).float().squeeze(0).cpu().numpy().astype(np.uint8)
                
                # Convert to color mask
                color_mask = np.zeros((image_height, image_width, 3), dtype=np.uint8)
                color_mask[preds == 1] = color
                combined_mask = cv2.addWeighted(combined_mask, 1, color_mask, 1, 0)
            
            seg_image_resized = cv2.resize(combined_mask, (frame_width, frame_height))
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            blended_frame = cv2.addWeighted(frame_bgr, 1 - transparency, seg_image_resized, transparency, 0)
            out.write(blended_frame)

    cap.release()
    out.release()

if __name__ == "__main__":
    video_path = "prueba6.mp4"  # Path to the input video
    output_path = "final_video2.mp4"  # Path to save the segmented video

    # Load each model with its corresponding checkpoint
    road_lines_model = load_model('road_marks2_model.pth.tar')
    road_model = load_model('road_model.pth.tar')
    vehicles_model = load_model('vehicles2_model.pth.tar')
    pedestrian_model = load_model('pedestrian2_model.pth.tar')

    # Process the video with each model's predictions in different colors
    models = [pedestrian_model,road_lines_model, road_model, vehicles_model]
    colors = [PEDESTRIANS_COLOR,ROAD_LINES_COLOR, ROAD_COLOR, VEHICLES_COLOR]
    
    process_video(video_path, output_path, models, colors, DEVICE)
