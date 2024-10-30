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

def process_video(video_path, output_path, model, device, image_height=360, image_width=480, transparency=0.6):

   
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # Use the same Albumentations transformation pipeline used in training
    transform = A.Compose([
        A.Resize(height=image_height, width=image_width),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2()
    ])

    model.eval()

    frame_idx = 0  # To track the frame index for saving
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the OpenCV frame (BGR) to RGB for consistency
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Apply the Albumentations transform
            augmented = transform(image=frame_rgb)
            input_frame = augmented["image"].unsqueeze(0).to(device)

            # Model prediction
            preds = model(input_frame).squeeze(1)
            preds = torch.sigmoid(preds)
            preds = (preds > 0.5).float().squeeze(0).cpu().numpy().astype(np.uint8)
            

            # Convert predicted mask to RGB format
            seg_image = convert_class_to_rgb(preds, index_to_rgb)

            # Save the segmentation image for this frame in the `video_pred` folder as PNG
           

            # Resize the segmentation image to match the original frame size and write to video
            seg_image_resized = cv2.resize(seg_image, (frame_width, frame_height))

            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            blended_frame = cv2.addWeighted(frame_bgr, 1- transparency, seg_image_resized, transparency,0)
            out.write(blended_frame)

            frame_idx += 1  # Increment frame index for the next frame

    cap.release()
    out.release()

if __name__ == "__main__":
    video_path = "prueba5.mp4"  # Path to the input video
    output_path = "pedest2_marks.mp4"  # Path to save the segmented video

    # Load your trained U-Net model
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    checkpoint_path = 'pedestrians_model.pth.tar'  # Adjust as needed
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(checkpoint['state_dict'])

    # Process the video
    process_video(video_path, output_path, model, DEVICE)
