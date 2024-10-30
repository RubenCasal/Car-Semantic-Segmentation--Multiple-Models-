import torch
import os
import numpy as np
from PIL import Image
from utils import convert_class_to_rgb, index_to_rgb
from model import UNET
from albumentations.pytorch import ToTensorV2
import albumentations as A

# Hyperparameters
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMAGE_HEIGHT = 260
IMAGE_WIDTH = 440

# Function to predict a single image and save the prediction
def predict_single_image(model, image_path, save_path, transform, device):
    model.eval()  # Set model to evaluation mode

    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image = np.array(image)

    # Apply the transformation
    augmented = transform(image=image)
    image_tensor = augmented["image"].unsqueeze(0).to(device)  # Add batch dimension and move to device

    with torch.no_grad():
        # Make prediction
        preds = model(image_tensor)
        preds = torch.argmax(preds, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)  # Get predicted class labels

    # Convert the predicted mask to RGB format
    rgb_pred = convert_class_to_rgb(preds, index_to_rgb)

    # Save the RGB prediction image
    rgb_pred_img = Image.fromarray(rgb_pred)
    rgb_pred_img.save(save_path)
    print(f"Prediction saved as {save_path}")

def main():
    # Define the transformation (same as in training)
    transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    # Load the trained U-Net model
    model = UNET(in_channels=3, out_channels=19).to(DEVICE)

    # Load the model checkpoint if available
    checkpoint_path = 'checkpoint2.pth.tar'  # Adjust as needed
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(checkpoint['state_dict'])

    # Path to the input image and the path to save the predicted image
    image_path = "only_image.png"  # Path to the input image
    save_path = "only_pred2.png"    # Path to save the predicted image

    # Predict and save the result
    predict_single_image(model, image_path, save_path, transform, DEVICE)

if __name__ == "__main__":
    main()
