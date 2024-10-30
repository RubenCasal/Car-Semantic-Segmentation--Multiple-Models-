import torch
import torchvision
from dataset import CarvanaDataset
from torch.utils.data import DataLoader
import numpy as np 
import os
from PIL import Image
from torchmetrics import JaccardIndex

def save_checkpoint(state, filename="road_marks3.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
        train_ds = CarvanaDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

        train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

        val_ds = CarvanaDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

        val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

        return train_loader, val_loader


def check_accuracy(loader, model, device='cuda'):
    num_correct = 0
    num_pixels = 0
    
    model.eval()

    with torch.no_grad():
        num_pixels = 0
        num_correct = 0
        dice_score = 0
        for x,y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
          
            ## Change to argmax Its a multiclass segmentation
            preds = model(x)
          
            preds = torch.sigmoid(preds)
            preds = (preds > 0.5).float()

            num_correct += torch.eq(preds, y).sum().item()
            num_pixels  += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )
            
            
    accuracy = num_correct / num_pixels * 100
    
           

    print(f"{num_correct}/{num_pixels} Accuracy: {accuracy:.2f}% ")
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()  # Set back to training mode



index_to_rgb = {
    0: (0, 255, 102), 
    1: (0, 0, 0) 
}


def convert_class_to_rgb(class_mask, index_to_rgb):

    height, width = class_mask.shape
    rgb_mask = np.zeros((height, width, 3), dtype=np.uint8)  # 3 channels for RGB

    # Loop over the class indices and their corresponding RGB values
    for class_index, rgb_value in index_to_rgb.items():
        # Create a boolean mask for pixels that belong to this class
        mask = class_mask == class_index
        
        # Assign the corresponding RGB color to those pixels in the rgb_mask
        rgb_mask[mask] = rgb_value  # Broadcast the RGB value across the third dimension

    return rgb_mask
def save_predictions_as_imgs(loader, model, folder="prueba_imagenes/", device="cuda"):
    # Create the folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)

    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = model(x)
            preds = torch.sigmoid(preds)
            preds = (preds > 0.5).float()  # Convert to binary (0 or 1)

        # Move tensors back to CPU and convert to numpy arrays for saving
        preds_np = preds.cpu().numpy().astype(np.uint8)  # Shape [batch_size, height, width]
        y_np = y.cpu().numpy().astype(np.uint8)          # Shape [batch_size, height, width]

        # Save each image in the batch
        for i in range(preds_np.shape[0]):
            # Predicted mask: Convert class indices to RGB colors
            pred_rgb = convert_class_to_rgb(preds_np[i], index_to_rgb)
            pred_img = Image.fromarray(pred_rgb)
            pred_img.save(f"{folder}/pred_{idx}_{i}.png")

            # Ground truth mask: Convert class indices to RGB colors
            target_rgb = convert_class_to_rgb(y_np[i], index_to_rgb)
            target_img = Image.fromarray(target_rgb)
            target_img.save(f"{folder}/target_{idx}_{i}.png")

    model.train()