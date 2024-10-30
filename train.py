import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from jaccard_loss import JaccardLoss
from utils import(
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs
)

#Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 8
NUM_EPOCHS = 240
NUM_WORKERS = 1
IMAGE_HEIGHT = 360
IMAGE_WIDTH = 480
#IMAGE_HEIGHT = 480v b
#IMAGE_WIDTH = 640  v
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = './road_marks/train_images'
TRAIN_MASK_DIR = 'road_marks/train_masks'
VAL_IMG_DIR = 'road_marks/val_images'
VAL_MASK_DIR = 'road_marks/val_masks'
bce_loss = nn.BCEWithLogitsLoss()
jaccard_loss = JaccardLoss()

def combined_loss(preds, targets):
    #return bce_loss(preds,targets)
    #dice_loss = 1 - (2 * (preds * targets).sum()) / ((preds + targets).sum() + 1e-8)
    #return dice_loss
    return 0.5 * bce_loss(preds, targets) + 0.5 * jaccard_loss(preds, targets)


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)  # Add channel dimension to targets
      
        
        #forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())


def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.ShiftScaleRotate(shift_limit = 0.05, scale_limit=0.1, rotate_limit=10, p=0.7),
            A.ElasticTransform(alpha=0.5, sigma=30.0, alpha_affine=None, p=0.4),
            A.RandomRain(p=0.2, brightness_coefficient=0.9, drop_width=1, blur_value=5),
            A.RandomFog(p=0.2, fog_coef_lower=0.3, fog_coef_upper=0.5, alpha_coef=0.1),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
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
    
    model = UNET(in_channels=3, out_channels=1).to(DEVICE) # for multiclass change the number of output channelss
   
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY
    )
    
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(NUM_EPOCHS):
        print(f"######### Epoch Number {epoch} ##############")
        train_fn(train_loader, model, optimizer, combined_loss, scaler)

        #save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        save_checkpoint(checkpoint)

        #check accuracy
        check_accuracy(val_loader,model, device=DEVICE)

        # print some examples to a folder

    save_predictions_as_imgs(
        val_loader, model, folder="saved_images/", device=DEVICE
    )


if __name__ == '__main__':
    main()
