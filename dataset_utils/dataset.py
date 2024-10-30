import os 
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class CarvanaDataset(Dataset):
    def __init__(self,image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)


    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace('.jpg', '_mask.gif'))

        # Load the image and the mask
        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('RGB'))

     
        label_mapping = {
    (0, 255, 102): 0,    
    (0, 0, 0): 1,   
}
    
        # Initialize the remapped mask with -1 (indicating unlabeled areas)
        remapped_mask = np.full((mask.shape[0], mask.shape[1]), -1, dtype=np.int8)

        # Map each RGB value to its corresponding class index
        for rgb_value, class_index in label_mapping.items():
            matches = np.all(mask == rgb_value, axis=-1)
            remapped_mask[matches] = class_index
       

        # If transformations are provided, apply them
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=remapped_mask)
            image = augmentations["image"]
            remapped_mask = augmentations["mask"]

        return image, remapped_mask
