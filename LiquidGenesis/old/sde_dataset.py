import torch
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import torchvision.transforms as T

class SDUDataset(Dataset):
    def __init__(self, img_dir, mask_dir, depth_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.depth_dir = depth_dir
        self.transform = transform

        self.files = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]
        base = os.path.splitext(img_name)[0]

        # Carica RGB
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        # Carica maschera (PNG con 3 classi)
        mask_path = os.path.join(self.mask_dir, base + ".png")
        mask = Image.open(mask_path)

        # Carica profondità
        depth_path = os.path.join(self.depth_dir, base + ".npy")
        depth = np.load(depth_path)

        if self.transform:
            image = self.transform(image)
            mask = torch.from_numpy(np.array(mask)).long()
            depth = torch.from_numpy(depth).unsqueeze(0).float()  # (1, H, W)

        return image, mask, depth
