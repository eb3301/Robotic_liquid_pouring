import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class LiquidVolumeDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=None, target_size=(160, 214)):
        """
        csv_path: CSV con colonne:
            liquid_path, container_path, liquid_mask_path, container_mask_path, volume_ml
        root_dir: cartella radice contenente i file .npy
        transform: augmentations da applicare alle depth map concatenate
        target_size: (H, W) finale delle mappe
        """
        self.root_dir = root_dir
        self.transform = transform
        self.target_size = target_size

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            self.samples = list(reader)

        if not self.samples:
            raise RuntimeError(f"Nessun campione trovato in {csv_path}")

    def _log_to_linear(self, depth):
        """Rileva se la depth è in scala log e applica exp() se necessario."""
        mask = depth > 0
        if not mask.any():
            return depth
        max_val = np.max(depth[mask])
        mean_val = np.mean(depth[mask])
        if max_val < 20 and mean_val < 10:
            depth_exp = np.exp(depth)
            if np.mean(depth_exp[mask]) > mean_val * 2:
                return depth_exp
        return depth

    def _process_depth_with_mask(self, depth, mask):
        """Normalizza e scala una depth map usando solo i pixel della maschera."""
        depth = self._log_to_linear(depth)

        valid_pixels = (mask > 0) & (depth != 0)
        if valid_pixels.any():
            mean = np.mean(depth[valid_pixels])
            std = np.std(depth[valid_pixels])
            depth_norm = depth.copy()
            depth_norm[valid_pixels] = (depth[valid_pixels] - mean + 10) / (std + 1e-6)
            gt_depth = np.median(depth[valid_pixels])
            depth_scaled = depth_norm * gt_depth
        else:
            depth_norm = depth
            depth_scaled = depth

        return depth_norm, depth_scaled

    def _resize_tensor(self, arr):
        """Converte un np.array 2D in tensor e ridimensiona a target_size."""
        # Forza arr a 2D
        if arr.ndim > 2:
            arr = np.squeeze(arr)
        if arr.ndim != 2:
            raise ValueError(f"Array in input non è 2D: shape={arr.shape}")

        t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        t = F.interpolate(t, size=self.target_size, mode="bilinear", align_corners=False)
        return t.squeeze(0)  # [1,H,W]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx]

        # Caricamento file
        liquid_depth = np.squeeze(np.load(os.path.join(self.root_dir, row["liquid_path"])).astype(np.float32))
        container_depth = np.squeeze(np.load(os.path.join(self.root_dir, row["container_path"])).astype(np.float32))
        liquid_mask = np.squeeze(np.load(os.path.join(self.root_dir, row["liquid_mask_path"])).astype(np.float32))
        container_mask = np.squeeze(np.load(os.path.join(self.root_dir, row["container_mask_path"])).astype(np.float32))
        # Filtra depth map con maschere
        liquid_depth_masked = liquid_depth * liquid_mask
        container_depth_masked = container_depth * container_mask

        # Normalizzazione e scaling sui pixel della maschera
        liquid_norm, liquid_scaled = self._process_depth_with_mask(liquid_depth_masked, liquid_mask)
        container_norm, container_scaled = self._process_depth_with_mask(container_depth_masked, container_mask)
        # Crop wb ratio
        av_w_b_ratio=0.15
        container_mask_binary = (container_mask > 0.5).astype(np.uint8) * 255
        centroid = get_centroid(container_mask_binary)
        crop_size = find_zoom_to_ratio(container_mask_binary, centroid, av_w_b_ratio)
        #print(f"wbratio: {white_black_ratio(container_mask)}")
        liquid_norm=crop_centered(liquid_norm, centroid, crop_size)
        container_norm=crop_centered(container_norm, centroid, crop_size)
        liquid_scaled=crop_centered(liquid_scaled, centroid, crop_size)
        container_scaled=crop_centered(container_scaled, centroid, crop_size)
        # Resize e conversione in tensor
        liquid_norm_t = self._resize_tensor(liquid_norm)       # [1,H,W]
        container_norm_t = self._resize_tensor(container_norm)
        liquid_scaled_t = self._resize_tensor(liquid_scaled)
        container_scaled_t = self._resize_tensor(container_scaled)

        # Concateno i 4 canali: norm_liquid, norm_container, scaled_liquid, scaled_container
        masks_tensor = torch.cat([
            liquid_norm_t,
            container_norm_t,
            liquid_scaled_t,
            container_scaled_t
        ], dim=0)  # [4,H,W]

        volume = torch.tensor(float(row["volume_ml"]), dtype=torch.float32)

        # Augmentation
        if self.transform:
            masks_tensor = self.transform(masks_tensor)

        return masks_tensor, volume

def get_centroid(binary_img):
    # Assumo immagine 2D numpy con valori 0 (nero) e 255 (bianco)
    y, x = np.nonzero(binary_img)   # coordinate pixel bianchi
    if len(x) == 0 or len(y) == 0:
        raise RuntimeError("No container found in mask")
    cx = int(np.mean(x))
    cy = int(np.mean(y))
    return cx, cy

def crop_centered(img, center, size):
    cx, cy = center
    h, w = img.shape
    
    half = size // 2
    x1, x2 = max(0, cx - half), min(w, cx + half)
    y1, y2 = max(0, cy - half), min(h, cy + half)
    
    return img[y1:y2, x1:x2]

def white_black_ratio(img):
    white = np.count_nonzero(img)
    total = img.size
    return white / total  # rapporto w/b

def find_zoom_to_ratio(binary_img, center, target_ratio, initial_size=50, step=1, max_size=5000):
    size = initial_size
    best_crop = None
    best_diff = float("inf")
    
    while size < max_size:
        crop = crop_centered(binary_img, center, size)
        ratio = white_black_ratio(crop)
        diff = abs(ratio - target_ratio)
        
        if diff < best_diff:
            best_diff = diff
            # best_crop = crop
            best_size = size
        
        size += step
    
    return best_size
