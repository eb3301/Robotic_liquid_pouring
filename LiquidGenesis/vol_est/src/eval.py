import torch
import numpy as np
import cv2
import os
import csv
import random
import torch.nn.functional as F

from model import VolumeNN, VolumeNetPret

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DIR = "/home/edo/thesis/LiquidGenesis/vol_est"
MODEL_PATH = DIR + "/checkpoints/best_model_ResNet_2.pth"
sim=False
newdataset=True
if sim:
    ROOT_DIR = DIR + "/processed/"
    CSV_PATH = DIR + "/src/samples.csv"
    if newdataset: 
        ROOT_DIR = DIR + "/dataset/"
        CSV_PATH = DIR + "/src/samples_new.csv"
else:
    idx="011"
    ROOT_DIR = "/home/edo/thesis/LiquidGenesis/cont_pos/data/" + str(idx) + "/"


def log_to_linear(depth):
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

def process_depth_with_mask(depth, mask):
    """Normalizza e scala una depth map usando solo i pixel della maschera."""
    depth = log_to_linear(depth)

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

def resize_tensor(arr, target_size):
    """Converte un np.array 2D in tensor e ridimensiona a target_size."""
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Array in input non è 2D: shape={arr.shape}")

    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    t = F.interpolate(t, size=target_size, mode="bilinear", align_corners=False)
    return t.squeeze(0)  # [1,H,W]

def preprocess(liquid_path, container_path, liquid_mask_path, container_mask_path, target_size):
    # Caricamento file
    liquid_depth = np.squeeze(np.load(liquid_path).astype(np.float32))
    container_depth = np.squeeze(np.load(container_path).astype(np.float32))
    liquid_mask = np.squeeze(np.load(liquid_mask_path).astype(np.float32))
    container_mask = np.squeeze(np.load(container_mask_path).astype(np.float32))

    print(f"container depth:" + str(type(container_depth)) + f" - {container_depth.shape} + max:{np.max(container_depth)} + min:{np.min(container_depth)} + mean:{np.mean(container_depth)}" )
    print(f"container mask:" + str(type(container_mask)) + f" - {container_mask.shape} + max:{np.max(container_mask)} + min:{np.min(container_mask)}")
    # print(np.max(liquid_depth))
    # print(np.min(liquid_depth))
    # print(np.max(container_depth))
    # print(np.min(container_depth))
    # Filtra depth map con maschere
    liquid_depth_masked = liquid_depth * liquid_mask
    container_depth_masked = container_depth * container_mask

    # Normalizzazione e scaling
    liquid_norm, liquid_scaled = process_depth_with_mask(liquid_depth_masked, liquid_mask)
    container_norm, container_scaled = process_depth_with_mask(container_depth_masked, container_mask)
    print(f"container depth norm:" + str(type(container_norm)) + f" - {container_norm.shape} + {np.max(container_norm)} + min:{np.min(container_norm)} + mean:{np.mean(container_norm)}" )
    print(f"container depth scaled:" + str(type(container_scaled)) + f" - {container_scaled.shape} + {np.max(container_scaled)} + min:{np.min(container_scaled)} + mean:{np.mean(container_scaled)}")
    # Zoom e Crop per avere dimensioni simili a quelle del dataset
    av_w_b_ratio=0.15
    container_mask_binary = (container_mask > 0.5).astype(np.uint8) * 255
    centroid = get_centroid(container_mask_binary)
    crop_size = find_zoom_to_ratio(container_mask_binary, centroid, av_w_b_ratio)
    print(f"wbratio: {white_black_ratio(container_mask)}")
    liquid_norm=crop_centered(liquid_norm, centroid, crop_size)
    container_norm=crop_centered(container_norm, centroid, crop_size)
    liquid_scaled=crop_centered(liquid_scaled, centroid, crop_size)
    container_scaled=crop_centered(container_scaled, centroid, crop_size)

    # Resize e conversione in tensor
    liquid_norm_t = resize_tensor(liquid_norm, target_size)
    container_norm_t = resize_tensor(container_norm, target_size)
    liquid_scaled_t = resize_tensor(liquid_scaled, target_size)
    container_scaled_t = resize_tensor(container_scaled, target_size)

    # Concateno i 4 canali
    masks_tensor = torch.cat([
        liquid_norm_t,
        container_norm_t,
        liquid_scaled_t,
        container_scaled_t
    ], dim=0).unsqueeze(0)  # [1,4,H,W]

    return masks_tensor

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


def main():
    if sim:
        # Carica un sample random dal CSV
        with open(CSV_PATH, newline="") as f:
            reader = csv.DictReader(f)
            samples = list(reader)
        idx = random.randint(0, len(samples) - 1)
        row = samples[idx]

        # Percorsi ai file
        liquid_path = os.path.join(ROOT_DIR, row["liquid_path"])
        container_path = os.path.join(ROOT_DIR, row["container_path"])
        liquid_mask_path = os.path.join(ROOT_DIR, row["liquid_mask_path"])
        container_mask_path = os.path.join(ROOT_DIR, row["container_mask_path"])

        vol = float(row["volume_ml"])
        id = row['id']
    else:
        liquid_path = os.path.join(ROOT_DIR, "Input_ContentDepth_segmented.npy")
        container_path = os.path.join(ROOT_DIR, "Input_EmptyVessel_Depth.npy")
        liquid_mask_path = os.path.join(ROOT_DIR, "Input_ContentMaskClean.npy")
        container_mask_path = os.path.join(ROOT_DIR, "Input_VesselMask.npy")
        with open(ROOT_DIR + "Input_vol_liquid.txt", "r") as file: vol = float(file.read())
        id = 0

    # Modello
    model = VolumeNetPret(backbone_name="ResNet18", input_channels=4, pretrained=True).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Preprocess
    target_size = (256, 192)
    masks_tensor = preprocess(liquid_path, container_path, liquid_mask_path, container_mask_path, target_size).to(DEVICE)

    with torch.no_grad():
        pred_volume = model(masks_tensor).item()

    print(f"Sample: {id}")
    print(f"Volume stimato: {pred_volume:.2f} ml")
    print(f"Volume reale: {vol:.2f} ml")


if __name__ == "__main__":
    main()
