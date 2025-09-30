import torch
import numpy as np
import cv2
import os
import argparse
import csv
import random

from model import VolumeNN, VolumeNetPret

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "/home/edo/thesis/LiquidGenesis/vol_est/checkpoints/best_model_ResNet_1.pth"
ROOT_DIR = "/home/edo/thesis/LiquidGenesis/vol_est/processed/"
CSV_PATH = "/home/edo/thesis/LiquidGenesis/vol_est/src/samples_new.csv"
with open(CSV_PATH, newline="") as f:
            reader = csv.DictReader(f)
            samples = list(reader)
idx=random.randint(0, len(samples))
row=samples[idx]
DIR = row["id"] + "/"
LIQUID_PATH=ROOT_DIR + DIR + "Input_ContentDepth_segmented.npy"
CONTAINER_PATH=ROOT_DIR + DIR + "Input_EmptyVessel_Depth_segmented.npy"
SIZE=(256, 192)


vol=row["volume_ml"]



def load_mask(path, target_size=(128, 128)):
    """Carica .npy e ridimensiona a target_size"""
    arr = np.load(path).astype(np.float32)
    if target_size is not None and arr.shape != target_size:
        arr = cv2.resize(arr, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
    return arr

def preprocess(liquid_path, vessel_path, target_size=(128, 128)):
    liquid = load_mask(liquid_path, target_size)
    vessel = load_mask(vessel_path, target_size)

    # Normalizzazione rispetto al max comune
    max_val = max(np.max(liquid), np.max(vessel), 1e-6)
    liquid /= max_val
    vessel /= max_val

    masks = np.stack([liquid, vessel], axis=0)  # shape (2, H, W)
    masks_tensor = torch.from_numpy(masks).float().unsqueeze(0)  # shape (1, 2, H, W)
    return masks_tensor

def main(MODEL_PATH,CONTAINER_PATH,LIQUID_PATH,SIZE):
    # Modello
    # model = VolumeNN(input_channels=4, size= TARGET_SIZE)
    model = VolumeNetPret(backbone_name="ResNet18", input_channels=4, pretrained=True).to(DEVICE)

    checkpoint = torch.load(MODEL_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Preprocess
    masks_tensor = preprocess(LIQUID_PATH, CONTAINER_PATH, target_size=(SIZE[0], SIZE[1])).to(DEVICE)

    with torch.no_grad():
        pred_volume = model(masks_tensor).item()

    print(f"Volume stimato: {pred_volume:.2f} ml")
    print(f"volume reale: " + vol + " ml")

if __name__ == "__main__":
    main(MODEL_PATH,CONTAINER_PATH,LIQUID_PATH,SIZE)
