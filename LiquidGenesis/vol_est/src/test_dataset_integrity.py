import torch
import numpy as np
import os
import csv
import numpy as np

DIR = "/home/edo/thesis/LiquidGenesis/vol_est"
ROOT_DIR = DIR + "/dataset/"
CSV_PATH = DIR + "/src/samples_new.csv"

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

def white_black_ratio(img):
    white = np.count_nonzero(img)
    total = img.size
    return white / total  # rapporto w/b

def main():
    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        samples = list(reader)

    container_depth_max=[]
    container_depth_min=[]
    container_depth_mean=[]
    liquid_depth_max=[]
    liquid_depth_min=[]
    liquid_depth_mean=[]
    wbratio=[]
    for idx in range(len(samples)-1):
        row = samples[idx]

        # Percorsi ai file
        liquid_path = os.path.join(ROOT_DIR, row["liquid_path"])
        container_path = os.path.join(ROOT_DIR, row["container_path"])
        liquid_mask_path = os.path.join(ROOT_DIR, row["liquid_mask_path"])
        container_mask_path = os.path.join(ROOT_DIR, row["container_mask_path"])

        vol = float(row["volume_ml"])
        id = row['id']

        # Caricamento file
        liquid_depth = np.squeeze(np.load(liquid_path).astype(np.float32))
        container_depth = np.squeeze(np.load(container_path).astype(np.float32))
        liquid_mask = np.squeeze(np.load(liquid_mask_path).astype(np.float32))
        container_mask = np.squeeze(np.load(container_mask_path).astype(np.float32))

        container_depth_max.append(np.max(container_depth))
        container_depth_min.append(np.min(container_depth))
        container_depth_mean.append(np.mean(container_depth))
        liquid_depth_max.append(np.max(liquid_depth))
        liquid_depth_min.append(np.min(liquid_depth))
        liquid_depth_mean.append(np.mean(liquid_depth))

        wbratio.append(white_black_ratio(container_mask))
        #print(f"container depth:" + str(type(container_depth)) + f" - {container_depth.shape} + max:{np.max(container_depth)} + min:{np.min(container_depth)} + mean:{np.mean(container_depth)}" )
        #print(f"container mask:" + str(type(container_mask)) + f" - {container_mask.shape} + max:{np.max(container_mask)} + min:{np.min(container_mask)}")
        # Zoom e Crop per avere dimensioni simili a quelle del dataset
        #print(f"wbratio: {white_black_ratio(container_mask)}") 
    av_max_cont=np.mean(container_depth_max)
    av_min_cont=np.mean(container_depth_min)
    av_mean_cont=np.mean(container_depth_mean)
    av_max_liq=np.mean(liquid_depth_max)
    av_min_liq=np.mean(liquid_depth_min)
    av_mean_liq=np.mean(liquid_depth_mean)
    av_wbratio=np.mean(wbratio)
    print(f"cont - max: {av_max_cont} min: {av_min_cont} mean: {av_mean_cont}")
    print(f"liq - max: {av_max_liq} min: {av_min_liq} mean: {av_mean_liq}")
    print(av_wbratio)

    import matplotlib.pyplot as plt
    plt.plot(wbratio, container_depth_mean, marker="o",)

    # aggiungi etichette e titolo
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Plot di due liste")

    # aggiungi legenda
    plt.legend()

    # mostra il grafico
    plt.show()


if __name__ == "__main__":
    main()