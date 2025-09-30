import os
import csv

ROOT_DIR = "/home/edo/thesis/LiquidGenesis/vol_est/dataset"
OUTPUT_CSV = "/home/edo/thesis/LiquidGenesis/vol_est/src/samples_new+real+video.csv"

# ROOT_DIR = "/home/edo/thesis/LiquidGenesis/cont_pos/data"
# OUTPUT_CSV = "/home/edo/thesis/LiquidGenesis/vol_est/src/samples_new+real.csv"

LIQUID_FILE = "Input_ContentDepth_segmented.npy"
VESSEL_FILE = "Input_EmptyVessel_Depth.npy" #"Input_EmptyVessel_Depth_segmented.npy"
LIQUID_MASK_FILE = "Input_ContentMaskClean.npy"
VESSEL_MASK_FILE = "Input_VesselMask.npy"
VOLUME_FILE = "Input_vol_liquid.txt"
RGB_FILE="rgb.png" #"Input_RGBImage.png"
VOLUME_VES_FILE ="Input_vol_vessel.txt"

def main():
    rows = []
    for sample_name in sorted(os.listdir(ROOT_DIR)):
        sample_path = os.path.join(ROOT_DIR, sample_name)
        if not os.path.isdir(sample_path):
            continue

        liquid_path = os.path.join(sample_path, LIQUID_FILE)
        vessel_path = os.path.join(sample_path, VESSEL_FILE)
        liquid_mask_path = os.path.join(sample_path, LIQUID_MASK_FILE)
        vessel_mask_path = os.path.join(sample_path, VESSEL_MASK_FILE)
        volume_path = os.path.join(sample_path, VOLUME_FILE)
        volume_ves_path = os.path.join(sample_path, VOLUME_VES_FILE)
        rgb_path=os.path.join(sample_path,RGB_FILE)

        # check file exist
        required_files = [liquid_path, vessel_path, liquid_mask_path, vessel_mask_path, volume_path,volume_ves_path, rgb_path]
        if not all(os.path.exists(f) for f in required_files):
            print(f"[SKIP] Mancano file in: {sample_name}")
            continue

        # Legge volume
        with open(volume_path, "r") as f:
            volume_str = f.read().strip()
        try:
            volume_ml = float(volume_str)
        except ValueError:
            print(f"[ERRORE] Volume non numerico in {volume_path}: '{volume_str}'")
            continue

        with open(volume_ves_path, "r") as f:
            volume_str = f.read().strip()
        try:
            vol_ves = float(volume_str)
        except ValueError:
            print(f"[ERRORE] Volume non numerico in {volume_path}: '{volume_str}'")
            continue

        rows.append({
            "id": sample_name,
            "liquid_path": os.path.relpath(liquid_path, start=ROOT_DIR),
            "container_path": os.path.relpath(vessel_path, start=ROOT_DIR),
            "liquid_mask_path": os.path.relpath(liquid_mask_path, start=ROOT_DIR),
            "container_mask_path": os.path.relpath(vessel_mask_path, start=ROOT_DIR),
            "volume_ml": volume_ml,
            "vol_ves": vol_ves,
            "rgb_path": rgb_path,
        })

    # CSV
    out_path = OUTPUT_CSV 
    with open(out_path, "w", newline="") as csvfile:
        fieldnames = [
            "id",
            "liquid_path",
            "container_path",
            "liquid_mask_path",
            "container_mask_path",
            "volume_ml",
            "vol_ves",
            "rgb_path"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Creato {out_path} con {len(rows)} campioni.")

if __name__ == "__main__":
    main()
