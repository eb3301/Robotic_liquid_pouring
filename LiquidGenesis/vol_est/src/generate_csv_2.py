import os
import csv

# Percorso alla cartella che contiene le sottocartelle dei campioni
ROOT_DIR = "/home/edo/thesis/LiquidGenesis/vol_est/processed"
OUTPUT_CSV = "/home/edo/thesis/LiquidGenesis/vol_est/src/samples.csv"

# Nome dei file chiave in formato .npy
LIQUID_FILE = "Input_ContentDepth_segmented.npy"
VESSEL_FILE = "Input_EmptyVessel_Depth_segmented.npy"
VOLUME_FILE = "Input_vol_liquid.txt"

def main():
    rows = []
    for sample_name in sorted(os.listdir(ROOT_DIR)):
        sample_path = os.path.join(ROOT_DIR, sample_name)
        if not os.path.isdir(sample_path):
            continue

        liquid_path = os.path.join(sample_path, LIQUID_FILE)
        vessel_path = os.path.join(sample_path, VESSEL_FILE)
        volume_path = os.path.join(sample_path, VOLUME_FILE)

        if not (os.path.exists(liquid_path) and os.path.exists(vessel_path) and os.path.exists(volume_path)):
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

        rows.append({
            "id": sample_name,
            "liquid_path": os.path.relpath(liquid_path, start=ROOT_DIR),
            "container_path": os.path.relpath(vessel_path, start=ROOT_DIR),
            "volume_ml": volume_ml
        })

    # Scrive CSV
    out_path = os.path.join(ROOT_DIR, OUTPUT_CSV)
    with open(out_path, "w", newline="") as csvfile:
        fieldnames = ["id", "liquid_path", "container_path", "volume_ml"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Creato {out_path} con {len(rows)} campioni.")

if __name__ == "__main__":
    main()
