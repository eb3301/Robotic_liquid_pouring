import os
import json
import cv2
import numpy as np
import open3d as o3d
import depthai as dai
import Visualization as vis

# Parametri
base_dir = "/home/edo/thesis/LiquidGenesis/cont_pos_&_vol_est/process_data"
idx = "1"

# Mappatura dei file attesi
file_map = {
    f"{idx}_saved_rgb.png": "rgb_path",
    f"{idx}_mask.png": "mask_path",
    f"{idx}_pointcloud_completa.ply": "ptcd_c_path",
    f"{idx}_pointcloud_filtrata.ply": "ptcd_path",
    f"{idx}_coord_centroide.json": "coord_cent"
}

# Variabili per i percorsi
paths = {}

# Scansione cartella e assegnazione
for name in sorted(os.listdir(base_dir)):
    if name.startswith(idx) and name in file_map:
        paths[file_map[name]] = os.path.join(base_dir, name)

# Visualizzazione
for key in ["rgb_path", "mask_path"]:
    if key in paths:
        img = cv2.imread(paths[key])
        if img is not None:
            cv2.imshow(key, img)
            cv2.waitKey(0)
        else:
            print(f"Errore: impossibile aprire {paths[key]}")

for key in ["ptcd_c_path", "ptcd_path"]:
    if key in paths:
        pcd = o3d.io.read_point_cloud(paths[key])
        o3d.visualization.draw_geometries([pcd])

if "coord_cent" in paths:
    with open(paths["coord_cent"], 'r') as f:
        data = json.load(f)
    print(json.dumps(data, indent=4))

cv2.destroyAllWindows()
