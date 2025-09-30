import torch
import numpy as np
import os
import csv
import model
import Visualization as vis
import cv2
from tqdm import tqdm 

DIR = "/home/edo/thesis/LiquidGenesis/vol_est"
ROOT_DIR = DIR + "/processed/"
CSV_PATH = DIR + "/src/samples_redo.csv"
path="/home/edo/thesis/LiquidGenesis/vol_est/dataset"

def CNN_predict(rgb_frame):

    Trained_model_path = "/home/edo/thesis/LiquidGenesis/cont_pos/logs/Default.torch"
    UseGPU = False
    MinSize = 600
    MaxSize = 1000
    XYZList = ["VesselXYZ","ContentXYZ","VesselOpening_XYZ"] # XYZ maps to predict
    MaskList = ["VesselMask","ContentMaskClean","VesselOpeningMask"] # Segmentation mask to predict
    XYZ2Mask={"VesselXYZ":"VesselMask","ContentXYZ":"ContentMaskClean","VesselOpening_XYZ":"VesselOpeningMask"} # Connect XYZ map to segmentation mask
    #XYZ2Color={"VesselXYZ":[255,0,0],"ContentXYZ":[0,255,0],"VesselOpening_XYZ":[0,0,255]} # Colors where eac object will appear on XYZ point cloud
    
    # ========== Segmentazione con rete ==========
    Net = model.Net(MaskList=MaskList, XYZList=XYZList)
    if UseGPU:
        Net.load_state_dict(torch.load(Trained_model_path))
    else:
        Net.load_state_dict(torch.load(Trained_model_path, map_location=torch.device('cpu')))
    Net.eval()

    # Resize immagine per rete
    Im = vis.ResizeToMaxSize(rgb_frame, MaxSize)
    Im = np.expand_dims(Im, axis=0)

    ###############################Run Net and make prediction###########################################################################
    with torch.no_grad():
        PrdXYZ, PrdProb, PrdMask = Net.forward(Images=Im,TrainMode=False,UseGPU=UseGPU) # Run net inference and get prediction

    #----------------------------Convert Prediction to numpy-------------------------------------------
    Prd={}
    for key in PrdXYZ:
        Prd[key]=(PrdXYZ[key].transpose(1,2).transpose(2, 3)).data.cpu().numpy()
    for key in PrdMask:
        Prd[key]=(PrdMask[key]).data.cpu().numpy()

    return Prd

def filter_and_scale_depth(Prd, threshold=0.95):
    content_depth = Prd["ContentXYZ"][0, :, :, 2]  # shape (H,W)
    vessel_depth  = Prd["VesselXYZ"][0, :, :, 2]

    content_mask = Prd["ContentMaskClean"][0, :, :]  # shape (H,W)
    vessel_mask  = Prd["VesselMask"][0, :, :]
 
    content_mask_bin = content_mask > threshold
    vessel_mask_bin = vessel_mask > threshold

    corr= min(np.min(content_depth),np.min(vessel_depth)) # correzione per portare il minimo a zero

    h,w = content_mask.shape
    filtered_content_depth = []
    filtered_vessel_depth = []
    for y in range(h):
        for x in range(w):
            if content_mask_bin[y, x]:
                filtered_content_depth.append(content_depth[y,x]-corr)
            else:
                filtered_content_depth.append(0)

            if vessel_mask_bin[y, x]:
                filtered_vessel_depth.append(vessel_depth[y,x]-corr)
            else:
                filtered_vessel_depth.append(0)

    filtered_content_depth=np.array(filtered_content_depth)
    filtered_vessel_depth=np.array(filtered_vessel_depth)
    
    filtered_content_depth.resize(h,w)
    filtered_vessel_depth.resize(h,w)

    size_res=(640,480)
    filtered_content_depth = cv2.resize(filtered_content_depth, size_res, interpolation=cv2.INTER_NEAREST)
    filtered_vessel_depth = cv2.resize(filtered_vessel_depth, size_res, interpolation=cv2.INTER_NEAREST)

    content_mask = cv2.resize(content_mask, size_res, interpolation=cv2.INTER_NEAREST)
    vessel_mask = cv2.resize(vessel_mask, size_res, interpolation=cv2.INTER_NEAREST)
    content_mask = (content_mask > 0.5).astype(np.uint8)
    vessel_mask  = (vessel_mask  > 0.5).astype(np.uint8)

    return filtered_content_depth, filtered_vessel_depth, content_mask, vessel_mask

def salva_output(rgb_frame, filtered_content_depth, filtered_vessel_depth, content_mask, vessel_mask, vol, vol_ves, path):
    os.makedirs(path, exist_ok=True)

    # Salvataggio in npy
    np.save(os.path.join(path, "Input_ContentDepth_segmented.npy"), filtered_content_depth)
    np.save(os.path.join(path, "Input_EmptyVessel_Depth.npy"), filtered_vessel_depth)
    np.save(os.path.join(path, "Input_ContentMaskClean.npy"), content_mask)
    np.save(os.path.join(path, "Input_VesselMask.npy"), vessel_mask)

    # RGB
    cv2.imwrite(os.path.join(path, "rgb.png"), rgb_frame)
    # Depth
    filtered_content_depth = cv2.normalize(filtered_content_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    filtered_vessel_depth = cv2.normalize(filtered_vessel_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(os.path.join(path, "content_depth.png"), filtered_content_depth)
    cv2.imwrite(os.path.join(path, "vessel_depth.png"), filtered_vessel_depth)

    # Mask
    cv2.imwrite(os.path.join(path, "content_mask.png"), (content_mask * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(path, "vessel_mask.png"), (vessel_mask * 255).astype(np.uint8))
    
    with open(path + "/Input_vol_liquid.txt", "w") as f:
        f.write(str(vol))
    with open(path + "/Input_vol_vessel.txt", "w") as f:
        f.write(str(vol_ves))
    
    #print("Salvataggio file completato!")

def main():
    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        samples = list(reader)

    for idx, row in enumerate(tqdm(samples, desc="Processing samples")):
        path_i = os.path.join(path, str(idx))
        if os.path.exists(path_i):
            print(f"Skipping sample {idx}, already processed.")
            continue
        
        rgb_path=os.path.join(ROOT_DIR, row["rgb_path"])
        img=cv2.imread(rgb_path)
        if img.ndim == 2:  # grayscale
            img = np.expand_dims(img, -1)
        
        vol_liq=float(row["volume_ml"])
        vol_ves=float(row["volume_ml"])
        Prd=CNN_predict(img)
        filtered_content_depth, filtered_vessel_depth, content_mask, vessel_mask = filter_and_scale_depth(Prd)
        path_i=path + "/" + str(idx) + "/"
        salva_output(img, filtered_content_depth, filtered_vessel_depth, content_mask, vessel_mask, vol_liq, vol_ves, path_i)

if __name__ == "__main__":
    main()
