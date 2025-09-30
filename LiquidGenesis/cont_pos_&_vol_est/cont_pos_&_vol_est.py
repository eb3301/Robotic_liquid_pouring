import numpy as np
import torch
import Visualization as vis
import cv2
import depthai as dai
import open3d as o3d
import json
import os

import torch.nn.functional as F
from model import VolumeNN, VolumeNetPret, Net

def oak_capture():
    # Closer-in minimum depth, disparity range is doubled (from 95 to 190):
    extended_disparity = False
    subpixel = True
    lr_check = True
    # ========== Acquisizione da OAK-D ==========
    # Crea una pipeline DepthAI
    pipeline = dai.Pipeline()

    # Camera RGB:
    cam_rgb = pipeline.createColorCamera()
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setPreviewSize(640, 480)
    cam_rgb.setInterleaved(False)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam_rgb.setFps(30)

    # Camere per depth:
    # Define sources and outputs
    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoRight = pipeline.create(dai.node.MonoCamera)
    depth = pipeline.create(dai.node.StereoDepth)
    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoLeft.setCamera("left")
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setCamera("right")

    depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
    depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)     # Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
    depth.setLeftRightCheck(lr_check)
    depth.setExtendedDisparity(extended_disparity)
    depth.setSubpixel(subpixel)

    depth.setDepthAlign(dai.CameraBoardSocket.CAM_A)

    config = depth.initialConfig.get()
    config.postProcessing.speckleFilter.enable = False
    config.postProcessing.speckleFilter.speckleRange = 50
    config.postProcessing.temporalFilter.enable = True
    config.postProcessing.spatialFilter.enable = True
    config.postProcessing.spatialFilter.holeFillingRadius = 2
    config.postProcessing.spatialFilter.numIterations = 1
    config.postProcessing.thresholdFilter.minRange = 400
    config.postProcessing.thresholdFilter.maxRange = 15000
    config.postProcessing.decimationFilter.decimationFactor = 1
    depth.initialConfig.set(config)

    # Pointcloud
    pointcloud: dai.node.PointCloud = pipeline.create(dai.node.PointCloud)
    pointcloud.setNumFramesPool(12) # 12 frame è il max
    sync = pipeline.create(dai.node.Sync)
    
    # Link sorgenti
    monoLeft.out.link(depth.left)
    monoRight.out.link(depth.right)
    depth.depth.link(pointcloud.inputDepth)

    # Link rgb e pcl al Sync (così ottieni un singolo messaggio sincronizzato)
    cam_rgb.isp.link(sync.inputs["rgb"])
    pointcloud.outputPointCloud.link(sync.inputs["pcl"])

    # Single XLinkOut per il messaggio sincronizzato
    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("pcl")
    xout.input.setBlocking(False)
    sync.out.link(xout.input)

    # Avvia la pipeline
    with dai.Device(pipeline) as device:
        q = device.getOutputQueue(name="pcl", maxSize=4, blocking=False)
        
        print("Acquisizione immagini (RGB + Depth) dalla OAK-D")

        for _ in range (100):
            ptcd_msg = q.get()
            if ptcd_msg is None:
                continue
            ptcd_pt = ptcd_msg["pcl"]
            ptcd_color = ptcd_msg["rgb"]
            ptcd_colorFrame = ptcd_color.getCvFrame()
            cvRGBFrame = cv2.cvtColor(ptcd_colorFrame, cv2.COLOR_BGR2RGB)
            rgb_frame = ptcd_colorFrame

            if ptcd_pt:
                ptcd_points = ptcd_pt.getPoints().astype(np.float64)
                ptcd_colors = (cvRGBFrame.reshape(-1, 3) / 255.0).astype(np.float64)
                
        #calib = device.readCalibration()
        #intrinsics = np.array(calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B, rgb_frame.shape[1], rgb_frame.shape[0]))


    return rgb_frame, ptcd_points, ptcd_colors

def CNN_predict(rgb_frame,target_class):

    Trained_model_path = "/home/edo/thesis/LiquidGenesis/cont_pos/logs/Default.torch"
    UseGPU = False
    MinSize = 600
    MaxSize = 1000
    XYZList = ["VesselXYZ","ContentXYZ","VesselOpening_XYZ"] # XYZ maps to predict
    MaskList = ["VesselMask","ContentMaskClean","VesselOpeningMask"] # Segmentation mask to predict
    XYZ2Mask={"VesselXYZ":"VesselMask","ContentXYZ":"ContentMaskClean","VesselOpening_XYZ":"VesselOpeningMask"} # Connect XYZ map to segmentation mask
    #XYZ2Color={"VesselXYZ":[255,0,0],"ContentXYZ":[0,255,0],"VesselOpening_XYZ":[0,0,255]} # Colors where eac object will appear on XYZ point cloud

    target_xyz_key = f"{target_class}XYZ"
    target_mask_key = XYZ2Mask[target_xyz_key]
    
    # ========== Segmentazione con rete ==========
    CNNet = Net(MaskList=MaskList, XYZList=XYZList)
    if UseGPU:
        Net.load_state_dict(torch.load(Trained_model_path))
    else:
        CNNet.load_state_dict(torch.load(Trained_model_path, map_location=torch.device('cpu')))
    CNNet.eval()

    # Resize immagine per rete
    Im = vis.ResizeToMaxSize(rgb_frame, MaxSize)
    Im = np.expand_dims(Im, axis=0)

    ###############################Run Net and make prediction###########################################################################
    with torch.no_grad():
        PrdXYZ, PrdProb, PrdMask = CNNet.forward(Images=Im,TrainMode=False,UseGPU=UseGPU) # Run net inference and get prediction

    #----------------------------Convert Prediction to numpy-------------------------------------------
    Prd={}
    for key in PrdXYZ:
        Prd[key]=(PrdXYZ[key].transpose(1,2).transpose(2, 3)).data.cpu().numpy()
    for key in PrdMask:
        Prd[key]=(PrdMask[key]).data.cpu().numpy()

    return Prd

def filter_and_scale_points(frame, color, Prd, threshold=0.95):
    mask  = Prd["VesselMask"][0, :, :]
    mask = (mask > threshold).astype(np.uint8) * 255
    mask = cv2.resize(mask, (1920, 1080), interpolation=cv2.INTER_NEAREST)
    h, w = mask.shape

    if frame.shape[0] != mask.shape[0]*mask.shape[1]:
        print("size di maschera e pointcloud non coincidono")
        raise RuntimeError()

    frame.resize(h,w,3)
    color.resize(h,w,3)
    mask_bin = mask > threshold

    points = []
    colors = []
    for y in range(h):
        for x in range(w):
            if mask_bin[y, x]:
                if frame[y,x,2]/1000 < 0.7 :#and frame[y,x,2]/1000 > 0.01:
                    points.append(frame[y,x,:]/1000)
                    # Z = ptcd_frame[y,x,2] / 1000.0 
                    # if Z == 0: 
                    #     continue
                    # X = (x - cx) * Z / fx
                    # Y = (y - cy) * Z / fy
                    # points.append([X, Y, Z])
                    colors.append(color[y,x,:])

    points=np.array(points)
    colors=np.array(colors)

    frame.resize(h*w,3)
    color.resize(h*w,3)

    if points.size == 0:
        cv2.imshow("mask", mask)
        cv2.waitKey(0)  
        cv2.destroyAllWindows()
        raise RuntimeError("La point cloud filtrata è vuota")

    return points, colors

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

def salva_output(rgb_frame, ptcd, ptcdf, centroid, filtered_content_depth, filtered_vessel_depth, content_mask, vessel_mask, vol, vol_ves, path):
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

    # Pointcloud
    o3d.io.write_point_cloud(path + f"/pointcloud_completa.ply", ptcd)
    o3d.io.write_point_cloud(path + f"/pointcloud_filtrata.ply", ptcdf)

    # Centroide
    with open(path + "/coord_centroide.json", "w") as f:
        json.dump({
            "X": float(centroid[0]),
            "Y": float(centroid[1]),
            "Z": float(centroid[2])
        }, f, indent=2)
    
    with open(path + "/Input_vol_liquid.txt", "w") as f:
        f.write(str(vol))
    with open(path + "/Input_vol_vessel.txt", "w") as f:
        f.write(str(vol_ves))
    
    
    
    
    print("Salvataggio file completato!")


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

def preprocess(filtered_content_depth, filtered_vessel_depth, content_mask, vessel_mask, target_size):    
    # print(np.max(filtered_content_depth))
    # print(np.min(filtered_content_depth))
    # print(np.max(filtered_vessel_depth))
    # print(np.min(filtered_vessel_depth))
    # Filtra depth map con maschere

    filtered_content_depth = filtered_content_depth.astype(np.float32)
    filtered_vessel_depth = filtered_vessel_depth.astype(np.float32)
    content_mask = content_mask.astype(np.float32)
    vessel_mask = vessel_mask.astype(np.float32)


    filtered_content_depth_masked = filtered_content_depth * content_mask
    filtered_vessel_depth_masked = filtered_vessel_depth * vessel_mask

    # Normalizzazione e scaling
    liquid_norm, liquid_scaled = process_depth_with_mask(filtered_content_depth_masked, content_mask)
    container_norm, container_scaled = process_depth_with_mask(filtered_vessel_depth_masked, vessel_mask)

    # Zoom e Crop per avere dimensioni simili a quelle del dataset
    av_w_b_ratio=0.15 # !!! deve eesere uguale a quello della dataset creation !!!
    _, vessel_mask_binary = cv2.threshold(vessel_mask, 0.5, 255, cv2.THRESH_BINARY)
    centroid = get_centroid(vessel_mask_binary)
    crop_size = find_zoom_to_ratio(vessel_mask_binary, centroid, av_w_b_ratio)
    print(f"wbratio: {white_black_ratio(vessel_mask)}")
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

# Main 
def main():
    # Variabili
    salva=True
    show=False
    idx="50"
    path="/home/edo/thesis/LiquidGenesis/cont_pos/data/"+idx
    vol=80
    vol_ves=100
    DIR = "/home/edo/thesis/LiquidGenesis/vol_est"
    MODEL_PATH = DIR + "/checkpoints/best_model_ResNet_1.pth"
    ########################### Cont pos #######################################
    

    # Cattura immagini:
    rgb_frame, ptcd_points, ptcd_colors = oak_capture()
    target_class = "Vessel"  # Altri: "Content", "VesselOpening"
    # Segmentazione:
    Prd = CNN_predict(rgb_frame, target_class)
    # Filtra pointcloud applicando maschera di segmentazione:
    filtered_ptcd_points, filtered_ptcd_colors=filter_and_scale_points(ptcd_points, ptcd_colors, Prd)
    # FIltra depth applicando maschera di segmentazione:
    filtered_content_depth, filtered_vessel_depth, content_mask, vessel_mask = filter_and_scale_depth(Prd)
    # Calcola centroide:
    centroid = np.mean(filtered_ptcd_points, axis=0) # x orizz camera, y vert camera, z profondità
    print(f"Coordinate 3D ({target_class}): X={centroid[0]:.3f} m, Y={centroid[1]:.3f} m, Z={centroid[2]:.3f} m")
    # Crea point cloud:
    ptcd= o3d.geometry.PointCloud()
    ptcd.points = o3d.utility.Vector3dVector(ptcd_points)
    ptcd.colors = o3d.utility.Vector3dVector(ptcd_colors)

    ptcdf= o3d.geometry.PointCloud()
    ptcdf.points = o3d.utility.Vector3dVector(filtered_ptcd_points)
    ptcdf.colors = o3d.utility.Vector3dVector(filtered_ptcd_colors)

    # Salvataggio e visualizzazione
    if salva:
        salva_output(rgb_frame, ptcd, ptcdf, centroid, filtered_content_depth, filtered_vessel_depth, content_mask, vessel_mask, vol, vol_ves, path)
    if show:
        disp=(vessel_mask * 255).astype(np.uint8)
        cv2.imshow("mask", disp)
        cv2.waitKey(0)  
        cv2.destroyAllWindows()

        o3d.visualization.draw_geometries([ptcd], f"Point Cloud originale - {target_class}")
        o3d.visualization.draw_geometries([ptcdf], f"Point Cloud filtrata - {target_class}")

    ################################# Vol est #######################################    
    
    # ROOT_DIR = "/home/edo/thesis/LiquidGenesis/cont_pos/data/4/"

    # liquid_path = os.path.join(ROOT_DIR, "Input_ContentDepth_segmented.npy")
    # container_path = os.path.join(ROOT_DIR, "Input_EmptyVessel_Depth.npy")
    # content_mask_path = os.path.join(ROOT_DIR, "Input_ContentMaskClean.npy")
    # vessel_mask_path = os.path.join(ROOT_DIR, "Input_VesselMask.npy")
    # vol= 0.0
    # id = 0

    # Modello
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VolumeNetPret(backbone_name="ResNet18", input_channels=4, pretrained=True).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Preprocess
    target_size = (256, 192)
    
    masks_tensor = preprocess(filtered_content_depth, filtered_vessel_depth, content_mask, vessel_mask, target_size).to(DEVICE)

    with torch.no_grad():
        pred_volume = model(masks_tensor).item()

    print(f"Sample: {idx}")
    print(f"Volume stimato: {pred_volume:.2f} ml")
    print(f"Volume reale: {vol:.2f} ml")


if __name__ == "__main__":
    main()
