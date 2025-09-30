import numpy as np
import model
import torch
import Visualization as vis
import cv2
import depthai as dai
import open3d as o3d
import json
import os


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
                if frame[y,x,2]/1000 < 0.7 and frame[y,x,2]/1000 > 0.01:
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

    filtered_content_depth = cv2.resize(filtered_content_depth, (1920, 1080), interpolation=cv2.INTER_NEAREST)
    filtered_vessel_depth = cv2.resize(filtered_vessel_depth, (1920, 1080), interpolation=cv2.INTER_NEAREST)

    content_mask = cv2.resize(content_mask, (1920, 1080), interpolation=cv2.INTER_NEAREST)
    vessel_mask = cv2.resize(vessel_mask, (1920, 1080), interpolation=cv2.INTER_NEAREST)
    content_mask = (content_mask > 0.5).astype(np.uint8)
    vessel_mask  = (vessel_mask  > 0.5).astype(np.uint8)

    return filtered_content_depth, filtered_vessel_depth, content_mask, vessel_mask

def salva_output( rgb_frame, ptcd, ptcdf, centroid, filtered_content_depth, filtered_vessel_depth, content_mask, vessel_mask, vol, path):
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
    
    
    print("Salvataggio file completato!")

# ========= Main ==========
def main():
    # Variabili
    salva=True
    show=True
    idx="4"
    path="/home/edo/thesis/LiquidGenesis/cont_pos/data/"+idx
    vol=40

    # Cattura immagini:
    rgb_frame, ptcd_points, ptcd_colors = oak_capture()
    target_class = "Vessel"  # Altri: "Content", "VesselOpening"
    # Segmentazione:
    Prd = CNN_predict(rgb_frame)
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

    # ====== Salvataggio (opzionale) ====== 
    if salva:
        salva_output(rgb_frame, ptcd, ptcdf, centroid, filtered_content_depth, filtered_vessel_depth, content_mask, vessel_mask, vol, path)
    
    # ==== Visualizzazione ====
    if show:
        disp=(vessel_mask * 255).astype(np.uint8)
        cv2.imshow("mask", disp)
        cv2.waitKey(0)  
        cv2.destroyAllWindows()

        o3d.visualization.draw_geometries([ptcd], f"Point Cloud originale - {target_class}")
        o3d.visualization.draw_geometries([ptcdf], f"Point Cloud filtrata - {target_class}")

if __name__ == "__main__":
    main()
