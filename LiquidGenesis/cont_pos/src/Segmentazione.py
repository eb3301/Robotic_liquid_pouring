import numpy as np
import FCN_NetModel as NET_FCN  # The net Class
import torch
import Visualization as vis
import cv2
import depthai as dai

# ========== Parametri ==========
Trained_model_path = "/home/edo/thesis/LiquidGenesis/cont_pos_&_vol_est/logs/Default.torch"
UseGPU = False
DisplayXYZPointCloud = False
DisplayVesselOpeningOnPointCloud = True
MinSize = 600
MaxSize = 1000
salvaIM=False

# Closer-in minimum depth, disparity range is doubled (from 95 to 190):
extended_disparity = False
# Better accuracy for longer distance, fractional disparity 32-levels:
subpixel = False
# Better handling for occlusions:
lr_check = True

save_depth_path = "/home/edo/thesis/LiquidGenesis/cont_pos_&_vol_est/saved_depth.png"
save_rgb_path = "/home/edo/thesis/LiquidGenesis/cont_pos_&_vol_est/saved_rgb.png"

XYZList = ["VesselXYZ","ContentXYZ","VesselOpening_XYZ"] # XYZ maps to predict
MaskList = ["VesselMask","ContentMaskClean","VesselOpeningMask"] # Segmentation mask to predict
XYZ2Mask={"VesselXYZ":"VesselMask","ContentXYZ":"ContentMaskClean","VesselOpening_XYZ":"VesselOpeningMask"} # Connect XYZ map to segmentation mask
XYZ2Color={"VesselXYZ":[255,0,0],"ContentXYZ":[0,255,0],"VesselOpening_XYZ":[0,0,255]} # Colors where eac object will appear on XYZ point cloud
# ========== Acquisizione da OAK-D ==========
# Crea una pipeline DepthAI
pipeline = dai.Pipeline()

# Camera RGB:
cam_rgb = pipeline.createColorCamera()
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
cam_rgb.setPreviewSize(640, 480)
cam_rgb.setInterleaved(False)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)

# Camere per stereo:
# Define sources and outputs
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
depth = pipeline.create(dai.node.StereoDepth)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setCamera("left")
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setCamera("right")

# Create a node that will produce the depth map (using disparity output as it's easier to visualize depth this way)
depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
# Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
depth.setLeftRightCheck(lr_check)
depth.setExtendedDisparity(extended_disparity)
depth.setSubpixel(subpixel)

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

# Output
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

xout_depth = pipeline.createXLinkOut()
xout_depth.setStreamName("depth")
monoLeft.out.link(depth.left)
monoRight.out.link(depth.right)
depth.disparity.link(xout_depth.input)

# Avvia la pipeline
with dai.Device(pipeline) as device:
    q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    q_depth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    
    print("Acquisizione immagini (RGB + Depth) dalla OAK-D")

    for _ in range (100):
        rgb_frame = q_rgb.get().getCvFrame()
        depth_frame = q_depth.get().getFrame()
        # Normalizza la profondità per visualizzazione
        depth_vis = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)
        depth_vis = np.uint8(depth_vis)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    if salvaIM:
        # Salva RGB e depth
        cv2.imwrite(save_rgb_path, rgb_frame)
        cv2.imwrite(save_depth_path, depth_vis)
        print(f"Immagini salvate con successo")



# ========== Segmentazione con rete ==========
Net = NET_FCN.Net(MaskList=MaskList, XYZList=XYZList)
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
#-----------------------------Display 3d point cloud----------------------------------------------------------
if DisplayXYZPointCloud:
    import open3d as o3d
    points = np.zeros([20000,3],np.float32) # list of points for XYZ point cloud
    colors = np.zeros([20000,3],np.float32) # Point color for above list

    xyzMap={} # Maps from which points will be sampled
    h,w=Im.shape[1:3]
    xyzMap["VesselXYZ"] = Prd["VesselXYZ"][0]
    xyzMap["ContentXYZ"] = Prd["ContentXYZ"][0]
    xyzMap["VesselOpening_XYZ"] =Prd["VesselOpening_XYZ"][0]


    #vis.show(Im[0].astype(np.uint8)," Close window to continue")

    tt=0
    print(f"Segmentazione in corso")
    while True: # Sample points for point cloud
           nm = list(xyzMap)[np.random.randint(len(list(xyzMap)))]
           x = np.random.randint(xyzMap[nm].shape[1])
           y = np.random.randint(xyzMap[nm].shape[0])
           if (Prd[XYZ2Mask[nm]][0,y,x])>0.95:
                     if np.abs(xyzMap[nm][y,x]).sum()>0:
                           points[tt]=xyzMap[nm][y,x]
                           colors[tt]=XYZ2Color[nm]
                           tt+=1
                           if tt>=points.shape[0]: break
    print(f"Segmentazione completata")
    
    #...................Display point cloud.........................................................................................
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd],"Red=Vessel, Green=Content, Blue=Opening")

#-----------------------------Display XYZ map and images-----------------------------------------
for key in Prd:
    print(key, Prd[key][0].max(),Prd[key][0].min())
    #---------------Normalize value to the range of RGB image 0-255--------------------------------------------
    tmIm = Prd[key][0].copy()
    if Prd[key][0].max()>255 or Prd[key][0].min()<0 or np.ndim(Prd[key][0])==2:
        if tmIm.max()>tmIm.min(): #
            tmIm[tmIm>1000]=0
            tmIm = tmIm-tmIm.min()
            tmIm = tmIm/tmIm.max()*255
        print(key,"New", tmIm.max(), tmIm.min())
        if np.ndim(tmIm)==2:
            tmIm=cv2.cvtColor(tmIm.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        # Remove region out side of the object mask from the XYZ mask
        if key in XYZ2Mask:
            print(key)
            for i in range(tmIm.shape[2]):
                tmIm[:,:,i][Prd[XYZ2Mask[key]][0]==0]=0
#--------------------------------display------------------------------------------------------------
    im=cv2.resize(Im[0].astype(np.uint8),(tmIm.shape[1],tmIm.shape[0]))
    vis.show(np.hstack([tmIm,im]),key+ " Max=" + str(Prd[key][0].max()) + " Min=" + str(Prd[key][0].min())+ " Close window to continue")
    