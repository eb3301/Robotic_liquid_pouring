import cv2
import depthai as dai
import numpy as np
import open3d as o3d

# Crea una pipeline DepthAI
pipeline = dai.Pipeline()

# Aggiungi le camere
cam_rgb = pipeline.createColorCamera()
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
cam_rgb.setPreviewSize(640, 480)
cam_rgb.setInterleaved(False)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)

cam_left = pipeline.createMonoCamera()
cam_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
cam_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)

cam_right = pipeline.createMonoCamera()
cam_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
cam_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

# Stereo depth
stereo = pipeline.createStereoDepth()
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
cam_left.out.link(stereo.left)
cam_right.out.link(stereo.right)

# Output
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

xout_depth = pipeline.createXLinkOut()
xout_depth.setStreamName("depth")
stereo.depth.link(xout_depth.input)

# Avvia la pipeline
with dai.Device(pipeline) as device:
    q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    q_depth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    print("Premi 'q' per uscire, 'p' per visualizzare la point cloud")

    while True:
        in_rgb = q_rgb.get()
        in_depth = q_depth.get()

        frame = in_rgb.getCvFrame()
        depth_frame = in_depth.getFrame()

        # Normalizza la profondità per visualizzazione
        depth_vis = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)
        depth_vis = np.uint8(depth_vis)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        # Mostra le immagini
        cv2.imshow("RGB", frame)
        cv2.imshow("Depth", depth_vis)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('p'):
            # Genera e visualizza la point cloud
            # Parametri intrinseci della camera (approssimati per 640x480)
            fx = 461.9
            fy = 461.9
            cx = 320
            cy = 240

            h, w = depth_frame.shape
            # Crea meshgrid di coordinate
            i, j = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
            z = depth_frame.astype(np.float32) / 1000.0  # da mm a metri
            x = (i - cx) * z / fx
            y = (j - cy) * z / fy

            # Maschera per punti validi
            valid = z > 0
            xyz = np.stack((x, y, z), axis=2)[valid]

            # Colori
            rgb = frame[valid]
            rgb = rgb[:, ::-1] / 255.0  # BGR->RGB e normalizza

            # Crea point cloud Open3D
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd.colors = o3d.utility.Vector3dVector(rgb)

            o3d.visualization.draw_geometries([pcd], window_name="Point Cloud")

    cv2.destroyAllWindows()
