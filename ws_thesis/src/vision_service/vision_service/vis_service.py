#!/home/edo/ros2_venv/bin/python

import rclpy
from rclpy.node import Node
import numpy as np
import torch
from interfaces.srv import Perception
from vision_service.cont_pos_vol_est import (
    oak_capture,
    CNN_predict,
    filter_and_scale_points,
    filter_and_scale_depth,
    preprocess,
    VolumeNetPret,
)

class PerceptionService(Node):
    def __init__(self):
        super().__init__('perception_service')

        # Parametri per percorsi modello (override da CLI/params se vuoi)
        self.declare_parameter('vol_model_path', '/home/edo/thesis/LiquidGenesis/vol_est/checkpoints/best_model_ResNet_1.pth')
        self.declare_parameter('target_size_h', 256)
        self.declare_parameter('target_size_w', 192)

        # Precarica modello volume
        vol_model_path = self.get_parameter('vol_model_path').get_parameter_value().string_value
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = VolumeNetPret(backbone_name='ResNet18', input_channels=4, pretrained=True).to(self.device)
        ckpt = torch.load(vol_model_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()

        self.srv = self.create_service(Perception, 'estimate_perception', self.callback)
        self.get_logger().info("Vision service ready")

    def callback(self, request: Perception.Request, response: Perception.Response):
        try:
            # Cattura img
            rgb_frame, ptcd_points, ptcd_colors = oak_capture()

            # Segmentazione (classe 'Vessel' per isolare il contenitore)
            prd = CNN_predict(rgb_frame, target_class='Vessel')

            # Point cloud filtrata + centroide
            filt_pts, _ = filter_and_scale_points(ptcd_points, ptcd_colors, prd)
            centroid = np.mean(filt_pts, axis=0).astype(float)
            response.centroid = [centroid[0], centroid[1], centroid[2]]

            # Volume (opzionale)
            volume = 0.0
            if request.estimate_volume:
                f_content, f_vessel, c_mask, v_mask = filter_and_scale_depth(prd)
                H = self.get_parameter('target_size_h').get_parameter_value().integer_value
                W = self.get_parameter('target_size_w').get_parameter_value().integer_value
                masks_tensor = preprocess(f_content, f_vessel, c_mask, v_mask, target_size=(H, W)).to(self.device)
                with torch.no_grad():
                    volume = float(self.model(masks_tensor).item())

            response.volume = volume
            response.success = True
            response.message = 'ok'
        except Exception as e:
            response.centroid = [0.0, 0.0, 0.0]
            response.volume = 0.0
            response.success = False
            response.message = str(e)
        return response

def main():
    rclpy.init()
    node = PerceptionService()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
