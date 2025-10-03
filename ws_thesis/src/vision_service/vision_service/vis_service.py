#!/home/edo/ros2_venv/bin/python

import rclpy
import rclpy.time
from rclpy.node import Node
import numpy as np
import torch
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped, Point
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
        self.get_logger().info("Vision service starting")
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

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread = True)

        self.srv = self.create_service(Perception, 'estimate_perception', self.callback)
        self.get_logger().info("Vision service ready")

    def callback(self, request: Perception.Request, response: Perception.Response):
        try:
            # Cattura img
            rgb_frame, ptcd_points, ptcd_colors = oak_capture()
            print("Acquisizione Completata")
            # Segmentazione (classe 'Vessel' per isolare il contenitore)
            prd = CNN_predict(rgb_frame, target_class='Vessel')

            # Point cloud filtrata + centroide
            filt_pts, _ = filter_and_scale_points(ptcd_points, ptcd_colors, prd)
            #print(f"punti filt:{filt_pts}")
            if filt_pts is None or len(filt_pts) == 0:
                centroid=np.array([0.0,0.0,0.0])
            else:
                centroid = np.mean(filt_pts, axis=0).astype(float)
            #print(f"centroid {centroid}")


            point_msg = PointStamped()
            point_msg.header.stamp = self.get_clock().now().to_msg()   # timestamp
            point_msg.header.frame_id = "camera_frame"                 # frame di riferimento

            point_msg.point = Point(x=float(centroid[0]),
                                    y=float(centroid[1]),
                                    z=float(centroid[2]))

            # Trasformazione da camera_frame a world:
            to_frame_rel = 'world'
            from_frame_rel = 'camera_frame'
            time=rclpy.time.Time() 
            # Wait for the transform asynchronously
            tf_future = self.buffer.wait_for_transform_async(
            target_frame=to_frame_rel,
            source_frame=from_frame_rel,
            time=time
            )
            rclpy.spin_until_future_complete(self, tf_future, timeout_sec=1)

            # Lookup tansform
            try:
                t = self.buffer.lookup_transform(to_frame_rel,
                                                from_frame_rel,
                                                time)
                # Do the transform
                transformed_point_msg = tf2_geometry_msgs.do_transform_point(point_msg, t)
            except Exception as e:
                self.get_logger().info(f"No transform found: {str(e)}")
                return

            response.centroid = [transformed_point_msg.point.x, transformed_point_msg.point.y, transformed_point_msg.point.z]

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
