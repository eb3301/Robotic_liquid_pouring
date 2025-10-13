""" Static transform publisher acquired via MoveIt 2 hand-eye calibration """
""" EYE-IN-HAND: tool0 -> oak_rgb_camera_optical_frame """
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    nodes = [
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            output="log",
            arguments=[
                "--frame-id",
                "tool0",
                "--child-frame-id",
                "oak_rgb_camera_optical_frame",
                "--x",
                "0.00247969",
                "--y",
                "-0.0556562",
                "--z",
                "0.164238",
                "--qx",
                "0.017052",
                "--qy",
                "0.0984414",
                "--qz",
                "0.991566",
                "--qw",
                "0.0825533",
                # "--roll",
                # "2.94773",
                # "--pitch",
                # "3.0915",
                # "--yaw",
                # "-0.161256",
            ],
        ),
    ]
    return LaunchDescription(nodes)
