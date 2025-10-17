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
                "-0.00179206",
                "--y",
                "-0.0547317",
                "--z",
                "0.143389",
                "--qx",
                "-0.00395043",
                "--qy",
                "-0.00693709",
                "--qz",
                "0.999968",
                "--qw",
                "0.000852198",
                # "--roll",
                # "0.0138679",
                # "--pitch",
                # "-0.00791251",
                # "--yaw",
                # "3.13994",
            ],
        ),
    ]
    return LaunchDescription(nodes)
