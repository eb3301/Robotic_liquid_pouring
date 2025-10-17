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
                "-0.000566925",
                "--y",
                "-0.0567256",
                "--z",
                "0.143985",
                "--qx",
                "-0.00438874",
                "--qy",
                "-0.00309819",
                "--qz",
                "0.999984",
                "--qw",
                "0.00159895",
                # "--roll",
                # "0.00618254",
                # "--pitch",
                # "-0.00878736",
                # "--yaw",
                # "3.13842",
            ],
        ),
    ]
    return LaunchDescription(nodes)
