#!/usr/bin/env python3
"""
Launch file for vision-based navigation system
Launches camera, YOLO detector, vision waypoint follower, and motor controller
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os


def generate_launch_description():
    # Declare launch arguments
    yolo_model_path_arg = DeclareLaunchArgument(
        'yolo_model_path',
        default_value='yolov8n.pt',
        description='Path to YOLO model weights file (e.g., best.pt from training)'
    )

    waypoints_path_arg = DeclareLaunchArgument(
        'waypoints_path',
        default_value='waypoints.txt',
        description='Path to waypoints file'
    )

    landmarks_path_arg = DeclareLaunchArgument(
        'landmarks_path',
        default_value='landmarks.json',
        description='Path to landmarks configuration file'
    )

    use_vision_correction_arg = DeclareLaunchArgument(
        'use_vision_correction',
        default_value='true',
        description='Enable vision-based position correction'
    )

    camera_enabled_arg = DeclareLaunchArgument(
        'camera_enabled',
        default_value='true',
        description='Launch camera node'
    )

    publish_debug_image_arg = DeclareLaunchArgument(
        'publish_debug_image',
        default_value='true',
        description='Publish debug images with YOLO detections'
    )

    # Get launch configurations
    yolo_model_path = LaunchConfiguration('yolo_model_path')
    waypoints_path = LaunchConfiguration('waypoints_path')
    landmarks_path = LaunchConfiguration('landmarks_path')
    use_vision_correction = LaunchConfiguration('use_vision_correction')
    camera_enabled = LaunchConfiguration('camera_enabled')
    publish_debug_image = LaunchConfiguration('publish_debug_image')

    # Camera node (include camera launch file if available)
    # You can replace this with your camera launch file
    camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('robot_vision'),
                'launch',
                'robot_vision_camera.launch.py'
            ])
        ]),
        condition=lambda context: context.perform_substitution(camera_enabled).lower() == 'true'
    )

    # YOLO detector node
    yolo_detector_node = Node(
        package='robot_control',
        executable='yolo_detector',
        name='yolo_detector',
        output='screen',
        parameters=[{
            'model_path': yolo_model_path,
            'confidence_threshold': 0.6,
            'target_class': 'Coke',
            'known_object_height': 0.123,  # Coke can height in meters
            'image_topic': '/camera/image_raw',
            'camera_info_topic': '/camera/camera_info',
            'publish_debug_image': publish_debug_image,
            'detection_rate_hz': 5.0,
        }],
        remappings=[
            ('/camera/image_raw', '/camera/image_raw'),
            ('/camera/camera_info', '/camera/camera_info'),
        ]
    )

    # Vision waypoint follower node
    vision_waypoint_follower_node = Node(
        package='robot_control',
        executable='vision_waypoint_follower',
        name='vision_waypoint_follower',
        output='screen',
        parameters=[{
            'waypoints_path': waypoints_path,
            'landmarks_path': landmarks_path,
            'wheel_base': 0.04,
            'car_width': 0.168,
            'max_speed_mps': 0.72,
            'max_angular_speed_radps': 3.14159,
            'max_norm_cmd': 0.5,
            'kv': 0.8,
            'kh': 0.8,
            'pos_tolerance': 0.05,
            'yaw_tolerance': 0.1,
            'rate_hz': 20.0,
            'use_vision_correction': use_vision_correction,
            'vision_correction_weight': 0.3,
            'min_detection_confidence': 0.6,
            'max_landmark_distance': 3.0,
            'use_imu': True,
        }]
    )

    # Motor controller node
    motor_control_node = Node(
        package='robot_control',
        executable='motor_control',
        name='motor_control',
        output='screen',
        parameters=[{
            'serial_port': '/dev/ttyTHS1',
            'baudrate': 115200,
        }]
    )

    return LaunchDescription([
        # Launch arguments
        yolo_model_path_arg,
        waypoints_path_arg,
        landmarks_path_arg,
        use_vision_correction_arg,
        camera_enabled_arg,
        publish_debug_image_arg,

        # Nodes
        camera_launch,
        yolo_detector_node,
        vision_waypoint_follower_node,
        motor_control_node,
    ])