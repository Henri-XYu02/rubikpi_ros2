from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # motor_node = Node(
    #     package='robot_control',
    #     executable='motor_control',
    #     name='motor_controller_node',
    #     output='screen',
    #     emulate_tty=True,
    #     parameters=[],
    # )

    follower_node = Node(
        package='robot_control',
        executable='vision_waypoint_follower',
        name='vision_waypoint_follower_node',
        output='screen',
        emulate_tty=True,
        parameters=[{
            # Override at launch if needed
            'waypoints_path': 'waypoints.txt',
            # Example tuning overrides (optional):
            # 'kv': 1.0,
            # 'kh': 2.5,
            # 'wheel_base': 0.16,
            # 'max_speed_mps': 0.4,
        }],
    )

# --- 3. AprilTag Node ---
    # This replaces the 'ros2 run' command you were using
    apriltag_config_path = '/home/ubuntu/ros2_ws/rubikpi_ros2/apriltag_ros/cfg/tags_36h11.yaml'
    
    apriltag_node = Node(
        package='apriltag_ros', # Make sure this is your package name
        executable='apriltag_node',
        name='apriltag_node',
        output='screen',
        emulate_tty=True,
        parameters=[apriltag_config_path],
        remappings=[
            # Remap to your camera's topics
            ('image_rect', '/camera/image_raw'),
            ('camera_info', '/camera/camera_info')
        ]
    )

# --- 4. Static Transform: Robot-to-Camera (base_link -> camera_frame) ---
    # This transform now includes the rotation from ROS-frame (base_link)
    # to OpenCV optical-frame (camera_frame)
    #
    # This rotation (Yaw=-90, Pitch=+90) maps:
    #   base_link X (fwd)   -> camera_frame Z (fwd)
    #   base_link Y (left)  -> camera_frame X (right) ... as -Y
    #   base_link Z (up)    -> camera_frame Y (down) ... as -Z
    # This matches your hypothesis.
    
    robot_to_camera_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_camera_tf',
        arguments=[
            '--x', '0.1',        # Your translation x (fwd)
            '--y', '0.0',        # Your translation y (left)
            '--z', '0.05',       # Your translation z (up)
            '--yaw', '-1.5708',  # -90 degrees
            '--pitch', '0.0', # +90 degrees
            '--roll', '-1.5708',
            '--frame-id', 'base_link',
            '--child-frame-id', 'camera_frame' # Matches your camera_info
        ]
    )

    # --- 5. Static Transforms: World-to-Tags (map -> tag_X) ---
    # map_to_tag0_tf = Node(
    #     package='tf2_ros',
    #     executable='static_transform_publisher',
    #     name='map_to_tag0_tf',
    #     arguments=[
    #         '--x', '2.0',        # From your log
    #         '--y', '0.0',
    #         '--z', '0.0',
    #         '--roll', '0.0',
    #         '--pitch', '0.0',
    #         '--yaw', '3.14',
    #         '--frame-id', 'map',
    #         '--child-frame-id', 'tag_0'
    #     ]
    # )

    # map_to_tag1_tf = Node(
    #     package='tf2_ros',
    #     executable='static_transform_publisher',
    #     name='map_to_tag1_tf',
    #     arguments=[
    #         '--x', '1.0',        # From your log
    #         '--y', '3.0',
    #         '--z', '0.0',
    #         '--roll', '0.0',
    #         '--pitch', '0.0',
    #         '--yaw', '0.0',
    #         '--frame-id', 'map',
    #         '--child-frame-id', 'tag_1'
    #     ]
    # )

    # map_to_tag2_tf = Node(
    #     package='tf2_ros',
    #     executable='static_transform_publisher',
    #     name='map_to_tag2_tf',
    #     arguments=[
    #         '--x', '0.0',        # From your log
    #         '--y', '1.0',
    #         '--z', '0.0',
    #         '--roll', '0.0',
    #         '--pitch', '0.0',
    #         '--yaw', '0.0',
    #         '--frame-id', 'map',
    #         '--child-frame-id', 'tag_2'
    #     ]
    # )
    
    # map_to_tag3_tf = Node(
    #     package='tf2_ros',
    #     executable='static_transform_publisher',
    #     name='map_to_tag3_tf',
    #     arguments=[
    #         '-0.5', '0.0', '0.1', # Example: x, y, z
    #         '0', '0', '0',        # Example: roll, pitch, yaw
    #         'map', 'tag_3'        # CHECK THIS FRAME NAME
    #     ]
    # )

    # --- Add all nodes to the launch description ---
    return LaunchDescription([
        # motor_node,
        follower_node,
        apriltag_node,
        robot_to_camera_tf,
        # map_to_tag0_tf,
        # map_to_tag1_tf,
        # map_to_tag2_tf
    ])


