from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    
    # --- 1. Motor Controller Node ---
    motor_node = Node(
        package='robot_control', 
        executable='motor_control',
        name='motor_controller_node',
        output='screen',
        emulate_tty=True,
    )

    # --- 2. Waypoint Follower Node (Closed-Loop) ---
    follower_node = Node(
        package='robot_control', 
        executable='waypoint_follower',
        name='waypoint_follower_node',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'waypoints_path': 'waypoints.txt',
        }],
    )

    # --- 3. AprilTag Node ---
    apriltag_node = Node(
        package='apriltag_ros', # <<< CHANGE THIS
        executable='apriltag_node', 
        name='apriltag_node',
        output='screen',
        emulate_tty=True,
        remappings=[
            # <<< CHANGE THESE to your camera topics
            ('image_rect', '/camera/image_rect'), 
            ('camera_info', '/camera/camera_info')
        ]
    )

    # --- 4. Static Transform: Robot-to-Camera (base_link -> camera_link) ---
    robot_to_camera_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_camera_tf',
        # <<< EDIT THIS with your robot's (x, y, z, roll, pitch, yaw)
        arguments=['0.1', '0', '0.0', '0', '0', '0', 'base_link', 'camera_link'] 
    )

    # --- 5. Static Transforms: World-to-Tags (map -> tag_frame) ---
    # YOU MUST MEASURE AND ADD ONE FOR EACH TAG.
    # The 'map' frame's origin (0,0,0) is your robot's starting pose.
    # The child frame name (e.g., 'tag36h11:0') must match the
    # 'family:id' published by the AprilTagNode.

    # <<< EDIT THIS: Tag ID 0 >>>
    map_to_tag0_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_to_tag0_tf',
        # arguments: [x, y, z, roll, pitch, yaw] 'map' 'tag36h11:0'
        arguments=['2.0', '0', '0.0', '0', '0', '0.0', 'map', 'tag36h11:0'] 
    )
    
    # <<< EDIT THIS: Tag ID 1 >>>
    map_to_tag1_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_to_tag1_tf',
        # arguments: [x, y, z, roll, pitch, yaw] 'map' 'tag36h11:1'
        arguments=['1.0', '3.0', '0.0', '0', '0', '1.57', 'map', 'tag36h11:1']
    )

    # # <<< EDIT THIS: Tag ID 2 >>>
    # map_to_tag2_tf = Node(
    #     package='tf2_ros',
    #     executable='static_transform_publisher',
    #     name='map_to_tag2_tf',
    #     # arguments: [x, y, z, roll, pitch, yaw] 'map' 'tag36h11:2'
    #     arguments=['0.0', '1.0', '0.1', '0', '0', '3.14', 'map', 'tag36h11:2']
    # )
    
    # # <<< EDIT THIS: Tag ID 3 >>>
    # map_to_tag3_tf = Node(
    #     package='tf2_ros',
    #     executable='static_transform_publisher',
    #     name='map_to_tag3_tf',
    #     # arguments: [x, y, z, roll, pitch, yaw] 'map' 'tag36h11:3'
    #     arguments=['-0.5', '0.0', '0.1', '0', '0', '0', 'map', 'tag36h11:3']
    # )

    return LaunchDescription([
        motor_node,
        follower_node,
        apriltag_node,
        robot_to_camera_tf,
        
        # Add all your tag transforms to the launch
        map_to_tag0_tf,
        map_to_tag1_tf
        # map_to_tag2_tf,
        # map_to_tag3_tf
    ])