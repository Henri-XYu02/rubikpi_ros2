from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """
    Launch file for hw_1_solution closed-loop package with AprilTag detection.
    Starts four nodes in sequence:
    1. motor_control - communicates with robot hardware via serial
    2. velocity_mapping - converts Twist commands to motor commands
    3. apriltag_ros (external) - must be launched separately or added here
    4. hw1_closedloop - Closed-loop PID-based waypoint navigation with AprilTag correction

    Note: You need to launch the camera and apriltag_ros nodes separately:
        Terminal 1: ros2 launch robot_vision robot_vision_camera.launch.py
        Terminal 2: ros2 launch apriltag_ros apriltag_launch.py
        Terminal 3: ros2 launch hw_1_solution hw1_closedloop.launch.py
    """

    # Node 1: Motor Controller (communicates with robot hardware)
    motor_controller = Node(
        package='hw_1_solution',
        executable='motor_control',
        name='motor_control',
        output='screen',
        emulate_tty=True,
    )

    # Node 2: Velocity to Motor Mapping
    velocity_mapping = Node(
        package='hw_1_solution',
        executable='velocity_mapping',
        name='velocity_mapping',
        output='screen',
        emulate_tty=True,
    )

    # Node 3: HW1 Closed-Loop Solution (Waypoint Navigation with AprilTag Correction)
    hw1_closedloop = Node(
        package='hw_1_solution',
        executable='hw1_closedloop',
        name='hw1_closedloop',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'use_apriltag_correction': True,
            'apriltag_correction_weight': 0.5,
            'min_detection_confidence': 0.5,
            'max_tag_distance': 3.0,
        }]
    )

    return LaunchDescription([
        motor_controller,
        velocity_mapping,
        hw1_closedloop,
    ])
