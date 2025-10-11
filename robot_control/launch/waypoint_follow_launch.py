from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    motor_node = Node(
        package='robot_control',
        executable='motor_control',
        name='motor_controller_node',
        output='screen',
        emulate_tty=True,
        parameters=[],
    )

    follower_node = Node(
        package='robot_control',
        executable='waypoint_follower',
        name='waypoint_follower_node',
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

    return LaunchDescription([
        motor_node,
        follower_node,
    ])


