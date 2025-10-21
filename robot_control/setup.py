from setuptools import find_packages, setup

package_name = 'robot_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
            'launch/robot_teleop_launch.py',
            'launch/waypoint_follow_launch.py',
            'launch/vision_navigation_launch.py'
        ]),
        ('share/' + package_name + '/config', [
            'config/landmarks.json',
            'config/waypoints_example.txt'
        ]),
    ],
    install_requires=[
        'setuptools',
        'pyserial>=3.5',
        'pynput>=1.7.0',
        'numpy>=1.20.0',
        'opencv-python>=4.5.0',
        'ultralytics>=8.0.0',
    ],
    zip_safe=True,
    maintainer='root',
    maintainer_email='chl235@ucsd.edu',
    description='Robot motor controller with keyboard input for ROS2',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'motor_control = robot_control.motor_control:main',
            'keyboard_control = robot_control.keyboard_control:main',
            'velocity_control = robot_control.velocity_control:main',
            'waypoint_follower = robot_control.waypoint_follower:main',
            'yolo_detector = robot_control.yolo_detector:main',
            'vision_waypoint_follower = robot_control.vision_waypoint_follower:main',
        ],
    },
)