ros2 launch robot_vision_camera robot_vision_camera.launch.py

ros2 run apriltag_ros apriltag_node --ros-args \
-r image_rect:=/camera/image_raw \
-r camera_info:=/camera/camera_info \
--params-file /home/ubuntu/ros2_ws/rubikpi_ros2/apriltag_ros/cfg/tags_36h11.yaml

ros2 launch apriltag_ros apriltag_launch.py

ros2 run hw_1_solution hw1_closedloop
ros2 run hw_1_solution velocity_mapping

or

ros2 run robot_control vision_waypoint_follower

ros2 run hw_1_solution motor_control