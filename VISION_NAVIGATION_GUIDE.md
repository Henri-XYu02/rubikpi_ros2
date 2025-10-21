# Vision-Based Navigation with YOLO Landmark Detection

This guide explains how to use the vision-based navigation system with YOLO object detection for landmark-based localization on the RubikPi robot.

## System Overview

The vision navigation system consists of four main components:

1. **Camera Node** ([robot_vision_camera.cpp](robot_vision/src/robot_vision_camera.cpp)) - Captures images from the camera
2. **YOLO Detector Node** ([yolo_detector.py](robot_control/robot_control/yolo_detector.py)) - Detects Coke cans and estimates their 3D positions
3. **Vision Waypoint Follower** ([vision_waypoint_follower.py](robot_control/robot_control/vision_waypoint_follower.py)) - Navigates waypoints with vision-based position correction
4. **Motor Controller** ([motor_control.py](robot_control/robot_control/motor_control.py)) - Sends commands to the robot motors

## Architecture

```
┌─────────────────┐
│  Camera Node    │ → /camera/image_raw
└─────────────────┘   /camera/camera_info
         │
         v
┌─────────────────┐
│ YOLO Detector   │ → /yolo/detections (distance, bearing, confidence)
└─────────────────┘   /yolo/debug_image (optional visualization)
         │
         v
┌──────────────────────────┐
│ Vision Waypoint Follower │ → /motor_commands
│  - Dead reckoning        │
│  - Vision correction     │
│  - Path following        │
└──────────────────────────┘
         │
         v
┌─────────────────┐
│ Motor Controller│ → Hardware
└─────────────────┘
```

## Prerequisites

### 1. Install Dependencies

On the RubikPi, install the required Python packages:

```bash
pip install ultralytics opencv-python numpy
```

### 2. Train or Download YOLO Model

You have a YOLO dataset in `yolo_dataset/` directory. To train a model:

```bash
cd yolo_dataset
yolo train data=data.yaml model=yolov8n.pt epochs=100 imgsz=640
```

This will create a model file at `runs/detect/train/weights/best.pt`.

Alternatively, if you already have a trained model, place it in a known location (e.g., `~/models/coke_detector.pt`).

### 3. Calibrate Camera

The camera is already calibrated - parameters are in [robot_vision/config/camera_parameter.yaml](robot_vision/config/camera_parameter.yaml:1-28).

Key parameters:
- **Resolution:** 1280×720
- **Focal length:** fx=490.98, fy=492.13 pixels
- **Principal point:** cx=639.43, cy=357.23 pixels

If you need to recalibrate:
```bash
# Use ROS camera calibration package or OpenCV calibration tools
ros2 run camera_calibration cameracalibrator --size 8x6 --square 0.024 image:=/camera/image_raw
```

## Setup Instructions

### 1. Prepare Landmark Map

Place Coke cans at known locations in your environment and measure their positions. Update [robot_control/config/landmarks.json](robot_control/config/landmarks.json):

```json
{
  "landmarks": [
    {"id": 0, "x": 1.0, "y": 0.5, "description": "First landmark"},
    {"id": 1, "x": 2.0, "y": 1.0, "description": "Second landmark"}
  ]
}
```

**Coordinate system:**
- Origin (0, 0) = Robot's starting position
- X-axis = Forward direction
- Y-axis = Left direction
- Units = meters

**How to measure:**
1. Place robot at starting position facing forward
2. Place Coke cans at measured positions
3. Use a measuring tape to get x (forward) and y (left) distances
4. Update landmarks.json with actual positions

### 2. Create Waypoints

Create a waypoints file (e.g., `waypoints.txt`) with format: `x, y, theta`

Example [robot_control/config/waypoints_example.txt](robot_control/config/waypoints_example.txt):
```
# x, y, theta (in meters and radians)
1.0, 0.0, 0.0      # Move forward 1m
1.0, 0.5, 1.57     # Move to (1.0, 0.5), face left
2.0, 1.0, 0.0      # Move to (2.0, 1.0), face forward
0.0, 0.0, 0.0      # Return to start
```

### 3. Build the Package

```bash
cd ~/ros2_ws  # or your workspace
colcon build --packages-select robot_control
source install/setup.bash
```

## Running the System

### Option 1: Full System Launch (Recommended)

Launch all components together:

```bash
ros2 launch robot_control vision_navigation_launch.py \
  yolo_model_path:=~/models/best.pt \
  waypoints_path:=waypoints.txt \
  landmarks_path:=landmarks.json \
  use_vision_correction:=true \
  publish_debug_image:=true
```

### Option 2: Individual Nodes (For Testing)

Launch components separately in different terminals:

**Terminal 1: Camera**
```bash
ros2 launch robot_vision robot_vision_camera.launch.py
```

**Terminal 2: YOLO Detector**
```bash
ros2 run robot_control yolo_detector \
  --ros-args \
  -p model_path:=~/models/best.pt \
  -p confidence_threshold:=0.6 \
  -p target_class:=Coke \
  -p publish_debug_image:=true
```

**Terminal 3: Vision Waypoint Follower**
```bash
ros2 run robot_control vision_waypoint_follower \
  --ros-args \
  -p waypoints_path:=waypoints.txt \
  -p landmarks_path:=landmarks.json \
  -p use_vision_correction:=true \
  -p vision_correction_weight:=0.3
```

**Terminal 4: Motor Controller**
```bash
ros2 run robot_control motor_control
```

## Configuration Parameters

### YOLO Detector Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_path` | `yolov8n.pt` | Path to YOLO model weights |
| `confidence_threshold` | `0.5` | Minimum detection confidence (0-1) |
| `target_class` | `Coke` | Object class to detect |
| `known_object_height` | `0.123` | Coke can height in meters (330ml = 12.3cm) |
| `detection_rate_hz` | `5.0` | Detection processing rate |
| `publish_debug_image` | `true` | Publish annotated images |

### Vision Waypoint Follower Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_vision_correction` | `true` | Enable landmark-based position correction |
| `vision_correction_weight` | `0.3` | How much to trust vision (0-1) |
| `min_detection_confidence` | `0.6` | Minimum confidence to use for correction |
| `max_landmark_distance` | `3.0` | Ignore landmarks beyond this distance (m) |
| `pos_tolerance` | `0.05` | Waypoint reached threshold (m) |
| `yaw_tolerance` | `0.1` | Heading tolerance (rad, ~6°) |
| `max_speed_mps` | `0.72` | Maximum linear speed (m/s) |
| `use_imu` | `true` | Use IMU for orientation |

## Monitoring and Debugging

### View Camera Images

```bash
# Raw camera image
ros2 run rqt_image_view rqt_image_view /camera/image_raw

# YOLO debug image with detections
ros2 run rqt_image_view rqt_image_view /yolo/debug_image
```

### Monitor Detection Data

```bash
ros2 topic echo /yolo/detections
```

Output format: `[num_detections, x1, y1, dist1, bearing1, conf1, x2, y2, dist2, bearing2, conf2, ...]`

### View Robot Position

Check the vision waypoint follower logs for position estimates:
```bash
ros2 node info /vision_waypoint_follower_node
```

### Use Foxglove for Visualization

```bash
ros2 launch foxglove_bridge foxglove_bridge_launch.xml
# Open browser to http://localhost:8765
```

## How It Works

### 1. Camera Acquisition

The camera node captures images at 10 FPS using GStreamer:
- Input format: NV12 from qtiqmmfsrc
- Output: RGB8 raw images + JPEG compressed
- Publishes camera intrinsics for 3D calculations

See [robot_vision_camera.cpp:379-463](robot_vision/src/robot_vision_camera.cpp#L379-L463) for implementation.

### 2. YOLO Detection

The YOLO detector:
1. Subscribes to camera images
2. Runs YOLOv8 inference at 5 Hz
3. Filters detections for "Coke" class
4. Estimates 3D position using camera intrinsics

**Distance estimation:**
```python
distance = (known_height * focal_length_y) / bbox_height_pixels
# For 330ml Coke can: height = 0.123m
```

**Bearing estimation:**
```python
bearing = atan2(pixel_x - cx, fx)
```

See [yolo_detector.py:210-235](robot_control/robot_control/yolo_detector.py#L210-L235) for 3D position calculation.

### 3. Landmark-Based Localization

When a Coke can is detected:

1. Calculate where the landmark would be in world coordinates based on current position estimate
2. Find the closest known landmark in the map
3. If match is found (within 0.5m), calculate corrected robot position:
   ```python
   x_corrected = landmark_x_true - distance * cos(global_bearing)
   y_corrected = landmark_y_true - distance * sin(global_bearing)
   ```
4. Blend with dead reckoning estimate using `vision_correction_weight`

See [vision_waypoint_follower.py:203-248](robot_control/robot_control/vision_waypoint_follower.py#L203-L248) for localization algorithm.

### 4. Waypoint Following

The controller uses a simple state machine:
- **Far from waypoint:** Rotate to face target, then drive forward
- **Near waypoint (< pos_tolerance):** Mark as reached, move to next
- **All waypoints complete:** Stop

Control law:
```python
if distance < pos_tolerance:
    # Waypoint reached
    v = 0, w = 0
elif abs(heading_error) > yaw_tolerance:
    # Rotate in place
    v = 0, w = max_angular_speed * sign(heading_error)
else:
    # Drive forward with heading correction
    v = max_speed
    w = max_angular_speed * heading_error / pi
```

## Tuning Tips

### Improve Detection Accuracy

1. **Retrain YOLO with more data** - Add images from your actual environment
2. **Adjust lighting** - Ensure good illumination, avoid shadows
3. **Increase confidence threshold** - Set to 0.7-0.8 for higher precision
4. **Calibrate object height** - Measure your actual Coke can height

### Improve Localization

1. **Place landmarks strategically** - Distribute evenly around the path
2. **Increase vision_correction_weight** - If vision is reliable (0.4-0.6)
3. **Decrease vision_correction_weight** - If detections are noisy (0.1-0.2)
4. **Use IMU** - Improves orientation estimate (set `use_imu:=true`)

### Improve Path Following

1. **Reduce speed** - Lower `max_speed_mps` for sharper turns
2. **Tighten tolerances** - Smaller `pos_tolerance` for precise navigation
3. **Increase control rate** - Higher `rate_hz` for smoother control

## Troubleshooting

### No Detections

- **Check camera:** `ros2 topic echo /camera/image_raw --once`
- **Check YOLO model:** Verify model path is correct
- **Check class name:** Ensure `target_class` matches your dataset (case-sensitive!)
- **Lower confidence:** Try `confidence_threshold:=0.3`

### Inaccurate Distance Estimation

- **Verify object height:** Measure your Coke can and update `known_object_height`
- **Check camera calibration:** Verify focal lengths in camera_parameter.yaml
- **Test at known distances:** Place can at 1m, 2m, 3m and check estimates

### Robot Drifts Off Path

- **Enable vision correction:** `use_vision_correction:=true`
- **Add more landmarks:** Place more Coke cans along the path
- **Check landmark positions:** Verify landmarks.json coordinates are accurate
- **Use IMU:** `use_imu:=true` for better orientation

### Vision Corrections Too Aggressive

- **Reduce weight:** `vision_correction_weight:=0.1`
- **Increase confidence threshold:** `min_detection_confidence:=0.8`
- **Reduce max distance:** `max_landmark_distance:=2.0`

## File Reference

### Source Files

- [yolo_detector.py](robot_control/robot_control/yolo_detector.py) - YOLO detection node
- [vision_waypoint_follower.py](robot_control/robot_control/vision_waypoint_follower.py) - Vision-based navigation
- [robot_vision_camera.cpp](robot_vision/src/robot_vision_camera.cpp) - Camera acquisition
- [vision_navigation_launch.py](robot_control/launch/vision_navigation_launch.py) - Launch file

### Configuration Files

- [landmarks.json](robot_control/config/landmarks.json) - Landmark map
- [waypoints_example.txt](robot_control/config/waypoints_example.txt) - Example waypoints
- [camera_parameter.yaml](robot_vision/config/camera_parameter.yaml) - Camera calibration
- [data.yaml](yolo_dataset/data.yaml) - YOLO dataset configuration

## Next Steps

1. **Train your YOLO model** using the dataset in `yolo_dataset/`
2. **Set up your environment** - Place Coke cans at measured positions
3. **Create landmarks.json** - Record actual landmark positions
4. **Create waypoints.txt** - Define your navigation path
5. **Test incrementally:**
   - First: Camera only
   - Second: Camera + YOLO detector
   - Third: Add waypoint follower without vision correction
   - Finally: Enable vision correction

## Advanced Features (Future Enhancements)

Potential improvements you could add:

1. **Multi-landmark trilateration** - Use multiple landmarks simultaneously for better accuracy
2. **Extended Kalman Filter (EKF)** - Fuse dead reckoning, IMU, and vision optimally
3. **Loop closure detection** - Correct accumulated drift when revisiting landmarks
4. **Dynamic landmark mapping** - Build map while navigating (SLAM)
5. **Obstacle avoidance** - Use YOLO to detect and avoid obstacles
6. **Path planning** - A* or RRT for complex environments

## Support

For issues or questions:
- Check logs: `ros2 node info /yolo_detector_node`
- Review camera calibration: [camera_parameter.yaml](robot_vision/config/camera_parameter.yaml)
- Verify YOLO training: Check dataset statistics in `yolo_dataset/`

Happy navigating! 🤖🥤
