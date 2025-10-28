# AprilTag-Based Navigation Guide

Complete guide for using AprilTag markers for precise robot localization and waypoint navigation.

## Overview

AprilTag-based navigation provides **more accurate localization** than YOLO detection because:
- AprilTags provide full 6-DOF pose (position + orientation)
- Detection is robust to lighting and viewing angle
- Sub-millimeter accuracy possible with proper calibration
- No ML training required - works out of the box

## System Architecture

```
┌─────────────────┐
│  Camera Node    │ → /camera/image_raw
└─────────────────┘   /camera/camera_info
         │
         v
┌─────────────────┐
│ AprilTag Node   │ → /detections (AprilTagDetectionArray)
└─────────────────┘   /tf (TF2 transforms: camera -> tag_N)
         │
         v
┌───────────────────────────┐
│ AprilTag Waypoint Follower│
│  - Subscribes to TF2      │ → /motor_commands
│  - Corrects position      │
│  - Follows waypoints      │
└───────────────────────────┘
         │
         v
┌─────────────────┐
│ Motor Controller│
└─────────────────┘
```

## Prerequisites

### 1. Print AprilTags

**Download and print AprilTag markers:**

```bash
# Clone AprilTag image generator
git clone https://github.com/AprilRobotics/apriltag-imgs.git

# Or download pre-generated images
# https://github.com/AprilRobotics/apriltag-imgs/tree/master/tag36h11
```

**Print guidelines:**
- Use **tag36h11 family** (configured in your system)
- Print on white paper with black ink
- Ensure high contrast and sharp edges
- Mount on rigid surface (foam board, cardboard)
- Measure the tag size accurately (outer black border edge to edge)

**Default tag size:** 165mm (0.165m) - configured in [apriltag_ros/cfg/tags_36h11.yaml](apriltag_ros/cfg/tags_36h11.yaml:5)

**Recommended tags to print:**
- Tag ID 0, 1, 2, 3, 4 (at minimum)
- More tags = better localization coverage

### 2. Update AprilTag Configuration

Edit [apriltag_ros/cfg/tags_36h11.yaml](apriltag_ros/cfg/tags_36h11.yaml) to include your tag IDs:

```yaml
/**:
    ros__parameters:
        family: 36h11
        size: 0.165             # Your measured tag size in meters
        max_hamming: 1

        detector:
            threads: 4
            decimate: 2.0
            blur: 0.0
            refine: True
            sharpening: 0.4
            debug: False

        pose_estimation_method: "pnp"

        tag:
            ids: [0, 1, 2, 3, 4]           # Add all your tag IDs
            frames: [tag_0, tag_1, tag_2, tag_3, tag_4]
            sizes: [0.165, 0.165, 0.165, 0.165, 0.165]  # Individual sizes if different
```

## Setup Process

### Step 1: Place AprilTags in Environment

1. **Position tags strategically:**
   - Distribute along your planned path
   - Place at different heights (robot camera level ±20cm)
   - Ensure tags are visible from multiple angles
   - Avoid placing behind obstacles

2. **Orient tags properly:**
   - Tags should face the areas where robot will travel
   - Tilting slightly downward can help with visibility

3. **Measure positions accurately:**
   - Use measuring tape from robot starting position
   - Record X (forward), Y (left) coordinates in meters
   - Mark starting position clearly

**Example placement:**
```
        Y (left)
        ^
        |
   Tag1 •  (2.0, 1.0)
        |
   Tag0 •  (1.0, 0.5)
        |
        +----------> X (forward)
      (0,0) Robot Start
```

### Step 2: Create AprilTag Map

Edit [robot_control/config/apriltag_map.json](robot_control/config/apriltag_map.json):

```json
{
  "tags": [
    {
      "id": 0,
      "x": 1.0,
      "y": 0.5,
      "yaw": 0.0,
      "description": "Near wall, facing forward"
    },
    {
      "id": 1,
      "x": 2.0,
      "y": 1.0,
      "yaw": 1.57,
      "description": "Corner position, facing left"
    },
    {
      "id": 2,
      "x": 1.5,
      "y": 2.0,
      "yaw": 3.14,
      "description": "Far wall, facing backward"
    }
  ]
}
```

**Coordinate system:**
- **Origin (0,0):** Robot starting position
- **X-axis:** Forward direction (robot front)
- **Y-axis:** Left direction (robot left side)
- **Yaw:** Tag orientation in radians
  - 0 = facing forward (same as robot start)
  - π/2 (1.57) = facing left
  - π (3.14) = facing backward
  - 3π/2 (4.71) = facing right

### Step 3: Create Waypoints

Create [waypoints.txt](robot_control/config/waypoints_example.txt):

```
# Format: x, y, theta (in meters and radians)
# Lines starting with # are comments

# Move forward toward tag 0
1.0, 0.3, 0.0

# Move near tag 1
2.0, 0.8, 1.57

# Return to start
0.0, 0.0, 0.0
```

**Tips for waypoint placement:**
- Place waypoints where at least one AprilTag is visible
- Avoid waypoints too far from any tag (>3m)
- Robot should see tags from waypoint position

### Step 4: Build and Test

```bash
cd ~/ros2_ws
colcon build --packages-select robot_control apriltag_ros
source install/setup.bash
```

## Running the System

### Full System Launch

```bash
ros2 launch robot_control apriltag_navigation_launch.py \
  waypoints_path:=waypoints.txt \
  apriltag_map_path:=apriltag_map.json \
  use_apriltag_correction:=true \
  apriltag_correction_weight:=0.5
```

**Parameters:**
- `waypoints_path`: Path to waypoints file
- `apriltag_map_path`: Path to AprilTag map JSON
- `use_apriltag_correction`: Enable AprilTag localization (true/false)
- `apriltag_correction_weight`: How much to trust AprilTags (0-1)
  - 0.0 = Pure dead reckoning (no correction)
  - 0.5 = Balanced (recommended)
  - 1.0 = Trust AprilTags completely

### Manual Node Launch (for debugging)

**Terminal 1: Camera**
```bash
ros2 launch robot_vision robot_vision_camera.launch.py
```

**Terminal 2: AprilTag Detector**
```bash
ros2 launch apriltag_ros apriltag_launch.py
```

**Terminal 3: AprilTag Waypoint Follower**
```bash
ros2 run robot_control apriltag_waypoint_follower \
  --ros-args \
  -p waypoints_path:=waypoints.txt \
  -p apriltag_map_path:=apriltag_map.json \
  -p use_apriltag_correction:=true \
  -p apriltag_correction_weight:=0.5
```

**Terminal 4: Motor Controller**
```bash
ros2 run robot_control motor_control
```

## Monitoring and Debugging

### View AprilTag Detections

```bash
# View detection messages
ros2 topic echo /detections

# Output shows detected tag IDs, corners, and decision margin (confidence)
```

### View TF Transforms

```bash
# List all transforms
ros2 run tf2_ros tf2_echo camera_frame tag_0

# This shows the transform from camera to tag 0
# Translation: x, y, z (meters)
# Rotation: quaternion (orientation)
```

### Visualize in RViz

```bash
ros2 run rviz2 rviz2
```

**Add displays:**
1. **TF** - Shows all coordinate frames
2. **Image** - Subscribe to `/camera/image_raw`
3. **Camera** - Shows camera frustum

**Set fixed frame to:** `camera_frame` or `world`

### Monitor Position Corrections

Check the logs from `apriltag_waypoint_follower`:

```
[INFO] AprilTag 0 correction: Pos=(1.023,0.487), Dist=0.98m, Bearing=12.3°
[INFO] WP1/4: Pos=(1.02,0.49), Yaw=5.2°, Dist=0.05m, Corr=3
```

- **Corr=3** means 3 AprilTag corrections have been applied

## How It Works

### 1. AprilTag Detection

The AprilTag detector ([AprilTagNode.cpp](apriltag_ros/src/AprilTagNode.cpp)):
1. Subscribes to camera images
2. Detects AprilTag markers (36h11 family)
3. Estimates 6-DOF pose using PnP algorithm
4. Publishes TF transforms: `camera_frame` → `tag_N`

### 2. Localization Algorithm

When a tag is detected ([apriltag_waypoint_follower.py:328-355](robot_control/robot_control/apriltag_waypoint_follower.py#L328-L355)):

```python
# 1. Get tag position in camera frame from TF
tag_x_cam, tag_y_cam, tag_z_cam = lookup_tf(camera_frame, tag_N)

# 2. Calculate bearing and distance
bearing_cam = atan2(tag_y_cam, tag_x_cam)
distance = sqrt(tag_x_cam^2 + tag_y_cam^2)

# 3. Get known tag position from map
tag_x_world, tag_y_world = apriltag_map[tag_id]

# 4. Calculate global bearing
global_bearing = robot_yaw + bearing_cam

# 5. Compute robot position
robot_x_corrected = tag_x_world - distance * cos(global_bearing)
robot_y_corrected = tag_y_world - distance * sin(global_bearing)

# 6. Weighted fusion with dead reckoning
robot_x = (1 - weight) * x_dead_reckoning + weight * x_corrected
robot_y = (1 - weight) * y_dead_reckoning + weight * y_corrected
```

### 3. Control Loop

Standard waypoint following with corrected position:
1. Compute error to current waypoint
2. Generate velocity commands
3. Send to motors
4. Integrate velocities (dead reckoning)
5. Apply AprilTag corrections when tags are visible

## Configuration Parameters

### AprilTag Waypoint Follower Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_apriltag_correction` | `true` | Enable AprilTag-based localization |
| `apriltag_correction_weight` | `0.5` | Correction weight (0-1) |
| `min_detection_confidence` | `0.5` | Min decision margin to use detection |
| `max_tag_distance` | `3.0` | Ignore tags beyond this distance (m) |
| `pos_tolerance` | `0.05` | Waypoint reached threshold (m) |
| `yaw_tolerance` | `0.1` | Heading tolerance (rad) |
| `max_speed_mps` | `0.72` | Maximum linear speed (m/s) |
| `camera_frame` | `camera_frame` | TF frame name for camera |
| `base_frame` | `base_link` | TF frame name for robot base |
| `world_frame` | `world` | TF frame name for world |
| `use_imu` | `true` | Use IMU for orientation |

### AprilTag Detector Parameters

Edit [apriltag_ros/cfg/tags_36h11.yaml](apriltag_ros/cfg/tags_36h11.yaml):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `size` | `0.165` | Tag edge size in meters |
| `max_hamming` | `1` | Max bit errors tolerated |
| `threads` | `4` | Number of detector threads |
| `decimate` | `2.0` | Image decimation for speed |
| `refine` | `True` | Refine corner positions |
| `pose_estimation_method` | `pnp` | Use PnP for pose (more accurate) |

## Tuning Guide

### Improve Localization Accuracy

**1. Increase AprilTag correction weight (0.5 → 0.7)**
```bash
ros2 param set /apriltag_waypoint_follower_node apriltag_correction_weight 0.7
```
- **Effect:** Trust AprilTags more, dead reckoning less
- **When:** AprilTags are highly accurate and always visible

**2. Place more AprilTags**
- Add tags every 1-2 meters along path
- Ensures robot always sees at least one tag
- Redundancy improves robustness

**3. Calibrate camera precisely**
- See [robot_vision/CAMERA_CALIBRATION_GUIDE.md](robot_vision/CAMERA_CALIBRATION_GUIDE.md)
- Better calibration = more accurate tag pose estimation
- Target < 0.5 pixel reprojection error

**4. Increase tag size**
- Larger tags = better detection at distance
- Standard: 165mm, can use up to 300mm
- Update `size` parameter in apriltag config

### Handle Difficult Lighting

**1. Adjust camera brightness**
```yaml
# In robot_vision_camera.launch.py
brightness: 0.05   # Increase for dark environments
contrast: 1.2      # Increase for low contrast
```

**2. Reduce decimation**
```yaml
# In tags_36h11.yaml
decimate: 1.0   # Process full resolution (slower but more accurate)
```

**3. Enable sharpening**
```yaml
sharpening: 0.6   # Enhance edges (helps with blur)
```

### Speed Up Detection

**1. Increase decimation**
```yaml
decimate: 3.0   # Faster but less accurate at distance
```

**2. Reduce threads**
```yaml
threads: 2   # Use fewer CPU cores
```

**3. Lower resolution**
- Reduce camera resolution to 640x480
- Faster processing, but reduced range

## Troubleshooting

### No AprilTag Detections

**Check:**
```bash
# Are tags being detected?
ros2 topic echo /detections

# Are TF transforms published?
ros2 run tf2_ros tf2_echo camera_frame tag_0
```

**Solutions:**
1. **Verify tag IDs match configuration**
   - Check printed tag ID matches `tags_36h11.yaml`

2. **Check tag size**
   - Measure tag edge-to-edge (outer black square)
   - Update `size` parameter if different from 165mm

3. **Improve lighting**
   - Ensure even lighting (no shadows on tags)
   - Avoid glare or reflections

4. **Check camera focus**
   - Tags must be in focus
   - Move closer if tags are blurry

### Poor Localization (Large Position Errors)

**Symptoms:**
- Robot deviates from path
- Corrections seem to make it worse
- Position jumps around

**Solutions:**
1. **Verify AprilTag map coordinates**
   - Re-measure tag positions with tape measure
   - Ensure X/Y axes are correct (X=forward, Y=left)

2. **Reduce correction weight**
   ```bash
   apriltag_correction_weight:=0.3   # Trust dead reckoning more
   ```

3. **Check camera calibration**
   - Recalibrate if reprojection error > 1.0 pixel
   - See camera calibration guide

4. **Increase confidence threshold**
   ```bash
   min_detection_confidence:=0.8   # Only use high-confidence detections
   ```

### Robot Oscillates or Stops

**Symptoms:**
- Robot moves back and forth
- Stops before reaching waypoint
- Jerky motion

**Solutions:**
1. **Loosen tolerances**
   ```bash
   pos_tolerance:=0.1      # Accept 10cm error
   yaw_tolerance:=0.2      # Accept ~11° heading error
   ```

2. **Reduce speed**
   ```bash
   max_speed_mps:=0.5
   ```

3. **Check waypoint positions**
   - Ensure waypoints are reachable
   - Not too close to obstacles

### TF Transform Errors

**Error:** `Lookup failed for tag_0`

**Solutions:**
1. **Tag not detected yet**
   - Wait for robot to see the tag
   - Move robot to position where tag is visible

2. **Frame names don't match**
   - Check `camera_frame` parameter matches actual frame name
   - List frames: `ros2 run tf2_ros tf2_echo`

3. **AprilTag detector not running**
   ```bash
   ros2 node list | grep apriltag
   ```

## Best Practices

### Tag Placement

✓ **Do:**
- Place tags at robot camera height (±20cm)
- Distribute evenly along path
- Orient tags toward expected robot positions
- Use rigid mounting (no flex or movement)
- Protect from dirt and damage

✗ **Don't:**
- Place tags too high or too low
- Cluster all tags in one area
- Mount on moving or flexible surfaces
- Use damaged or poorly printed tags
- Place where tags can get occluded

### System Configuration

✓ **Do:**
- Start with `apriltag_correction_weight:=0.5`
- Calibrate camera before deploying
- Test with simple paths first
- Monitor correction statistics in logs
- Use IMU if available

✗ **Don't:**
- Set weight to 1.0 unless tags are perfect
- Skip camera calibration
- Create waypoints far from all tags (>3m)
- Ignore large position jumps in logs

### Operational Tips

✓ **Do:**
- Check tag visibility before starting
- Clean tags regularly
- Verify tag map matches physical layout
- Test in actual lighting conditions
- Have spare printed tags

✗ **Don't:**
- Run in changing lighting (sunrise/sunset)
- Move tags without updating map
- Expect perfect performance first try
- Deploy without testing waypoint path

## Performance Expectations

### Typical Accuracy

| Metric | Value | Notes |
|--------|-------|-------|
| **Position accuracy** | ±5cm | With good calibration and tag placement |
| **Heading accuracy** | ±3° | Using IMU fusion |
| **Detection range** | 0.5-3m | Depends on tag size and camera resolution |
| **Detection rate** | 10-30 Hz | Depends on CPU and detector settings |
| **Waypoint tolerance** | 5cm | Default setting |

### Comparison to YOLO

| Feature | AprilTag | YOLO |
|---------|----------|------|
| **Accuracy** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Robustness** | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Setup effort** | ⭐⭐ (print tags) | ⭐⭐⭐⭐ (train model) |
| **Flexibility** | ⭐⭐⭐ (need tags) | ⭐⭐⭐⭐⭐ (any object) |
| **Computation** | ⭐⭐⭐⭐⭐ | ⭐⭐ (GPU helps) |

**Recommendation:** Use AprilTags for structured environments, YOLO for unstructured/dynamic environments.

## Advanced Features

### Multi-Tag Fusion

To use multiple tags simultaneously (future enhancement):
```python
# Average corrections from multiple visible tags
for each visible tag:
    compute robot_pos_from_tag
    weights.append(1.0 / distance)  # Closer tags weighted more

robot_pos_corrected = weighted_average(all_corrections, weights)
```

### Loop Closure

Detect when returning to previously visited location:
```python
if distance_to_start < threshold and tag_0_visible:
    apply_large_correction()  # Fix accumulated drift
```

### Adaptive Weighting

Adjust correction weight based on conditions:
```python
if decision_margin > 0.8 and distance < 1.5:
    weight = 0.7  # High confidence, close range
elif decision_margin < 0.6 or distance > 2.5:
    weight = 0.2  # Low confidence or far away
```

## File Reference

### Source Code
- [apriltag_waypoint_follower.py](robot_control/robot_control/apriltag_waypoint_follower.py) - Main navigation node
- [AprilTagNode.cpp](apriltag_ros/src/AprilTagNode.cpp) - AprilTag detector
- [apriltag_navigation_launch.py](robot_control/launch/apriltag_navigation_launch.py) - Launch file

### Configuration
- [apriltag_map.json](robot_control/config/apriltag_map.json) - Tag positions in world frame
- [tags_36h11.yaml](apriltag_ros/cfg/tags_36h11.yaml) - AprilTag detector settings
- [waypoints_example.txt](robot_control/config/waypoints_example.txt) - Example waypoints

### Documentation
- [CAMERA_CALIBRATION_GUIDE.md](robot_vision/CAMERA_CALIBRATION_GUIDE.md) - Camera calibration
- [VISION_NAVIGATION_GUIDE.md](VISION_NAVIGATION_GUIDE.md) - YOLO-based navigation (alternative)

## Quick Reference

```bash
# Print AprilTags
git clone https://github.com/AprilRobotics/apriltag-imgs.git
# Print tags from: apriltag-imgs/tag36h11/

# Measure and update map
nano robot_control/config/apriltag_map.json

# Create waypoints
nano waypoints.txt

# Build
colcon build --packages-select robot_control

# Run
ros2 launch robot_control apriltag_navigation_launch.py

# Monitor
ros2 topic echo /detections
ros2 run tf2_ros tf2_echo camera_frame tag_0

# Tune
ros2 param set /apriltag_waypoint_follower_node apriltag_correction_weight 0.7
```

---

**Ready to navigate!** 🤖📍
