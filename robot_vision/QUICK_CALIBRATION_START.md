# Quick Start: Camera Calibration

## TL;DR - 5 Minute Setup

### 1. Print Checkerboard
- Print this: https://raw.githubusercontent.com/opencv/opencv/master/doc/pattern.png
- Or Google "9x6 checkerboard calibration pattern"
- Glue to cardboard for flatness
- Measure square size (usually 24mm)

### 2. Capture Images (20-30 images)

```bash
# Terminal 1: Start camera
ros2 launch robot_vision robot_vision_camera.launch.py

# Terminal 2: Capture images
cd ~/ros2_ws/src/rubikpi_ros2/robot_vision/scripts
python3 capture_calibration_images.py

# In the window that opens:
# - Press SPACE to capture (move checkerboard to different positions)
# - Press 'q' when done (after 20-30 images)
```

**Tip:** Move checkerboard to different:
- Distances (near, medium, far)
- Angles (tilted, rotated)
- Positions (center, corners, edges)

### 3. Run Calibration

```bash
cd ~/ros2_ws/src/rubikpi_ros2/robot_vision/scripts

python3 calibrate_camera.py \
  --images_dir calibration_images \
  --output camera_calibration.yaml \
  --pattern_width 9 \
  --pattern_height 6 \
  --square_size 0.024
```

### 4. Copy Calibration File

```bash
cp camera_calibration.yaml ../config/camera_parameter.yaml
```

### 5. Test

```bash
ros2 launch robot_vision robot_vision_camera.launch.py image_rectify:=true
```

## What the Numbers Mean

Good calibration:
- **Reprojection error < 0.5 pixels** ✓ Excellent
- **Reprojection error < 1.0 pixel** ✓ Good
- **Reprojection error > 1.0 pixel** ✗ Recalibrate

## Common Issues

**"No corners found"**
- Use a 9x6 checkerboard (standard)
- Ensure good lighting
- Pattern must be flat

**High error (> 1.0 pixel)**
- Capture more images (30+)
- Cover all areas of image
- Measure square size accurately

**Script errors**
- Install dependencies: `pip install opencv-python numpy pyyaml`

## Full Documentation

See [CAMERA_CALIBRATION_GUIDE.md](CAMERA_CALIBRATION_GUIDE.md) for complete instructions.

## File Locations

- **Capture script:** `scripts/capture_calibration_images.py`
- **Calibration script:** `scripts/calibrate_camera.py`
- **Current calibration:** `config/camera_parameter.yaml`
- **Output directory:** `calibration_images/` (created in current directory)

## Parameters Reference

### Capture Script
```bash
python3 capture_calibration_images.py
  --ros-args
  -p image_topic:=/camera/image_raw          # ROS topic to subscribe
  -p output_dir:=calibration_images          # Where to save images
  -p image_prefix:=calib                      # Filename prefix
```

### Calibration Script
```bash
python3 calibrate_camera.py
  --images_dir calibration_images             # Input images
  --output camera_calibration.yaml            # Output file
  --pattern_width 9                           # Checkerboard interior corners (horizontal)
  --pattern_height 6                          # Checkerboard interior corners (vertical)
  --square_size 0.024                         # Square size in meters
  --show_corners                              # Display detected corners (optional)
```

### Common Checkerboard Sizes

| Squares | Interior Corners | Pattern Width | Pattern Height |
|---------|------------------|---------------|----------------|
| 10x7    | 9x6             | 9             | 6              |
| 9x7     | 8x6             | 8             | 6              |
| 8x6     | 7x5             | 7             | 5              |

**Note:** OpenCV detects interior corners, not squares!

## Quick Validation

After calibration, check the output:

```yaml
# Good values for 1280x720 camera:
image_width: 1280
image_height: 720

camera_matrix:
  fx: 450-550 pixels    # Focal length X
  fy: 450-550 pixels    # Focal length Y (similar to fx)
  cx: 600-680 pixels    # Principal point X (near width/2)
  cy: 330-390 pixels    # Principal point Y (near height/2)

distortion_coefficients:
  k1: -0.5 to 0.5       # Radial distortion
  k2: -2.0 to 0.0       # Radial distortion
  # Other values usually small
```

If values are way off, recalibrate with more images.

---

**Ready?** Start with Terminal 1 above! 🎥
