# Camera Calibration Guide for RubikPi

This guide walks you through calibrating your camera using OpenCV and a checkerboard pattern.

## Overview

Camera calibration determines the camera's intrinsic parameters (focal length, principal point, distortion coefficients) which are essential for accurate 3D vision tasks like object detection and localization.

## What You'll Need

### 1. Checkerboard Pattern

**Print a checkerboard pattern:**
- Download: https://raw.githubusercontent.com/opencv/opencv/master/doc/pattern.png
- Or use this 9x6 pattern: https://markhedleyjones.com/projects/calibration-checkerboard-collection
- Print on A4 or Letter paper
- Mount on flat, rigid surface (cardboard, foam board)
- Measure the square size accurately (typically 24mm or 25mm)

**Pattern specifications:**
- **9x6 checkerboard** = 8x5 interior corners (what OpenCV detects)
- Must be perfectly flat (no wrinkles or bends)
- High contrast black and white

### 2. Software Setup

Make the scripts executable:
```bash
cd ~/ros2_ws/src/rubikpi_ros2/robot_vision/scripts
chmod +x capture_calibration_images.py
chmod +x calibrate_camera.py
```

## Calibration Process

### Step 1: Start the Camera

In **Terminal 1**, launch the camera node:
```bash
cd ~/ros2_ws
source install/setup.bash
ros2 launch robot_vision robot_vision_camera.launch.py
```

Verify camera is working:
```bash
ros2 topic list | grep camera
# Should see:
# /camera/image_raw
# /camera/image_raw/compressed
# /camera/camera_info
```

### Step 2: Capture Calibration Images

In **Terminal 2**, run the image capture script:
```bash
cd ~/ros2_ws
source install/setup.bash

# Create directory for calibration images
mkdir -p ~/calibration_images

# Run capture script
ros2 run robot_vision capture_calibration_images \
  --ros-args \
  -p image_topic:=/camera/image_raw \
  -p output_dir:=/home/ubuntu/calibration_images
```

Or if you prefer Python directly:
```bash
python3 robot_vision/scripts/capture_calibration_images.py
```

**Instructions:**
1. A window will open showing the live camera feed
2. Hold the checkerboard in front of the camera
3. Press **SPACE** to capture an image
4. Move the checkerboard to a different position/angle
5. Repeat until you have 20-30 images
6. Press **'q'** to quit

**Tips for good calibration:**
- Capture images with checkerboard at different:
  - **Distances** (close, medium, far)
  - **Angles** (tilted left/right, up/down, rotated)
  - **Positions** (center, edges, corners of image)
- Fill the entire frame at least once
- Keep checkerboard flat and still
- Ensure good lighting (no shadows on pattern)
- Aim for 20-30 good images

### Step 3: Run Camera Calibration

After capturing images, run the calibration script:

```bash
cd ~/ros2_ws/src/rubikpi_ros2/robot_vision/scripts

python3 calibrate_camera.py \
  --images_dir ~/calibration_images \
  --output ~/camera_calibration.yaml \
  --pattern_width 9 \
  --pattern_height 6 \
  --square_size 0.024 \
  --show_corners
```

**Parameters:**
- `--images_dir`: Directory with captured images
- `--output`: Output YAML file path
- `--pattern_width`: Number of interior corners horizontally (9 for standard 10x7 checkerboard)
- `--pattern_height`: Number of interior corners vertically (6 for standard 10x7 checkerboard)
- `--square_size`: Size of each square in meters (measure with ruler!)
- `--show_corners`: Display detected corners (optional)

**Example for different checkerboard sizes:**

| Checkerboard Size | Interior Corners | Command |
|-------------------|------------------|---------|
| 10x7 squares | 9x6 | `--pattern_width 9 --pattern_height 6` |
| 9x7 squares | 8x6 | `--pattern_width 8 --pattern_height 6` |
| 8x6 squares | 7x5 | `--pattern_width 7 --pattern_height 5` |

### Step 4: Review Calibration Results

The script will output:

```
============================================================
CALIBRATION SUMMARY
============================================================

Image size: 1280 x 720

Camera Matrix:
  fx = 490.98 pixels
  fy = 492.13 pixels
  cx = 639.43 pixels
  cy = 357.23 pixels

Distortion Coefficients (plumb_bob model):
  k1 = 0.487826
  k2 = -1.851282
  p1 = 1.209359
  p2 = -0.068021
  k3 = 0.000000

Reprojection Error: 0.42 pixels
Mean Error: 0.38 pixels

Quality Assessment:
  ✓ Excellent calibration (error < 0.5 pixels)
============================================================
```

**Quality criteria:**
- **Excellent:** Reprojection error < 0.5 pixels
- **Good:** Reprojection error < 1.0 pixel
- **Poor:** Reprojection error > 1.0 pixel → Recalibrate

### Step 5: Update Camera Configuration

Copy the calibration file to your robot_vision config directory:

```bash
cp ~/camera_calibration.yaml ~/ros2_ws/src/rubikpi_ros2/robot_vision/config/camera_parameter.yaml
```

Verify the file format matches ROS requirements:
```bash
cat ~/ros2_ws/src/rubikpi_ros2/robot_vision/config/camera_parameter.yaml
```

### Step 6: Test Calibration

Restart the camera with rectification enabled:

```bash
ros2 launch robot_vision robot_vision_camera.launch.py image_rectify:=true
```

Compare original vs rectified images:
```bash
# Terminal 1: View raw image
ros2 run rqt_image_view rqt_image_view /camera/image_raw

# Terminal 2: View rectified image (if published separately)
# Or check that straight lines appear straight in the image
```

## Understanding Calibration Parameters

### Camera Matrix (Intrinsics)

```
K = | fx   0   cx |
    | 0   fy   cy |
    | 0    0    1 |
```

- **fx, fy**: Focal lengths in pixels (should be similar for square pixels)
- **cx, cy**: Principal point (optical center), usually near image center
- **Typical values for 1280x720:** fx/fy ≈ 500-600, cx ≈ 640, cy ≈ 360

### Distortion Coefficients

**Plumb Bob model:** `[k1, k2, p1, p2, k3]`

- **k1, k2, k3**: Radial distortion coefficients
  - Correct barrel/pincushion distortion
  - Wide-angle lenses have larger values
- **p1, p2**: Tangential distortion coefficients
  - Correct lens misalignment
  - Usually small values

### Interpreting Values

**Focal length (fx, fy):**
- Higher values = narrower field of view (telephoto)
- Lower values = wider field of view (wide-angle)
- For your camera at 1280x720, ~490-500 pixels is typical

**Radial distortion (k1):**
- Positive k1 = Pincushion distortion
- Negative k1 = Barrel distortion
- Your camera has k1 = 0.48 (slight pincushion)

## Troubleshooting

### "No corners found" in most images

**Solutions:**
1. Check checkerboard pattern is correct size (interior corners!)
2. Ensure good lighting (uniform, bright)
3. Make sure pattern is flat and in focus
4. Try adjusting pattern size parameters
5. Print a larger checkerboard

### High reprojection error (> 1.0 pixel)

**Solutions:**
1. Capture more images (30-40)
2. Ensure better coverage of image area
3. Check checkerboard is perfectly flat
4. Verify square size measurement is accurate
5. Remove blurry or poorly detected images

### Camera matrix values seem wrong

**Check:**
- fx and fy should be similar (within 5%)
- cx should be near image_width/2
- cy should be near image_height/2
- If values are wildly off, recalibrate with more images

### Script can't find checkerboard

Try manual corner detection test:
```python
import cv2
import numpy as np

img = cv2.imread('calibration_images/calib_001.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

print(f"Corners found: {ret}")
if ret:
    img_corners = cv2.drawChessboardCorners(img, (9, 6), corners, ret)
    cv2.imshow('Corners', img_corners)
    cv2.waitKey(0)
```

## Alternative: Using ROS Camera Calibration Tool

ROS provides a GUI calibration tool:

```bash
# Install if not already installed
sudo apt install ros-humble-camera-calibration

# Run calibration (move checkerboard around until bars turn green)
ros2 run camera_calibration cameracalibrator \
  --size 8x6 \
  --square 0.024 \
  --no-service-check \
  image:=/camera/image_raw \
  camera:=/camera
```

This GUI tool:
- Shows live feedback on calibration quality
- Automatically determines when you have enough coverage
- Saves calibration in ROS format

## Verifying Calibration Quality

### Visual Test

Take a picture of straight lines (door frame, table edge, etc.) and check:
- Straight lines should appear straight after rectification
- No visible barrel or pincushion distortion at edges

### Quantitative Test

Measure a known distance using calibrated camera:
```python
# If you know object is 1 meter away and 0.3m tall
# Calculate expected pixel height:
# pixel_height = (real_height * fy) / distance
# = (0.3 * 492) / 1.0 = 147.6 pixels

# Compare with actual pixel height in image
```

## Re-calibration Schedule

Recalibrate if:
- Camera lens is adjusted or changed
- Camera is dropped or impacted
- After 6-12 months of use
- 3D measurements seem inaccurate
- Significant temperature changes

## Advanced: Stereo Calibration

For stereo vision or depth estimation, you'll need to calibrate multiple cameras together. This requires:
- Simultaneous capture from both cameras
- `cv2.stereoCalibrate()` function
- More complex setup

## Files Created

After calibration, you'll have:

```
~/calibration_images/
├── calib_000_20240123_143022.png
├── calib_001_20240123_143025.png
├── ...
└── calib_025_20240123_143156.png

~/camera_calibration.yaml        # Generated calibration
~/ros2_ws/src/rubikpi_ros2/robot_vision/config/
└── camera_parameter.yaml        # Active calibration used by robot
```

## Next Steps

After calibration:
1. ✓ Enable image rectification in launch file
2. ✓ Test YOLO detection accuracy with calibrated camera
3. ✓ Verify distance estimation to objects
4. ✓ Run vision-based navigation

## Resources

- **OpenCV Calibration Tutorial:** https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
- **Checkerboard Patterns:** https://markhedleyjones.com/projects/calibration-checkerboard-collection
- **ROS Camera Calibration:** http://wiki.ros.org/camera_calibration
- **Understanding Camera Calibration:** https://learnopencv.com/camera-calibration-using-opencv/

---

**Questions?** Check the calibration summary output for quality metrics and ensure reprojection error is below 0.5 pixels for best results.
