#!/usr/bin/env python3
"""
YOLO-based object detector for ROS2
Detects Coke cans using a trained YOLO model and publishes detection results
with estimated 3D positions relative to the robot.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header, Float32MultiArray
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
from typing import Optional, List, Tuple
import os

# Try to import ultralytics YOLO, provide helpful error if not available
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class DetectedObject:
    """Represents a detected object with bounding box and 3D position estimate"""
    def __init__(self, class_id: int, class_name: str, confidence: float,
                 bbox: Tuple[float, float, float, float],
                 position_3d: Optional[Tuple[float, float, float]] = None,
                 landmark_id: Optional[int] = None):
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox  # (x1, y1, x2, y2) in pixels
        self.position_3d = position_3d  # (x, y, z) in meters relative to camera
        self.landmark_id = landmark_id


class YOLODetectorNode(Node):
    """
    YOLO detector node that:
    1. Subscribes to camera images and camera info
    2. Runs YOLO detection on each frame
    3. Estimates 3D position of detected objects using camera intrinsics
    4. Publishes detections for navigation system
    """

    def __init__(self):
        super().__init__('yolo_detector_node')

        # Parameters
        self.declare_parameter('model_path', 'yolov8n.pt')  # Path to YOLO model
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('target_class', 'Coke')  # Class name to detect
        self.declare_parameter('known_object_height', 0.123)  # Coke can height in meters (standard 330ml)
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('detection_rate_hz', 5.0)  # Process at 5 Hz to save resources

        # Read parameters
        self.model_path = self.get_parameter('model_path').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.target_class = self.get_parameter('target_class').value
        self.known_height = self.get_parameter('known_object_height').value
        self.image_topic = self.get_parameter('image_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.publish_debug = self.get_parameter('publish_debug_image').value
        self.detection_rate = self.get_parameter('detection_rate_hz').value

        # Check if YOLO is available
        if not YOLO_AVAILABLE:
            self.get_logger().error('ultralytics YOLO not available! Install with: pip install ultralytics')
            raise RuntimeError('ultralytics not installed')

        # Load YOLO model
        try:
            if not os.path.isabs(self.model_path):
                # Try to find model in package share or current directory
                cwd_path = os.path.join(os.getcwd(), self.model_path)
                if os.path.exists(cwd_path):
                    self.model_path = cwd_path

            self.get_logger().info(f'Loading YOLO model from {self.model_path}...')
            self.model = YOLO(self.model_path)
            self.get_logger().info(f'YOLO model loaded successfully')
            self.get_logger().info(f'Model classes: {self.model.names}')
        except Exception as e:
            self.get_logger().error(f'Failed to load YOLO model: {e}')
            raise

        # CV Bridge for image conversion
        self.bridge = CvBridge()

        # Camera intrinsics (will be updated from camera_info)
        self.camera_matrix = None
        self.dist_coeffs = None
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.camera_calibrated = False

        # Latest image
        self.latest_image = None
        self.latest_image_stamp = None

        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            10
        )

        # Publishers
        # Publish detections as Float32MultiArray with format:
        # [num_detections,
        #  det1_x, det1_y, det1_distance, det1_bearing, det1_confidence,
        #  det2_x, det2_y, det2_distance, det2_bearing, det2_confidence,
        #  ...]
        self.detection_pub = self.create_publisher(
            Float32MultiArray,
            'yolo/detections',
            10
        )

        if self.publish_debug:
            self.debug_image_pub = self.create_publisher(
                Image,
                'yolo/debug_image',
                10
            )

        # Timer for detection processing
        self.detection_timer = self.create_timer(
            1.0 / self.detection_rate,
            self.process_detection
        )

        self.get_logger().info(f'YOLO detector initialized. Detecting class: {self.target_class}')
        self.get_logger().info(f'Detection rate: {self.detection_rate} Hz, Confidence threshold: {self.confidence_threshold}')

    def camera_info_callback(self, msg: CameraInfo):
        """Extract camera intrinsics from CameraInfo message"""
        if not self.camera_calibrated:
            # Camera matrix from projection matrix P (3x4)
            # P = [fx  0  cx tx]
            #     [0  fy cy ty]
            #     [0  0  1  0 ]
            self.fx = msg.p[0]
            self.fy = msg.p[5]
            self.cx = msg.p[2]
            self.cy = msg.p[6]

            self.camera_matrix = np.array([
                [self.fx, 0, self.cx],
                [0, self.fy, self.cy],
                [0, 0, 1]
            ], dtype=np.float32)

            # Distortion coefficients (k1, k2, t1, t2, k3)
            self.dist_coeffs = np.array(msg.d, dtype=np.float32)

            self.camera_calibrated = True
            self.get_logger().info(f'Camera calibrated: fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}')

    def image_callback(self, msg: Image):
        """Store latest image for processing"""
        self.latest_image = msg
        self.latest_image_stamp = msg.header.stamp

    def pixel_to_bearing(self, pixel_x: float) -> float:
        """Convert pixel x coordinate to bearing angle in radians"""
        if not self.camera_calibrated:
            return 0.0
        angle = math.atan2(pixel_x - self.cx, self.fx)
        return angle

    def estimate_distance(self, bbox_height_pixels: float) -> float:
        """
        Estimate distance to object using known real-world height
        distance = (real_height * focal_length) / pixel_height
        """
        if not self.camera_calibrated or bbox_height_pixels < 1:
            return 0.0
        distance = (self.known_height * self.fy) / bbox_height_pixels
        return distance

    def bbox_to_3d_position(self, bbox: Tuple[float, float, float, float]) -> Tuple[float, float, float]:
        """
        Convert bounding box to 3D position relative to camera
        Returns (x, y, z) where:
        - x: forward distance from camera (depth)
        - y: lateral offset (positive = left)
        - z: vertical offset (positive = up)
        """
        x1, y1, x2, y2 = bbox

        # Center of bounding box
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0

        # Height of bounding box in pixels
        height_pixels = y2 - y1

        # Estimate distance (depth)
        distance = self.estimate_distance(height_pixels)

        # Bearing angle (horizontal angle from camera center)
        bearing = self.pixel_to_bearing(center_x)

        # Calculate 3D position
        # In camera frame: x=forward, y=left, z=up
        x_cam = distance * math.cos(bearing)  # Forward component
        y_cam = distance * math.sin(bearing)  # Lateral component

        # Vertical position (simplified - assumes camera is level)
        # Elevation angle
        elevation = math.atan2(center_y - self.cy, self.fy)
        z_cam = distance * math.sin(elevation)

        return (x_cam, y_cam, z_cam)

    def process_detection(self):
        """Process latest image with YOLO detector"""
        if self.latest_image is None:
            return

        if not self.camera_calibrated:
            self.get_logger().warn('Camera not calibrated yet, skipping detection', throttle_duration_sec=5.0)
            return

        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(self.latest_image, desired_encoding='rgb8')

            # Run YOLO detection
            results = self.model(cv_image, conf=self.confidence_threshold, verbose=False)

            # Process detections
            detections: List[DetectedObject] = []

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    cls_name = self.model.names[cls_id]
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    # Filter for target class
                    if cls_name == self.target_class:
                        # Estimate 3D position
                        pos_3d = self.bbox_to_3d_position((x1, y1, x2, y2))

                        detection = DetectedObject(
                            class_id=cls_id,
                            class_name=cls_name,
                            confidence=conf,
                            bbox=(x1, y1, x2, y2),
                            position_3d=pos_3d
                        )
                        detections.append(detection)

            # Publish detections
            self.publish_detections(detections)

            # Publish debug image if enabled
            if self.publish_debug and len(detections) > 0:
                self.publish_debug_image(cv_image, detections)

            if len(detections) > 0:
                self.get_logger().info(f'Detected {len(detections)} {self.target_class} objects')
                for det in detections:
                    x, y, z = det.position_3d
                    bearing = self.pixel_to_bearing((det.bbox[0] + det.bbox[2]) / 2.0)
                    distance = math.hypot(x, y)
                    self.get_logger().info(
                        f'  -> Distance: {distance:.2f}m, Bearing: {math.degrees(bearing):.1f}°, '
                        f'Position: ({x:.2f}, {y:.2f}, {z:.2f}), Confidence: {det.confidence:.2f}'
                    )

        except Exception as e:
            self.get_logger().error(f'Error processing detection: {e}')

    def publish_detections(self, detections: List[DetectedObject]):
        """Publish detections as Float32MultiArray"""
        msg = Float32MultiArray()

        # Format: [num_detections, det1_x, det1_y, det1_distance, det1_bearing, det1_confidence, ...]
        data = [float(len(detections))]

        for det in detections:
            if det.position_3d is not None:
                x, y, z = det.position_3d
                distance = math.hypot(x, y)
                bearing = self.pixel_to_bearing((det.bbox[0] + det.bbox[2]) / 2.0)

                data.extend([
                    float(x),          # Forward distance
                    float(y),          # Lateral offset
                    float(distance),   # Euclidean distance
                    float(bearing),    # Bearing angle in radians
                    float(det.confidence)
                ])

        msg.data = data
        self.detection_pub.publish(msg)

    def publish_debug_image(self, cv_image: np.ndarray, detections: List[DetectedObject]):
        """Publish annotated debug image"""
        debug_img = cv_image.copy()

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Draw bounding box
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label
            if det.position_3d is not None:
                x, y, z = det.position_3d
                distance = math.hypot(x, y)
                bearing = self.pixel_to_bearing((det.bbox[0] + det.bbox[2]) / 2.0)
                label = f'{det.class_name} {det.confidence:.2f} | {distance:.2f}m @ {math.degrees(bearing):.0f}°'
            else:
                label = f'{det.class_name} {det.confidence:.2f}'

            # Text background
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(debug_img, (x1, y1 - text_height - 5), (x1 + text_width, y1), (0, 255, 0), -1)
            cv2.putText(debug_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Convert back to ROS Image
        debug_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding='rgb8')
        debug_msg.header.stamp = self.latest_image_stamp
        debug_msg.header.frame_id = 'camera_frame'
        self.debug_image_pub.publish(debug_msg)


def main(args=None):
    rclpy.init(args=args)
    node = YOLODetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()