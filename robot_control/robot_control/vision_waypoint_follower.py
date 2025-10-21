#!/usr/bin/env python3
"""
Vision-based waypoint follower using YOLO landmark detection
Extends the basic waypoint follower with visual landmark-based localization
to correct dead-reckoning drift.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Imu
import math
import os
import json
from typing import List, Tuple, Dict, Optional
import numpy as np


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]"""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class VisionWaypointFollowerNode(Node):
    """
    Vision-enhanced waypoint follower that:
    1. Follows waypoints using differential drive control
    2. Uses dead reckoning for position estimation
    3. Corrects position using visual landmarks (Coke cans) detected by YOLO
    4. Optionally uses IMU for improved orientation estimation
    """

    def __init__(self):
        super().__init__('vision_waypoint_follower_node')

        # ===== Parameters =====
        # Waypoint and control parameters
        self.declare_parameter('waypoints_path', 'waypoints.txt')
        self.declare_parameter('landmarks_path', 'landmarks.json')
        self.declare_parameter('wheel_base', 0.04)  # meters (wheel radius)
        self.declare_parameter('car_width', 0.168)  # meters
        self.declare_parameter('max_speed_mps', 0.72)
        self.declare_parameter('max_angular_speed_radps', math.pi)
        self.declare_parameter('max_norm_cmd', 0.5)
        self.declare_parameter('kv', 0.8)  # linear velocity gain
        self.declare_parameter('kh', 0.8)  # heading gain
        self.declare_parameter('pos_tolerance', 0.05)  # meters
        self.declare_parameter('yaw_tolerance', 0.1)  # radians (~11 deg)
        self.declare_parameter('rate_hz', 20.0)

        # Vision-based localization parameters
        self.declare_parameter('use_vision_correction', True)
        self.declare_parameter('vision_correction_weight', 0.3)  # How much to trust vision vs dead reckoning
        self.declare_parameter('min_detection_confidence', 0.6)
        self.declare_parameter('max_landmark_distance', 3.0)  # meters - ignore landmarks farther than this
        self.declare_parameter('use_imu', False)

        # Read parameters
        self.wheel_base = float(self.get_parameter('wheel_base').value)
        self.max_speed_mps = float(self.get_parameter('max_speed_mps').value)
        self.max_angular_speed_radps = float(self.get_parameter('max_angular_speed_radps').value)
        self.max_norm_cmd = float(self.get_parameter('max_norm_cmd').value)
        self.kv = float(self.get_parameter('kv').value)
        self.kh = float(self.get_parameter('kh').value)
        self.pos_tol = float(self.get_parameter('pos_tolerance').value)
        self.yaw_tol = float(self.get_parameter('yaw_tolerance').value)
        self.rate_hz = float(self.get_parameter('rate_hz').value)

        self.use_vision = bool(self.get_parameter('use_vision_correction').value)
        self.vision_weight = float(self.get_parameter('vision_correction_weight').value)
        self.min_confidence = float(self.get_parameter('min_detection_confidence').value)
        self.max_landmark_dist = float(self.get_parameter('max_landmark_distance').value)
        self.use_imu = bool(self.get_parameter('use_imu').value)

        self.linear_min = 0.1
        self.angular_min = 0.25

        # Load waypoints
        waypoints_path = self.get_parameter('waypoints_path').value
        self.waypoints: List[Tuple[float, float, float]] = self.load_waypoints(waypoints_path)
        if not self.waypoints:
            self.get_logger().error(f'No waypoints loaded from {waypoints_path}. Stopping.')
            raise RuntimeError('waypoints_empty')

        # Load landmark map (known positions of Coke cans in world frame)
        landmarks_path = self.get_parameter('landmarks_path').value
        self.landmarks: Dict[int, Tuple[float, float]] = self.load_landmarks(landmarks_path)
        self.get_logger().info(f'Loaded {len(self.landmarks)} known landmarks')

        # ===== State =====
        # Estimated pose (x, y, yaw)
        self.x_est = 0.0
        self.y_est = 0.0
        self.yaw = 0.0

        # IMU-based yaw (if using IMU)
        self.imu_yaw = None
        self.imu_available = False

        # Tracking
        self.current_idx = 0
        self.last_time = self.get_clock().now()

        # Statistics
        self.vision_corrections = 0
        self.total_detections = 0

        # ===== Publishers =====
        self.cmd_pub = self.create_publisher(Float32MultiArray, 'motor_commands', 10)

        # ===== Subscribers =====
        self.detection_sub = self.create_subscription(
            Float32MultiArray,
            'yolo/detections',
            self.detection_callback,
            10
        )

        if self.use_imu:
            self.imu_sub = self.create_subscription(
                Imu,
                'imu/data',
                self.imu_callback,
                10
            )

        # ===== Timer =====
        self.timer = self.create_timer(1.0 / self.rate_hz, self.control_step)

        self.get_logger().info(f'Vision-based waypoint follower initialized')
        self.get_logger().info(f'Loaded {len(self.waypoints)} waypoints, {len(self.landmarks)} landmarks')
        self.get_logger().info(f'Vision correction: {self.use_vision}, Weight: {self.vision_weight}')
        self.get_logger().info(f'Control rate: {self.rate_hz:.1f} Hz')

    def load_waypoints(self, path: str) -> List[Tuple[float, float, float]]:
        """Load waypoints from file (same as original waypoint follower)"""
        waypoints: List[Tuple[float, float, float]] = []
        try:
            if not os.path.isabs(path):
                cwd_path = os.path.join(os.getcwd(), path)
                if os.path.exists(cwd_path):
                    path = cwd_path
            with open(path, 'r') as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith('#'):
                        continue
                    parts = [p.strip() for p in s.split(',')]
                    if len(parts) != 3:
                        continue
                    x = float(parts[0])
                    y = float(parts[1])
                    a = float(parts[2])
                    waypoints.append((x, y, a))
        except Exception as e:
            self.get_logger().error(f'Failed to load waypoints from {path}: {e}')
        return waypoints

    def load_landmarks(self, path: str) -> Dict[int, Tuple[float, float]]:
        """
        Load landmark positions from JSON file
        Format: {"landmarks": [{"id": 0, "x": 1.0, "y": 0.5}, ...]}
        """
        landmarks: Dict[int, Tuple[float, float]] = {}
        try:
            if not os.path.isabs(path):
                cwd_path = os.path.join(os.getcwd(), path)
                if os.path.exists(cwd_path):
                    path = cwd_path

            if not os.path.exists(path):
                self.get_logger().warn(f'Landmarks file not found: {path}. Vision correction will be disabled.')
                return landmarks

            with open(path, 'r') as f:
                data = json.load(f)
                for lm in data.get('landmarks', []):
                    lm_id = int(lm['id'])
                    lm_x = float(lm['x'])
                    lm_y = float(lm['y'])
                    landmarks[lm_id] = (lm_x, lm_y)
        except Exception as e:
            self.get_logger().error(f'Failed to load landmarks from {path}: {e}')
        return landmarks

    def imu_callback(self, msg: Imu):
        """Extract yaw from IMU orientation (quaternion)"""
        # Convert quaternion to yaw
        # yaw = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
        qx = msg.orientation.x
        qy = msg.orientation.y
        qz = msg.orientation.z
        qw = msg.orientation.w

        # Compute yaw (rotation around z-axis)
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        self.imu_yaw = math.atan2(siny_cosp, cosy_cosp)
        self.imu_available = True

    def detection_callback(self, msg: Float32MultiArray):
        """
        Process YOLO detections and update position estimate
        Message format: [num_detections, det1_x, det1_y, det1_distance, det1_bearing, det1_confidence, ...]
        """
        if not self.use_vision or len(self.landmarks) == 0:
            return

        data = msg.data
        if len(data) < 1:
            return

        num_detections = int(data[0])
        if num_detections == 0:
            return

        # Parse detections (each detection has 5 values)
        detections = []
        for i in range(num_detections):
            idx = 1 + i * 5
            if idx + 4 < len(data):
                det_x = data[idx]          # Forward distance
                det_y = data[idx + 1]      # Lateral offset
                det_dist = data[idx + 2]   # Euclidean distance
                det_bearing = data[idx + 3]  # Bearing angle
                det_conf = data[idx + 4]   # Confidence

                # Filter by confidence and distance
                if det_conf >= self.min_confidence and det_dist <= self.max_landmark_dist:
                    detections.append({
                        'x': det_x,
                        'y': det_y,
                        'distance': det_dist,
                        'bearing': det_bearing,
                        'confidence': det_conf
                    })

        self.total_detections += len(detections)

        if len(detections) > 0:
            # Use detections to update position estimate
            self.update_position_from_landmarks(detections)

    def update_position_from_landmarks(self, detections: List[Dict]):
        """
        Update robot position estimate using detected landmarks
        Uses trilateration or weighted averaging based on multiple landmark observations
        """
        if len(self.landmarks) == 0:
            return

        # For simplicity, we'll use the closest detected landmark
        # In a full implementation, you'd use multiple landmarks with trilateration or EKF

        # Convert detections from camera frame to world frame
        # Detection gives us: distance and bearing in robot's current frame
        # We need to find which landmark this corresponds to

        # Simple approach: assume closest landmark in robot's current estimated position
        # is the one we're seeing

        for detection in detections:
            det_distance = detection['distance']
            det_bearing = detection['bearing']

            # Global bearing (robot's yaw + detection bearing)
            global_bearing = normalize_angle(self.yaw + det_bearing)

            # Estimated landmark position in world frame based on current pose estimate
            lm_x_est = self.x_est + det_distance * math.cos(global_bearing)
            lm_y_est = self.y_est + det_distance * math.sin(global_bearing)

            # Find closest known landmark
            closest_lm_id = None
            min_dist = float('inf')
            for lm_id, (lm_x, lm_y) in self.landmarks.items():
                dist = math.hypot(lm_x - lm_x_est, lm_y - lm_y_est)
                if dist < min_dist:
                    min_dist = dist
                    closest_lm_id = lm_id

            # If we found a close match (within 0.5m), use it for correction
            if closest_lm_id is not None and min_dist < 0.5:
                lm_x_true, lm_y_true = self.landmarks[closest_lm_id]

                # Calculate corrected robot position based on known landmark position
                # Robot is at: landmark_pos - (distance * direction)
                x_corrected = lm_x_true - det_distance * math.cos(global_bearing)
                y_corrected = lm_y_true - det_distance * math.sin(global_bearing)

                # Weighted update (blend current estimate with vision-based correction)
                self.x_est = (1.0 - self.vision_weight) * self.x_est + self.vision_weight * x_corrected
                self.y_est = (1.0 - self.vision_weight) * self.y_est + self.vision_weight * y_corrected

                self.vision_corrections += 1

                self.get_logger().info(
                    f'Vision correction using landmark {closest_lm_id}: '
                    f'Position updated to ({self.x_est:.3f}, {self.y_est:.3f}), '
                    f'Correction: ({x_corrected - self.x_est:.3f}, {y_corrected - self.y_est:.3f})'
                )

    def diff2velocity_simple(self, dist: float, heading_error: float,
                            gtheta: float, theta: float) -> Tuple[float, float]:
        """
        Compute velocity commands based on position and heading error
        (Same as original waypoint follower)
        """
        if dist < self.pos_tol:
            # Waypoint reached - move to next
            self.current_idx += 1
            self.get_logger().info(
                f'Reached waypoint {self.current_idx}/{len(self.waypoints)} - '
                f'Vision corrections: {self.vision_corrections}/{self.total_detections}'
            )
            v_cmd, w_cmd = 0.0, 0.0
        elif abs(heading_error) > self.yaw_tol:
            # Heading correction needed - rotate in place
            v_cmd, w_cmd = 0.0, math.copysign(self.max_angular_speed_radps, heading_error)
        else:
            # Move forward with proportional heading correction
            v_cmd = self.max_speed_mps
            w_cmd = math.copysign(self.max_angular_speed_radps * abs(heading_error) / math.pi, heading_error)

        v_cmd = v_cmd * self.kv
        w_cmd = w_cmd * self.kh
        return v_cmd, w_cmd

    def velocity2input(self, v_cmd: float, w_cmd: float) -> Tuple[float, float]:
        """
        Convert velocity commands to motor inputs
        (Same as original waypoint follower)
        """
        if v_cmd == 0 and w_cmd == 0:
            return 0.0, 0.0
        if w_cmd == 0:
            # Pure linear motion
            l1 = self.linear_min + v_cmd * (0.5 - self.linear_min) / self.max_speed_mps
            l1 = l1 * v_cmd / abs(v_cmd)
            return l1, l1
        elif v_cmd == 0:
            # Pure rotation
            r2 = math.copysign(self.angular_min + abs(w_cmd) * (0.5 - self.angular_min) / self.max_angular_speed_radps, w_cmd)
            return -r2, r2
        else:
            # Combined linear + angular
            l1, r1 = v_cmd * (0.5 - self.linear_min) / self.max_speed_mps, v_cmd * (0.5 - self.linear_min) / self.max_speed_mps
            l2, r2 = -1 * w_cmd * (0.5 - self.angular_min) / self.max_angular_speed_radps, w_cmd * (0.5 - self.angular_min) / self.max_angular_speed_radps
            l = l1 + l2 + self.linear_min
            r = r1 + r2 + self.linear_min
            l = max(-self.max_norm_cmd, min(self.max_norm_cmd, l))
            r = max(-self.max_norm_cmd, min(self.max_norm_cmd, r))
            return l, r

    def control_step(self):
        """Main control loop - execute at configured rate"""
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        if dt < 0.0001:
            return
        self.last_time = now

        # Check if all waypoints completed
        if self.current_idx >= len(self.waypoints):
            self.publish_cmd(0.0, 0.0)
            self.get_logger().info(
                f'All waypoints completed! Total vision corrections: {self.vision_corrections}/{self.total_detections}',
                throttle_duration_sec=5.0
            )
            return

        # Get desired waypoint
        gx, gy, gtheta = self.waypoints[self.current_idx]

        # Compute errors
        dx = gx - self.x_est
        dy = gy - self.y_est
        dist = math.hypot(dx, dy)

        # Use IMU yaw if available and enabled, otherwise use integrated yaw
        theta = self.imu_yaw if (self.use_imu and self.imu_available) else self.yaw

        # Heading to goal
        path_angle = math.atan2(dy, dx)
        heading_error = normalize_angle(path_angle - theta)

        # Compute velocity commands
        v_cmd, w_cmd = self.diff2velocity_simple(dist, heading_error, gtheta, theta)

        # Convert to motor commands and publish
        l_norm, r_norm = self.velocity2input(v_cmd, w_cmd)
        self.publish_cmd(l_norm, r_norm)

        # Integrate position estimate using commanded velocities
        self.x_est += v_cmd * math.cos(theta) * dt
        self.y_est += v_cmd * math.sin(theta) * dt

        # Integrate yaw using angular velocity (if not using IMU)
        if not self.use_imu or not self.imu_available:
            self.yaw += w_cmd * dt
            self.yaw = normalize_angle(self.yaw)

        # Log status
        self.get_logger().info(
            f'WP {self.current_idx+1}/{len(self.waypoints)}: '
            f'Pos=({self.x_est:.2f},{self.y_est:.2f}), Yaw={math.degrees(theta):.1f}°, '
            f'Dist={dist:.2f}m, HErr={math.degrees(heading_error):.1f}°, '
            f'Cmd=({l_norm:.2f},{r_norm:.2f}), VCorr={self.vision_corrections}'
        )

    def publish_cmd(self, left: float, right: float):
        """Publish motor commands"""
        msg = Float32MultiArray()
        msg.data = [float(left), float(right)]
        self.cmd_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = VisionWaypointFollowerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_cmd(0.0, 0.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        return 0


if __name__ == '__main__':
    main()