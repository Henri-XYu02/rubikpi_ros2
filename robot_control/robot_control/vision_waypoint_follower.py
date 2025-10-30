#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from std_msgs.msg import Float32MultiArray
from apriltag_msgs.msg import AprilTagDetectionArray
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster, Buffer, TransformListener
import math
import os
from typing import List, Tuple
from math import sin, cos, atan2
from geometry_msgs.msg import TransformStamped
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
    Waypoint follower with AprilTag-based vision correction.

    Operation:
    - Dead reckoning phase (0.5s): Move using dead reckoning
    - Correction phase (0.5s): Stop and update pose using AprilTag TF

    TF Chain for localization:
    odom -> tag_X (static, from tf_publisher)
    tag_X -> camera_frame (dynamic, from AprilTag detector)
    camera_frame -> base_link (static, from tf_publisher)

    By looking up odom -> base_link, we get the robot's pose in odom frame.
    """

    def __init__(self):
        super().__init__('vision_waypoint_follower_node')

        # Parameters
        self.declare_parameter('waypoints_path', 'waypoints.txt')
        self.declare_parameter('wheel_base', 0.04)
        self.declare_parameter('car_width', 0.168)
        self.declare_parameter('max_speed_mps', 0.68)
        self.declare_parameter('max_angular_speed_radps', 4/3*math.pi)
        self.declare_parameter('max_norm_cmd', 0.5)
        self.declare_parameter('kv', 0.6)
        self.declare_parameter('kh', 0.6)
        self.declare_parameter('pos_tolerance', 0.05)
        self.declare_parameter('yaw_tolerance', 0.1)
        self.declare_parameter('rate_hz', 10.0)
        self.declare_parameter('dead_reckoning_duration', 1)  # seconds
        self.declare_parameter('correction_duration', 1)  # seconds
        self.declare_parameter('min_detection_confidence', 0.5)

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
        self.dead_reckoning_duration = float(self.get_parameter('dead_reckoning_duration').value)
        self.correction_duration = float(self.get_parameter('correction_duration').value)
        self.min_detection_confidence = float(self.get_parameter('min_detection_confidence').value)
        self.apriltag_map = self.load_apriltag_map()

        self.linear_min = 0.1
        self.angular_min = 0.25
        self.points = []

        # Load waypoints
        waypoints_path = self.get_parameter('waypoints_path').value
        self.waypoints: List[Tuple[float, float, float]] = self.load_waypoints(waypoints_path)
        if not self.waypoints:
            self.get_logger().error(f'No waypoints loaded from {waypoints_path}. Stopping.')
            raise RuntimeError('waypoints_empty')

        # Publisher for motor commands
        self.cmd_pub = self.create_publisher(Float32MultiArray, 'motor_commands', 10)

        # TF2 for AprilTag localization
        self.tf_buffer = Buffer()
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscribe to AprilTag detections
        self.apriltag_sub = self.create_subscription(
            AprilTagDetectionArray,
            '/detections',
            self.apriltag_callback,
            10
        )

        # Frame names (must match tf_publisher.py)
        self.odom_frame = 'map'
        self.base_frame = 'base_link'

        # Estimated pose (updated by dead reckoning and AprilTag correction)
        self.x_est = 0.0
        self.y_est = 0.0
        self.yaw = 0.0

        # Detected AprilTags
        self.detected_tags = []

        # State machine: 'dead_reckoning' or 'correction'
        self.state = 'dead_reckoning'
        self.state_start_time = self.get_clock().now()

        # Tracking state
        self.current_waypoint_idx = 0
        self.last_time = self.get_clock().now()
        self.timer = self.create_timer(1.0 / self.rate_hz, self.control_step)

        # Statistics
        self.correction_count = 0
        self.total_corrections_applied = 0

        self.get_logger().info(f'Loaded {len(self.waypoints)} waypoints.')
        self.get_logger().info(f'Vision-based follower starting at {self.rate_hz:.1f} Hz')
        self.get_logger().info(f'Dead reckoning: {self.dead_reckoning_duration}s, Correction: {self.correction_duration}s')

    def load_apriltag_map(self):
        """
        Load AprilTag map from file or use default map.
        Returns dict: {tag_id: (x, y, yaw)}
        """
        # Try to load from file
        try:
            import json
            with open('/home/ubuntu/ros2_ws/rubikpi_ros2/robot_control/apriltag_map.json', 'r') as f:
                data = json.load(f)
                apriltag_map = {}
                for tag in data['tags']:
                    apriltag_map[tag['id']] = (tag['x'], tag['y'], tag['yaw'])
                self.get_logger().info(f'Loaded AprilTag map with {len(apriltag_map)} tags')
                self.get_logger().info(f'AprilTag map: {apriltag_map}')
                return apriltag_map
        except Exception as e:
            self.get_logger().warn(f'Could not load apriltag_map.json: {e}')
            exit(1)

    def load_waypoints(self, path: str) -> List[Tuple[float, float, float]]:
        """
        Load waypoints from a text file.
        Each line: x, y, theta (in radians)
        """
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

    def apriltag_callback(self, msg: AprilTagDetectionArray):
        """
        Callback for AprilTag detections.
        Store which tags are currently visible with sufficient confidence.
        """
        self.detected_tags = []
        for detection in msg.detections:
            tag_id = detection.id
            decision_margin = detection.decision_margin

            # Only use high-confidence detections
            if decision_margin >= self.min_detection_confidence:
                self.detected_tags.append(tag_id)

        if len(self.detected_tags) > 0:
            self.get_logger().debug(f'Detected tags: {self.detected_tags}')

    def quaternion_to_yaw(self, qx: float, qy: float, qz: float, qw: float) -> float:
        """
        Convert quaternion to yaw angle (rotation around Z-axis)
        """
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = atan2(siny_cosp, cosy_cosp)
        return yaw

    def broadcast_apriltag_odom_transforms(self, full_tag_id: str):
        """
        Broadcast static transforms from odom frame to each AprilTag.

        This creates the TF chain:
        odom <- tag_X <- camera_frame <- base_link

        By looking up odom -> base_link reversely handled by tf2, we can localize the robot in the odom frame.
        """
        static_transforms = []

        for tag_id, (x, y, yaw) in self.apriltag_map.items():
            if str(tag_id) != full_tag_id:
                continue
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = f'tag_{tag_id}'
            t.child_frame_id = self.odom_frame  # odom frame

            # Position of the tag in odom frame
            t.transform.translation.x = -1 *float(x)
            t.transform.translation.y = -1 * float(y)
            t.transform.translation.z = 0.0  # Assuming tags are on the ground plane

            # Orientation of the tag in odom frame
            qx, qy, qz, qw = self.euler_to_quaternion(0.0, 0.0, float(-yaw))
            t.transform.rotation.x = qx
            t.transform.rotation.y = qy
            t.transform.rotation.z = qz
            t.transform.rotation.w = qw

            static_transforms.append(t)

        if len(static_transforms) > 0:
            self.static_tf_broadcaster.sendTransform(static_transforms)
            self.get_logger().info(f'Published {len(static_transforms)} static transforms: tag_{tag_id} -> odom')

    def euler_to_quaternion(self, roll, pitch, yaw):
        """
        Convert Euler angles (roll, pitch, yaw) to quaternion (x, y, z, w)
        """
        cy = cos(yaw * 0.5)
        sy = sin(yaw * 0.5)
        cp = cos(pitch * 0.5)
        sp = sin(pitch * 0.5)
        cr = cos(roll * 0.5)
        sr = sin(roll * 0.5)
        
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        
        return qx, qy, qz, qw

    def apply_apriltag_correction(self):
        """
        Use AprilTag observations to correct robot pose estimate via TF lookup.

        TF Chain: odom -> tag_X -> camera_frame -> base_link

        By looking up the transform from odom to base_link, we directly get
        the robot's pose in the odom frame.

        Process:
        1. For each detected tag, lookup transform: odom -> base_link (via tag)
        2. Extract robot position (x, y) and orientation (yaw) in odom frame
        3. Average multiple tag observations if available
        4. Update robot state estimate
        """
        if len(self.detected_tags) == 0:
            self.get_logger().info('No AprilTags detected. Skipping correction.')
            return

        self.get_logger().info(f'Attempting correction with tags: {self.detected_tags}')

        # Collect pose estimates from each detected tag
        pose_estimates = []

        for tag_id in self.detected_tags:
            try:
                # publish the static TF from tag_id to map
                self.broadcast_apriltag_odom_transforms(str(tag_id))
                
                # Look up transform from odom to base_link through the tag
                # TF chain: odom -> tag_X -> camera_frame -> base_link
                transform = self.tf_buffer.lookup_transform(
                    self.odom_frame,
                    self.base_frame,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.1)
                )

                # Extract robot position in odom frame
                robot_x = transform.transform.translation.x
                robot_y = transform.transform.translation.y

                # Extract robot orientation (yaw) in odom frame
                qx = transform.transform.rotation.x
                qy = transform.transform.rotation.y
                qz = transform.transform.rotation.z
                qw = transform.transform.rotation.w

                # Convert quaternion to yaw
                robot_yaw = self.quaternion_to_yaw(qx, qy, qz, qw)

                pose_estimates.append({
                    'tag_id': tag_id,
                    'x': robot_x,
                    'y': robot_y,
                    'yaw': robot_yaw
                })

                self.get_logger().info(
                    f'Tag {tag_id}: Estimated pose = ({robot_x:.3f}, {robot_y:.3f}, {math.degrees(robot_yaw):.1f}°)'
                )

            except Exception as e:
                self.get_logger().warn(f'Failed to lookup transform for tag {tag_id}: {e}')
                continue

        # Update pose estimate using average of all tag observations
        if len(pose_estimates) > 0:
            # Simple average (could be weighted by distance or confidence)
            avg_x = sum(est['x'] for est in pose_estimates) / len(pose_estimates)
            avg_y = sum(est['y'] for est in pose_estimates) / len(pose_estimates)

            # For yaw, use circular mean (via sin/cos components)
            avg_yaw_sin = sum(math.sin(est['yaw']) for est in pose_estimates) / len(pose_estimates)
            avg_yaw_cos = sum(math.cos(est['yaw']) for est in pose_estimates) / len(pose_estimates)
            avg_yaw = atan2(avg_yaw_sin, avg_yaw_cos)

            # Log the correction
            old_x, old_y, old_yaw = self.x_est, self.y_est, self.yaw

            # Update pose estimate
            self.x_est = avg_x
            self.y_est = avg_y
            self.yaw = avg_yaw

            self.correction_count += 1
            self.total_corrections_applied += 1

            self.get_logger().info(
                f'CORRECTION #{self.total_corrections_applied}: '
                f'Pose updated from ({old_x:.3f}, {old_y:.3f}, {math.degrees(old_yaw):.1f}°) '
                f'to ({self.x_est:.3f}, {self.y_est:.3f}, {math.degrees(self.yaw):.1f}°) '
                f'using {len(pose_estimates)} tag(s)'
            )
        else:
            self.get_logger().warn('No valid pose estimates from AprilTags.')

    def diff2velocity_simple(self, dist: float, heading_error: float, gtheta: float, theta: float) -> Tuple[float, float]:
        """
        Simple proportional controller to compute linear and angular velocity commands.
        """
        if dist < self.pos_tol:
            # Waypoint reached - move to next waypoint
            self.current_waypoint_idx += 1
            self.get_logger().info(f'Reached waypoint {self.current_waypoint_idx}/{len(self.waypoints)}')
            v_cmd, w_cmd = 0.0, 0.0
        elif abs(heading_error) > self.yaw_tol:
            # Large heading error - rotate in place
            v_cmd, w_cmd = 0.0, math.copysign(self.max_angular_speed_radps, heading_error)
        else:
            # Proceed forward with heading correction
            v_cmd = self.max_speed_mps
            w_cmd = 0 # math.copysign(self.max_angular_speed_radps * abs(heading_error) / math.pi, heading_error)

        v_cmd = v_cmd * self.kv
        w_cmd = w_cmd * self.kh
        return v_cmd, w_cmd

    def velocity2input(self, v_cmd: float, w_cmd: float) -> Tuple[float, float]:
        """
        Convert linear and angular velocity commands to normalized motor inputs.
        """
        if v_cmd == 0 and w_cmd == 0:
            return 0.0, 0.0
        if w_cmd == 0:
            # Pure forward/backward
            l1 = self.linear_min + v_cmd * (0.5 - self.linear_min) / self.max_speed_mps
            l1 = l1 * v_cmd / abs(v_cmd)
            r1 = l1
            return l1, r1
        elif v_cmd == 0:
            # Pure rotation
            r2 = math.copysign(self.angular_min + abs(w_cmd) * (0.5 - self.angular_min) / self.max_angular_speed_radps, w_cmd)
            l2 = -1 * r2
            return l2, r2
        else:
            # Mixture of both
            l1, r1 = v_cmd * (0.5 - self.linear_min) / self.max_speed_mps, v_cmd * (0.5 - self.linear_min) / self.max_speed_mps
            l2, r2 = -1 * w_cmd * (0.5 - self.angular_min) / self.max_angular_speed_radps, w_cmd * (0.5 - self.angular_min) / self.max_angular_speed_radps
            l_thres, r_thres = self.linear_min, self.linear_min
            l = l1 + l2 + l_thres
            r = r1 + r2 + r_thres
            l = max(-self.max_norm_cmd, min(self.max_norm_cmd, l))
            r = max(-self.max_norm_cmd, min(self.max_norm_cmd, r))
            return l, r

    def control_step(self):
        """
        Main control loop with state machine.

        States:
        - 'dead_reckoning': Move for 0.5s using dead reckoning
        - 'correction': Stop for 0.5s and update pose using AprilTag TF
        """
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        if dt < 0.0001:
            return
        self.last_time = now

        # Check if all waypoints are completed
        if self.current_waypoint_idx >= len(self.waypoints):
            self.publish_cmd(0.0, 0.0)
            self.get_logger().info(
                f'All waypoints completed! Total corrections applied: {self.total_corrections_applied}'
            )
            return
        
        self.points.append([self.x_est, self.y_est, self.yaw])

        # State machine timing
        time_in_state = (now - self.state_start_time).nanoseconds / 1e9

        # State transitions
        if self.state == 'dead_reckoning' and time_in_state >= self.dead_reckoning_duration:
            # Transition to correction phase
            self.state = 'correction'
            self.state_start_time = now
            self.correction_count = 0
            self.get_logger().info('=== ENTERING CORRECTION PHASE ===')
            self.publish_cmd(0.0, 0.0)  # Stop the robot
            return

        elif self.state == 'correction' and time_in_state >= self.correction_duration:
            # Transition to dead reckoning phase
            self.state = 'dead_reckoning'
            self.state_start_time = now
            self.get_logger().info('=== ENTERING DEAD RECKONING PHASE ===')

        # Execute state-specific behavior
        if self.state == 'correction':
            # Stay stopped and apply AprilTag correction once
            self.apply_apriltag_correction()
            self.correction_count += 1
            self.publish_cmd(0.0, 0.0)
            return

        elif self.state == 'dead_reckoning':
            # Normal waypoint following with dead reckoning
            # Desired waypoint
            gx, gy, gtheta = self.waypoints[self.current_waypoint_idx]

            # Compute control based on current estimated state
            dx = gx - self.x_est
            dy = gy - self.y_est
            dist = math.hypot(dx, dy)

            # Use current estimated yaw
            theta = self.yaw

            # Heading to goal
            path_angle = math.atan2(dy, dx)
            heading_error = normalize_angle(path_angle - theta)

            # Linear and angular velocity commands (m/s, rad/s)
            v_cmd, w_cmd = self.diff2velocity_simple(dist, heading_error, gtheta, theta)

            # Convert to motor commands and publish
            l_norm, r_norm = self.velocity2input(v_cmd, w_cmd)
            self.publish_cmd(l_norm, r_norm)

            # Integrate position and orientation estimate using commanded velocities (dead reckoning)
            self.x_est += v_cmd * math.cos(theta) * dt
            self.y_est += v_cmd * math.sin(theta) * dt
            self.yaw += w_cmd * dt
            self.yaw = normalize_angle(self.yaw)

            # self.get_logger().info(
            #     f'[DR] x: {self.x_est:.3f}, y: {self.y_est:.3f}, yaw: {math.degrees(self.yaw):.1f}°, '
            #     f'v: {v_cmd:.2f}, w: {w_cmd:.2f}, dist: {dist:.3f}m, heading_err: {math.degrees(heading_error):.1f}°'
            # )

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
        # save node.points to npy file
        np.save('robot_path.npy', np.array(node.points))
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        return 0


if __name__ == '__main__':
    main()