import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from std_msgs.msg import Float32MultiArray
from apriltag_msgs.msg import AprilTagDetectionArray
from tf2_ros import Buffer, TransformListener
import math
import os
from typing import List, Tuple, Dict, Optional
from math import sin, cos, atan2, sqrt


def normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class WaypointFollowerNode(Node):
    def __init__(self):
        super().__init__('waypoint_follower_node')

        # Parameters
        # differential drive robot parameters
        self.declare_parameter('waypoints_path', 'waypoints.txt')
        self.declare_parameter('apriltag_map_path', 'apriltag_map.json')
        self.declare_parameter('wheel_base', 0.04)  # meters (wheel radius)
        self.declare_parameter('car_width', 0.168)  # meters (car width)
        self.declare_parameter('max_speed_mps', 0.68)  # approximate max linear speed
        self.declare_parameter('max_angular_speed_radps', math.pi * 3 / 2)  # approximate max angular speed
        self.declare_parameter('max_norm_cmd', 0.5)  # matches motor_controller clamp
        self.declare_parameter('kv', 0.8)  # linear gain
        self.declare_parameter('kh', 0.8)  # heading gain
        self.declare_parameter('pos_tolerance', 0.05)  # meters
        self.declare_parameter('yaw_tolerance', 0.1)  # radians (~11 degrees)
        self.declare_parameter('rate_hz', 20.0)
        
        # AprilTag parameters
        self.declare_parameter('use_apriltag_localization', True)
        self.declare_parameter('apriltag_correction_weight', 0.3)  # fusion weight
        self.declare_parameter('apriltag_yaw_weight', 0.3)  # yaw fusion weight
        self.declare_parameter('min_detection_confidence', 0.5)
        self.declare_parameter('max_tag_distance', 2.0)  # meters
        self.declare_parameter('camera_frame', 'camera_frame')

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
        
        # AprilTag parameters
        self.use_apriltag = bool(self.get_parameter('use_apriltag_localization').value)
        self.apriltag_pos_weight = float(self.get_parameter('apriltag_correction_weight').value)
        self.apriltag_yaw_weight = float(self.get_parameter('apriltag_yaw_weight').value)
        self.min_detection_confidence = float(self.get_parameter('min_detection_confidence').value)
        self.max_tag_distance = float(self.get_parameter('max_tag_distance').value)
        self.camera_frame = self.get_parameter('camera_frame').value

        self.linear_min = 0.1
        self.angular_min = 0.25

        # Load waypoints
        waypoints_path = self.get_parameter('waypoints_path').value
        self.waypoints: List[Tuple[float, float, float]] = self.load_waypoints(waypoints_path)
        if not self.waypoints:
            self.get_logger().error(f'No waypoints loaded from {waypoints_path}. Stopping.')
            raise RuntimeError('waypoints_empty')

        # Load AprilTag map
        self.apriltag_map: Dict[int, Tuple[float, float, float]] = {}
        if self.use_apriltag:
            apriltag_map_path = self.get_parameter('apriltag_map_path').value
            self.apriltag_map = self.load_apriltag_map(apriltag_map_path)
            self.get_logger().info(f'Loaded {len(self.apriltag_map)} AprilTags from map')

        # Publisher for motor commands
        self.cmd_pub = self.create_publisher(Float32MultiArray, 'motor_commands', 10)

        # Subscriber for AprilTag detections
        if self.use_apriltag:
            self.apriltag_sub = self.create_subscription(
                AprilTagDetectionArray,
                '/detections',
                self.apriltag_callback,
                10
            )
            
            # TF2 buffer for looking up tag positions
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)
            
            self.detected_tags: List[int] = []

        # Estimated pose (x, y from integrated v; yaw from integrated angular velocity)
        self.x_est = 0.0
        self.y_est = 0.0
        self.yaw = 0.0

        # Tracking state
        self.current_waypoint_idx = 0
        self.last_time = self.get_clock().now()
        self.timer = self.create_timer(1.0 / self.rate_hz, self.control_step)

        self.get_logger().info(f'Loaded {len(self.waypoints)} waypoints. Starting follower at {self.rate_hz:.1f} Hz')
        if self.use_apriltag:
            self.get_logger().info(f'AprilTag localization enabled with {len(self.apriltag_map)} tags')

    def load_waypoints(self, path: str) -> List[Tuple[float, float, float]]:
        """
        Load waypoints from a text file.
        Each line in the file should contain three comma-separated values: x, y, theta (in radians).
        Comments and empty lines are ignored.
        :param path: Path to the waypoint file.
        :return: List of waypoints as (x, y, theta) tuples.
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

    def load_apriltag_map(self, path: str) -> Dict[int, Tuple[float, float, float]]:
        """
        Load AprilTag map from JSON file.
        Expected format: {"tags": [{"id": 0, "x": 1.0, "y": 2.0, "yaw": 0.0}, ...]}
        :param path: Path to the AprilTag map file.
        :return: Dict mapping tag_id to (x, y, yaw).
        """
        import json
        apriltag_map = {}
        try:
            if not os.path.isabs(path):
                cwd_path = os.path.join(os.getcwd(), path)
                if os.path.exists(cwd_path):
                    path = cwd_path
            
            with open(path, 'r') as f:
                data = json.load(f)
                if 'tags' in data:
                    for tag in data['tags']:
                        tag_id = int(tag['id'])
                        x = float(tag['x'])
                        y = float(tag['y'])
                        yaw = float(tag.get('yaw', 0.0))
                        apriltag_map[tag_id] = (x, y, yaw)
        except Exception as e:
            self.get_logger().error(f'Failed to load AprilTag map from {path}: {e}')
        return apriltag_map

    def apriltag_callback(self, msg: AprilTagDetectionArray):
        """
        Callback for AprilTag detections.
        Updates list of currently detected tags.
        """
        self.detected_tags = []
        for detection in msg.detections:
            tag_id = detection.id
            decision_margin = detection.decision_margin
            if decision_margin >= self.min_detection_confidence:
                self.detected_tags.append(tag_id)

    def estimate_yaw_from_apriltag(self) -> Optional[float]:
        """
        Estimate robot yaw using bearing to known tag location.
        
        Formula: yaw = expected_bearing - bearing_cam
        where:
          - expected_bearing = atan2(tag_y - robot_y, tag_x - robot_x)
          - bearing_cam = atan2(tag_x_cam, tag_z_cam) in camera frame
        
        Returns: estimated_yaw or None if estimation failed
        """
        if len(self.detected_tags) == 0:
            return None
        
        tag_id = self.detected_tags[0]
        
        if tag_id not in self.apriltag_map:
            return None
        
        try:
            # Get tag position in world frame from map
            tag_x_world, tag_y_world, _ = self.apriltag_map[tag_id]
            
            # Calculate expected bearing (from robot to tag)
            expected_bearing = atan2(
                tag_y_world - self.y_est,
                tag_x_world - self.x_est
            )
            
            # Get tag position in camera frame via TF
            transform = self.tf_buffer.lookup_transform(
                self.camera_frame,
                f'tag_{tag_id}',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            
            tag_x_cam = transform.transform.translation.x
            tag_z_cam = transform.transform.translation.z
            
            # Calculate measured bearing in camera frame
            bearing_cam = atan2(tag_x_cam, tag_z_cam)
            
            # Solve for yaw: expected_bearing = yaw + bearing_cam
            estimated_yaw = expected_bearing - bearing_cam
            estimated_yaw = normalize_angle(estimated_yaw)
            
            self.get_logger().info(
                f'Yaw estimation: tag_id={tag_id}, expected_bearing={math.degrees(expected_bearing):.1f}°, '
                f'bearing_cam={math.degrees(bearing_cam):.1f}°, estimated_yaw={math.degrees(estimated_yaw):.1f}°'
            )
            
            return estimated_yaw
            
        except Exception as e:
            self.get_logger().debug(f'Failed to estimate yaw from tag {tag_id}: {e}')
            return None

    def apply_apriltag_correction(self):
        """
        Apply AprilTag-based localization corrections to position and yaw estimates.
        
        PROCESS:
        1. Estimate yaw from bearing (corrects dead reckoning drift)
        2. Use corrected yaw for position calculation
        3. Fuse position and yaw with dead reckoning estimates
        """
        if not self.use_apriltag or len(self.detected_tags) == 0:
            return
        
        # STEP 1: Estimate and update yaw
        estimated_yaw = self.estimate_yaw_from_apriltag()
        
        if estimated_yaw is not None:
            # Fuse with dead reckoning
            old_yaw = self.yaw
            self.yaw = (1 - self.apriltag_yaw_weight) * self.yaw + self.apriltag_yaw_weight * estimated_yaw
            self.yaw = normalize_angle(self.yaw)

            self.get_logger().info(
                f'Yaw updated: {math.degrees(old_yaw):.1f}° → {math.degrees(self.yaw):.1f}°'
            )

        # STEP 2: Correct position based on detected tags
        position_corrections = []
        
        for tag_id in self.detected_tags:
            if tag_id not in self.apriltag_map:
                continue
            
            try:
                # Get tag position in world frame
                tag_x_world, tag_y_world, _ = self.apriltag_map[tag_id]
                
                # Get tag position in camera frame
                transform = self.tf_buffer.lookup_transform(
                    self.camera_frame,
                    f'tag_{tag_id}',
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.1)
                )
                
                tag_x_cam = transform.transform.translation.x
                tag_y_cam = transform.transform.translation.y
                tag_z_cam = transform.transform.translation.z
                
                # Calculate 2D distance
                distance_2d = sqrt(tag_x_cam**2 + tag_z_cam**2)
                distance_3d = sqrt(tag_x_cam**2 + tag_y_cam**2 + tag_z_cam**2)
                
                # Filter by distance
                if distance_3d > self.max_tag_distance:
                    self.get_logger().debug(f'Tag {tag_id} too far: {distance_3d:.2f}m')
                    continue
                
                # Calculate bearing in camera frame
                bearing_cam = atan2(tag_x_cam, tag_z_cam)
                
                # Convert to global bearing using corrected yaw
                global_bearing = self.yaw + bearing_cam
                
                # Calculate robot position (opposite direction from tag)
                robot_x_corrected = tag_x_world - distance_2d * cos(global_bearing)
                robot_y_corrected = tag_y_world - distance_2d * sin(global_bearing)
                
                position_corrections.append((robot_x_corrected, robot_y_corrected, distance_2d))
                
                self.get_logger().info(
                    f'Tag {tag_id}: distance={distance_2d:.2f}m, corrected_pos=({robot_x_corrected:.2f}, {robot_y_corrected:.2f})'
                )
                
            except Exception as e:
                self.get_logger().debug(f'Failed to process tag {tag_id}: {e}')
                continue
        
        # STEP 3: Fuse position corrections with dead reckoning
        if len(position_corrections) > 0:
            # Weight by inverse distance (closer tags are more reliable)
            total_weight = 0.0
            weighted_x = 0.0
            weighted_y = 0.0
            
            for x, y, dist in position_corrections:
                weight = 1.0 / max(dist, 0.1)
                weighted_x += x * weight
                weighted_y += y * weight
                total_weight += weight
            
            if total_weight > 0:
                avg_x = weighted_x / total_weight
                avg_y = weighted_y / total_weight
                
                # Fuse with dead reckoning
                old_x, old_y = self.x_est, self.y_est
                self.x_est = (1 - self.apriltag_pos_weight) * self.x_est + self.apriltag_pos_weight * avg_x
                self.y_est = (1 - self.apriltag_pos_weight) * self.y_est + self.apriltag_pos_weight * avg_y
                
                self.get_logger().info(
                    f'Position corrected: ({old_x:.2f}, {old_y:.2f}) → ({self.x_est:.2f}, {self.y_est:.2f})'
                )
    
    def diff2velocity_simple(self, dist: float, heading_error: float, gtheta: float, theta: float) -> Tuple[float, float]:
        """
        Simple proportional controller to compute linear and angular velocity commands.
        :param dist: Distance to the goal waypoint.
        :param heading_error: Heading error to the goal waypoint.
        :param gtheta: Goal orientation (not used in this simple controller).
        :param theta: Current orientation (not used in this simple controller).
        :return: Tuple of (v_cmd, w_cmd) in m/s and rad/s.
        """
        if dist < self.pos_tol:
            # Waypoint fully reached - move to next waypoint
            self.current_waypoint_idx += 1
            self.get_logger().info(f'Reached waypoint {self.current_waypoint_idx}/{len(self.waypoints)}')
            v_cmd, w_cmd = 0.0, 0.0
        elif abs(heading_error) > self.yaw_tol:
            # Large heading error - rotate in place
            v_cmd, w_cmd = 0.0, math.copysign(self.max_angular_speed_radps, heading_error)
        else:
            # Proceed forward with heading correction
            v_cmd = self.max_speed_mps
            w_cmd = math.copysign(self.max_angular_speed_radps * abs(heading_error) / math.pi, heading_error)
        
        v_cmd = v_cmd * self.kv
        w_cmd = w_cmd * self.kh
        return v_cmd, w_cmd
    
    def velocity2input(self, v_cmd: float, w_cmd: float) -> Tuple[float, float]:
        """
        Convert linear and angular velocity commands to normalized motor inputs.
        :param v_cmd: Desired linear velocity in m/s. 
        :param w_cmd: Desired angular velocity in rad/s.
        :return: Tuple of (left_motor_input, right_motor_input) in range [-max_norm_cmd, max_norm_cmd].
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
            # Mixture of linear and angular
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
        Main control loop step.
        1. Apply AprilTag corrections if available
        2. Compute time delta since last step
        3. If all waypoints are completed, stop and return
        4. Get current goal waypoint
        5. Compute distance and heading error to goal
        6. Compute velocity commands using diff2velocity_simple
        7. Convert to motor commands and publish
        8. Integrate position and orientation estimate using commanded velocities
        9. Log current state
        :return: None
        """
        # Apply AprilTag corrections
        if self.use_apriltag:
            self.apply_apriltag_correction()
        
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        if dt < 0.0001:
            return
        self.last_time = now

        # Check if all waypoints are completed
        if self.current_waypoint_idx >= len(self.waypoints):
            self.publish_cmd(0.0, 0.0)
            self.get_logger().info('All waypoints completed!')
            return

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

        # Integrate position and orientation estimate using commanded velocities
        self.x_est += v_cmd * math.cos(theta) * dt
        self.y_est += v_cmd * math.sin(theta) * dt
        
        # Integrate yaw using angular velocity
        self.yaw += w_cmd * dt
        self.yaw = normalize_angle(self.yaw)
        
        self.get_logger().info(
            f"x: {self.x_est:.2f}, y: {self.y_est:.2f}, yaw: {math.degrees(self.yaw):.1f}°, "
            f"v_cmd: {v_cmd:.2f}, w_cmd: {w_cmd:.2f}, l_norm: {l_norm:.2f}, r_norm: {r_norm:.2f}, "
            f"heading_error: {math.degrees(heading_error):.1f}°, dist: {dist:.2f}"
        )

    def publish_cmd(self, left: float, right: float):
        msg = Float32MultiArray()
        msg.data = [float(left), float(right)]
        self.cmd_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = WaypointFollowerNode()
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