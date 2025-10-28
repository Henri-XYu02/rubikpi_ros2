#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
from apriltag_msgs.msg import AprilTagDetectionArray
import numpy as np
from math import sin, cos, atan2, sqrt
import json

"""
Closed-loop waypoint follower using AprilTag detection for localization.
This version improves upon the open-loop dead reckoning solution by fusing
AprilTag observations to correct position estimates.
"""

class PIDcontroller:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.target = None
        self.I = np.array([0.0, 0.0])
        self.lastError = np.array([0.0, 0.0])
        self.timestep = 0.1
        self.maximumValue = 0.2

    def setTarget(self, state):
        """
        set the target pose.
        """
        self.I = np.array([0.0, 0.0])
        self.lastError = np.array([0.0, 0.0])
        self.target = np.array(state)

    def getError(self, currentState, targetState, drive_backwards):
        """
        return the error between current and target state
        for differential drive: distance error and heading error
        """
        delta_x = targetState[0] - currentState[0]
        delta_y = targetState[1] - currentState[1]

        distance = np.sqrt(delta_x**2 + delta_y**2)

        angle_to_target = np.arctan2(delta_y, delta_x)

        if drive_backwards:
            desired_heading = angle_to_target + np.pi
            desired_heading = (desired_heading + np.pi) % (2 * np.pi) - np.pi
            distance = -distance
        else:
            desired_heading = angle_to_target

        heading_error = desired_heading - currentState[2]
        heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi

        if abs(distance) < 0.05:
            heading_error = targetState[2] - currentState[2]
            heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi
            distance = 0.0

        return np.array([distance, heading_error])

    def setMaximumUpdate(self, mv):
        """
        set maximum velocity for stability.
        """
        self.maximumValue = mv

    def update(self, currentState, drive_backwards):
        """
        calculate the update value based on PID control
        Returns: [linear_velocity, angular_velocity]
        """
        e = self.getError(currentState, self.target, drive_backwards)

        P = self.Kp * e
        self.I = self.I + self.Ki * e * self.timestep
        I = self.I
        D = self.Kd * (e - self.lastError)
        result = P + I + D

        self.lastError = e

        if abs(result[0]) > self.maximumValue:
            result[0] = np.sign(result[0]) * self.maximumValue

        max_angular = 1.5
        if abs(result[1]) > max_angular:
            result[1] = np.sign(result[1]) * max_angular

        if abs(e[0]) < 0.05:
            result[0] = 0.0
        return result


class Hw1ClosedLoopNode(Node):
    def __init__(self):
        super().__init__('hw1_closedloop_node')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # TF2 for AprilTag transforms
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscribe to AprilTag detections to know which tags are visible
        self.apriltag_sub = self.create_subscription(
            AprilTagDetectionArray,
            '/detections',
            self.apriltag_callback,
            10
        )

        # Frame names
        self.odom_frame = 'odom'
        self.base_frame = 'base_link'
        self.camera_frame = 'camera_frame'

        # Waypoints (same as open-loop for testing)
        self.waypoints = np.array([
            [1, 0.0, 0.0],
            [1, -1, -np.pi/4],
        ])

        # AprilTag map: tag_id -> (x, y, yaw) in world frame
        # This should be loaded from a JSON file in practice
        self.apriltag_map = self.load_apriltag_map()

        # PID controller
        self.pid = PIDcontroller(0.5, 0.01, 0.005)

        # State estimation
        self.current_state = np.array([0.0, 0.0, 0.0])  # [x, y, yaw]

        # AprilTag correction parameters
        self.use_apriltag_correction = True
        self.apriltag_correction_weight = 0.5  # 0=dead reckoning only, 1=trust tags completely
        self.min_detection_confidence = 0.5
        self.max_tag_distance = 3.0
        self.correction_count = 0

        # Detected tags
        self.detected_tags = []

        # Waypoint tracking
        self.current_waypoint_idx = 0
        self.waypoint_reached = False
        self.tolerance = 0.1
        self.angle_tolerance = 0.1

        # Control parameters
        self.drive_backwards = False
        self.dt = 0.1
        self.control_timer = self.create_timer(self.dt, self.control_loop)

        # Control stages
        self.stage = 'rotate_to_goal'
        self.stage_pid = PIDcontroller(0.5, 0.01, 0.005)
        self.fixed_rotation_vel = 0.785

        self.get_logger().info('Closed-loop AprilTag waypoint follower started')
        self.get_logger().info(f'AprilTag correction: {self.use_apriltag_correction}')
        self.get_logger().info(f'Correction weight: {self.apriltag_correction_weight}')

    def load_apriltag_map(self):
        """
        Load AprilTag map from file or use default map.
        Returns dict: {tag_id: (x, y, yaw)}
        """
        # Try to load from file
        try:
            with open('/home/ubuntu/ros2_ws/rubikpi_ros2/hw_1_solution/apriltag_map.json', 'r') as f:
                data = json.load(f)
                apriltag_map = {}
                for tag in data['tags']:
                    apriltag_map[tag['id']] = (tag['x'], tag['y'], tag['yaw'])
                self.get_logger().info(f'Loaded AprilTag map with {len(apriltag_map)} tags')
                return apriltag_map
        except Exception as e:
            self.get_logger().warn(f'Could not load apriltag_map.json: {e}')
            exit(1)

    def apriltag_callback(self, msg):
        """
        Callback for AprilTag detections.
        Store which tags are currently visible.
        """
        self.detected_tags = []
        self.get_logger().info(f"Message received: {msg}")
        for detection in msg.detections:
            tag_id = detection.id
            decision_margin = detection.decision_margin

            # Only use high-confidence detections
            if decision_margin >= self.min_detection_confidence:
                self.detected_tags.append(tag_id)

        if len(self.detected_tags) > 0:
            self.get_logger().debug(f'Detected tags: {self.detected_tags}')

    def estimate_yaw_from_apriltag(self):
        """
        METHOD 2: Estimate robot yaw using bearing to known tag location.
        
        This corrects dead reckoning by comparing:
        - Expected bearing: direction from robot position to tag (world frame)
        - Measured bearing: where tag appears in camera frame
        
        Formula: yaw = expected_bearing - bearing_cam
        
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
            
            # Calculate expected bearing (what we should see if position estimate is correct)
            expected_bearing = atan2(
                tag_y_world - self.current_state[1],
                tag_x_world - self.current_state[0]
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
            
            # Normalize to [-π, π]
            estimated_yaw = (estimated_yaw + np.pi) % (2 * np.pi) - np.pi
            
            self.get_logger().info(
                f'YAW ESTIMATION: expected_bearing={np.rad2deg(expected_bearing):.1f}°, '
                f'bearing_cam={np.rad2deg(bearing_cam):.1f}°, '
                f'estimated_yaw={np.rad2deg(estimated_yaw):.1f}°'
            )
            
            return estimated_yaw
            
        except Exception as e:
            self.get_logger().debug(f'Failed to estimate yaw: {e}')
            return None

    def apply_apriltag_correction(self):
        """
        Use AprilTag observations to correct robot pose estimate.
        Fuses dead reckoning with AprilTag measurements for BOTH position and yaw.
        
        PROCESS:
        1. Estimate yaw from bearing (corrects dead reckoning drift)
        2. Use corrected yaw for position calculation
        3. Update both position and yaw consistently
        """
        if not self.use_apriltag_correction or len(self.detected_tags) == 0:
            return
        
        self.get_logger().info(f'Applying AprilTag correction with tags: {self.detected_tags}')

        # STEP 1: Estimate and update yaw from bearing
        estimated_yaw = self.estimate_yaw_from_apriltag()
        
        if estimated_yaw is not None:
            # Fuse estimated yaw with dead reckoning
            alpha_yaw = 0.1  # 30% AprilTag, 70% dead reckon (conservative)
            
            old_yaw = self.current_state[2]
            self.current_state[2] = (1 - alpha_yaw) * self.current_state[2] + alpha_yaw * estimated_yaw
            self.current_state[2] = (self.current_state[2] + np.pi) % (2 * np.pi) - np.pi
            
            self.get_logger().info(
                f'YAW UPDATED: {np.rad2deg(old_yaw):.1f}° → {np.rad2deg(self.current_state[2]):.1f}° '
                f'(estimated: {np.rad2deg(estimated_yaw):.1f}°)'
            )

        # STEP 2: Process each detected tag for position correction
        corrections = []

        for tag_id in self.detected_tags:
            # Check if we know this tag's position
            if tag_id not in self.apriltag_map:
                self.get_logger().debug(f'Tag {tag_id} not in map, skipping')
                continue

            # Get tag position in world frame from map
            tag_x_world, tag_y_world, tag_yaw_world = self.apriltag_map[tag_id]
            self.get_logger().debug(f'Tag {tag_id} world pos: ({tag_x_world}, {tag_y_world}), yaw: {tag_yaw_world}')

            try:
                # Look up transform from camera to tag
                transform = self.tf_buffer.lookup_transform(
                    self.camera_frame,
                    f'tag_{tag_id}',
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.1)
                )

                # Extract tag position in camera frame
                tag_x_cam = transform.transform.translation.x
                tag_y_cam = transform.transform.translation.y
                tag_z_cam = transform.transform.translation.z

                self.get_logger().debug(f'Tag {tag_id} camera pos: ({tag_x_cam}, {tag_y_cam}, {tag_z_cam})')

                # Calculate 2D distance (horizontal plane only)
                distance_2d = sqrt(tag_x_cam**2 + tag_z_cam**2)
                distance_3d = sqrt(tag_x_cam**2 + tag_y_cam**2 + tag_z_cam**2)

                # Ignore tags that are too far away
                if distance_3d > self.max_tag_distance:
                    self.get_logger().debug(f'Tag {tag_id} too far ({distance_3d:.2f}m), skipping')
                    continue

                # Calculate bearing to tag in camera frame
                bearing_cam = atan2(tag_x_cam, tag_z_cam)
                
                self.get_logger().debug(f'Tag {tag_id} distance_2d: {distance_2d:.2f}m, bearing_cam: {np.rad2deg(bearing_cam):.1f}°')
                
                # Convert to global bearing using CORRECTED yaw (from yaw estimation)
                global_bearing = self.current_state[2] + bearing_cam
                
                self.get_logger().debug(f'Tag {tag_id} global_bearing: {np.rad2deg(global_bearing):.1f}° (with corrected yaw={np.rad2deg(self.current_state[2]):.1f}°)')

                # Calculate robot position based on tag observation
                robot_x_corrected = tag_x_world - distance_2d * cos(global_bearing)
                robot_y_corrected = tag_y_world - distance_2d * sin(global_bearing)

                corrections.append((robot_x_corrected, robot_y_corrected, distance_2d))

                self.get_logger().debug(
                    f'Tag {tag_id}: bearing_cam={np.rad2deg(bearing_cam):.1f}°, '
                    f'corrected_pos=({robot_x_corrected:.2f}, {robot_y_corrected:.2f})'
                )

            except Exception as e:
                self.get_logger().debug(f'Failed to lookup transform for tag {tag_id}: {e}')
                continue

        # STEP 3: Apply position corrections using weighted average
        if len(corrections) > 0:
            # Weight corrections by inverse distance (closer tags are more reliable)
            total_weight = 0.0
            weighted_x = 0.0
            weighted_y = 0.0

            for x, y, dist in corrections:
                weight = 1.0 / max(dist, 0.1)  # Avoid division by zero
                weighted_x += x * weight
                weighted_y += y * weight
                total_weight += weight

            if total_weight > 0:
                avg_x = weighted_x / total_weight
                avg_y = weighted_y / total_weight

                # Fuse with dead reckoning estimate
                alpha = self.apriltag_correction_weight
                
                old_x = self.current_state[0]
                old_y = self.current_state[1]
                
                self.current_state[0] = (1 - alpha) * self.current_state[0] + alpha * avg_x
                self.current_state[1] = (1 - alpha) * self.current_state[1] + alpha * avg_y

                self.correction_count += 1

                self.get_logger().info(
                    f'Applied AprilTag correction from {len(corrections)} tag(s). '
                    f'Position: ({old_x:.2f}, {old_y:.2f}) → ({self.current_state[0]:.2f}, {self.current_state[1]:.2f}), '
                    f'Yaw: {np.rad2deg(self.current_state[2]):.1f}°'
                )

    def update_dead_reckoning(self, linear_vel, angular_vel):
        """
        Update robot pose using dead reckoning
        """
        self.current_state[0] += linear_vel * np.cos(self.current_state[2]) * self.dt
        self.current_state[1] += linear_vel * np.sin(self.current_state[2]) * self.dt
        self.current_state[2] += angular_vel * self.dt
        self.current_state[2] = (self.current_state[2] + np.pi) % (2 * np.pi) - np.pi
        
    def broadcast_tf(self):
        """
        Broadcast TF transform from odom to base_link
        """
        current_time = self.get_clock().now()
        
        t = TransformStamped()
        t.header.stamp = current_time.to_msg()
        t.header.frame_id = self.odom_frame
        t.child_frame_id = self.base_frame
        
        t.transform.translation.x = self.current_state[0]
        t.transform.translation.y = self.current_state[1]
        t.transform.translation.z = 0.0
        
        qx, qy, qz, qw = self.euler_to_quaternion(0, 0, self.current_state[2])
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        
        self.tf_broadcaster.sendTransform(t)
        
    def euler_to_quaternion(self, roll, pitch, yaw):
        """
        Convert Euler angles to quaternion
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
    
    def should_drive_backwards(self, current_wp):
        """
        Determine if robot should drive backwards considering both:
        1. Angle to target position
        2. Target orientation at waypoint
        
        Returns True if backwards is better, False for forward
        """
        delta_x = current_wp[0] - self.current_state[0]
        delta_y = current_wp[1] - self.current_state[1]
        angle_to_target = np.arctan2(delta_y, delta_x)
        
        angle_diff_to_pos = angle_to_target - self.current_state[2]
        angle_diff_to_pos = (angle_diff_to_pos + np.pi) % (2 * np.pi) - np.pi
        
        angle_diff_to_orient = current_wp[2] - self.current_state[2]
        angle_diff_to_orient = (angle_diff_to_orient + np.pi) % (2 * np.pi) - np.pi
        
        forward_rotation = abs(angle_diff_to_pos) + abs(current_wp[2] - angle_to_target)
        
        backward_angle_to_pos = angle_to_target + np.pi
        backward_angle_to_pos = (backward_angle_to_pos + np.pi) % (2 * np.pi) - np.pi
        backward_initial_rotation = backward_angle_to_pos - self.current_state[2]
        backward_initial_rotation = (backward_initial_rotation + np.pi) % (2 * np.pi) - np.pi
        
        backward_rotation = abs(backward_initial_rotation) + abs(current_wp[2] - backward_angle_to_pos)
        
        drive_backwards = backward_rotation < forward_rotation
        return drive_backwards
    
    def get_desired_heading_to_goal(self, current_wp, drive_backwards):
        """
        Get the desired heading to face towards (or away from) the goal
        """
        delta_x = current_wp[0] - self.current_state[0]
        delta_y = current_wp[1] - self.current_state[1]
        angle_to_target = np.arctan2(delta_y, delta_x)
        
        if drive_backwards:
            desired_heading = angle_to_target + np.pi
            desired_heading = (desired_heading + np.pi) % (2 * np.pi) - np.pi
        else:
            desired_heading = angle_to_target
        
        return desired_heading
    
    def get_rotation_direction(self, heading_error):
        """
        Determine rotation direction based on heading error.
        Returns: angular velocity with fixed magnitude but correct direction
        """
        if heading_error > 0:
            return self.fixed_rotation_vel
        else:
            return -self.fixed_rotation_vel
        
    def control_loop(self):
        """
        Main control loop with three stages: rotate to goal, drive, rotate to orientation
        """
        # self.apply_apriltag_correction()
        
        if self.current_waypoint_idx >= len(self.waypoints):
            self.get_logger().info('All waypoints reached! Stopping robot.')
            self.stop_robot()
            self.broadcast_tf()
            return

        current_wp = self.waypoints[self.current_waypoint_idx]
        
        if not self.waypoint_reached:
            self.pid.setTarget(current_wp)
            self.drive_backwards = self.should_drive_backwards(current_wp)
            self.waypoint_reached = True
            self.stage = 'rotate_to_goal'
            self.stage_pid.setTarget(current_wp)

        delta_x = current_wp[0] - self.current_state[0]
        delta_y = current_wp[1] - self.current_state[1]
        position_error = np.sqrt(delta_x**2 + delta_y**2)
        twist_msg = Twist()
        
        # Stage 1: Rotate to face the goal (or away if driving backwards)
        if self.stage == 'rotate_to_goal':
            desired_heading = self.get_desired_heading_to_goal(current_wp, self.drive_backwards)
            heading_error = desired_heading - self.current_state[2]
            heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi
            
            if abs(heading_error) < 0.05:
                self.stage = 'drive'
                twist_msg.angular.z = 0.0
            else:
                twist_msg.angular.z = float(self.get_rotation_direction(heading_error))
        
        # Stage 2: Drive towards/away from the goal
        elif self.stage == 'drive':
            self.apply_apriltag_correction()
            position_error = np.sqrt(delta_x**2 + delta_y**2)
            desired_heading = self.get_desired_heading_to_goal(current_wp, self.drive_backwards)
            heading_error = desired_heading - self.current_state[2]
            heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi
            self.get_logger().info(f'Position error: {position_error:.2f}m, Heading error: {np.rad2deg(heading_error):.1f}°')
            if position_error < self.tolerance:
                self.stage = 'rotate_to_orient'
                twist_msg.linear.x = 0.0
            elif abs(heading_error) > 0.05:
                self.stage = 'rotate_to_goal'
                twist_msg.linear.x = 0.0
                self.get_logger().info(f'Heading error too large: {np.rad2deg(heading_error):.1f}°, switching to rotate_to_goal')
                twist_msg.angular.z = float(self.get_rotation_direction(heading_error))
            else:
                update_value = self.pid.update(self.current_state, self.drive_backwards)
                twist_msg.linear.x = float(update_value[0])
        
        # Stage 3: Rotate to target orientation
        elif self.stage == 'rotate_to_orient':
            heading_error = current_wp[2] - self.current_state[2]
            heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi
            if abs(heading_error) < self.angle_tolerance:
                self.current_waypoint_idx += 1
                self.waypoint_reached = False
                twist_msg.angular.z = 0.0
            else:
                twist_msg.angular.z = float(self.get_rotation_direction(heading_error))
        
        self.update_dead_reckoning(twist_msg.linear.x, twist_msg.angular.z)
        self.broadcast_tf()
        self.cmd_vel_pub.publish(twist_msg)

    def stop_robot(self):
        """
        Stop the robot by publishing zero velocities
        """
        twist_msg = Twist()
        self.cmd_vel_pub.publish(twist_msg)


def main(args=None):
    rclpy.init(args=args)
    node = Hw1ClosedLoopNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Stopped by keyboard interrupt')
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()