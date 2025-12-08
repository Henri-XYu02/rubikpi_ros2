#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from geometry_msgs.msg import TransformStamped
from tf2_ros import Buffer, TransformListener, TransformBroadcaster
import numpy as np
import os
import yaml
import time
import math
import random
from ament_index_python.packages import get_package_share_directory
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

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
        self.I = np.array([0.0, 0.0])
        self.lastError = np.array([0.0, 0.0])
        self.target = np.array(state)

    def getError(self, currentState, targetState, drive_backwards):
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

    def update(self, currentState, drive_backwards):
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

class Hw2SolutionNode(Node):
    def __init__(self):
        super().__init__('hw2_solution_node')
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.odom_frame = 'odom'
        self.base_frame = 'base_link'

        self.tag_positions = {}

        # Roomba Logic State
        self.sweep_orientation = 'horizontal'
        self.waypoints = np.array([])
        self.current_waypoint_idx = 0

        self.pid = PIDcontroller(0.8, 0.005, 0.001)

        # Initial state
        self.current_state = np.array([0.5, 0.5, 0.0])
        self.obs_current_state = np.array([0.5, 0.5, 0.0])

        self.waypoint_reached = False
        self.tolerance = 0.15
        self.angle_tolerance = 0.1

        # Localization Vars
        self.last_tag_detection_time = time.time() # Initialize to now so we don't start in panic mode
        self.using_tag_localization = False

        self.load_tag_configurations()

        self.drive_backwards = False
        self.dt = 0.1
        self.control_timer = self.create_timer(self.dt, self.control_loop)
        self.localization_timer = self.create_timer(0.1, self.localization_update)

        self.stage = 'rotate_to_goal'
        self.stage_pid = PIDcontroller(0.8, 0.05, 0.001)
        self.fixed_rotation_vel = 0.785

        self.trajectory = []

        # Recovery Behavior State Variables
        self.recovery_mode = 'IDLE' # Options: IDLE, REVERSE, PAUSE_1, ROTATE, PAUSE_2
        self.recovery_state_start_time = 0.0

        # Generate initial batch
        self.generate_next_batch()

    def generate_next_batch(self):
        if self.sweep_orientation == 'horizontal':
            self.get_logger().info('Generating HORIZONTAL sweep...')
            new_waypoints = self.generate_coverage_waypoints(mode='horizontal')
            self.sweep_orientation = 'vertical'
        else:
            self.get_logger().info('Generating VERTICAL sweep...')
            new_waypoints = self.generate_coverage_waypoints(mode='vertical')
            self.sweep_orientation = 'horizontal'

        self.waypoints = new_waypoints
        self.current_waypoint_idx = 0
        self.waypoint_reached = False
        self.get_logger().info(f'New batch ready: {len(self.waypoints)} waypoints: {self.waypoints}')

    def generate_coverage_waypoints(self, mode='horizontal'):
        waypoints = []
        x_min, x_max = 0.0, 2.5
        y_min, y_max = 0.0, 2.5
        margin = 0.4

        x_start, x_end = x_min + margin, x_max - margin
        y_start, y_end = y_min + margin, y_max - margin

        stripe_spacing = 0.5 + random.uniform(-0.05, 0.05)

        if mode == 'horizontal':
            y_current = y_start
            direction = 1
            while y_current <= y_end:
                if direction == 1:
                    waypoints.append([x_end, y_current, 0.0])
                else:
                    waypoints.append([x_start, y_current, np.pi])

                y_current += stripe_spacing
                if y_current <= y_end:
                    if direction == 1:
                         waypoints.append([x_end, y_current, -np.pi/2])
                    else:
                         waypoints.append([x_start, y_current, -np.pi/2])
                    direction *= -1

        elif mode == 'vertical':
            x_current = x_start
            direction = 1
            while x_current <= x_end:
                if direction == 1:
                    waypoints.append([x_current, y_end, np.pi/2])
                else:
                    waypoints.append([x_current, y_start, -np.pi/2])

                x_current += stripe_spacing
                if x_current <= x_end:
                    if direction == 1:
                        waypoints.append([x_current, y_end, 0.0])
                    else:
                        waypoints.append([x_current, y_start, 0.0])
                    direction *= -1

        return np.array(waypoints)

    def load_tag_configurations(self):
        try:
            package_share_dir = get_package_share_directory('hw_2_solution')
            yaml_path = os.path.join(package_share_dir, 'configs', 'apriltags_position.yaml')
            with open(yaml_path, 'r') as file:
                data = yaml.safe_load(file)
            tags_data = data.get('apriltags', [])
            for tag in tags_data:
                tag_id = tag.get('id')
                if tag_id is None: continue
                self.tag_positions[tag_id] = {
                    'x': float(tag['x']), 'y': float(tag['y']), 'z': float(tag['z']),
                    'qx': float(tag['qx']), 'qy': float(tag['qy']), 'qz': float(tag['qz']), 'qw': float(tag['qw'])
                }
        except Exception as e:
            self.get_logger().error(f'Failed to load tag configs: {str(e)}')

    def update_dead_reckoning(self, linear_vel, angular_vel):
        self.current_state[0] += linear_vel * np.cos(self.current_state[2]) * self.dt
        self.current_state[1] += linear_vel * np.sin(self.current_state[2]) * self.dt
        self.current_state[2] += angular_vel * self.dt
        self.current_state[2] = (self.current_state[2] + np.pi) % (2 * np.pi) - np.pi

    def broadcast_tf(self):
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
        cy = math.cos(yaw * 0.5); sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5); sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5); sr = math.sin(roll * 0.5)
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        return qx, qy, qz, qw

    def should_drive_backwards(self, current_wp):
        return False

    def get_desired_heading_to_goal(self, current_wp, drive_backwards):
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
        if heading_error > 0: return self.fixed_rotation_vel
        else: return -self.fixed_rotation_vel

    def compute_and_publish_robot_pose_from_tag(self, tag_id, tag_observation):
        tag_map = self.tag_positions[tag_id]
        tag_map_pos = np.array([tag_map['x'], tag_map['y'], tag_map['z']])
        tag_map_rot = Rotation.from_quat([tag_map['qx'], tag_map['qy'], tag_map['qz'], tag_map['qw']])
        obs_pos = np.array([
            tag_observation.transform.translation.x,
            tag_observation.transform.translation.y,
            tag_observation.transform.translation.z
        ])
        obs_rot = Rotation.from_quat([
            tag_observation.transform.rotation.x,
            tag_observation.transform.rotation.y,
            tag_observation.transform.rotation.z,
            tag_observation.transform.rotation.w
        ])
        tag_to_robot_rot = obs_rot.inv()
        tag_to_robot_pos = -tag_to_robot_rot.apply(obs_pos)
        robot_map_rot = tag_map_rot * tag_to_robot_rot
        robot_map_pos = tag_map_pos + tag_map_rot.apply(tag_to_robot_pos)
        yaw = robot_map_rot.as_euler('xyz')[2]
        self.current_state = 0.5 * (self.current_state + np.array([robot_map_pos[0], robot_map_pos[1], yaw]))
        self.obs_current_state = 0.5 * (self.obs_current_state + np.array([robot_map_pos[0], robot_map_pos[1], yaw]))

    def control_loop(self):
        twist_msg = Twist()
        current_time = time.time()

        # 1. Check if we are lost (no tags for > 2 seconds)
        # Note: We give a grace period at startup
        time_since_last_tag = current_time - self.last_tag_detection_time
        is_lost = time_since_last_tag > 2.0

        if is_lost:
            # === RECOVERY BEHAVIOR ===
            if self.recovery_mode == 'IDLE':
                self.get_logger().warn("No tags detected! Entering recovery mode.")
                self.recovery_mode = 'REVERSE'
                self.recovery_state_start_time = current_time

            elapsed_state_time = current_time - self.recovery_state_start_time

            if self.recovery_mode == 'REVERSE':
                twist_msg.linear.x = -0.15  # Back up slowly
                if elapsed_state_time > 1.5: # Duration of reverse
                    self.recovery_mode = 'PAUSE_1'
                    self.recovery_state_start_time = current_time

            elif self.recovery_mode == 'PAUSE_1':
                twist_msg.linear.x = 0.0
                twist_msg.angular.z = 0.0
                if elapsed_state_time > 2.0: # Pause for 2 seconds
                    self.recovery_mode = 'ROTATE'
                    self.recovery_state_start_time = current_time

            elif self.recovery_mode == 'ROTATE':
                twist_msg.angular.z = 0.6  # Rotate to scan
                if elapsed_state_time > 2.0: # Duration of rotation scan
                    self.recovery_mode = 'PAUSE_2'
                    self.recovery_state_start_time = current_time

            elif self.recovery_mode == 'PAUSE_2':
                twist_msg.linear.x = 0.0
                twist_msg.angular.z = 0.0
                if elapsed_state_time > 2.0:
                    self.recovery_mode = 'REVERSE' # Loop back to start
                    self.recovery_state_start_time = current_time

            # Publish and exit early - do not do waypoint logic
            self.update_dead_reckoning(twist_msg.linear.x, twist_msg.angular.z)
            self.broadcast_tf()
            self.cmd_vel_pub.publish(twist_msg)
            return

        # === NORMAL BEHAVIOR (WAYPOINTS) ===

        # If we just recovered, reset the recovery state
        if self.recovery_mode != 'IDLE':
            self.get_logger().info("Tags found! Resuming normal navigation.")
            self.recovery_mode = 'IDLE'

        # Continuously generate new points
        if self.current_waypoint_idx >= len(self.waypoints):
            self.get_logger().info("Reached all waypoints! Generating new batch...")
            self.generate_next_batch()
            return

        current_wp = self.waypoints[self.current_waypoint_idx]

        if not self.waypoint_reached:
            self.pid.setTarget(current_wp)
            self.drive_backwards = self.should_drive_backwards(current_wp)
            self.waypoint_reached = True
            self.stage = 'rotate_to_goal'
            self.stage_pid.setTarget(current_wp)

        delta_x = current_wp[0] - self.obs_current_state[0]
        delta_y = current_wp[1] - self.obs_current_state[1]

        if self.stage == 'rotate_to_goal':
            desired_heading = self.get_desired_heading_to_goal(current_wp, self.drive_backwards)
            heading_error = desired_heading - self.current_state[2]
            heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi

            if abs(heading_error) < 0.05:
                self.stage = 'drive'
                twist_msg.angular.z = 0.0
            else:
                twist_msg.angular.z = float(self.get_rotation_direction(heading_error))

        elif self.stage == 'drive':
            position_error = np.sqrt(delta_x**2 + delta_y**2)
            if position_error < self.tolerance:
                self.stage = 'rotate_to_orient'
                twist_msg.linear.x = 0.0
            else:
                desired_heading = self.get_desired_heading_to_goal(current_wp, self.drive_backwards)
                heading_error = desired_heading - self.current_state[2]
                heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi
                if abs(heading_error) > 0.2:
                    self.stage = 'rotate_to_goal'
                    twist_msg.linear.x = 0.0
                else:
                    update_value = self.pid.update(self.current_state, self.drive_backwards)
                    twist_msg.linear.x = float(update_value[0])

        elif self.stage == 'rotate_to_orient':
            heading_error = current_wp[2] - self.current_state[2]
            heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi
            if abs(heading_error) < self.angle_tolerance:
                self.get_logger().info(f"Reached another waypoint! Waypoint {self.current_waypoint_idx}")
                self.current_waypoint_idx += 1
                self.waypoint_reached = False
                twist_msg.angular.z = 0.0
            else:
                twist_msg.angular.z = float(self.get_rotation_direction(heading_error))

        self.update_dead_reckoning(twist_msg.linear.x, twist_msg.angular.z)
        self.broadcast_tf()
        self.cmd_vel_pub.publish(twist_msg)

        self.trajectory.append([
            self.current_state[0], self.current_state[1], self.current_state[2], time.time()
        ])

    def localization_update(self):
        current_time = time.time()
        closest_tag_id = None
        closest_observation = None
        closest_distance = float('inf')
        for tag_id in self.tag_positions.keys():
            try:
                tag_frame = f'tag_{tag_id}'
                observation = self.tf_buffer.lookup_transform('base_link', tag_frame, rclpy.time.Time())
                transform_time = rclpy.time.Time.from_msg(observation.header.stamp)
                time_diff = (self.get_clock().now() - transform_time).nanoseconds / 1e9
                if time_diff > 0.25: continue
                dx = observation.transform.translation.x
                dy = observation.transform.translation.y
                dz = observation.transform.translation.z
                distance = np.sqrt(dx*dx + dy*dy + dz*dz)
                if distance < closest_distance:
                    closest_distance = distance
                    closest_tag_id = tag_id
                    closest_observation = observation
            except Exception: continue

        if closest_observation is not None:
            self.compute_and_publish_robot_pose_from_tag(closest_tag_id, closest_observation)
            self.last_tag_detection_time = current_time # Update time whenever we see a tag
            self.using_tag_localization = True
        else:
            if (current_time - self.last_tag_detection_time) > 1.0:
                self.using_tag_localization = False

    def stop_robot(self):
        twist_msg = Twist()
        self.cmd_vel_pub.publish(twist_msg)

    def plot_trajectory(self):
        self.get_logger().info('Saving trajectory plot...')
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            traj = np.array(self.trajectory)
            if len(traj) > 0:
                ax.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=1)
                ax.set_xlim(-0.5, 3.0)
                ax.set_ylim(-0.5, 3.0)
                ax.set_title('Continuous Coverage Trajectory')
            plt.savefig('continuous_trajectory.png')
            plt.close(fig)
        except Exception as e:
            self.get_logger().error(str(e))

def main(args=None):
    rclpy.init(args=args)
    node = Hw2SolutionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Stopped by keyboard interrupt')
        node.plot_trajectory()
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
