import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from std_msgs.msg import Float32MultiArray
import math
import os
from typing import List, Tuple

# ROS TF2 imports
import tf2_ros
from tf2_ros import TransformException
from geometry_msgs.msg import TransformStamped, Quaternion


def euler_from_quaternion(quat: Quaternion) -> Tuple[float, float, float]:
    """
    Convert a ROS Quaternion message to RPY (roll, pitch, yaw) angles.
    Yaw is the rotation around the Z-axis.
    """
    # Using a simple conversion (can also use tf_transformations library)
    x = quat.x
    y = quat.y
    z = quat.z
    w = quat.w

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians

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
        self.declare_parameter('wheel_base', 0.04)  # meters (wheel radius)
        self.declare_parameter('car_width', 0.168)  # meters (car width)
        self.declare_parameter('max_speed_mps', 0.72)  # approximate max linear speed
        self.declare_parameter('max_angular_speed_radps', math.pi)  # approximate max angular speed
        self.declare_parameter('max_norm_cmd', 0.5)  # matches motor_controller clamp
        self.declare_parameter('kv', 0.8)  # linear gain
        self.declare_parameter('kh', 0.8)  # heading gain
        self.declare_parameter('pos_tolerance', 0.05)  # meters
        self.declare_parameter('yaw_tolerance', 0.1)  # radians (~11 degrees)
        self.declare_parameter('rate_hz', 20.0)

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

        self.linear_min = 0.1
        self.angular_min = 0.25

        # Load waypoints
        waypoints_path = self.get_parameter('waypoints_path').value
        self.waypoints: List[Tuple[float, float, float]] = self.load_waypoints(waypoints_path)
        if not self.waypoints:
            self.get_logger().error(f'No waypoints loaded from {waypoints_path}. Stopping.')
            raise RuntimeError('waypoints_empty')

        # Publisher for motor commands
        self.cmd_pub = self.create_publisher(Float32MultiArray, 'motor_commands', 10)

        # TF2 Listener
        # Used to look up transforms (like map -> base_link)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Estimated pose (x, y from integrated v; yaw from integrated angular velocity)
        self.x_est = 0.0
        self.y_est = 0.0
        self.yaw = 0.0

        # Tracking state
        self.current_waypoint_idx = 0
        self.last_time = self.get_clock().now()
        self.timer = self.create_timer(1.0 / self.rate_hz, self.control_step)

        self.get_logger().info(f'Loaded {len(self.waypoints)} waypoints. Starting follower at {self.rate_hz:.1f} Hz')
        self.get_logger().info('Waypoint follower is in CLOSED_LOOP mode (AprilTag only).')


    # Pose Correction 
    def correct_pose_from_tf(self):
        """
        Try to get the robot's pose from the TF tree (map -> base_link).
        If successful, update self.x_est, self.y_est, and self.yaw.
        This "resets" the dead-reckoning drift.
        """
        try:
            # We want the transform from the fixed 'map' frame to the robot's 'base_link' frame
            # The timestamp rclpy.time.Time() means "get the latest available transform"
            t = self.tf_buffer.lookup_transform(
                'map',          # Target frame
                'base_link',    # Source frame
                rclpy.time.Time(),
                timeout=Duration(seconds=0.1) # Don't wait too long
            )

            # --- POSE CORRECTION ---
            # If lookup_transform succeeded, a tag is visible and TF is working
            old_x, old_y, old_yaw = self.x_est, self.y_est, self.yaw
            
            self.x_est = t.transform.translation.x
            self.y_est = t.transform.translation.y
            
            # Convert quaternion to euler angles (Roll, Pitch, Yaw)
            _roll, _pitch, self.yaw = euler_from_quaternion(t.transform.rotation)

            self.get_logger().info(
                f'POSE CORRECTION: Drift corrected by TF. '
                f'Old: ({old_x:.2f}, {old_y:.2f}, {old_yaw:.2f}), '
                f'New: ({self.x_est:.2f}, {self.y_est:.2f}, {self.yaw:.2f})'
            )
            return True

        except TransformException as ex:
            # This is NOT an error. It just means no tag is visible right now.
            self.get_logger().warn(
                f'Could not get pose from TF: {ex}. '
                'Falling back to dead-reckoning.'
            )
            return False
        

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
                # Resolve relative to package share or cwd; default to cwd
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
    
    
    def diff2velocity_simple(self, dist: float, heading_error: float, gtheta: float, theta: float) -> Tuple[float, float]:
        """
        Simple proportional controller to compute linear and angular velocity commands.
        :param dist: Distance to the goal waypoint.
        :param heading_error: Heading error to the goal waypoint.
        :param gtheta: Goal orientation (not used in this simple controller).
        :param theta: Current orientation (not used in this simple controller).
        :return: Tuple of (v_cmd, w_cmd) in m/s and rad/s.
        1. If within position tolerance, stop and move to next waypoint.
        2. If heading error is large, rotate in place.
        3. Otherwise, move forward with speed proportional to heading error.
        4. Scale commands by kv and kh gains.
        5. Clamp commands to max speed limits.
        6. Return (v_cmd, w_cmd).
        P.S. a.  The control_step function will handle publishing the commands.
             b. This function does not handle the case of final orientation at the last waypoint.
        
        """
        if dist < self.pos_tol:
            # Waypoint fully reached - move to next waypoint
            self.current_waypoint_idx += 1
            self.get_logger().info(f'Reached waypoint {self.current_waypoint_idx}/{len(self.waypoints)}')
            # Return stop command - the control_step will handle publishing
            v_cmd, w_cmd = 0.0, 0.0
        elif abs(heading_error) > self.yaw_tol:
            # Large heading error - rotate in place
            v_cmd, w_cmd = 0.0, math.copysign(self.max_angular_speed_radps, heading_error)
        else:
            # Proceed forward
            v_cmd = self.max_speed_mps
                # also add some angle adjustment while we are going "straight" so we can compensate for any accumulating misalignments
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
        Note: This function assumes a linear mapping from velocity commands to motor inputs.
        """
        # assume that the mapping from L, R = x, x to linear speed is a linear function: y = (x - self.linear_min) / (0.5 - self.linear_min) * self.max_linear_speed
        # assume that mapping from L, R = x, -x to angular speed is a linear function: y = (x - self.angular_min) / (0.5 - self.angular_min) * self.max_linear_speed
        # but we don't know whether we can do a simple mixture addition of both
        if v_cmd == 0 and w_cmd == 0:
            return 0.0, 0.0
        if w_cmd == 0:
            # pure forward/backward. it's either .1 or 80% * (inputPower / .72metersPerSec). 
            # get rid of deadzone (0.1). scale linearly to max speed
            l1 = self.linear_min + v_cmd * (0.5 - self.linear_min) / self.max_speed_mps
            l1 = l1 * v_cmd / abs(v_cmd) #if the waypoint is behind us and we don't turn then just go backward
                                         # but currect strategy doesn't move backward. Just turns
            r1 = l1
            return l1, r1
        elif v_cmd == 0:
            r2 = math.copysign(self.angular_min + abs(w_cmd) * (0.5 - self.angular_min) / self.max_angular_speed_radps, w_cmd)
            l2 = -1 * r2
            return l2, r2
        else:
            # mixture of both
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
        1. Compute time delta since last step.
        2. Try to correct pose from TF.
        3. If all waypoints are completed, stop and return.
        4. Get current goal waypoint.
        5. Compute distance and heading error to goal.
        6. Compute velocity commands using diff2velocity_simple.
        7. Convert to motor commands and publish.
        8. Dead-Reckoning Integration: Integrate position and orientation estimate using commanded velocities.
        9. Log current state.
        :return: None
        """
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        if dt < 0.0001:
            return
        self.last_time = now

        # Try to correct pose 
        self.correct_pose_from_tf()
        # Now, self.x_est, self.y_est, self.yaw are the "best" available pose

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

        # Dead-Reckoning Intergration. Integrate position and orientation estimate using commanded velocities
        self.x_est += v_cmd * math.cos(theta) * dt
        self.y_est += v_cmd * math.sin(theta) * dt
        
        # Integrate yaw using angular velocity
        self.yaw += w_cmd * dt
        self.yaw = normalize_angle(self.yaw)
        self.get_logger().info(f"x: {self.x_est:.2f}, y: {self.y_est:.2f}, yaw: {self.yaw:.2f}, v_cmd: {v_cmd:.2f}, w_cmd: {w_cmd:.2f}, l_norm: {l_norm:.2f}, r_norm: {r_norm:.2f}, heading_error: {heading_error:.2f}, dist: {dist:.2f}")

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


