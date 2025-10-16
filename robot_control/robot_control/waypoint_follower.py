import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from std_msgs.msg import Float32MultiArray
import math
import os
from typing import List, Tuple


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

        self.linear_threshold = 0.1
        self.angular_threshold = 0.25

        # Load waypoints
        waypoints_path = self.get_parameter('waypoints_path').value
        self.waypoints: List[Tuple[float, float, float]] = self.load_waypoints(waypoints_path)
        if not self.waypoints:
            self.get_logger().error(f'No waypoints loaded from {waypoints_path}. Stopping.')
            raise RuntimeError('waypoints_empty')

        # Publisher for motor commands
        self.cmd_pub = self.create_publisher(Float32MultiArray, 'motor_commands', 10)

        # Estimated pose (x, y from integrated v; yaw from integrated angular velocity)
        self.x_est = 0.0
        self.y_est = 0.0
        self.yaw = 0.0

        # Tracking state
        self.current_idx = 0
        self.last_time = self.get_clock().now()
        self.timer = self.create_timer(1.0 / self.rate_hz, self.control_step)

        self.get_logger().info(f'Loaded {len(self.waypoints)} waypoints. Starting follower at {self.rate_hz:.1f} Hz')

    def load_waypoints(self, path: str) -> List[Tuple[float, float, float]]:
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
        if dist < self.pos_tol:
            # Waypoint fully reached - move to next waypoint
            self.current_idx += 1
            self.get_logger().info(f'Reached waypoint {self.current_idx}/{len(self.waypoints)}')
            # Return stop command - the control_step will handle publishing
            v_cmd, w_cmd = 0.0, 0.0
        elif abs(heading_error) > self.yaw_tol:
            v_cmd, w_cmd = 0.0, math.copysign(self.max_angular_speed_radps, heading_error)
        else:
            v_cmd = self.max_speed_mps
            w_cmd = math.copysign(self.max_angular_speed_radps * abs(heading_error) / math.pi, heading_error)
        v_cmd = v_cmd * self.kv
        w_cmd = w_cmd * self.kh
        return v_cmd, w_cmd
    
    def velocity2input(self, v_cmd: float, w_cmd: float) -> Tuple[float, float]:
        # assume that the mapping from L, R = x, x to linear speed is a linear function: y = (x - self.linear_threshold) / (0.5 - self.linear_threshold) * self.max_linear_speed
        # assume that mapping from L, R = x, -x to angular speed is a linear function: y = (x - self.angular_threshold) / (0.5 - self.angular_threshold) * self.max_linear_speed
        # but we don't know whether we can do a simple mixture addition of both
        if v_cmd == 0 and w_cmd == 0:
            return 0.0, 0.0
        if w_cmd == 0:
            l1 = self.linear_threshold + v_cmd * (0.5 - self.linear_threshold) / self.max_speed_mps
            l1 = l1 * v_cmd / abs(v_cmd)
            r1 = l1
            return l1, r1
        elif v_cmd == 0:
            r2 = math.copysign(self.angular_threshold + abs(w_cmd) * (0.5 - self.angular_threshold) / self.max_angular_speed_radps, w_cmd)
            l2 = -1 * r2
            return l2, r2
        else:
            l1, r1 = v_cmd * (0.5 - self.linear_threshold) / self.max_speed_mps, v_cmd * (0.5 - self.linear_threshold) / self.max_speed_mps
            l2, r2 = -1 * w_cmd * (0.5 - self.angular_threshold) / self.max_angular_speed_radps, w_cmd * (0.5 - self.angular_threshold) / self.max_angular_speed_radps
            l_thres, r_thres = self.linear_threshold, self.linear_threshold
            l = l1 + l2 + l_thres
            r = r1 + r2 + r_thres
            l = max(-self.max_norm_cmd, min(self.max_norm_cmd, l))
            r = max(-self.max_norm_cmd, min(self.max_norm_cmd, r))
            return l, r

    def control_step(self):
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        if dt < 0.0001:
            return
        self.last_time = now

        # Check if all waypoints are completed
        if self.current_idx >= len(self.waypoints):
            self.publish_cmd(0.0, 0.0)
            self.get_logger().info('All waypoints completed!')
            return

        # Desired waypoint
        gx, gy, gtheta = self.waypoints[self.current_idx]

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


