#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from geometry_msgs.msg import TransformStamped
from tf2_ros import Buffer, TransformListener, TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import math
from scipy.spatial.transform import Rotation
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


"""
EKF-SLAM implementation for landmark mapping while navigating waypoints

State vector (without velocity):
x = [x_robot, y_robot, yaw_robot, x_lm1, y_lm1, x_lm2, y_lm2, ...]
Robot state: 3 dimensions [x, y, yaw]
Each landmark: 2 dimensions [x, y]
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

    def setMaximumUpdate(self, mv):
        self.maximumValue = mv

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


class EKFSLAMNode(Node):
    """
    EKF-SLAM Node for simultaneous localization and mapping using AprilTags.
    
    State Representation (without velocity):
    - Robot state: [x, y, yaw] (3 dimensions)
    - Each landmark: [x, y] (2 dimensions)
    - Full state: [x_r, y_r, yaw_r, x_lm1, y_lm1, ..., x_lmN, y_lmN]
    
    EKF maintains:
    - xEst: State estimate vector (mean)
    - PEst: Covariance matrix (uncertainty)
    """
    
    def __init__(self):
        """
        Initialize EKF-SLAM node with state, covariance, and ROS interfaces.
        """
        super().__init__('ekf_slam_node')
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # TF (Transform) interfaces for coordinate frame management
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Coordinate frames
        self.odom_frame = 'odom'  # Global reference frame
        self.base_frame = 'base_link'  # Robot's local frame
        
        # EKF-SLAM State (3D robot state: x, y, yaw)
        # State vector: x = [x_robot, y_robot, yaw_robot, x_lm1, y_lm1, ...]
        self.xEst = np.zeros((3, 1))  # Initial state: robot at origin with zero yaw
        self.PEst = np.eye(3) * 0.05   # Initial covariance (low uncertainty)
        
        # Landmark database: maps landmark_index -> tag_id
        # This allows us to associate tag observations with state vector entries
        self.landmark_tags = {}
        
        # Process noise covariance Q (uncertainty in motion model)
        # Q represents how much we trust our motion model
        # Diagonal matrix: [σ_x², σ_y², σ_yaw²]
        self.Q = np.diag([0.10, 0.10, np.deg2rad(10.0)]) ** 2
        
        # Measurement noise covariance R (uncertainty in sensor)
        # R represents how much we trust our sensor measurements
        # For TF observations: [σ_dx², σ_dy²]
        self.R_tf = np.diag([0.2, 0.2]) ** 2
        
        # Navigation waypoints [x, y, yaw]
        # self.waypoints = np.array([
        #     [0.0,0.0, 0.0],
        #     [0.5, 0.0, np.pi/2],
        #     [0.5, 0.5, np.pi],
        #     [0.0, 0.5, -np.pi/2],
        #     [0.0,0.0, 0.0],
        #     [0.5, 0.0, np.pi/2],
        #     [0.5, 0.5, np.pi],
        #     [0.0, 0.5, -np.pi/2],
        # ])
        self.plotted = False

        # # 8-point octagon waypoints
        self.waypoints = np.array([
            [0.5000, 0.0000, 0.3927],  # WP6 - Bottom
            [0.8536, 0.1464, 1.1781],  # WP7 - Lower right
            [1.0000, 0.5000, 1.9635],   # WP8 - Return to start
            [0.8536, 0.8536, 2.7489],  # WP1 - Upper right
            [0.5000, 1.0000, -2.7489], # WP2 - Top
            [0.1464, 0.8536, -1.9635], # WP3 - Upper left
            [0.0000, 0.5000, -1.1781], # WP4 - Left side
            [0.1464, 0.1464, -0.3927], # WP5 - Lower left
            [0.5000, 0.0000, 0.3927],  # WP6 - Bottom
            [0.8536, 0.1464, 1.1781],  # WP7 - Lower right
            [1.0000, 0.5000, 1.9635],   # WP8 - Return to start
            [0.8536, 0.8536, 2.7489],  # WP1 - Upper right
            [0.5000, 1.0000, -2.7489], # WP2 - Top
            [0.1464, 0.8536, -1.9635], # WP3 - Upper left
            [0.0000, 0.5000, -1.1781], # WP4 - Left side
            [0.1464, 0.1464, -0.3927], # WP5 - Lower left
        ])

        # self.waypoints = np.array([
        #     [0.0, 0.0, 0.0], 
        #     [1.0, 0.0, 0],
        # ])
        
        # Navigation state variables
        self.pid = PIDcontroller(0.8, 0.003, 0.001)
        self.current_waypoint_idx = 0
        self.waypoint_reached = False
        self.tolerance = 0.1  # Position tolerance (meters)
        self.angle_tolerance = 0.1  # Orientation tolerance (radians)
        self.drive_backwards = False
        self.stage = 'rotate_to_goal'  # Navigation state machine
        self.fixed_rotation_vel = 0.785  # Fixed angular velocity for pure rotation
        
        self.last_tag_detection_time = 0.0

        # ========== Trajectory tracking for plotting ==========
        self.trajectory_x = []
        self.trajectory_y = []
        self.trajectory_yaw = []
        
        # Control loop timing
        self.dt = 0.1  # Time step (seconds)
        self.control_timer = self.create_timer(self.dt, self.control_loop)
        
        self.get_logger().info('EKF-SLAM Node initialized (3D state: x, y, yaw)')


    # ==================== Plot ==================================
    def save_final_plot(self):
        """Save the final trajectory plot with landmark uncertainty ellipses"""
        plt.ioff()  # Turn off interactive mode
        
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Plot trajectory
        ax.plot(self.trajectory_x, self.trajectory_y, 'b-', linewidth=2, label='Robot Trajectory')
        ax.plot(self.trajectory_x[0], self.trajectory_y[0], 'go', markersize=12, label='Start')
        ax.plot(self.trajectory_x[-1], self.trajectory_y[-1], 'ro', markersize=12, label='End')
        
        # Plot waypoints
        wp_x = self.waypoints[:, 0]
        wp_y = self.waypoints[:, 1]
        ax.plot(wp_x, wp_y, 'g^', markersize=12, label='Waypoints', alpha=0.6)
        
        for i, wp in enumerate(self.waypoints):
            ax.text(wp[0] + 0.05, wp[1] + 0.05, f'WP{i}', fontsize=10, fontweight='bold')
        
        # Plot landmarks with uncertainty ellipses
        if len(self.landmark_tags) > 0:
            for lm_idx, tag_id in self.landmark_tags.items():
                lm_id = 3 + lm_idx * 2
                x = self.xEst[lm_id, 0]
                y = self.xEst[lm_id + 1, 0]
                
                # Get covariance for this landmark
                cov_xx = self.PEst[lm_id, lm_id]
                cov_yy = self.PEst[lm_id + 1, lm_id + 1]
                cov_xy = self.PEst[lm_id, lm_id + 1]
                
                # Compute eigenvalues and eigenvectors for ellipse
                cov_matrix = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]])
                eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
                
                # Get ellipse parameters (95% confidence interval, chi-square = 5.991)
                angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
                width = 2 * np.sqrt(5.991 * eigenvalues[0])
                height = 2 * np.sqrt(5.991 * eigenvalues[1])
                
                # Plot landmark
                ax.plot(x, y, 'ms', markersize=15, label='Landmark' if lm_idx == 0 else '')
                
                # Plot uncertainty ellipse
                ellipse = Ellipse((x, y), width, height, angle=angle, 
                                 facecolor='magenta', alpha=0.2, edgecolor='purple', linewidth=2)
                ax.add_patch(ellipse)
                
                # Add label
                std_x = math.sqrt(cov_xx)
                std_y = math.sqrt(cov_yy)
                ax.text(x + 0.1, y + 0.1, 
                       f'Tag{tag_id}\n({x:.2f}, {y:.2f})\nσ=({std_x:.3f}, {std_y:.3f})',
                       fontsize=9, color='purple', fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel('X Position (m)', fontsize=12)
        ax.set_ylabel('Y Position (m)', fontsize=12)
        ax.set_title('EKF-SLAM: Final Robot Trajectory and Landmark Map', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig('./ekf_slam_trajectory.png', dpi=300, bbox_inches='tight')
        
        self.get_logger().info('Saved final plot to ekf_slam_trajectory.png')
        
        # Show the final plot
        plt.show()
      
    # ==================== EKF-SLAM Functions ====================
    
    def motion_model_slam(self, x, u):
        """
        SLAM motion model: predicts next state given current state and control.
        
        Motion Model Equations (discrete-time, without velocity in state):
        x_{t+1} = x_t + v*dt*cos(yaw)
        y_{t+1} = y_t + v*dt*sin(yaw)
        yaw_{t+1} = yaw_t + ω*dt
        landmarks remain static: x_lm_{t+1} = x_lm_t
        
        In matrix form: x_{t+1} = F*x_t + B*u
        
        Parameters:
        -----------
        x : np.array (n, 1)
            Current state vector [x_r, y_r, yaw, x_lm1, y_lm1, ...]
            Robot state is now only 3D (no velocity)
        u : np.array (2, 1)
            Control input [v, ω] (linear velocity, angular velocity)
            
        Returns:
        --------
        x_new : np.array (n, 1)
            Predicted next state
        """
        n = len(x)
        
        # State transition matrix F (landmarks don't move, so F is identity)
        F_slam = np.eye(n)
        
        # Control input matrix B (only affects robot state, not landmarks)
        B_slam = np.zeros((n, 2))
        yaw = x[2, 0]
        
        # Robot state is at indices 0, 1, 2 (x, y, yaw)
        B_slam[0:3, 0:2] = np.array([
            [self.dt * math.cos(yaw), 0],  # x position change from linear velocity
            [self.dt * math.sin(yaw), 0],  # y position change from linear velocity
            [0.0, self.dt]                 # yaw change from angular velocity
        ])
        
        # Apply motion model: x_new = F*x + B*u
        x_new = F_slam @ x + B_slam @ u
        
        # Normalize yaw to [-π, π]
        x_new[2, 0] = (x_new[2, 0] + np.pi) % (2 * np.pi) - np.pi
        
        return x_new
    
    def jacob_f_slam(self, x, u):
        """
        Compute Jacobian of motion model (for covariance propagation).
        
        The Jacobian F_j = ∂f/∂x represents how small changes in state affect
        the predicted state. Used in EKF predict step: P_pred = F_j*P*F_j^T + Q
        
        Partial derivatives (for 3D robot state):
        ∂x/∂x = 1
        ∂x/∂y = 0
        ∂x/∂yaw = -v*dt*sin(yaw)
        
        ∂y/∂x = 0
        ∂y/∂y = 1
        ∂y/∂yaw = v*dt*cos(yaw)
        
        ∂yaw/∂x = 0
        ∂yaw/∂y = 0
        ∂yaw/∂yaw = 1
        
        All landmark derivatives are 0 (landmarks don't move)
        
        Parameters:
        -----------
        x : np.array (n, 1)
            Current state vector
        u : np.array (2, 1)
            Control input [v, ω]
            
        Returns:
        --------
        jF : np.array (n, n)
            Jacobian matrix of motion model
        """
        n = len(x)
        jF = np.eye(n)  # Start with identity (landmarks unchanged)
        
        # Only robot state has non-trivial derivatives
        yaw = x[2, 0]
        v = u[0, 0]
        
        # Robot state Jacobian (3x3 block)
        jF[0:3, 0:3] = np.array([
            # [∂x/∂x, ∂x/∂y, ∂x/∂yaw]
            [1.0, 0.0, -self.dt * v * math.sin(yaw)],
            # [∂y/∂x, ∂y/∂y, ∂y/∂yaw]
            [0.0, 1.0, self.dt * v * math.cos(yaw)],
            # [∂yaw/∂x, ∂yaw/∂y, ∂yaw/∂yaw]
            [0.0, 0.0, 1.0]
        ])
        
        return jF
    
    def observation_model_landmark_tf(self, x, landmark_id):
        """
        Observation model: predict what sensor would measure given current state.
        
        For TF observations from base_link to landmark:
        1. Get landmark position in global frame: (x_lm, y_lm)
        2. Get robot pose in global frame: (x_r, y_r, yaw_r)
        3. Transform landmark to robot's local frame
        
        Observation Model Equations:
        dx_global = x_lm - x_r
        dy_global = y_lm - y_r
        dx_local = dx_global*cos(-yaw_r) - dy_global*sin(-yaw_r)
        dy_local = dx_global*sin(-yaw_r) + dy_global*cos(-yaw_r)
        
        h(x) = [dx_local, dy_local]
        
        Parameters:
        -----------
        x : np.array (n, 1)
            Current state vector
        landmark_id : int
            Index of landmark in state vector (0, 1, 2, ...)
            
        Returns:
        --------
        z : np.array (2, 1)
            Predicted measurement [dx_local, dy_local]
        """
        # Calculate index in state vector: robot has 3 states, each landmark has 2
        lm_id = 3 + landmark_id * 2
        
        # Extract robot pose
        x_robot = x[0, 0]
        y_robot = x[1, 0]
        yaw_robot = x[2, 0]
        
        # Extract landmark position
        x_landmark = x[lm_id, 0]
        y_landmark = x[lm_id + 1, 0]
        
        # Transform landmark from global to robot's local frame
        dx_global = x_landmark - x_robot
        dy_global = y_landmark - y_robot
        
        # Rotation matrix
        # R(-yaw) = [cos(-yaw)  -sin(-yaw)]
        #           [sin(-yaw)   cos(-yaw)]
        dx_local = dx_global * math.cos(-yaw_robot) - dy_global * math.sin(-yaw_robot)
        dy_local = dx_global * math.sin(-yaw_robot) + dy_global * math.cos(-yaw_robot)
        
        z = np.array([[dx_local], [dy_local]])
        return z
    
    def jacob_h_landmark_tf(self, x, landmark_id):
        """
        Compute Jacobian of observation model (for Kalman gain calculation).
        
        The Jacobian H = ∂h/∂x represents how measurements change with state.
        Used in EKF update step for:
        - Innovation covariance: S = H*P*H^T + R
        - Kalman gain: K = P*H^T*S^(-1)
        
        For TF observation model h(x) = [dx_local, dy_local]:
        
        Derivatives w.r.t robot pose (3D state):
        ∂(dx_local)/∂x_r = -cos(-yaw)
        ∂(dx_local)/∂y_r = -sin(-yaw)
        ∂(dx_local)/∂yaw = -dx_global*sin(-yaw) + dy_global*cos(-yaw)
        
        ∂(dy_local)/∂x_r = sin(-yaw)
        ∂(dy_local)/∂y_r = -cos(-yaw)
        ∂(dy_local)/∂yaw = dx_global*cos(-yaw) + dy_global*sin(-yaw)
        
        Derivatives w.r.t landmark position:
        ∂(dx_local)/∂x_lm = cos(-yaw)
        ∂(dx_local)/∂y_lm = sin(-yaw)
        ∂(dy_local)/∂x_lm = -sin(-yaw)
        ∂(dy_local)/∂y_lm = cos(-yaw)
        
        Parameters:
        -----------
        x : np.array (n, 1)
            Current state vector
        landmark_id : int
            Index of landmark in state vector
            
        Returns:
        --------
        jH : np.array (2, n)
            Jacobian matrix of observation model
            Only non-zero for robot pose and this landmark's position
        """
        lm_id = 3 + landmark_id * 2
        
        x_robot = x[0, 0]
        y_robot = x[1, 0]
        yaw = x[2, 0]
        
        x_lm = x[lm_id, 0]
        y_lm = x[lm_id + 1, 0]
        
        dx_global = x_lm - x_robot
        dy_global = y_lm - y_robot
        
        cos_yaw = math.cos(-yaw)
        sin_yaw = math.sin(-yaw)
        
        n = len(x)
        jH = np.zeros((2, n))  # Sparse matrix: most entries are zero
        
        # dx_local = dx_global * cos(-yaw) - dy_global * sin(-yaw)
        # dy_local = dx_global * sin(-yaw) + dy_global * cos(-yaw)
        
        # Derivatives w.r.t robot position (columns 0, 1, 2)
        jH[0, 0] = -cos_yaw  # ∂(dx_local)/∂x_robot
        jH[0, 1] = sin_yaw  # ∂(dx_local)/∂y_robot
        jH[0, 2] = dx_global * sin_yaw + dy_global * cos_yaw  # ∂(dx_local)/∂yaw
        
        jH[1, 0] = -sin_yaw   # ∂(dy_local)/∂x_robot
        jH[1, 1] = -cos_yaw  # ∂(dy_local)/∂y_robot
        jH[1, 2] = -dx_global * cos_yaw + dy_global * sin_yaw  # ∂(dy_local)/∂yaw
        
        # Derivatives w.r.t landmark position (columns lm_id, lm_id+1)
        jH[0, lm_id] = cos_yaw      # ∂(dx_local)/∂x_landmark
        jH[0, lm_id + 1] = -sin_yaw  # ∂(dx_local)/∂y_landmark
        
        jH[1, lm_id] = sin_yaw     # ∂(dy_local)/∂x_landmark
        jH[1, lm_id + 1] = cos_yaw  # ∂(dy_local)/∂y_landmark


        return jH
    
    def tf_to_global(self, x_robot, y_robot, yaw_robot, tf_obs):
        """
        Convert TF observation (in robot frame) to global landmark position.
        
        Inverse of observation model - used when initializing new landmarks.
        
        Transformation:
        [x_lm]   [x_r]       [cos(yaw)  -sin(yaw)] [dx_local]
        [y_lm] = [y_r] + R = [sin(yaw)   cos(yaw)] [dy_local]
        
        Parameters:
        -----------
        x_robot, y_robot, yaw_robot : float
            Robot pose in global frame
        tf_obs : np.array (2, 1)
            TF observation [dx_local, dy_local] in robot's frame
            
        Returns:
        --------
        x_landmark, y_landmark : float
            Landmark position in global frame
        """
        dx_local = tf_obs[0, 0]
        dy_local = tf_obs[1, 0]
        
        # Rotate from robot frame to global frame using rotation matrix R(yaw)
        dx_global = dx_local * math.cos(yaw_robot) - dy_local * math.sin(yaw_robot)
        dy_global = dx_local * math.sin(yaw_robot) + dy_local * math.cos(yaw_robot)
        
        # Translate by robot position
        x_landmark = x_robot + dx_global
        y_landmark = y_robot + dy_global
        
        return x_landmark, y_landmark
    
    def add_new_landmark_tf(self, xEst, PEst, tf_obs, tag_id):
        """
        Add newly discovered landmark to state vector and covariance matrix.
        
        This is called when a landmark is observed for the first time.
        The state vector and covariance matrix are expanded:
        
        Old state: [x_r, y_r, yaw, x_lm1, y_lm1, ..., x_lmN, y_lmN]
        New state: [..., x_lmN, y_lmN, x_lm(N+1), y_lm(N+1)]
        
        Old covariance: P (m×m)
        New covariance: [P      0  ]  ((m+2)×(m+2))
                        [0   P_new]
        
        Parameters:
        -----------
        xEst : np.array (n, 1)
            Current state estimate
        PEst : np.array (n, n)
            Current covariance matrix
        tf_obs : np.array (2, 1)
            TF observation of new landmark
        tag_id : int
            AprilTag ID of the landmark
            
        Returns:
        --------
        xEst : np.array (n+2, 1)
            Expanded state estimate with new landmark
        P_new : np.array (n+2, n+2)
            Expanded covariance matrix with new landmark
        """
        x_robot = xEst[0, 0]
        y_robot = xEst[1, 0]
        yaw_robot = xEst[2, 0]
        
        # Initialize landmark position from first observation
        x_lm, y_lm = self.tf_to_global(x_robot, y_robot, yaw_robot, tf_obs)
        
        # Expand state vector: append [x_lm, y_lm]
        xEst = np.vstack([xEst, [[x_lm], [y_lm]]])
        
        # Expand covariance matrix
        n = len(xEst)
        P_new = np.zeros((n, n))
        
        # Copy existing covariance
        P_new[0:len(PEst), 0:len(PEst)] = PEst
        
        # Initialize new landmark uncertainty (high initial uncertainty)
        INIT_LANDMARK_COV = 0.2  # Standard deviation in meters
        P_new[-2, -2] = INIT_LANDMARK_COV ** 2  # Variance in x
        P_new[-1, -1] = INIT_LANDMARK_COV ** 2  # Variance in y
        
        self.get_logger().info(
            f'Added new landmark: tag_{tag_id} at ({x_lm:.2f}, {y_lm:.2f})'
        )
        
        return xEst, P_new
    
    def associate_landmark_tf(self, tag_id):
        """
        Data association: determine if observed tag matches known landmark.
        
        Since we have unique AprilTag IDs, data association is trivial:
        just check if tag_id exists in our landmark database.
        
        For more complex scenarios (e.g., feature-based SLAM without IDs),
        you would use Mahalanobis distance or nearest neighbor matching.
        
        Parameters:
        -----------
        tag_id : int
            AprilTag ID from observation
            
        Returns:
        --------
        landmark_id : int
            Index in state vector (-1 if new landmark)
        """
        for lm_idx, known_tag_id in self.landmark_tags.items():
            if known_tag_id == tag_id:
                return lm_idx  # Found existing landmark
        return -1  # New landmark
    
    def ekf_slam_update(self, u, tf_observations):
        """
        Main EKF-SLAM algorithm: predict + update steps.
        
        EKF PREDICT STEP:
        -----------------
        x_pred = f(x_est, u)              # Predict state using motion model
        F = ∂f/∂x                         # Compute Jacobian
        P_pred = F*P_est*F^T + Q          # Predict covariance
        
        EKF UPDATE STEP (for each measurement):
        ----------------------------------------
        For new landmarks:
            - Initialize landmark position
            - Expand state vector and covariance
        
        For known landmarks:
            - z_pred = h(x_pred)          # Predict measurement
            - y = z - z_pred              # Innovation (measurement residual)
            - H = ∂h/∂x                   # Compute Jacobian
            - S = H*P_pred*H^T + R        # Innovation covariance
            - K = P_pred*H^T*S^(-1)       # Kalman gain
            - x_est = x_pred + K*y        # Update state (including robot pose!)
            - P_est = (I - K*H)*P_pred    # Update covariance
        
        Parameters:
        -----------
        u : np.array (2, 1)
            Control input [v, ω] (linear velocity, angular velocity)
        tf_observations : list of tuples
            Each tuple: (tag_id, tf_obs) where tf_obs is [dx, dy]
        """
        # ========== PREDICT STEP ==========
        # Predict next state using motion model
        xPred = self.motion_model_slam(self.xEst, u)
        
        # Compute Jacobian of motion model
        jF = self.jacob_f_slam(self.xEst, u)
        
        # Build process noise matrix Q (only robot state has process noise)
        n = len(self.xEst)
        Q_slam = np.zeros((n, n))
        Q_slam[0:3, 0:3] = self.Q  # Process noise only for robot (3x3)
        
        # Predict covariance: P_pred = F*P*F^T + Q
        PPred = jF @ self.PEst @ jF.T + Q_slam
        
        # ========== UPDATE STEP ==========
        # Process each landmark observation
        for tag_id, tf_obs in tf_observations:
            # Data association: is this a new or known landmark?
            lm_idx = self.associate_landmark_tf(tag_id)
            
            if lm_idx == -1:
                # NEW LANDMARK: Initialize and add to state
                xPred, PPred = self.add_new_landmark_tf(xPred, PPred, tf_obs, tag_id)
                new_lm_idx = (len(xPred) - 3) // 2 - 1
                self.landmark_tags[new_lm_idx] = tag_id
            else:
                # KNOWN LANDMARK: Perform EKF update
                
                # Compute Jacobian of observation model: H = ∂h/∂x
                jH = self.jacob_h_landmark_tf(xPred, lm_idx)
                
                # Predict what we should observe: z_pred = h(x_pred)
                z_pred = self.observation_model_landmark_tf(xPred, lm_idx)
                
                # Innovation (measurement residual): y = z - z_pred
                y = tf_obs - z_pred
                
                # Innovation covariance: S = H*P*H^T + R
                S = jH @ PPred @ jH.T + self.R_tf
                
                # Kalman gain: K = P*H^T*S^(-1)
                # K determines how much to trust the measurement vs. prediction
                # K affects ENTIRE state, including robot pose!
                K = PPred @ jH.T @ np.linalg.inv(S)
                
                # Update state estimate: x_new = x_pred + K*y
                # This updates robot x, y, yaw AND landmark positions
                xPred = xPred + K @ y
                
                # Normalize yaw angle to [-π, π]
                xPred[2, 0] = (xPred[2, 0] + np.pi) % (2 * np.pi) - np.pi
                
                # Update covariance: P_new = (I - K*H)*P_pred
                # This represents reduction in uncertainty due to measurement
                PPred = (np.eye(len(xPred)) - K @ jH) @ PPred

        # ========== Record trajectory for plotting ==========
        self.trajectory_x.append(self.xEst[0, 0])
        self.trajectory_y.append(self.xEst[1, 0])
        self.trajectory_yaw.append(self.xEst[2, 0])

        # Store updated state and covariance
        self.xEst = xPred
        self.PEst = PPred
        landmarks_info = [(self.landmark_tags[k], (f"{self.xEst[3 + k*2,0]:.2f}", f"{self.xEst[4 + k*2,0]:.2f}")) for k in self.landmark_tags.keys()]
        self.get_logger().info(f'EKF-SLAM update: Position ({self.xEst[0,0]:.2f}, {self.xEst[1,0]:.2f}), Yaw {math.degrees(self.xEst[2,0]):.1f}°, waypoint {self.current_waypoint_idx}')
        self.get_logger().info(f'landmarks: {len(self.landmark_tags)}, landmark positions: {landmarks_info}')
    
    def get_tf_observations(self):
        """
        Query TF tree for all visible AprilTag landmarks.
        
        Looks up transforms from 'base_link' to 'tag_X' for all possible tags.
        Filters by:
        - Recency: Skip transforms older than 0.3 seconds
        - Range: Only include tags within SENSOR_RANGE
        
        Returns:
        --------
        tf_observations : list of tuples
            Each tuple: (tag_id, tf_obs)
            where tf_obs is np.array([[dx], [dy]]) in base_link frame
        """
        tf_observations = []
        current_time = rclpy.time.Time()  # Get latest available transform

        # Try to detect tags 0-9 (adjust range based on your setup)
        for tag_id in range(10):
            try:
                # Look up transform from base_link to tag_{tag_id}
                transform = self.tf_buffer.lookup_transform(
                    'base_link',  # Target frame (robot)
                    f'tag_{tag_id}',  # Source frame (landmark)
                    current_time,  # Time (latest)
                    timeout=rclpy.duration.Duration(seconds=0.01)
                )
                
                # Check if transform is recent (reject stale data)
                transform_time = rclpy.time.Time.from_msg(transform.header.stamp)
                time_diff = (self.get_clock().now() - transform_time).nanoseconds / 1e9
                
                if time_diff > 0.2:  # Skip transforms older than 300ms
                    continue
                
                # Extract 2D position in base_link frame (ignore z for 2D SLAM)
                dx = transform.transform.translation.x
                dy = transform.transform.translation.y
                dz = transform.transform.translation.z
                
                # Calculate 3D distance for range filtering
                dist = math.sqrt(dx**2 + dy**2 + dz**2)
                SENSOR_RANGE = 2.5 # Maximum detection range (meters)
                SENSOR_MINIMUM = 0
                
                if dist < SENSOR_RANGE and dist > SENSOR_MINIMUM:
                    # Create observation: [dx, dy] in robot's local frame
                    tf_obs = np.array([[dx], [dy]])
                    tf_observations.append((tag_id, tf_obs))
                    
            except Exception:
                # Tag not visible or TF not available
                continue
        
        return tf_observations
    
    # ==================== Navigation Functions ====================
    
    def get_current_pose(self):
        """
        Extract current robot pose from EKF state estimate.
        
        Returns:
        --------
        pose : np.array [x, y, yaw]
            Robot pose in global (odom) frame
        """
        return np.array([
            self.xEst[0, 0],  # x position
            self.xEst[1, 0],  # y position
            self.xEst[2, 0]   # yaw orientation
        ])
    
    def should_drive_backwards(self, current_wp):
        """
        Determine optimal driving direction (forward or backward).
        
        For simplicity, always drive forward. Could be extended to compare
        total rotation needed for forward vs. backward approach.
        
        Parameters:
        -----------
        current_wp : np.array [x, y, yaw]
            Target waypoint
            
        Returns:
        --------
        bool : True if should drive backwards, False for forward
        """
        return False
    
    def get_desired_heading_to_goal(self, current_wp, drive_backwards):
        """
        Calculate desired heading to face toward (or away from) goal.
        
        Parameters:
        -----------
        current_wp : np.array [x, y, yaw]
            Target waypoint
        drive_backwards : bool
            Whether driving backwards
            
        Returns:
        --------
        desired_heading : float
            Target heading angle in radians (wrapped to [-π, π])
        """
        current_pose = self.get_current_pose()
        delta_x = current_wp[0] - current_pose[0]
        delta_y = current_wp[1] - current_pose[1]
        angle_to_target = np.arctan2(delta_y, delta_x)
        
        if drive_backwards:
            # Face away from target (add π)
            desired_heading = angle_to_target + np.pi
            desired_heading = (desired_heading + np.pi) % (2 * np.pi) - np.pi
        else:
            desired_heading = angle_to_target
        
        return desired_heading
    
    def get_rotation_direction(self, heading_error):
        """
        Determine angular velocity direction for pure rotation.
        
        Parameters:
        -----------
        heading_error : float
            Difference between desired and current heading (radians)
            Positive = need to rotate CCW, Negative = rotate CW
            
        Returns:
        --------
        angular_velocity : float
            Fixed magnitude, sign indicates direction
        """
        if heading_error > 0:
            return self.fixed_rotation_vel  # Counter-clockwise
        else:
            return -self.fixed_rotation_vel  # Clockwise
    
    def control_loop(self):
        """
        Main control loop: EKF-SLAM update + waypoint navigation.
        
        Control Flow:
        1. Get TF observations of landmarks
        2. Run EKF-SLAM update (predict + update)
        3. Broadcast updated robot pose as TF
        4. Execute waypoint navigation (three-stage controller)
        
        Three-Stage Navigation:
        - Stage 1 (rotate_to_goal): Rotate to face goal direction
        - Stage 2 (drive): Drive toward goal with heading correction
        - Stage 3 (rotate_to_orient): Rotate to final orientation
        """
        # ===== EKF-SLAM UPDATE =====
        
        current_time = time.time()
        
        if current_time - self.last_tag_detection_time > 0.2:
            self.last_tag_detection_time = current_time
            tf_observations = self.get_tf_observations()
        else:
            tf_observations = []
        # tf_observations = []
        
        
        # Create control input from last command (for prediction step)
        if hasattr(self, 'last_linear_vel') and hasattr(self, 'last_angular_vel'):
            u = np.array([[self.last_linear_vel], [self.last_angular_vel]])
        else:
            u = np.array([[0.0], [0.0]])
        
        # Run EKF-SLAM algorithm
        self.ekf_slam_update(u, tf_observations)
        
        # Publish updated pose
        self.broadcast_tf()
        
        # ===== WAYPOINT NAVIGATION =====
        
        # Check if all waypoints completed
        if self.current_waypoint_idx >= len(self.waypoints):
            if not self.plotted:
                self.save_final_plot()
                self.plotted = True
            self.get_logger().info('All waypoints reached!')
            self.stop_robot()
            return

        current_wp = self.waypoints[self.current_waypoint_idx]
        current_pose = self.get_current_pose()
        
        # Initialize for new waypoint
        if not self.waypoint_reached:
            self.pid.setTarget(current_wp)
            self.drive_backwards = self.should_drive_backwards(current_wp)
            self.waypoint_reached = True
            self.stage = 'rotate_to_goal'

        # Calculate position error
        delta_x = current_wp[0] - current_pose[0]
        delta_y = current_wp[1] - current_pose[1]
        position_error = np.sqrt(delta_x**2 + delta_y**2)
        
        twist_msg = Twist()
        
        # STAGE 1: Rotate to face the goal
        if self.stage == 'rotate_to_goal':
            desired_heading = self.get_desired_heading_to_goal(current_wp, self.drive_backwards)
            heading_error = desired_heading - current_pose[2]
            heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi
            
            if abs(heading_error) < 0.05:  # Aligned with goal
                self.stage = 'drive'
                twist_msg.angular.z = 0.0
            else:
                # Pure rotation at fixed velocity
                twist_msg.angular.z = float(self.get_rotation_direction(heading_error))
        
        # STAGE 2: Drive towards the goal
        elif self.stage == 'drive':
            if position_error < self.tolerance:  # Reached position
                self.stage = 'rotate_to_orient'
                twist_msg.linear.x = 0.0
            else:
                desired_heading = self.get_desired_heading_to_goal(current_wp, self.drive_backwards)
                heading_error = desired_heading - current_pose[2]
                heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi
                
                if abs(heading_error) > 0.2:  # Heading drift too large
                    self.stage = 'rotate_to_goal'
                    twist_msg.linear.x = 0.0
                else:
                    # PID control for smooth driving with heading correction
                    update_value = self.pid.update(current_pose, self.drive_backwards)
                    twist_msg.linear.x = float(update_value[0])
                    twist_msg.angular.z = float(update_value[1])
        
        # STAGE 3: Rotate to target orientation
        elif self.stage == 'rotate_to_orient':
            heading_error = current_wp[2] - current_pose[2]
            heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi
            
            if abs(heading_error) < self.angle_tolerance:  # Final orientation achieved
                self.current_waypoint_idx += 1
                self.waypoint_reached = False
                twist_msg.angular.z = 0.0
                self.get_logger().info(f'Reached waypoint {self.current_waypoint_idx}')
            else:
                # Pure rotation at fixed velocity
                twist_msg.angular.z = float(self.get_rotation_direction(heading_error))
        
        # Store velocities for next EKF prediction step
        self.last_linear_vel = twist_msg.linear.x
        self.last_angular_vel = twist_msg.angular.z
        
        # Publish velocity command to robot
        self.cmd_vel_pub.publish(twist_msg)
    
    def broadcast_tf(self):
        """
        Broadcast TF transform from odom to base_link.
        
        Publishes the estimated robot pose as a TF transform, allowing
        other nodes to transform between coordinate frames.
        """
        current_time = self.get_clock().now()
        
        t = TransformStamped()
        t.header.stamp = current_time.to_msg()
        t.header.frame_id = self.odom_frame  # Parent frame
        t.child_frame_id = self.base_frame   # Child frame
        
        # Set translation (position)
        t.transform.translation.x = float(self.xEst[0, 0])
        t.transform.translation.y = float(self.xEst[1, 0])
        t.transform.translation.z = 0.0  # 2D navigation
        
        # Set rotation (orientation as quaternion)
        yaw = float(self.xEst[2, 0])
        qx, qy, qz, qw = self.euler_to_quaternion(0, 0, yaw)
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        
        # Broadcast transform
        self.tf_broadcaster.sendTransform(t)
    
    def euler_to_quaternion(self, roll, pitch, yaw):
        """
        Convert Euler angles (roll, pitch, yaw) to quaternion (x, y, z, w).
        
        For 2D navigation, roll=0 and pitch=0, only yaw is used.
        
        Parameters:
        -----------
        roll, pitch, yaw : float
            Euler angles in radians
            
        Returns:
        --------
        qx, qy, qz, qw : float
            Quaternion components
        """
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        
        return qx, qy, qz, qw

    
    def stop_robot(self):
        """
        Stop the robot by publishing zero velocities.
        Also logs the final landmark map.
        """
        twist_msg = Twist()
        self.cmd_vel_pub.publish(twist_msg)
        
        # Log final map
        self.get_logger().info('='*50)
        self.get_logger().info('Final Landmark Map:')
        n_landmarks = (len(self.xEst) - 3) // 2  # Robot state is 3D now
        for i in range(n_landmarks):
            lm_id = 3 + i * 2  # Landmarks start at index 3
            tag_id = self.landmark_tags[i]
            x = self.xEst[lm_id, 0]
            y = self.xEst[lm_id + 1, 0]
            std_x = math.sqrt(self.PEst[lm_id, lm_id])
            std_y = math.sqrt(self.PEst[lm_id + 1, lm_id + 1])
            self.get_logger().info(
                f'  tag_{tag_id}: pos=({x:.3f}, {y:.3f}), '
                f'std=({std_x:.3f}, {std_y:.3f})'
            )
        self.get_logger().info('='*50)


def main(args=None):
    rclpy.init(args=args)
    node = EKFSLAMNode()
    
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