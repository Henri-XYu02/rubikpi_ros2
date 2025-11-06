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

"""
EKF-SLAM implementation for landmark mapping while navigating waypoints
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
    def __init__(self):
        super().__init__('ekf_slam_node')
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)
        
        self.odom_frame = 'odom'
        self.base_frame = 'base_link'
        
        # EKF-SLAM State
        # State: [x_robot, y_robot, yaw_robot, v_robot, x_lm1, y_lm1, x_lm2, y_lm2, ...]
        self.xEst = np.zeros((4, 1))  # Start with just robot state
        self.PEst = np.eye(4) * 0.1   # Initial covariance
        
        # Map: landmark_index -> tag_id
        self.landmark_tags = {}
        
        # Process and measurement noise matrices
        self.Q = np.diag([0.1, 0.1, np.deg2rad(5.0), 0.5]) ** 2  # Process noise
        self.R_tf = np.diag([0.08, 0.08]) ** 2  # TF measurement noise (x, y in meters)
        
        # Waypoints
        self.waypoints = np.array([
            [0.0, 0.0, 0.0], 
            [1.0, 0.0, np.pi/2],
            [1.0, 1.0, np.pi],
            [0.0, 1.0, -np.pi/2]
        ])
        
        # Navigation state
        self.pid = PIDcontroller(0.8, 0.01, 0.005)
        self.current_waypoint_idx = 0
        self.waypoint_reached = False
        self.tolerance = 0.15
        self.angle_tolerance = 0.1
        self.drive_backwards = False
        self.stage = 'rotate_to_goal'
        self.fixed_rotation_vel = 0.785
        
        # Timers
        self.dt = 0.1
        self.control_timer = self.create_timer(self.dt, self.control_loop)
        
        self.get_logger().info('EKF-SLAM Node initialized')
        
    # ==================== EKF-SLAM Functions ====================
    
    def motion_model_slam(self, x, u):
        """
        SLAM motion model: predicts next state given current state and control.
        
        Motion Model Equations (discrete-time):
        x_{t+1} = x_t + v*dt*cos(yaw)
        y_{t+1} = y_t + v*dt*sin(yaw)
        yaw_{t+1} = yaw_t + ω*dt
        v_{t+1} = v_t
        landmarks remain static: x_lm_{t+1} = x_lm_t
        
        In matrix form: x_{t+1} = F*x_t + B*u
        
        Parameters:
        -----------
        x : np.array (n, 1)
            Current state vector [x_r, y_r, yaw, v, x_lm1, y_lm1, ...]
        u : np.array (2, 1)
            Control input [v, ω] (linear velocity, angular velocity)
            
        Returns:
        --------
        x_new : np.array (n, 1)
            Predicted next state
        """
        n = len(x)
        
        # State transition matrix F (landmarks don't move, so F is mostly identity)
        F_slam = np.eye(n)
        
        # Robot state transition (velocity model)
        F_slam[0:4, 0:4] = np.array([
            [1.0, 0, 0, 0],  # x doesn't depend on other states directly
            [0, 1.0, 0, 0],  # y doesn't depend on other states directly
            [0, 0, 1.0, 0],  # yaw doesn't depend on other states directly
            [0, 0, 0, 0]     # velocity is replaced by control input
        ])
        
        # Control input matrix B (only affects robot state, not landmarks)
        B_slam = np.zeros((n, 2))
        yaw = x[2, 0]
        B_slam[0:4, 0:2] = np.array([
            [self.dt * math.cos(yaw), 0],  # x position change from linear velocity
            [self.dt * math.sin(yaw), 0],  # y position change from linear velocity
            [0.0, self.dt],                # yaw change from angular velocity
            [1.0, 0.0]                     # velocity becomes the control input
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
        
        Partial derivatives:
        ∂x/∂yaw = -v*dt*sin(yaw)
        ∂x/∂v = dt*cos(yaw)
        ∂y/∂yaw = v*dt*cos(yaw)
        ∂y/∂v = dt*sin(yaw)
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
        
        jF[0:4, 0:4] = np.array([
            [1.0, 0.0, -self.dt * v * math.sin(yaw), self.dt * math.cos(yaw)],
            [0.0, 1.0, self.dt * v * math.cos(yaw), self.dt * math.sin(yaw)],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
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
        # Calculate index in state vector: robot has 4 states, each landmark has 2
        lm_id = 4 + landmark_id * 2
        
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
        
        # Rotation matrix from global to local frame
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
        
        Derivatives w.r.t robot pose:
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
        lm_id = 4 + landmark_id * 2
        
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
        jH[0, 1] = -sin_yaw  # ∂(dx_local)/∂y_robot
        jH[0, 2] = -dx_global * sin_yaw + dy_global * cos_yaw  # ∂(dx_local)/∂yaw
        
        jH[1, 0] = sin_yaw   # ∂(dy_local)/∂x_robot
        jH[1, 1] = -cos_yaw  # ∂(dy_local)/∂y_robot
        jH[1, 2] = dx_global * cos_yaw + dy_global * sin_yaw  # ∂(dy_local)/∂yaw
        
        # Derivatives w.r.t landmark position (columns lm_id, lm_id+1)
        jH[0, lm_id] = cos_yaw      # ∂(dx_local)/∂x_landmark
        jH[0, lm_id + 1] = sin_yaw  # ∂(dx_local)/∂y_landmark
        
        jH[1, lm_id] = -sin_yaw     # ∂(dy_local)/∂x_landmark
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
        
        Old state: [x_r, y_r, yaw, v, x_lm1, y_lm1, ..., x_lmN, y_lmN]
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
        INIT_LANDMARK_COV = 0.5  # Standard deviation in meters
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
            - x_est = x_pred + K*y        # Update state
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
        Q_slam[0:4, 0:4] = self.Q  # Process noise only for robot
        
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
                new_lm_idx = (len(xPred) - 4) // 2 - 1
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
                K = PPred @ jH.T @ np.linalg.inv(S)
                
                # Update state estimate: x_new = x_pred + K*y
                xPred = xPred + K @ y
                
                # Normalize yaw angle to [-π, π]
                xPred[2, 0] = (xPred[2, 0] + np.pi) % (2 * np.pi) - np.pi
                
                # Update covariance: P_new = (I - K*H)*P_pred
                # This represents reduction in uncertainty due to measurement
                PPred = (np.eye(len(xPred)) - K @ jH) @ PPred
        
        # Store updated state and covariance
        self.xEst = xPred
        self.PEst = PPred
    
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
                    timeout=rclpy.duration.Duration(seconds=0.05)
                )
                
                # Check if transform is recent (reject stale data)
                transform_time = rclpy.time.Time.from_msg(transform.header.stamp)
                time_diff = (self.get_clock().now() - transform_time).nanoseconds / 1e9
                
                if time_diff > 0.3:  # Skip transforms older than 300ms
                    continue
                
                # Extract 2D position in base_link frame (ignore z for 2D SLAM)
                dx = transform.transform.translation.x
                dy = transform.transform.translation.y
                dz = transform.transform.translation.z
                
                # Calculate 3D distance for range filtering
                dist = math.sqrt(dx**2 + dy**2 + dz**2)
                SENSOR_RANGE = 5.0  # Maximum detection range (meters)
                
                if dist < SENSOR_RANGE:
                    # Create observation: [dx, dy] in robot's local frame
                    tf_obs = np.array([[dx], [dy]])
                    tf_observations.append((tag_id, tf_obs))
                    
            except Exception:
                # Tag not visible or TF not available
                continue
        
        return tf_observations
    
    # ==================== Navigation Functions ====================
    
    def get_current_pose(self):
        """Get current robot pose from EKF estimate"""
        return np.array([
            self.xEst[0, 0],  # x
            self.xEst[1, 0],  # y
            self.xEst[2, 0]   # yaw
        ])
    
    def should_drive_backwards(self, current_wp):
        """Determine if robot should drive backwards"""
        return False  # Keep it simple - always drive forward
    
    def get_desired_heading_to_goal(self, current_wp, drive_backwards):
        """Get the desired heading to face towards the goal"""
        current_pose = self.get_current_pose()
        delta_x = current_wp[0] - current_pose[0]
        delta_y = current_wp[1] - current_pose[1]
        angle_to_target = np.arctan2(delta_y, delta_x)
        
        if drive_backwards:
            desired_heading = angle_to_target + np.pi
            desired_heading = (desired_heading + np.pi) % (2 * np.pi) - np.pi
        else:
            desired_heading = angle_to_target
        
        return desired_heading
    
    def get_rotation_direction(self, heading_error):
        """Determine rotation direction"""
        if heading_error > 0:
            return self.fixed_rotation_vel
        else:
            return -self.fixed_rotation_vel
    
    def control_loop(self):
        """Main control loop with EKF-SLAM"""
        # Get TF observations
        tf_observations = self.get_tf_observations()
        
        # Control input from last command
        if hasattr(self, 'last_linear_vel') and hasattr(self, 'last_angular_vel'):
            u = np.array([[self.last_linear_vel], [self.last_angular_vel]])
        else:
            u = np.array([[0.0], [0.0]])
        
        # EKF-SLAM update
        self.ekf_slam_update(u, tf_observations)
        
        # Broadcast TF
        self.broadcast_tf()
        
        # Navigation logic
        if self.current_waypoint_idx >= len(self.waypoints):
            self.get_logger().info('All waypoints reached!')
            self.stop_robot()
            return

        current_wp = self.waypoints[self.current_waypoint_idx]
        current_pose = self.get_current_pose()
        
        if not self.waypoint_reached:
            self.pid.setTarget(current_wp)
            self.drive_backwards = self.should_drive_backwards(current_wp)
            self.waypoint_reached = True
            self.stage = 'rotate_to_goal'

        delta_x = current_wp[0] - current_pose[0]
        delta_y = current_wp[1] - current_pose[1]
        position_error = np.sqrt(delta_x**2 + delta_y**2)
        
        twist_msg = Twist()
        
        # Stage 1: Rotate to face the goal
        if self.stage == 'rotate_to_goal':
            desired_heading = self.get_desired_heading_to_goal(current_wp, self.drive_backwards)
            heading_error = desired_heading - current_pose[2]
            heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi
            
            if abs(heading_error) < 0.05:
                self.stage = 'drive'
                twist_msg.angular.z = 0.0
            else:
                twist_msg.angular.z = float(self.get_rotation_direction(heading_error))
        
        # Stage 2: Drive towards the goal
        elif self.stage == 'drive':
            if position_error < self.tolerance:
                self.stage = 'rotate_to_orient'
                twist_msg.linear.x = 0.0
            else:
                desired_heading = self.get_desired_heading_to_goal(current_wp, self.drive_backwards)
                heading_error = desired_heading - current_pose[2]
                heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi
                
                if abs(heading_error) > 0.2:
                    self.stage = 'rotate_to_goal'
                    twist_msg.linear.x = 0.0
                else:
                    update_value = self.pid.update(current_pose, self.drive_backwards)
                    twist_msg.linear.x = float(update_value[0])
                    twist_msg.angular.z = float(update_value[1])
        
        # Stage 3: Rotate to target orientation
        elif self.stage == 'rotate_to_orient':
            heading_error = current_wp[2] - current_pose[2]
            heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi
            
            if abs(heading_error) < self.angle_tolerance:
                self.current_waypoint_idx += 1
                self.waypoint_reached = False
                twist_msg.angular.z = 0.0
                self.get_logger().info(f'Reached waypoint {self.current_waypoint_idx}')
            else:
                twist_msg.angular.z = float(self.get_rotation_direction(heading_error))
        
        # Store velocities for next EKF prediction
        self.last_linear_vel = twist_msg.linear.x
        self.last_angular_vel = twist_msg.angular.z
        
        # Publish control command
        self.cmd_vel_pub.publish(twist_msg)
    
    def broadcast_tf(self):
        """Broadcast TF transform from odom to base_link"""
        current_time = self.get_clock().now()
        
        t = TransformStamped()
        t.header.stamp = current_time.to_msg()
        t.header.frame_id = self.odom_frame
        t.child_frame_id = self.base_frame
        
        t.transform.translation.x = float(self.xEst[0, 0])
        t.transform.translation.y = float(self.xEst[1, 0])
        t.transform.translation.z = 0.0
        
        yaw = float(self.xEst[2, 0])
        qx, qy, qz, qw = self.euler_to_quaternion(0, 0, yaw)
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        
        self.tf_broadcaster.sendTransform(t)
    
    def euler_to_quaternion(self, roll, pitch, yaw):
        """Convert Euler angles to quaternion"""
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
        """Stop the robot"""
        twist_msg = Twist()
        self.cmd_vel_pub.publish(twist_msg)
        
        # Log final map
        self.get_logger().info('='*50)
        self.get_logger().info('Final Landmark Map:')
        n_landmarks = (len(self.xEst) - 4) // 2
        for i in range(n_landmarks):
            lm_id = 4 + i * 2
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