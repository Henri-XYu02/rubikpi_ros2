#!/usr/bin/env python3
import numpy as np
import heapq
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import math

class PathPlanner:
    """
    Path planning using Voronoi diagrams (max safety) and Visibility graphs (min distance)
    """
    def __init__(self, workspace_bounds=(0, 2.5, 0, 2.5), robot_radius=0.2):
        """
        Initialize path planner
        
        Args:
            workspace_bounds: (x_min, x_max, y_min, y_max) in meters
            robot_radius: Robot radius for collision checking in meters
        """
        self.x_min, self.x_max, self.y_min, self.y_max = workspace_bounds
        self.robot_radius = robot_radius
        
        # Square obstacle definition (center of workspace)
        self.obstacle_center = np.array([1.25, 1.25])  # Center of 2.5m x 2.5m workspace
        self.obstacle_size = 0.3  # Side length of square obstacle
        
        # Calculate square bounds
        half_size = self.obstacle_size / 2
        self.obstacle_bounds = {
            'x_min': self.obstacle_center[0] - half_size,
            'x_max': self.obstacle_center[0] + half_size,
            'y_min': self.obstacle_center[1] - half_size,
            'y_max': self.obstacle_center[1] + half_size
        }
        
        
        # Square corners for reference
        self.obstacle_corners = self._get_obstacle_corners()

    def _get_obstacle_corners(self):
        """Get the four corners of the square obstacle"""
        half_size = self.obstacle_size / 2
        corners = [
            self.obstacle_center + np.array([-half_size, -half_size]),  # Bottom-left
            self.obstacle_center + np.array([half_size, -half_size]),   # Bottom-right
            self.obstacle_center + np.array([half_size, half_size]),    # Top-right
            self.obstacle_center + np.array([-half_size, half_size])    # Top-left
        ]
        return corners

    def create_square_boundary_points(self, center, size, points_per_edge=5):
        """Create points along all 4 edges of the square"""
        half_size = size / 2
        points = []
        
        # Bottom edge (left to right)
        for i in range(points_per_edge):
            t = i / (points_per_edge - 1)
            x = center[0] - half_size + t * size
            y = center[1] - half_size
            points.append([x, y])
        
        # Right edge (bottom to top)
        for i in range(1, points_per_edge):  # Skip corner (already added)
            t = i / (points_per_edge - 1)
            x = center[0] + half_size
            y = center[1] - half_size + t * size
            points.append([x, y])
        
        # Top edge (right to left)
        for i in range(1, points_per_edge):
            t = i / (points_per_edge - 1)
            x = center[0] + half_size - t * size
            y = center[1] + half_size
            points.append([x, y])
        
        # Left edge (top to bottom)
        for i in range(1, points_per_edge - 1):  # Skip both corners
            t = i / (points_per_edge - 1)
            x = center[0] - half_size
            y = center[1] + half_size - t * size
            points.append([x, y])
        
        return points
        
    
    def point_to_square_distance(self, point):
        """
        Calculate minimum distance from point to square obstacle boundary.
        Returns negative if inside the square.
        """
        # Distance to each edge of the square
        dx_min = point[0] - self.obstacle_bounds['x_min']
        dx_max = self.obstacle_bounds['x_max'] - point[0]
        dy_min = point[1] - self.obstacle_bounds['y_min']
        dy_max = self.obstacle_bounds['y_max'] - point[1]
        
        # Check if point is inside the square
        if dx_min >= 0 and dx_max >= 0 and dy_min >= 0 and dy_max >= 0:
            # Inside: return negative of minimum distance to any edge
            return -min(dx_min, dx_max, dy_min, dy_max)
        
        # Outside: calculate distance to nearest edge or corner
        # Clamp point to square boundary
        closest_x = np.clip(point[0], self.obstacle_bounds['x_min'], self.obstacle_bounds['x_max'])
        closest_y = np.clip(point[1], self.obstacle_bounds['y_min'], self.obstacle_bounds['y_max'])
        
        return np.linalg.norm(point - np.array([closest_x, closest_y]))
    
    def is_line_intersecting_square(self, p1, p2):
        """
        Check if line segment from p1 to p2 intersects with the square obstacle.
        """
        # Expand square by robot radius for collision checking
        expanded_bounds = {
            'x_min': self.obstacle_bounds['x_min'] - self.robot_radius,
            'x_max': self.obstacle_bounds['x_max'] + self.robot_radius,
            'y_min': self.obstacle_bounds['y_min'] - self.robot_radius,
            'y_max': self.obstacle_bounds['y_max'] + self.robot_radius
        }
        
        # Check if either endpoint is inside the expanded square
        for p in [p1, p2]:
            if (expanded_bounds['x_min'] <= p[0] <= expanded_bounds['x_max'] and
                expanded_bounds['y_min'] <= p[1] <= expanded_bounds['y_max']):
                return True
        
        # Check if line segment intersects any of the four edges of expanded square
        square_edges = [
            (np.array([expanded_bounds['x_min'], expanded_bounds['y_min']]),
             np.array([expanded_bounds['x_max'], expanded_bounds['y_min']])),  # Bottom
            (np.array([expanded_bounds['x_max'], expanded_bounds['y_min']]),
             np.array([expanded_bounds['x_max'], expanded_bounds['y_max']])),  # Right
            (np.array([expanded_bounds['x_max'], expanded_bounds['y_max']]),
             np.array([expanded_bounds['x_min'], expanded_bounds['y_max']])),  # Top
            (np.array([expanded_bounds['x_min'], expanded_bounds['y_max']]),
             np.array([expanded_bounds['x_min'], expanded_bounds['y_min']])),  # Left
        ]
        
        for edge_start, edge_end in square_edges:
            if self._line_segments_intersect(p1, p2, edge_start, edge_end):
                return True
        
        return False
    
    def _line_segments_intersect(self, p1, p2, p3, p4):
        """Check if line segment p1-p2 intersects with line segment p3-p4"""
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)
    
    def is_collision_free(self, p1, p2, check_distance=0.01):
        """
        Check if line segment from p1 to p2 is collision-free
        
        Args:
            p1, p2: Start and end points as numpy arrays
            check_distance: Distance between collision check points
        """
        # Check workspace bounds for both endpoints
        for p in [p1, p2]:
            if not (self.x_min <= p[0] <= self.x_max and 
                    self.y_min <= p[1] <= self.y_max):
                return False
        
        # Check if line intersects with square obstacle
        if self.is_line_intersecting_square(p1, p2):
            return False
        
        # Additional point-by-point checking for safety
        direction = p2 - p1
        distance = np.linalg.norm(direction)
        
        if distance < 1e-6:
            return True
        
        direction = direction / distance
        num_checks = int(distance / check_distance) + 1
        
        for i in range(num_checks + 1):
            t = min(i * check_distance, distance)
            point = p1 + direction * t
            
            # Check if point is too close to square obstacle
            dist_to_obs = self.point_to_square_distance(point)
            if dist_to_obs < self.robot_radius:
                return False
        
        return True
    
    def distance_to_obstacle(self, point):
        """Calculate minimum distance from point to square obstacle boundary"""
        return self.point_to_square_distance(point)
    
    def plan_voronoi_path(self, start, goal):
        """
        Plan maximum safety path using Voronoi diagram
        
        Args:
            start: Start position [x, y] or [x, y, theta]
            goal: Goal position [x, y] or [x, y, theta]
        
        Returns:
            waypoints: List of [x, y, theta] waypoints
        """
        start = np.array(start[:2])
        goal = np.array(goal[:2])
        
        # Create Voronoi diagram using square boundary and workspace corners
        points = []
        
        # Add workspace corners (important for bounding the diagram)
        corners = [
            [self.x_min, self.y_min],
            [self.x_min, self.y_max],
            [self.x_max, self.y_min],
            [self.x_max, self.y_max]
        ]
        points.extend(corners)
        
        # Add points along workspace edges for better Voronoi structure
        num_edge_points = 4
        for i in range(1, num_edge_points):
            t = i / num_edge_points
            # Bottom and top edges
            points.append([self.x_min + t * (self.x_max - self.x_min), self.y_min])
            points.append([self.x_min + t * (self.x_max - self.x_min), self.y_max])
            # Left and right edges
            points.append([self.x_min, self.y_min + t * (self.y_max - self.y_min)])
            points.append([self.x_max, self.y_min + t * (self.y_max - self.y_min)])
        
        # Add square obstacle boundary points (dense approximation)
        square_points = self.create_square_boundary_points(
            self.obstacle_center, 
            self.obstacle_size, 
            points_per_edge=6  # Balanced approximation
        )
        points.extend(square_points)
        
        points = np.array(points)
        
        # Generate Voronoi diagram
        vor = Voronoi(points)
        
        print(f"\nVoronoi Diagram Generation:")
        print(f"  Seed points used: {len(points)}")
        print(f"  Voronoi vertices generated: {len(vor.vertices)}")
        print(f"\nAll Voronoi vertices:")
        for i, vertex in enumerate(vor.vertices):
            in_bounds = (self.x_min <= vertex[0] <= self.x_max and 
                        self.y_min <= vertex[1] <= self.y_max)
            dist_to_obs = self.distance_to_obstacle(vertex)
            print(f"  {i}: [{vertex[0]:.3f}, {vertex[1]:.3f}] - "
                  f"in_bounds={in_bounds}, dist_to_obs={dist_to_obs:.3f}")
        
        # Extract valid Voronoi vertices (nodes) with relaxed constraints
        nodes = [start]
        margin = 0.05  # Small margin from workspace edges
        valid_voronoi_nodes = []
        
        for i, vertex in enumerate(vor.vertices):
            # Check if vertex is within workspace bounds (with margin)
            if (self.x_min + margin <= vertex[0] <= self.x_max - margin and 
                self.y_min + margin <= vertex[1] <= self.y_max - margin):
                # Check if vertex is not inside the obstacle
                dist_to_obs = self.distance_to_obstacle(vertex)
                # More relaxed: just need to be outside the expanded obstacle
                if dist_to_obs > 0:  # Outside obstacle
                    nodes.append(vertex)
                    valid_voronoi_nodes.append(i)
        
        nodes.append(goal)
        nodes = np.array(nodes)
        
        print(f"\nValid Voronoi nodes (used for path planning): {len(nodes) - 2}")  # Exclude start/goal
        print(f"Valid vertex indices: {valid_voronoi_nodes}")
        
        # If we have very few nodes, create a grid-based backup
        if len(nodes) < 10:
            print("\nAdding grid nodes for better connectivity...")
            grid_nodes = self._create_grid_nodes(4)  # 4x4 grid
            for node in grid_nodes:
                if self.distance_to_obstacle(node) > self.robot_radius:
                    nodes = np.vstack([nodes, node])
            print(f"Total nodes after grid augmentation: {len(nodes)}")
        
        # Build graph of collision-free edges with safety-based costs
        graph = {i: [] for i in range(len(nodes))}
        
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if self.is_collision_free(nodes[i], nodes[j]):
                    # Calculate geometric distance
                    dist = np.linalg.norm(nodes[i] - nodes[j])
                    
                    # For Voronoi, add cost penalty based on proximity to obstacle
                    # Check clearance at multiple points along the edge
                    num_samples = 5
                    min_clearance = float('inf')
                    
                    for k in range(num_samples + 1):
                        t = k / num_samples
                        sample_point = nodes[i] + t * (nodes[j] - nodes[i])
                        clearance = self.distance_to_obstacle(sample_point)
                        min_clearance = min(min_clearance, clearance)
                    
                    # Apply penalty based on clearance
                    # Paths with less clearance get higher cost
                    safety_threshold = self.robot_radius * 3  # Prefer 3x robot radius clearance
                    
                    if min_clearance < safety_threshold:
                        # Exponential penalty for getting too close
                        penalty_factor = 1.0 + 2.0 * (1.0 - min_clearance / safety_threshold) ** 2
                    else:
                        # Small bonus for being far from obstacles
                        penalty_factor = 0.9
                    
                    # Final cost includes safety penalty
                    weighted_cost = dist * penalty_factor
                    
                    graph[i].append((j, weighted_cost))
                    graph[j].append((i, weighted_cost))
        
        # Check connectivity
        start_connections = len(graph[0])
        goal_connections = len(graph[len(nodes) - 1])
        print(f"\nGraph connectivity:")
        print(f"  Start has {start_connections} connections")
        print(f"  Goal has {goal_connections} connections")
        
        # Run A* from start (index 0) to goal (index len(nodes)-1)
        path_indices = self._astar(graph, nodes, 0, len(nodes) - 1)
        
        if path_indices is None:
            print("No path found with Voronoi method!")
            return None
        
        # Convert to waypoints with orientations
        waypoints = self._indices_to_waypoints(nodes, path_indices)
        
        # Store nodes for visualization
        self.voronoi_nodes = nodes
        
        return waypoints
    
    def _min_distance_to_line_segment(self, point, line_start, line_end):
        """
        Calculate minimum distance from a point to a line segment.
        For a square obstacle, calculates the minimum distance from any point 
        on the obstacle boundary to the line segment.
        
        Args:
            point: Center of obstacle (or any reference point)
            line_start: Start of line segment
            line_end: End of line segment
        
        Returns:
            Minimum distance from obstacle to line segment
        """
        # For square obstacle, check distance from all corners and edges
        # to the line segment and return the minimum
        
        min_dist = float('inf')
        
        # Check distance from obstacle corners to line segment
        for corner in self.obstacle_corners:
            dist = self._point_to_line_segment_distance(corner, line_start, line_end)
            min_dist = min(min_dist, dist)
        
        # Also check if line segment passes through or near obstacle
        # by sampling points along the line segment
        num_samples = 10
        for k in range(num_samples + 1):
            t = k / num_samples
            sample_point = line_start + t * (line_end - line_start)
            dist_to_obs = self.distance_to_obstacle(sample_point)
            min_dist = min(min_dist, dist_to_obs)
        
        return min_dist
    
    def _point_to_line_segment_distance(self, point, line_start, line_end):
        """
        Calculate minimum distance from a point to a line segment.
        
        Args:
            point: Point to measure from
            line_start: Start of line segment  
            line_end: End of line segment
        
        Returns:
            Minimum distance from point to line segment
        """
        # Vector from line_start to line_end
        line_vec = line_end - line_start
        line_len_sq = np.dot(line_vec, line_vec)
        
        if line_len_sq < 1e-10:  # Line segment is essentially a point
            return np.linalg.norm(point - line_start)
        
        # Project point onto the line, clamped to segment
        # t represents position along line: 0 = start, 1 = end
        t = max(0, min(1, np.dot(point - line_start, line_vec) / line_len_sq))
        
        # Find closest point on line segment
        closest_point = line_start + t * line_vec
        
        # Return distance from point to closest point on segment
        return np.linalg.norm(point - closest_point)
    
    def _create_grid_nodes(self, grid_size):
        """Create a grid of nodes in the free space"""
        nodes = []
        step_x = (self.x_max - self.x_min) / (grid_size + 1)
        step_y = (self.y_max - self.y_min) / (grid_size + 1)
        
        for i in range(1, grid_size + 1):
            for j in range(1, grid_size + 1):
                x = self.x_min + i * step_x
                y = self.y_min + j * step_y
                node = np.array([x, y])
                
                # Only add if not too close to obstacle
                if self.distance_to_obstacle(node) > self.robot_radius:
                    nodes.append(node)
        
        return nodes
    
    def plan_visibility_path(self, start, goal):
        """
        Plan minimum distance path using Visibility graph
        
        Args:
            start: Start position [x, y] or [x, y, theta]
            goal: Goal position [x, y] or [x, y, theta]
        
        Returns:
            waypoints: List of [x, y, theta] waypoints
        """
        start = np.array(start[:2])
        goal = np.array(goal[:2])
        
        # Generate visibility graph nodes
        nodes = [start]
        
        # Reduced safety margin - visibility graph prioritizes distance over safety
        # Just enough clearance to avoid collision
        safe_distance = self.robot_radius + 0.02  # Minimal extra clearance (was 0.1)
        
        # Create vertices at corners of expanded square
        expanded_corners = [
            np.array([self.obstacle_bounds['x_min'] - safe_distance, 
                     self.obstacle_bounds['y_min'] - safe_distance]),  # SW
            np.array([self.obstacle_bounds['x_max'] + safe_distance, 
                     self.obstacle_bounds['y_min'] - safe_distance]),  # SE
            np.array([self.obstacle_bounds['x_max'] + safe_distance, 
                     self.obstacle_bounds['y_max'] + safe_distance]),  # NE
            np.array([self.obstacle_bounds['x_min'] - safe_distance, 
                     self.obstacle_bounds['y_max'] + safe_distance]),  # NW
        ]
        
        # Add midpoints on each side for better paths
        # These allow the path to hug the obstacle edges more closely
        midpoints = [
            np.array([self.obstacle_center[0], 
                     self.obstacle_bounds['y_min'] - safe_distance]),  # South
            np.array([self.obstacle_bounds['x_max'] + safe_distance, 
                     self.obstacle_center[1]]),                         # East
            np.array([self.obstacle_center[0], 
                     self.obstacle_bounds['y_max'] + safe_distance]),  # North
            np.array([self.obstacle_bounds['x_min'] - safe_distance, 
                     self.obstacle_center[1]]),                         # West
        ]
        
        for vertex in expanded_corners + midpoints:
            # Check if vertex is within workspace
            if (self.x_min <= vertex[0] <= self.x_max and 
                self.y_min <= vertex[1] <= self.y_max):
                nodes.append(vertex)
        
        nodes.append(goal)
        nodes = np.array(nodes)
        
        print(f"Visibility: Generated {len(nodes)} nodes")
        
        # Build visibility graph
        graph = {i: [] for i in range(len(nodes))}
        
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if self.is_collision_free(nodes[i], nodes[j]):
                    # Pure distance-based cost - no safety penalties
                    dist = np.linalg.norm(nodes[i] - nodes[j])
                    graph[i].append((j, dist))
                    graph[j].append((i, dist))
        
        # Check connectivity
        start_connections = len(graph[0])
        goal_connections = len(graph[len(nodes) - 1])
        print(f"Start has {start_connections} connections, Goal has {goal_connections} connections")
        
        # Run A* from start (index 0) to goal (index len(nodes)-1)
        path_indices = self._astar(graph, nodes, 0, len(nodes) - 1)
        
        if path_indices is None:
            print("No path found with Visibility graph method!")
            return None
        
        # Convert to waypoints with orientations
        waypoints = self._indices_to_waypoints(nodes, path_indices)
        
        return waypoints
    
    def _astar(self, graph, nodes, start_idx, goal_idx):
        """
        A* algorithm for finding shortest path
        
        Args:
            graph: Dictionary of {node_idx: [(neighbor_idx, cost), ...]}
            nodes: Array of node positions
            start_idx: Start node index
            goal_idx: Goal node index
        
        Returns:
            List of node indices forming the path, or None if no path exists
        """
        # Priority queue: (f_score, node_idx)
        open_set = [(0, start_idx)]
        came_from = {}
        
        g_score = {i: float('inf') for i in graph.keys()}
        g_score[start_idx] = 0
        
        f_score = {i: float('inf') for i in graph.keys()}
        f_score[start_idx] = np.linalg.norm(nodes[start_idx] - nodes[goal_idx])
        
        while open_set:
            current_f, current = heapq.heappop(open_set)
            
            if current == goal_idx:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]
            
            for neighbor, cost in graph[current]:
                tentative_g = g_score[current] + cost
                
                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    h = np.linalg.norm(nodes[neighbor] - nodes[goal_idx])
                    f_score[neighbor] = tentative_g + h
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return None  # No path found
    
    def _indices_to_waypoints(self, nodes, path_indices):
        """
        Convert path indices to waypoints with orientations
        
        Args:
            nodes: Array of node positions
            path_indices: List of indices into nodes array
        
        Returns:
            List of [x, y, theta] waypoints
        """
        waypoints = []
        
        for i, idx in enumerate(path_indices):
            x, y = nodes[idx]
            
            # Calculate orientation to next waypoint
            if i < len(path_indices) - 1:
                next_idx = path_indices[i + 1]
                dx = nodes[next_idx][0] - x
                dy = nodes[next_idx][1] - y
                theta = np.arctan2(dy, dx)
            else:
                # Last waypoint: maintain previous orientation or use goal orientation
                if i > 0:
                    prev_idx = path_indices[i - 1]
                    dx = x - nodes[prev_idx][0]
                    dy = y - nodes[prev_idx][1]
                    theta = np.arctan2(dy, dx)
                else:
                    theta = 0.0
            
            waypoints.append([x, y, theta])
        
        return waypoints
    
    def visualize_path(self, start, goal, voronoi_path=None, visibility_path=None, 
                       save_path='path_comparison.png'):
        """
        Visualize both paths for comparison
        
        Args:
            start: Start position [x, y] or [x, y, theta]
            goal: Goal position [x, y] or [x, y, theta]
            voronoi_path: Waypoints from Voronoi planning
            visibility_path: Waypoints from Visibility planning
            save_path: Path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot Voronoi path
        self._plot_single_path(ax1, start, goal, voronoi_path, 
                               "Maximum Safety Path (Voronoi)", 'green')
        
        # Plot Visibility path
        self._plot_single_path(ax2, start, goal, visibility_path, 
                               "Minimum Distance Path (Visibility)", 'blue')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Path visualization saved to {save_path}")
        plt.close()
    
    def _plot_single_path(self, ax, start, goal, path, title, path_color):
        """Helper function to plot a single path"""
        # Plot workspace bounds
        ax.plot([self.x_min, self.x_max, self.x_max, self.x_min, self.x_min],
                [self.y_min, self.y_min, self.y_max, self.y_max, self.y_min],
                'k-', linewidth=2, label='Workspace')
        
        # Plot square obstacle
        square_corners = self.obstacle_corners + [self.obstacle_corners[0]]  # Close the square
        square_x = [corner[0] for corner in square_corners]
        square_y = [corner[1] for corner in square_corners]
        ax.fill(square_x, square_y, color='red', alpha=0.5, label='Obstacle')
        ax.plot(square_x, square_y, 'r-', linewidth=2)
        
        # Plot safety boundary around obstacle (expanded square)
        safety_offset = self.robot_radius
        safety_corners = [
            [self.obstacle_bounds['x_min'] - safety_offset, 
             self.obstacle_bounds['y_min'] - safety_offset],
            [self.obstacle_bounds['x_max'] + safety_offset, 
             self.obstacle_bounds['y_min'] - safety_offset],
            [self.obstacle_bounds['x_max'] + safety_offset, 
             self.obstacle_bounds['y_max'] + safety_offset],
            [self.obstacle_bounds['x_min'] - safety_offset, 
             self.obstacle_bounds['y_max'] + safety_offset],
            [self.obstacle_bounds['x_min'] - safety_offset, 
             self.obstacle_bounds['y_min'] - safety_offset],
        ]
        safety_x = [corner[0] for corner in safety_corners]
        safety_y = [corner[1] for corner in safety_corners]
        ax.plot(safety_x, safety_y, 'r--', linewidth=1, alpha=0.3, label='Safety margin')
        
        # If this is a Voronoi plot and we have Voronoi nodes, plot them
        if 'Voronoi' in title and hasattr(self, 'voronoi_nodes'):
            # Plot all Voronoi graph nodes (excluding start and goal)
            voronoi_only = self.voronoi_nodes[1:-1]  # Exclude first (start) and last (goal)
            if len(voronoi_only) > 0:
                ax.scatter(voronoi_only[:, 0], voronoi_only[:, 1], 
                          c='cyan', s=30, alpha=0.6, marker='x', 
                          label='Voronoi nodes', zorder=3)
        
        # Plot start and goal
        ax.plot(start[0], start[1], 'go', markersize=15, label='Start', zorder=10)
        ax.plot(goal[0], goal[1], 'ro', markersize=15, label='Goal', zorder=10)
        
        # Plot path
        if path is not None:
            path_array = np.array(path)
            ax.plot(path_array[:, 0], path_array[:, 1], 
                   color=path_color, linewidth=3, marker='o', 
                   markersize=6, label='Path', zorder=5)
            
            # Calculate path length
            path_length = 0
            for i in range(len(path) - 1):
                path_length += np.linalg.norm(
                    np.array(path[i][:2]) - np.array(path[i+1][:2])
                )
            ax.text(0.02, 0.98, f'Path Length: {path_length:.2f}m', 
                   transform=ax.transAxes, fontsize=12,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.set_xlim(-0.2, self.x_max + 0.2)
        ax.set_ylim(-0.2, self.y_max + 0.2)


def main():
    """Example usage of path planning algorithms"""
    
    # Initialize planner
    planner = PathPlanner(workspace_bounds=(0, 2.5, 0, 2.5), robot_radius=0.2)
    
    # Define start and goal
    start = [0.3, 0.3, 0.0]  # Bottom left
    goal = [2.2, 2.2, np.pi/2]  # Top right
    
    print("="*60)
    print("PATH PLANNING WITH SQUARE OBSTACLE")
    print("="*60)
    print(f"Workspace: {planner.x_max}m x {planner.y_max}m")
    print(f"Square obstacle: {planner.obstacle_size}m x {planner.obstacle_size}m")
    print(f"Obstacle center: ({planner.obstacle_center[0]:.2f}, {planner.obstacle_center[1]:.2f})")
    print(f"Robot radius: {planner.robot_radius}m")
    print(f"Start: ({start[0]:.2f}, {start[1]:.2f})")
    print(f"Goal: ({goal[0]:.2f}, {goal[1]:.2f})")
    print()
    
    # Plan using Voronoi (maximum safety)
    print("-"*60)
    print("VORONOI PATH PLANNING (Maximum Safety)")
    print("-"*60)
    voronoi_waypoints = planner.plan_voronoi_path(start, goal)
    if voronoi_waypoints:
        voronoi_length = sum(np.linalg.norm(np.array(voronoi_waypoints[i][:2]) - 
                                            np.array(voronoi_waypoints[i+1][:2])) 
                            for i in range(len(voronoi_waypoints)-1))
        print(f"✓ Path found with {len(voronoi_waypoints)} waypoints")
        print(f"Path length: {voronoi_length:.3f}m")
        print("Waypoints:")
        for i, wp in enumerate(voronoi_waypoints):
            print(f"  {i}: [{wp[0]:.3f}, {wp[1]:.3f}, {wp[2]:.3f}]")
    else:
        print("✗ No path found!")
    print()
    
    # Plan using Visibility graph (minimum distance)
    print("-"*60)
    print("VISIBILITY GRAPH PLANNING (Minimum Distance)")
    print("-"*60)
    visibility_waypoints = planner.plan_visibility_path(start, goal)
    if visibility_waypoints:
        visibility_length = sum(np.linalg.norm(np.array(visibility_waypoints[i][:2]) - 
                                               np.array(visibility_waypoints[i+1][:2])) 
                               for i in range(len(visibility_waypoints)-1))
        print(f"✓ Path found with {len(visibility_waypoints)} waypoints")
        print(f"Path length: {visibility_length:.3f}m")
        print("Waypoints:")
        for i, wp in enumerate(visibility_waypoints):
            print(f"  {i}: [{wp[0]:.3f}, {wp[1]:.3f}, {wp[2]:.3f}]")
    else:
        print("✗ No path found!")
    print()
    
    # Visualize both paths
    print("-"*60)
    print("Generating visualization...")
    planner.visualize_path(start, goal, voronoi_waypoints, visibility_waypoints)
    print("Done!")
    print("="*60)
    
    # Return waypoints for use in robot controller
    return voronoi_waypoints, visibility_waypoints


if __name__ == '__main__':
    voronoi_path, visibility_path = main()