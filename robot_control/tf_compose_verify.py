#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from tf2_ros import TransformListener, Buffer
import tf2_geometry_msgs
from geometry_msgs.msg import TransformStamped
import math
import numpy as np


class TFComposer(Node):
    def __init__(self):
        super().__init__('tf_composer_node')
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Create a timer to check transforms periodically
        self.timer = self.create_timer(2.0, self.compose_and_verify)
        
    def compose_and_verify(self):
        """Compose transforms and compare with static transform"""
        
        try:
            # Get static transform: odom → tag_0
            static_odom_to_tag = self.tf_buffer.lookup_transform('odom', 'tag_0', rclpy.time.Time())
            
            # Get component transforms
            # When base_link == odom, we compose:
            # base_link (≈ odom) → camera_frame → tag_0
            base_to_camera = self.tf_buffer.lookup_transform('base_link', 'camera_frame', rclpy.time.Time())
            camera_to_tag = self.tf_buffer.lookup_transform('camera_frame', 'tag_0', rclpy.time.Time())
            
            # Compose: base_link → camera_frame → tag_0
            # This gives us base_link → tag_0, which should equal odom → tag_0 if base_link == odom
            composed_base_to_tag = self.compose_transforms(base_to_camera, camera_to_tag)
            
            self.get_logger().info("=" * 70)
            self.get_logger().info("STATIC TRANSFORM (odom → tag_0):")
            self.print_transform_detailed(static_odom_to_tag)
            
            self.get_logger().info("\n" + "-" * 70)
            self.get_logger().info("COMPOSED TRANSFORM (base_link → camera_frame → tag_0):")
            self.print_transform_detailed(composed_base_to_tag)
            
            self.get_logger().info("\n" + "-" * 70)
            self.get_logger().info("COMPONENT TRANSFORMS:")
            
            self.get_logger().info("\n1. base_link → camera_frame:")
            self.print_transform_detailed(base_to_camera)
            
            self.get_logger().info("\n2. camera_frame → tag_0:")
            self.print_transform_detailed(camera_to_tag)
            
            self.get_logger().info("\n" + "=" * 70)
            self.get_logger().info("VERIFICATION (Assuming base_link ≈ odom):")
            
            # Compare translations
            static_trans = static_odom_to_tag.transform.translation
            composed_trans = composed_base_to_tag.transform.translation
            
            trans_error = math.sqrt(
                (static_trans.x - composed_trans.x)**2 +
                (static_trans.y - composed_trans.y)**2 +
                (static_trans.z - composed_trans.z)**2
            )
            
            # Compare rotations (as quaternions)
            static_rot = static_odom_to_tag.transform.rotation
            composed_rot = composed_base_to_tag.transform.rotation
            
            # Quaternion distance
            rot_error = abs(1.0 - abs(
                static_rot.x * composed_rot.x +
                static_rot.y * composed_rot.y +
                static_rot.z * composed_rot.z +
                static_rot.w * composed_rot.w
            ))
            
            self.get_logger().info(f"\nTranslation Error: {trans_error:.6f} meters")
            self.get_logger().info(f"Rotation Error (quaternion distance): {rot_error:.6f}")
            
            if trans_error < 0.01 and rot_error < 0.01:
                self.get_logger().info("✓ TRANSFORMS MATCH! Your base_link→camera rotation is correct!")
            else:
                self.get_logger().warn("✗ TRANSFORMS DO NOT MATCH - rotation angles need adjustment")
                self.get_logger().info("\nDifference in Translation:")
                self.get_logger().info(f"  Δx = {static_trans.x - composed_trans.x:.6f}")
                self.get_logger().info(f"  Δy = {static_trans.y - composed_trans.y:.6f}")
                self.get_logger().info(f"  Δz = {static_trans.z - composed_trans.z:.6f}")
                
            self.get_logger().info("=" * 70 + "\n")
            
        except Exception as e:
            self.get_logger().warn(f"Transform lookup failed: {e}")
    
    def compose_transforms(self, t1: TransformStamped, t2: TransformStamped) -> TransformStamped:
        """
        Compose two transforms: T1 then T2
        Result is the transform from T1.header.frame_id to T2.child_frame_id
        
        Args:
            t1: Transform from frame A to frame B
            t2: Transform from frame B to frame C
            
        Returns:
            Transform from frame A to frame C
        """
        result = TransformStamped()
        result.header.frame_id = t1.header.frame_id
        result.child_frame_id = t2.child_frame_id
        result.header.stamp = t1.header.stamp
        
        # Compose translations
        # P_A = T1.translation + R1 * T2.translation
        trans1 = [t1.transform.translation.x, t1.transform.translation.y, t1.transform.translation.z]
        trans2 = [t2.transform.translation.x, t2.transform.translation.y, t2.transform.translation.z]
        
        rot1 = self.quat_to_matrix(
            t1.transform.rotation.x, t1.transform.rotation.y,
            t1.transform.rotation.z, t1.transform.rotation.w
        )
        
        trans2_rotated = rot1 @ np.array(trans2)
        composed_trans = np.array(trans1) + trans2_rotated
        
        result.transform.translation.x = float(composed_trans[0])
        result.transform.translation.y = float(composed_trans[1])
        result.transform.translation.z = float(composed_trans[2])
        
        # Compose rotations: quat_result = quat1 * quat2
        q1 = [t1.transform.rotation.x, t1.transform.rotation.y,
              t1.transform.rotation.z, t1.transform.rotation.w]
        q2 = [t2.transform.rotation.x, t2.transform.rotation.y,
              t2.transform.rotation.z, t2.transform.rotation.w]
        
        q_result = self.quat_multiply(q1, q2)
        result.transform.rotation.x = q_result[0]
        result.transform.rotation.y = q_result[1]
        result.transform.rotation.z = q_result[2]
        result.transform.rotation.w = q_result[3]
        
        return result
    
    def quat_to_matrix(self, qx, qy, qz, qw):
        """Convert quaternion to 3x3 rotation matrix"""
        matrix = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
        ])
        return matrix
    
    def quat_multiply(self, q1, q2):
        """Multiply two quaternions: q1 * q2
        q = [x, y, z, w]
        """
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        
        return [x, y, z, w]
    
    def print_transform_detailed(self, transform: TransformStamped):
        """Pretty print a transform with all details"""
        t = transform.transform.translation
        r = transform.transform.rotation
        
        self.get_logger().info(f"  {transform.header.frame_id} → {transform.child_frame_id}")
        self.get_logger().info(f"  Position: x={t.x:.6f}, y={t.y:.6f}, z={t.z:.6f}")
        self.get_logger().info(f"  Quaternion: x={r.x:.6f}, y={r.y:.6f}, z={r.z:.6f}, w={r.w:.6f}")
        
        # Convert quaternion to Euler angles
        roll, pitch, yaw = self.quaternion_to_euler(r.x, r.y, r.z, r.w)
        self.get_logger().info(f"  Euler: roll={math.degrees(roll):7.2f}°, "
                             f"pitch={math.degrees(pitch):7.2f}°, yaw={math.degrees(yaw):7.2f}°")
    
    def quaternion_to_euler(self, qx, qy, qz, qw):
        """Convert quaternion to Euler angles (roll, pitch, yaw)"""
        # Roll (x-axis rotation)
        sin_roll = 2.0 * (qw * qx + qy * qz)
        cos_roll = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = math.atan2(sin_roll, cos_roll)
        
        # Pitch (y-axis rotation)
        sin_pitch = 2.0 * (qw * qy - qz * qx)
        sin_pitch = max(-1.0, min(1.0, sin_pitch))
        pitch = math.asin(sin_pitch)
        
        # Yaw (z-axis rotation)
        sin_yaw = 2.0 * (qw * qz + qx * qy)
        cos_yaw = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(sin_yaw, cos_yaw)
        
        return roll, pitch, yaw


def main():
    rclpy.init()
    node = TFComposer()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('TF Composer stopped')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()