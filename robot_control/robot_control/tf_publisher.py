from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster, Buffer, TransformListener
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
import rclpy
from math import sin, cos
import json

# From odom to camera_frame + camera_frame to tag frame should roughly equals the static published tf from odom to tag frame
# for odom, points forward is x, left is y, up is z
# for camera, points forward is z, left or right is x, vertical up or down is y
# Not sure if the current static transform from base_link to camera_frame is correct
# AprilTag publishes the tf from the camera frame to the tag or vice versa.

# We need to know the odom -> baselink -> camera_frame -> tag_0
# AprilTag gives us camera_frame -> tag_0 or vice versa
# Please write static transform baselink -> camera_frame
# Please write static transform odom -> tag_0 as we already know the tag_0 positions
# By knowing those we can get odom -> baselink

from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster, Buffer, TransformListener
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
import rclpy
from math import sin, cos
import json


class TFBroadcaster(Node):
    def __init__(self):
        super().__init__('tf_broadcaster_node')
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)

        # Frame names
        self.odom_frame = 'odom'
        self.base_frame = 'base_link'
        self.camera_frame = 'camera_frame'

        self.apriltag_map = self.load_apriltag_map()

        # Broadcast static transform from base_link to camera_frame
        self.broadcast_static_camera_transform()

        # Broadcast static transforms from odom frame to AprilTags
        self.broadcast_apriltag_odom_transforms()

    def load_apriltag_map(self):
        """
        Load AprilTag map from file or use default map.
        Returns dict: {tag_id: (x, y, yaw)}
        """
        try:
            with open('/home/ubuntu/ros2_ws/rubikpi_ros2/robot_control/apriltag_map.json', 'r') as f:
                data = json.load(f)
                apriltag_map = {}
                for tag in data['tags']:
                    apriltag_map[tag['id']] = (tag['x'], tag['y'], tag['yaw'])
                self.get_logger().info(f'Loaded AprilTag map with {len(apriltag_map)} tags')
                return apriltag_map
        except Exception as e:
            self.get_logger().warn(f'Could not load apriltag_map.json: {e}')
            exit(1)

    def broadcast_static_camera_transform(self):
        """
        Broadcast static transform from base_link to camera_frame.

        Verified camera frame definition (from AprilTag measurements):
        - AprilTag in front of camera: z > 0 (forward)
        - AprilTag to the right of camera: x > 0 (right)
        - AprilTag below camera: y < 0 (down)
        
        Therefore:
        - Camera frame: x=right, y=down, z=forward
        - Robot/Base frame: x=forward, y=left, z=up

        Transformation equations:
        - X_robot = Z_camera  (camera forward → robot forward)
        - Y_robot = -X_camera (camera right → robot left)
        - Z_robot = -Y_camera (camera down → robot up)

        This is achieved with a -90° rotation around the Y-axis (pitch).
        """
        static_transform = TransformStamped()
        static_transform.header.stamp = self.get_clock().now().to_msg()
        static_transform.header.frame_id = self.base_frame
        static_transform.child_frame_id = self.camera_frame

        # Set translation (camera mounted at origin relative to base_link)
        # Adjust these if camera is physically offset from robot center
        static_transform.transform.translation.x = 0.1
        static_transform.transform.translation.y = 0.0
        static_transform.transform.translation.z = 0.05

        # Set rotation: -90° pitch (rotation around Y-axis)
        # This converts camera frame (x=right, y=down, z=forward)
        # to robot frame (x=forward, y=left, z=up)
        qx, qy, qz, qw = self.euler_to_quaternion(
            0.0,       # roll: 0°
            -1.5708,   # pitch: -90° in radians (Y-axis rotation)
            0.0        # yaw: 0°
        )

        static_transform.transform.rotation.x = qx
        static_transform.transform.rotation.y = qy
        static_transform.transform.rotation.z = qz
        static_transform.transform.rotation.w = qw

        # Publish the static transform
        self.static_tf_broadcaster.sendTransform([static_transform])
        self.get_logger().info('Static transform from base_link to camera_frame published')
        self.get_logger().info(f'  Camera frame: x=right, y=down, z=forward')
        self.get_logger().info(f'  Rotation: roll=0°, pitch=-90°, yaw=0°')

    def broadcast_apriltag_odom_transforms(self):
        """
        Broadcast static transforms from odom frame to each AprilTag.

        This creates the TF chain:
        odom -> tag_X -> camera_frame -> base_link

        When AprilTag is detected:
        odom -> base_link can be computed as:
        odom -> tag_0 -> camera_frame -> base_link

        where:
        - odom → tag_0 is published here (static, known positions)
        - camera_frame → base_link is inverse of base_link → camera_frame
        - tag_0 → camera_frame comes from AprilTag detector
        """
        static_transforms = []

        for tag_id, (x, y, yaw) in self.apriltag_map.items():
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = self.odom_frame
            t.child_frame_id = f'tag_{tag_id}'

            # Position of the tag in odom frame
            t.transform.translation.x = float(x)
            t.transform.translation.y = float(y)
            t.transform.translation.z = 0.0

            # Orientation of the tag in odom frame
            qx, qy, qz, qw = self.euler_to_quaternion(0.0, 0.0, float(yaw))
            t.transform.rotation.x = qx
            t.transform.rotation.y = qy
            t.transform.rotation.z = qz
            t.transform.rotation.w = qw

            static_transforms.append(t)

        if len(static_transforms) > 0:
            self.static_tf_broadcaster.sendTransform(static_transforms)
            self.get_logger().info(f'Published {len(static_transforms)} static transforms: odom → tag_X')

    def euler_to_quaternion(self, roll, pitch, yaw):
        """
        Convert Euler angles (roll, pitch, yaw) in radians to quaternion (x, y, z, w).
        
        Args:
            roll: Rotation around x-axis (radians)
            pitch: Rotation around y-axis (radians)
            yaw: Rotation around z-axis (radians)
            
        Returns:
            Tuple of (qx, qy, qz, qw)
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


def main():
    rclpy.init()
    node = TFBroadcaster()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('TF Broadcaster stopped by keyboard interrupt')
    finally:
        node.destroy_node()
        rclpy.shutdown()   


if __name__ == '__main__':
    main()