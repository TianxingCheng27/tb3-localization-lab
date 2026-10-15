"""Estimate pose from wheel angles; start at the agreed pose (0, 0, 0)."""
import math

import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState

from tb3_localization_lab.kinematics import integrate_pose, wheel_motion


class WheelOdometry(Node):
    def __init__(self):
        super().__init__('wheel_odometry')
        self.declare_parameter('wheel_radius', 0.033)
        self.declare_parameter('wheel_separation', 0.160)
        self.radius = float(self.get_parameter('wheel_radius').value)
        self.separation = float(self.get_parameter('wheel_separation').value)
        if not all(math.isfinite(v) and v > 0.0
                   for v in (self.radius, self.separation)):
            raise ValueError('Wheel geometry must be finite and positive.')
        self.x = self.y = self.yaw = 0.0
        self.previous = None
        # Publish an independent estimate without broadcasting TF.
        self.publisher = self.create_publisher(Odometry, '/odom_est', 10)
        self.subscription = self.create_subscription(
            JointState, '/joint_states', self.on_joint_state,
            qos_profile_sensor_data,
        )
        self.get_logger().info('Waiting for wheel angles; initial pose (0, 0, 0).')
    def on_joint_state(self, msg):
        # Find wheels by name, rather than assuming an array order.
        try:
            left = float(msg.position[msg.name.index('wheel_left_joint')])
            right = float(msg.position[msg.name.index('wheel_right_joint')])
        except (ValueError, IndexError, TypeError):
            self.get_logger().warning('Ignoring a message missing wheel positions.')
            return

        if not (math.isfinite(left) and math.isfinite(right)):
            return

        stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec

        # The first sample establishes the wheel reference.
        if self.previous is None or stamp_ns < self.previous[0]:
            if self.previous is not None:
                self.get_logger().warning('Time moved backwards; resetting estimate.')
            self.x = self.y = self.yaw = 0.0
            self.previous = (stamp_ns, left, right)
            self.publish_odometry(msg.header.stamp, 0.0, 0.0)
            return

        if stamp_ns == self.previous[0]:
            return

        previous_ns, previous_left, previous_right = self.previous
        dt = (stamp_ns - previous_ns) / 1_000_000_000.0

        # Wheel angles are expected to accumulate continuously.
        distance, turn = wheel_motion(
            left - previous_left, right - previous_right,
            self.radius, self.separation,
        )

        if not (math.isfinite(distance) and math.isfinite(turn)):
            return

        # Accumulate from the previous estimate, not from the origin.
        self.x, self.y, self.yaw = integrate_pose(
            self.x, self.y, self.yaw, distance, turn,
        )

        self.previous = (stamp_ns, left, right)
        self.publish_odometry(msg.header.stamp, distance / dt, turn / dt)
    def publish_odometry(self, stamp, linear_speed, angular_speed):
        odom = Odometry()
    # Preserve the measurement timestamp.
        odom.header.stamp = stamp
        odom.header.frame_id = 'odom_est'
        odom.child_frame_id = 'base_footprint'


# Store the estimated planar position.
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y


     # Convert planar heading into a quaternion.
        odom.pose.pose.orientation.z = math.sin(self.yaw / 2.0)
        odom.pose.pose.orientation.w = math.cos(self.yaw / 2.0)

    # Store forward speed and turning speed.
        odom.twist.twist.linear.x = linear_speed
        odom.twist.twist.angular.z = angular_speed
  
    
        # Covariance is not estimated in this baseline.
        self.publisher.publish(odom)


def main(args=None):
    # Initialize ROS communication.
    rclpy.init(args=args)
    node = None

    try:
        # Create the odometry node and process incoming messages.
        node = WheelOdometry()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # Release resources when the program stops.
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
