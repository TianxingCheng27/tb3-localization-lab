"""Run a short velocity-command test using simulation time.

Linear speed is in metres per second; angular speed is in radians per second.
This controller does not read wheel odometry or ground truth.
"""

import time

import rclpy
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions


class MotionTest(Node):
    def __init__(self):
        super().__init__('motion_test')

        # Require simulation time for this experiment.
        if not self.get_parameter('use_sim_time').value:
            raise RuntimeError('Run with use_sim_time:=true')

        # Configure the linear and angular test velocities once.
        self.declare_parameter('linear_speed', 0.10)
        self.declare_parameter('angular_speed', 0.0)
        self.linear_speed = float(self.get_parameter('linear_speed').value)
        self.angular_speed = float(self.get_parameter('angular_speed').value)

        self.publisher = self.create_publisher(
            TwistStamped, '/cmd_vel', 10)
        self.start_ns = None
        self.done = False

        # Run the callback every 0.05 simulation seconds.
        self.timer = self.create_timer(0.05, self.tick)

    def send_speed(self, speed, turn_rate=0.0):
        # Create a fresh command with the current simulation timestamp.
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.twist.linear.x = speed
        msg.twist.angular.z = turn_rate
        self.publisher.publish(msg)

    def tick(self):
        now_ns = self.get_clock().now().nanoseconds

        if self.start_ns is None:
            # Wait for the command bridge before starting the timer.
            if self.publisher.get_subscription_count() == 0:
                return
            self.start_ns = now_ns
            self.get_logger().info('Motion started.')

        elapsed = (now_ns - self.start_ns) / 1_000_000_000.0

        # Move for 4 seconds, then repeatedly send zero velocity.
        if 0.0 <= elapsed < 4.0:
            self.send_speed(self.linear_speed, self.angular_speed)
        else:
            self.send_speed(0.0)

        # Finish after stopping, or if time jumps before the start.
        self.done = elapsed >= 4.5 or elapsed < 0.0


def main(args=None):
    # Keep ROS communication available while handling Ctrl+C.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = None

    try:
        node = MotionTest()
        node.get_logger().info('Waiting for simulation and command bridge.')
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            # Cancel motion callbacks before sending final stop commands.
            node.timer.cancel()
            for _ in range(5):
                if not rclpy.ok():
                    break
                node.send_speed(0.0)
                # This short wall-clock wait is only for final stop delivery.
                time.sleep(0.05)
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
