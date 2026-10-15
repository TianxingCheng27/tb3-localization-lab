"""Drive straight, left, straight, right, straight using simulation time.

This is an open-loop command sequence. It never reads odometry, ground truth,
or wheel data. Distances and angles describe the nominal commanded path.
The actual path must be measured independently in Gazebo.
"""

import math
import time

import rclpy
from geometry_msgs.msg import TwistStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions


def build_segments(segment_length, linear_speed, angular_speed):
    """Return (name, duration_seconds, linear_m_s, angular_rad_s) tuples."""
    straight_time = segment_length / linear_speed
    turn_time = (math.pi / 2.0) / angular_speed

    return (
        ('Initial stop', 2.0, 0.0, 0.0),
        ('Straight 1 of 3', straight_time, linear_speed, 0.0),
        ('Pause after straight 1', 0.5, 0.0, 0.0),
        ('Left turn, 90 degrees', turn_time, 0.0, angular_speed),
        ('Pause after left turn', 0.5, 0.0, 0.0),
        ('Straight 2 of 3', straight_time, linear_speed, 0.0),
        ('Pause after straight 2', 0.5, 0.0, 0.0),
        ('Right turn, 90 degrees', turn_time, 0.0, -angular_speed),
        ('Pause after right turn', 0.5, 0.0, 0.0),
        ('Straight 3 of 3', straight_time, linear_speed, 0.0),
        ('Final stop', 2.0, 0.0, 0.0),
    )


class TrajectoryDriver(Node):
    def __init__(self):
        super().__init__('trajectory_driver')

        # Use Gazebo simulation time for every movement and pause.
        if self.get_parameter('use_sim_time').value is not True:
            raise RuntimeError('Run with --ros-args -p use_sim_time:=true')

        self.declare_parameter('segment_length', 2.0)
        self.declare_parameter('linear_speed', 0.15)
        self.declare_parameter('angular_speed', 0.4)

        segment_length = float(self.get_parameter('segment_length').value)
        linear_speed = float(self.get_parameter('linear_speed').value)
        angular_speed = float(self.get_parameter('angular_speed').value)

        if not all(math.isfinite(value) and value > 0.0 for value in
                   (segment_length, linear_speed, angular_speed)):
            raise ValueError(
                'Length and speed parameters must be finite and positive.'
            )

        self.segments = build_segments(
            segment_length, linear_speed, angular_speed,
        )
        self.total_duration = sum(segment[1] for segment in self.segments)

        self.publisher = self.create_publisher(
            TwistStamped, '/cmd_vel', 10,
        )

        self.start_ns = None
        self.previous_now_ns = None
        self.current_segment = None
        self.done = False

        # Run the callback every 0.05 simulation seconds.
        self.timer = self.create_timer(0.05, self.tick)

        self.get_logger().info(
            'Waiting for simulation time and a /cmd_vel subscriber. '
            f'Nominal duration: {self.total_duration:.6f} simulation seconds.'
        )

    def send_speed(self, linear_speed, angular_speed=0.0):
        """Publish a fresh velocity command."""
        message = TwistStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = 'base_footprint'
        message.twist.linear.x = linear_speed
        message.twist.angular.z = angular_speed
        self.publisher.publish(message)

    def tick(self):
        now_ns = self.get_clock().now().nanoseconds

        # Stop if simulation time moves backwards.
        if self.previous_now_ns is not None and now_ns < self.previous_now_ns:
            self.send_speed(0.0)
            self.done = True
            self.timer.cancel()
            self.get_logger().error(
                'Simulation time moved backwards. Stopped; restart this node '
                'after resetting the simulation.'
            )
            return

        self.previous_now_ns = now_ns

        if self.start_ns is None:
            # Waiting does not consume any part of the movement schedule.
            if now_ns <= 0 or self.publisher.get_subscription_count() == 0:
                return

            self.start_ns = now_ns
            self.get_logger().info(
                'Command subscriber ready; sequence started.'
            )

        elapsed = (now_ns - self.start_ns) / 1_000_000_000.0

        # Measure every segment boundary from the same starting time.
        segment_end = 0.0

        for index, (name, duration, linear_speed, angular_speed) in enumerate(
                self.segments):
            segment_end += duration

            if elapsed < segment_end:
                if index != self.current_segment:
                    self.current_segment = index
                    self.get_logger().info(
                        f'[sim +{elapsed:.3f} s] {name}: '
                        f'v={linear_speed:.3f} m/s, '
                        f'w={angular_speed:.3f} rad/s'
                    )

                self.send_speed(linear_speed, angular_speed)
                return

        self.send_speed(0.0)
        self.done = True
        self.timer.cancel()
        self.get_logger().info(
            f'[sim +{elapsed:.3f} s] Sequence complete; zero velocity sent.'
        )


def main(args=None):
    # Keep ROS active on Ctrl+C so final stop commands can be sent.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = None

    try:
        node = TrajectoryDriver()

        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)

    except (KeyboardInterrupt, ExternalShutdownException):
        pass

    finally:
        if node is not None:
            # Cancel movement callbacks before sending final stop commands.
            node.timer.cancel()

            for _ in range(5):
                if not rclpy.ok():
                    break

                node.send_speed(0.0)
                # This wall-clock delay only helps deliver cleanup commands.
                time.sleep(0.05)

            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
