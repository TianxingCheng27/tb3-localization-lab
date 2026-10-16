"""Publish cumulative wheel-angle measurements with an explicit error model.

The robot's physical motion is unchanged. Only /joint_states_noisy is produced;
ground truth and velocity commands are never read. Pure helpers can be imported
without ROS. A fixed seed reproduces only the same accepted input sequence.
"""

import math
from numbers import Integral
import sys

import numpy as np


LEFT = 'wheel_left_joint'
RIGHT = 'wheel_right_joint'


class EncoderModelError(ValueError):
    """A bad configuration or input makes the current experiment invalid."""


def finite_number(name, value):
    if isinstance(value, (bool, np.bool_)):
        raise EncoderModelError(f'{name} must be a finite number, not a boolean.')
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise EncoderModelError(f'{name} must be a finite number.') from exc
    if not math.isfinite(value):
        raise EncoderModelError(f'{name} must be finite.')
    return value


def integer_time(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
        raise EncoderModelError(f'{name} must be a nonnegative integer.')
    return int(value)


def extract_joint_sample(message):
    """Select the two wheel positions by name without changing the input message."""
    try:
        sec = integer_time('stamp.sec', message.header.stamp.sec)
        nanosec = integer_time('stamp.nanosec', message.header.stamp.nanosec)
        names = list(message.name)
        positions = message.position
    except (AttributeError, TypeError) as exc:
        raise EncoderModelError('Expected a JointState with a valid source header and wheel positions.') from exc
    if sec > 2_147_483_647 or nanosec >= 1_000_000_000:
        raise EncoderModelError('The source timestamp is outside the ROS Time field ranges.')
    if (not all(isinstance(name, str) for name in names)
            or names.count(LEFT) != 1 or names.count(RIGHT) != 1):
        raise EncoderModelError('Each JointState must contain both named wheel joints exactly once.')
    try:
        if len(positions) != len(names):
            raise EncoderModelError('JointState name and position arrays must have equal lengths.')
        left = finite_number('left wheel position', positions[names.index(LEFT)])
        right = finite_number('right wheel position', positions[names.index(RIGHT)])
    except (IndexError, TypeError) as exc:
        raise EncoderModelError('Invalid wheel-position array.') from exc
    return sec * 1_000_000_000 + nanosec, left, right


class EncoderMeasurementModel:
    """Convert wheel increments to corrupted travel, then accumulate measured angles."""

    def __init__(self, wheel_radius=.033, sigma_1m=.002, left_bias=-.005,
                 right_bias=.005, seed=0, max_source_gap=.1):
        self.wheel_radius = finite_number('wheel_radius', wheel_radius)
        self.sigma_1m = finite_number('sigma_1m', sigma_1m)
        left_bias = finite_number('left_bias', left_bias)
        right_bias = finite_number('right_bias', right_bias)
        max_source_gap = finite_number('max_source_gap', max_source_gap)
        self.seed = integer_time('seed', seed)
        if self.wheel_radius <= 0 or self.sigma_1m < 0 or min(left_bias, right_bias) <= -1:
            raise EncoderModelError('Require wheel_radius > 0, sigma_1m >= 0 and wheel biases > -1.')
        if max_source_gap <= 0:
            raise EncoderModelError('max_source_gap must be positive.')
        self.max_gap_ns = round(max_source_gap * 1_000_000_000)
        if self.max_gap_ns < 1:
            raise EncoderModelError('max_source_gap must cover at least one nanosecond.')
        self.bias = np.asarray([left_bias, right_bias], dtype=float)
        self.rng = np.random.default_rng(self.seed)
        self.previous_stamp_ns = None
        self.previous_angles = None
        self.measured_angles = None

    def update(self, stamp_ns, left_angle, right_angle):
        """Return two measured angles, or None for an equivalent duplicate timestamp.

        First input establishes both angle baselines without drawing noise.
        A fatal input raises EncoderModelError before changing the accepted state.
        """
        stamp_ns = integer_time('source timestamp', stamp_ns)
        angles = np.asarray([finite_number('left wheel position', left_angle),
                             finite_number('right wheel position', right_angle)])
        if self.previous_stamp_ns is None:
            self.previous_stamp_ns = stamp_ns
            self.previous_angles = angles.copy()
            self.measured_angles = angles.copy()
            return self.measured_angles.copy()
        elapsed_ns = stamp_ns - self.previous_stamp_ns
        if elapsed_ns < 0:
            raise EncoderModelError('Source time moved backwards; restart the complete experiment.')
        if elapsed_ns == 0:
            if np.allclose(angles, self.previous_angles, rtol=0., atol=1e-12):
                return None  # No state change, publication or random-number draw.
            raise EncoderModelError('Different wheel positions share one source timestamp.')
        if elapsed_ns > self.max_gap_ns:
            raise EncoderModelError(f'Source-message gap {elapsed_ns / 1e9:.6f} s exceeds '
                                    f'{self.max_gap_ns / 1e9:.6f} s.')
        with np.errstate(over='ignore', invalid='ignore'):
            delta_angle = angles - self.previous_angles
        if not np.isfinite(delta_angle).all() or np.any(np.abs(delta_angle) > math.pi):
            raise EncoderModelError('Wheel-angle jump exceeds pi radians; check reset or wrapped positions.')
        with np.errstate(over='ignore', invalid='ignore'):
            distance = self.wheel_radius * delta_angle
            measured_distance = (1.0 + self.bias) * distance
            if self.sigma_1m > 0:
                # Independent left/right draws, including zero-travel accepted intervals.
                # abs(distance)/(1 metre) is dimensionless; sigma_1m has units of metres.
                z = self.rng.standard_normal(2)
                measured_distance += self.sigma_1m * np.sqrt(np.abs(distance)) * z
            measured_angles = self.measured_angles + measured_distance / self.wheel_radius
        if not np.isfinite(measured_angles).all():
            raise EncoderModelError('Non-finite measured angle; this experiment must be restarted.')
        self.previous_stamp_ns = stamp_ns
        self.previous_angles = angles.copy()
        self.measured_angles = measured_angles
        return measured_angles.copy()


def create_node():
    """Keep ROS imports here so the measurement model can be tested independently."""
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from sensor_msgs.msg import JointState

    class EncoderModelNode(Node):
        def __init__(self):
            super().__init__('encoder_model')
            if self.get_parameter('use_sim_time').value is not True:
                raise EncoderModelError('Run this node with use_sim_time:=true.')
            defaults = dict(wheel_radius=.033, sigma_1m=.002, left_bias=-.005,
                            right_bias=.005, seed=0, max_source_gap=.1)
            for name, default in defaults.items():
                self.declare_parameter(name, default)
            parameters = {name: self.get_parameter(name).value for name in defaults}
            self.model = EncoderMeasurementModel(**parameters)
            self.failed = False
            # Sensor-style QoS matches the existing best-effort odometry subscriber.
            # Depth 50 buffers a short burst; it does not guarantee zero message loss.
            qos = QoSProfile(depth=50, reliability=ReliabilityPolicy.BEST_EFFORT,
                             durability=DurabilityPolicy.VOLATILE)
            self.publisher = self.create_publisher(JointState, '/joint_states_noisy', qos)
            self.subscription = self.create_subscription(
                JointState, '/joint_states', self.on_joint_state, qos)
            self.get_logger().info(
                f'Wheel measurement model: radius={parameters["wheel_radius"]} m, '
                f'sigma_1m={parameters["sigma_1m"]} m, left_bias={parameters["left_bias"]}, '
                f'right_bias={parameters["right_bias"]}, seed={parameters["seed"]}, '
                f'max_source_gap={parameters["max_source_gap"]} s. '
                'Fixed-seed reproducibility requires the same accepted input sequence.'
            )

        def on_joint_state(self, message):
            if self.failed:
                return
            try:
                stamp, left, right = extract_joint_sample(message)
                positions = self.model.update(stamp, left, right)
            except EncoderModelError as exc:
                self.failed = True
                self.get_logger().error(f'Encoder experiment stopped: {exc} '
                                        'Restart the full simulation and all experiment nodes.')
                self.destroy_subscription(self.subscription)
                return
            if positions is None:
                return
            output = JointState()
            output.header = message.header  # Preserve source time/frame without editing them.
            output.name = [LEFT, RIGHT]
            output.position = [float(value) for value in positions]
            # velocity and effort stay empty: this model defines only measured positions.
            self.publisher.publish(output)

    return EncoderModelNode()


def main(args=None):
    import rclpy
    from rclpy.executors import ExternalShutdownException

    node = None
    exit_code = 0
    try:
        rclpy.init(args=args)
        node = create_node()
        while rclpy.ok() and not node.failed:
            rclpy.spin_once(node, timeout_sec=.1)
        if node.failed:
            exit_code = 1
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception as exc:
        print(f'Encoder model failed: {exc}', file=sys.stderr)
        exit_code = 1
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())