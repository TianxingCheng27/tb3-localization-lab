"""Inspect a recorded command sequence using message header timestamps."""

import argparse
from dataclasses import dataclass, field
import math
from pathlib import Path
import sys


COMMAND_TOPIC = '/cmd_vel'
COMMAND_TYPE = 'geometry_msgs/msg/TwistStamped'
SPEED_TOLERANCE = 1e-9
EXPECTED_ACTIONS = ('STRAIGHT', 'LEFT', 'STRAIGHT', 'RIGHT', 'STRAIGHT')


def same_command(first, second):
    """Allow insignificant floating-point differences."""
    return all(math.isclose(a, b, rel_tol=0.0, abs_tol=SPEED_TOLERANCE)
               for a, b in zip(first, second))


def action_name(linear, angular):
    linear_zero = abs(linear) <= SPEED_TOLERANCE
    angular_zero = abs(angular) <= SPEED_TOLERANCE

    if linear_zero and angular_zero:
        return 'STOP'
    if angular_zero:
        return 'STRAIGHT' if linear > 0.0 else 'REVERSE'
    if linear_zero:
        return 'LEFT' if angular > 0.0 else 'RIGHT'
    return 'CURVE'


@dataclass
class Phase:
    linear: float
    angular: float
    start_ns: int
    end_ns: int
    first_bag_ns: int
    last_bag_ns: int
    count: int = 1

    @property
    def action(self):
        return action_name(self.linear, self.angular)


@dataclass
class CommandSummary:
    phases: list = field(default_factory=list)
    inspected: int = 0
    accepted: int = 0
    duplicates: int = 0
    duplicate_stops: int = 0
    duplicate_transitions: int = 0
    backwards: int = 0
    first_header_ns: int | None = None
    last_header_ns: int | None = None
    first_bag_ns: int | None = None
    last_bag_ns: int | None = None
    rollback: tuple | None = None

    def add(self, header_ns, bag_ns, linear, angular):
        """Consume one command; stop if its timestamp moves backwards."""
        if self.rollback is not None:
            raise RuntimeError('Cannot continue after a clock rollback.')

        if not all(math.isfinite(value) for value in (linear, angular)):
            raise ValueError('A command contains a non-finite speed.')

        self.inspected += 1
        command = (linear, angular)
        previous = self.phases[-1] if self.phases else None

        if self.last_header_ns is not None:
            if header_ns < self.last_header_ns:
                self.backwards += 1
                self.rollback = (self.last_header_ns, header_ns, bag_ns)
                return False

            if header_ns == self.last_header_ns:
                self.duplicates += 1

                if not same_command(
                        command, (previous.linear, previous.angular)):
                    self.duplicate_transitions += 1
                elif action_name(*command) == 'STOP':
                    self.duplicate_stops += 1

        if self.first_header_ns is None:
            self.first_header_ns = header_ns
            self.first_bag_ns = bag_ns

        self.last_header_ns = header_ns
        self.last_bag_ns = bag_ns
        self.accepted += 1

        if previous and same_command(
                command, (previous.linear, previous.angular)):
            previous.end_ns = header_ns
            previous.last_bag_ns = bag_ns
            previous.count += 1
        else:
            # A new command marks the end of the preceding phase.
            if previous:
                previous.end_ns = header_ns

            self.phases.append(Phase(
                linear, angular, header_ns, header_ns, bag_ns, bag_ns,
            ))

        return True

    @property
    def nonzero_actions(self):
        # Pauses are part of the planned trajectory.
        return tuple(
            phase.action for phase in self.phases
            if phase.action != 'STOP'
        )

    @property
    def pattern_matches(self):
        actions = self.nonzero_actions
        length = len(EXPECTED_ACTIONS)

        return [
            index for index in range(len(actions) - length + 1)
            if actions[index:index + length] == EXPECTED_ACTIONS
        ]

    @property
    def matches_one_sequence(self):
        return (
            self.rollback is None
            and self.duplicate_transitions == 0
            and self.nonzero_actions == EXPECTED_ACTIONS
        )


def inspect_commands(bag_dir):
    """Read commands; keep bag receive times separate from header times."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()

    try:
        reader.open(
            rosbag2_py.StorageOptions(
                uri=str(bag_dir), storage_id='mcap',
            ),
            rosbag2_py.ConverterOptions('cdr', 'cdr'),
        )

        topics = {
            item.name: item
            for item in reader.get_all_topics_and_types()
        }

        if COMMAND_TOPIC not in topics:
            raise ValueError(f'The bag does not contain {COMMAND_TOPIC}.')

        metadata = topics[COMMAND_TOPIC]

        if metadata.type != COMMAND_TYPE:
            raise ValueError(
                f'Expected {COMMAND_TYPE}, found {metadata.type}.'
            )

        if metadata.serialization_format != 'cdr':
            raise ValueError('This tool expects CDR serialization.')

        message_class = get_message(metadata.type)

        reader.set_filter(
            rosbag2_py.StorageFilter(topics=[COMMAND_TOPIC])
        )

        summary = CommandSummary()

        while reader.has_next():
            topic, data, bag_ns = reader.read_next()

            if topic != COMMAND_TOPIC:
                raise RuntimeError('Unexpected filtered topic.')

            message = deserialize_message(data, message_class)

            header_ns = (
                message.header.stamp.sec * 1_000_000_000
                + message.header.stamp.nanosec
            )

            if not summary.add(
                    header_ns, int(bag_ns),
                    float(message.twist.linear.x),
                    float(message.twist.angular.z)):
                break

        return summary

    finally:
        reader.close()


def seconds(timestamp_ns):
    return f'{timestamp_ns / 1_000_000_000:.9f}'


def print_summary(summary, bag_dir):
    print(f'Bag: {bag_dir}')
    print(f'Commands inspected: {summary.inspected}')
    print(f'Commands accepted: {summary.accepted}')
    print(f'Duplicate header timestamps: {summary.duplicates}')
    print(f'Repeated STOP timestamps: {summary.duplicate_stops}')
    print(f'Different commands at one timestamp: '
          f'{summary.duplicate_transitions}')
    print(f'Backwards timestamps: {summary.backwards}')

    if not summary.accepted:
        print('No command messages were available.')
        return 2

    print(f'First command time: {seconds(summary.first_header_ns)} s')
    print(f'Last command time: {seconds(summary.last_header_ns)} s')
    print()
    print('Phase boundaries use message header timestamps.')

    for index, phase in enumerate(summary.phases, 1):
        duration = (phase.end_ns - phase.start_ns) / 1_000_000_000

        print(
            f'{index:2d}. {phase.action:<9} '
            f'v={phase.linear: .3f} m/s, '
            f'w={phase.angular: .3f} rad/s, '
            f'start={seconds(phase.start_ns)} s, '
            f'end={seconds(phase.end_ns)} s, '
            f'duration={duration:.6f} s, '
            f'count={phase.count}'
        )

    print()
    print('Nonzero action order: '
          + (' -> '.join(summary.nonzero_actions) or '(none)'))

    if summary.duplicate_stops:
        print('Repeated STOP timestamps can occur during final cleanup.')

    if summary.rollback is not None:
        before, after, bag_ns = summary.rollback
        print(
            'ANALYSIS STOPPED: header time moved backwards from '
            f'{seconds(before)} to {seconds(after)} s; '
            f'bag receive time: {seconds(bag_ns)} s.'
        )
        return 2

    if summary.duplicate_transitions:
        print('Different commands share a timestamp; inspect this run.')
        return 2

    print(
        'Exact planned order '
        'STRAIGHT -> LEFT -> STRAIGHT -> RIGHT -> STRAIGHT: '
        + ('YES' if summary.matches_one_sequence else 'NO')
    )

    if len(summary.pattern_matches) > 1:
        print('Multiple planned sequences found; select one run explicitly.')
    elif not summary.matches_one_sequence:
        print('The sequence is incomplete or does not match one planned run.')

    print('The final phase ends at its last recorded command.')
    print('Inspection complete; the original bag is unchanged.')

    return 0 if summary.matches_one_sequence else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag_dir', type=Path)
    args = parser.parse_args(argv)
    bag_dir = args.bag_dir.expanduser().resolve()

    if not bag_dir.is_dir() or not (bag_dir / 'metadata.yaml').is_file():
        parser.error('Provide a finalized bag directory with metadata.yaml.')

    try:
        print('Reading recorded /cmd_vel messages...', flush=True)
        summary = inspect_commands(bag_dir)
        return print_summary(summary, bag_dir)

    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
