"""Evaluate one baseline run offline using source-message simulation timestamps.

Keep inspect_bag.py beside this file. The original bag is opened read-only;
this script creates CSV, JSON and plots in a separate analysis directory.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

import inspect_bag


GT = '/ground_truth/pose'
ODOM = '/odom_est'
TOPIC_TYPES = {GT: 'geometry_msgs/msg/PoseStamped', ODOM: 'nav_msgs/msg/Odometry'}
RAW_COLUMNS = ('header_time_ns', 'bag_time_ns', 'frame_id', 'child_frame_id',
               'x_m', 'y_m', 'z_m', 'qx', 'qy', 'qz', 'qw')
GUARD_NS = 100_000_000


def validate_schedule(summary, segment_length, tolerance):
    """Check completion against the recorded experiment's intended schedule."""
    expected = ('STOP', 'STRAIGHT', 'STOP', 'LEFT', 'STOP', 'STRAIGHT',
                'STOP', 'RIGHT', 'STOP', 'STRAIGHT', 'STOP')
    if not summary.matches_one_sequence or tuple(p.action for p in summary.phases) != expected:
        raise ValueError('Expected one complete five-action sequence with its STOP phases.')
    for index, phase in enumerate(summary.phases):
        duration = (phase.end_ns - phase.start_ns) / 1e9
        if index == 0:
            # Discovery may omit initial zero commands; require a stable reference window.
            if duration < 1.0:
                raise ValueError('At least one second of initial STOP commands is required.')
            continue
        if phase.action == 'STRAIGHT':
            target = segment_length / abs(phase.linear)
        elif phase.action in ('LEFT', 'RIGHT'):
            target = (math.pi / 2.0) / abs(phase.angular)
        else:
            target = 2.0 if index == len(summary.phases) - 1 else 0.5
        if abs(duration - target) > tolerance + 1e-9:
            raise ValueError(f'Phase {index + 1} duration {duration:.6f} s differs from '
                             f'its intended {target:.6f} s by more than {tolerance:.3f} s.')


def wrap(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def quaternion_yaw(x, y, z, w):
    """Use the complete normalized quaternion, including roll and pitch terms."""
    norm = math.hypot(x, y, z, w)
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError('A pose has an invalid quaternion.')
    x, y, z, w = (value / norm for value in (x, y, z, w))
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def read_poses(bag_dir, start_ns, end_ns):
    """Filter topics in storage; deserialize streams without retaining idle data."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    rows = {GT: [], ODOM: []}
    reader = rosbag2_py.SequentialReader()
    try:
        reader.open(rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id='mcap'),
                    rosbag2_py.ConverterOptions('cdr', 'cdr'))
        metadata = {item.name: item for item in reader.get_all_topics_and_types()}
        for topic, expected in TOPIC_TYPES.items():
            if topic not in metadata or metadata[topic].type != expected:
                raise ValueError(f'{topic} must have type {expected}.')
            if metadata[topic].serialization_format != 'cdr':
                raise ValueError(f'{topic} is not CDR serialized.')
        classes = {topic: get_message(kind) for topic, kind in TOPIC_TYPES.items()}
        reader.set_filter(rosbag2_py.StorageFilter(topics=list(TOPIC_TYPES)))
        while reader.has_next():
            topic, data, bag_ns = reader.read_next()
            message = deserialize_message(data, classes[topic])
            stamp_ns = (message.header.stamp.sec * 1_000_000_000
                        + message.header.stamp.nanosec)
            if start_ns <= stamp_ns <= end_ns:
                pose = message.pose if topic == GT else message.pose.pose
                p, q = pose.position, pose.orientation
                rows[topic].append((stamp_ns, int(bag_ns), message.header.frame_id,
                                    getattr(message, 'child_frame_id', ''),
                                    p.x, p.y, p.z, q.x, q.y, q.z, q.w))
        return rows
    finally:
        reader.close()


def prepare_series(rows, topic):
    """Validate source order; preserve all raw rows but deduplicate interpolation."""
    if len(rows) < 2:
        raise ValueError(f'{topic}: fewer than two samples in the guarded window.')
    stamps = np.asarray([row[0] for row in rows], dtype=np.int64)
    delta = np.diff(stamps)
    backwards, duplicates = int(np.sum(delta < 0)), int(np.sum(delta == 0))
    if backwards:
        raise ValueError(f'{topic}: {backwards} backwards header timestamps; analysis stopped.')
    frames, children = {row[2] for row in rows}, {row[3] for row in rows}
    if len(frames) != 1 or not next(iter(frames)) or len(children) != 1:
        raise ValueError(f'{topic}: empty or changing frame IDs; inspect the source data.')
    if topic == ODOM and not next(iter(children)):
        raise ValueError(f'{topic}: child_frame_id is empty.')
    if not np.isfinite(np.asarray([row[4:] for row in rows], dtype=float)).all():
        raise ValueError(f'{topic}: non-finite pose values.')
    poses = np.asarray([(row[4], row[5], quaternion_yaw(*row[7:11])) for row in rows])
    for index in np.flatnonzero(delta == 0):
        if (np.linalg.norm(poses[index + 1, :2] - poses[index, :2]) > 1e-8
                or abs(wrap(poses[index + 1, 2] - poses[index, 2])) > 1e-8):
            raise ValueError(f'{topic}: conflicting planar poses share a header timestamp.')
    keep = np.r_[delta != 0, True]  # Keep the last equivalent sample at each timestamp.
    stamps, poses = stamps[keep], poses[keep].copy()
    if len(stamps) < 2:
        raise ValueError(f'{topic}: fewer than two distinct header timestamps.')
    poses[:, 2] = np.unwrap(poses[:, 2])
    checks = dict(raw_samples=len(rows), unique_samples=len(stamps),
                  duplicate_header_timestamps=duplicates, backwards_header_timestamps=backwards,
                  frame_id=next(iter(frames)), child_frame_id=next(iter(children)),
                  max_header_gap_s=float(np.max(np.diff(stamps))) / 1e9,
                  first_header_ns=int(stamps[0]), last_header_ns=int(stamps[-1]))
    return stamps, poses, checks


def interpolate(stamps, poses, query_ns):
    """Interpolate unwrapped poses in relative seconds, explicitly forbidding extrapolation."""
    query_ns = np.atleast_1d(np.asarray(query_ns, dtype=np.int64))
    if not len(query_ns) or query_ns.min() < stamps[0] or query_ns.max() > stamps[-1]:
        raise ValueError('Required source-time coverage is missing; extrapolation is forbidden.')
    time = (stamps - stamps[0]).astype(float) / 1e9
    query = (query_ns - stamps[0]).astype(float) / 1e9
    return np.column_stack([np.interp(query, time, poses[:, axis]) for axis in range(3)])


def to_start_frame(poses, initial):
    """Apply one fixed inverse initial SE(2) transform; never fit the full path."""
    delta = np.asarray(poses, dtype=float) - initial
    c, s = math.cos(initial[2]), math.sin(initial[2])
    return np.column_stack((c * delta[:, 0] + s * delta[:, 1],
                            -s * delta[:, 0] + c * delta[:, 1], delta[:, 2]))


def evaluate(series, alignment_ns, first_motion_ns, endpoint_ns):
    gt_t, gt_pose, gt_checks = series[GT]
    od_t, od_pose, od_checks = series[ODOM]
    gt_initial = interpolate(gt_t, gt_pose, [alignment_ns])[0]
    od_initial = interpolate(od_t, od_pose, [alignment_ns])[0]
    query = gt_t[(gt_t >= first_motion_ns) & (gt_t <= endpoint_ns)]
    if len(query) < 2:
        raise ValueError('Fewer than two ground-truth evaluation samples.')
    gt = to_start_frame(interpolate(gt_t, gt_pose, query), gt_initial)
    odom = to_start_frame(interpolate(od_t, od_pose, query), od_initial)
    position_error = np.linalg.norm(odom[:, :2] - gt[:, :2], axis=1)
    heading_error = wrap(odom[:, 2] - gt[:, 2])
    gt_end = to_start_frame(interpolate(gt_t, gt_pose, [endpoint_ns]), gt_initial)[0]
    od_end = to_start_frame(interpolate(od_t, od_pose, [endpoint_ns]), od_initial)[0]
    initial_difference = od_initial - gt_initial
    initial_difference[2] = wrap(initial_difference[2])
    metrics = {
        'alignment': 'Each trajectory uses its own inverse initial SE(2) pose at t0; '
                     'relative-motion evaluation only. No full-trajectory fitting.',
        'alignment_time_ns': int(alignment_ns), 'evaluation_start_ns': int(first_motion_ns),
        'evaluation_end_ns': int(endpoint_ns), 'read_guard_s': GUARD_NS / 1e9,
        'evaluation_sampling': 'Unique GT header timestamps within the evaluation interval.',
        'evaluation_samples': len(query),
        'position_rmse_m': float(np.sqrt(np.mean(position_error ** 2))),
        'endpoint_position_error_m': float(np.linalg.norm(od_end[:2] - gt_end[:2])),
        'heading_rmse_deg': float(np.degrees(np.sqrt(np.mean(heading_error ** 2)))),
        'endpoint_heading_error_deg': float(np.degrees(wrap(od_end[2] - gt_end[2]))),
        'maximum_position_error_m': float(np.max(position_error)),
        'raw_initial_poses_x_m_y_m_yaw_rad': {GT: gt_initial.tolist(), ODOM: od_initial.tolist()},
        'raw_initial_odom_minus_gt': initial_difference.tolist(),
        'raw_initial_difference_note': 'Raw coordinate-value diagnostic; interpret only if the '
                                       'original frames are comparable. Not a calibrated transform.',
        'aligned_endpoint_x_m_y_m_yaw_rad': {GT: gt_end.tolist(), ODOM: od_end.tolist()},
        'reference_point_assumption': 'Assumes the Gazebo model origin and odometry base reference '
                                      'coincide in the plane; verify the model configuration.',
        'duplicate_policy': 'Preserve all raw rows; keep the last equivalent timestamp for '
                            'interpolation. Conflicting planar duplicates are rejected.',
        'topic_checks': {GT: gt_checks, ODOM: od_checks},
        'sampling_note': 'Gaps are observed header gaps; these checks do not prove zero message loss.',
    }
    return query, gt, odom, position_error, heading_error, metrics


def make_plot(output_dir, query, gt, odom, position_error, heading_error, metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11, 5.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.12, 1])
    trajectory = fig.add_subplot(grid[:, 0])
    position = fig.add_subplot(grid[0, 1])
    heading = fig.add_subplot(grid[1, 1], sharex=position)
    gt_end = np.asarray(metrics['aligned_endpoint_x_m_y_m_yaw_rad'][GT])
    od_end = np.asarray(metrics['aligned_endpoint_x_m_y_m_yaw_rad'][ODOM])
    trajectory.plot(np.r_[0., gt[:, 0], gt_end[0]], np.r_[0., gt[:, 1], gt_end[1]],
                    'k-', label='Ground truth')
    trajectory.plot(np.r_[0., odom[:, 0], od_end[0]], np.r_[0., odom[:, 1], od_end[1]],
                    color='tab:blue', linestyle='--', label='Wheel odometry')
    trajectory.scatter(0, 0, marker='o', color='tab:green', s=45, label='Aligned start')
    trajectory.scatter(gt_end[0], gt_end[1], marker='s', color='black', s=40, label='GT endpoint')
    trajectory.scatter(od_end[0], od_end[1], marker='x', color='tab:blue', s=55,
                       label='Odometry endpoint')
    trajectory.set(xlabel='X in initial frame (m)', ylabel='Y in initial frame (m)',
                   title='Trajectory comparison')
    trajectory.set_aspect('equal', adjustable='box')
    trajectory.legend(fontsize=8)
    time = (query - metrics['evaluation_start_ns']).astype(float) / 1e9
    position.plot(time, position_error, color='tab:red')
    position.set(ylabel='Position error (m)', title='Position error')
    heading.plot(time, np.degrees(heading_error), color='tab:purple')
    heading.set(xlabel='Time since first motion (s)', ylabel='Heading error (deg)',
                title='Wrapped heading error')
    for axis in (trajectory, position, heading):
        axis.grid(True, alpha=0.3)
    fig.suptitle('Baseline wheel odometry vs. independent simulator pose\n'
                 'One initial SE(2) alignment per trajectory; no path fitting', fontsize=12)
    fig.savefig(output_dir / 'trajectory_comparison.png', dpi=300)
    fig.savefig(output_dir / 'trajectory_comparison.pdf')
    plt.close(fig)


def write_outputs(output_dir, rows, result, bag_dir):
    query, gt, odom, position_error, heading_error, metrics = result
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics['source_bag'] = str(bag_dir)
    for topic, name in ((GT, 'raw_gt.csv'), (ODOM, 'raw_odom.csv')):
        with (output_dir / name).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(RAW_COLUMNS)
            writer.writerows(rows[topic])
    with (output_dir / 'comparison.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(('header_time_ns', 'time_since_first_motion_s', 'gt_x_m', 'gt_y_m',
                         'gt_yaw_rad', 'odom_x_m', 'odom_y_m', 'odom_yaw_rad',
                         'position_error_m', 'heading_error_rad'))
        for index, stamp in enumerate(query):
            writer.writerow((int(stamp), (int(stamp) - metrics['evaluation_start_ns']) / 1e9,
                             *gt[index], *odom[index], position_error[index], heading_error[index]))
    with (output_dir / 'metrics.json').open('w', encoding='utf-8') as stream:
        json.dump(metrics, stream, indent=2, allow_nan=False)
        stream.write('\n')
    make_plot(output_dir, *result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag_dir', type=Path, help='Finalized ROS 2 Jazzy MCAP bag directory')
    parser.add_argument('--output-dir', type=Path, help='Directory for generated analysis files')
    parser.add_argument('--segment-length', type=float, default=2.0,
                        help='Straight length in the recorded driver configuration (m)')
    parser.add_argument('--timing-tolerance', type=float, default=0.10,
                        help='Allowed command-duration difference (s; default: two 20 Hz periods)')
    parser.add_argument('--max-gap', type=float, default=0.10,
                        help='Maximum accepted source-message interval in either pose stream (s)')
    args = parser.parse_args(argv)
    bag_dir = args.bag_dir.expanduser().resolve()
    if not bag_dir.is_dir() or not (bag_dir / 'metadata.yaml').is_file():
        parser.error('bag_dir must contain metadata.yaml from a completed recording')
    output_dir = (args.output_dir or Path.home() / 'tb3_experiment_data' / 'analysis'
                  / bag_dir.name).expanduser().resolve()
    if output_dir == bag_dir or bag_dir in output_dir.parents:
        parser.error('The output directory must be outside the original bag directory')
    try:
        if not all(math.isfinite(value) and value > 0.0 for value in
                   (args.segment_length, args.timing_tolerance, args.max_gap)):
            raise ValueError('Analysis configuration values must be finite and positive.')
        print('Reading recorded command phases...', flush=True)
        summary = inspect_bag.inspect_commands(bag_dir)
        print('Command phases (source header time):')
        for index, phase in enumerate(summary.phases, 1):
            print(f'{index:2d} {phase.action:9s} v={phase.linear:.5f} m/s '
                  f'w={phase.angular:.5f} rad/s: {phase.start_ns / 1e9:.9f} '
                  f'to {phase.end_ns / 1e9:.9f} s; duration '
                  f'{(phase.end_ns - phase.start_ns) / 1e9:.6f} s')
        print(f'Command header duplicates: {summary.duplicates}; backwards: {summary.backwards}')
        if not summary.matches_one_sequence:
            raise ValueError('Commands do not identify exactly one valid STRAIGHT-LEFT-'
                             'STRAIGHT-RIGHT-STRAIGHT sequence; run inspect_bag.py first.')
        if summary.phases[0].action != 'STOP' or summary.phases[-1].action != 'STOP':
            raise ValueError('The sequence must include initial and final STOP phases.')
        validate_schedule(summary, args.segment_length, args.timing_tolerance)
        print('Command order and duration checks passed.', flush=True)
        first_motion_ns = next(phase.start_ns for phase in summary.phases if phase.action != 'STOP')
        t0, t1 = first_motion_ns - 1_000_000_000, summary.last_header_ns
        if t0 < summary.first_header_ns:
            raise ValueError('The recorded initial STOP does not provide one second before motion.')
        print(f'Alignment t0: {t0 / 1e9:.9f} s; evaluation: '
              f'{first_motion_ns / 1e9:.9f} to {t1 / 1e9:.9f} s')
        print('Reading only GT and odometry; aligning by header time, not bag receive time.',
              flush=True)
        rows = read_poses(bag_dir, t0 - GUARD_NS, t1 + GUARD_NS)
        series = {topic: prepare_series(rows[topic], topic) for topic in TOPIC_TYPES}
        for topic, (_, _, checks) in series.items():
            print(f'{topic}: {checks["raw_samples"]} raw samples; '
                  f'{checks["duplicate_header_timestamps"]} duplicate stamps; '
                  f'maximum header gap {checks["max_header_gap_s"]:.6f} s; '
                  f'frame={checks["frame_id"]!r}, child={checks["child_frame_id"]!r}')
            if checks['max_header_gap_s'] > args.max_gap + 1e-12:
                raise ValueError(f'{topic}: source gap {checks["max_header_gap_s"]:.6f} s '
                                 f'exceeds the configured {args.max_gap:.6f} s limit.')
        result = evaluate(series, t0, first_motion_ns, t1)
        result[-1]['command_validation'] = {
            'expected_straight_length_m': args.segment_length,
            'expected_turn_angle_deg': 90.0,
            'duration_tolerance_s': args.timing_tolerance,
            'maximum_accepted_pose_gap_s': args.max_gap,
            'phases': [dict(action=p.action, start_ns=p.start_ns, end_ns=p.end_ns,
                            linear_m_s=p.linear, angular_rad_s=p.angular) for p in summary.phases],
        }
        print('Writing CSV, metrics and plots...', flush=True)
        write_outputs(output_dir, rows, result, bag_dir)
        for key in ('position_rmse_m', 'endpoint_position_error_m', 'heading_rmse_deg'):
            print(f'{key}: {result[-1][key]:.9g}')
        print('Each path has one initial SE(2) alignment; raw initial poses remain in metrics.json.')
        print('Header-gap checks do not prove zero message loss. No added-noise evaluation is included.')
        print(f'Analysis files: {output_dir}')
        return 0
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
