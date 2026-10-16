"""Compare baseline and online corrupted-wheel odometry from one completed bag.

Keep inspect_bag.py and analyse_run.py beside this file. No ROS node is started.
The original bag and parameter dump are read-only; outputs use a new directory.
"""

import argparse
import csv
import hashlib
import json
import math
from numbers import Integral
from pathlib import Path
import sys

import numpy as np

import analyse_run as baseline
import inspect_bag


GT = '/ground_truth/pose'
BASELINE = '/odom_est'
NOISY = '/odom_noisy'
ESTIMATES = (BASELINE, NOISY)
TOPIC_TYPES = {
    GT: 'geometry_msgs/msg/PoseStamped',
    BASELINE: 'nav_msgs/msg/Odometry',
    NOISY: 'nav_msgs/msg/Odometry',
}
GUARD_NS = 100_000_000


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_params(path):
    """Validate and retain the actual encoder node's ros2 param dump."""
    import yaml

    path = Path(path).expanduser().resolve()
    source = path.read_bytes()
    text = source.decode('utf-8')
    document = yaml.safe_load(text)
    if not isinstance(document, dict):
        raise ValueError('The parameter dump must be a YAML mapping.')
    matches = [
        (name, value) for name, value in document.items()
        if isinstance(name, str) and name.strip('/') == 'encoder_model'
    ]
    if len(matches) != 1 or not isinstance(matches[0][1], dict):
        raise ValueError(
            'Expected exactly one encoder_model node in the parameter dump.'
        )
    node_name, node = matches[0]
    parameters = node.get('ros__parameters')
    if not isinstance(parameters, dict) or parameters.get('use_sim_time') is not True:
        raise ValueError(
            'encoder_model parameters must include use_sim_time: true.'
        )
    config = {'use_sim_time': True}
    for name in (
        'wheel_radius', 'sigma_1m', 'left_bias', 'right_bias', 'max_source_gap'
    ):
        value = parameters.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f'Parameter {name} must be a finite number.')
        config[name] = float(value)
    seed = parameters.get('seed')
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError('Parameter seed must be a nonnegative integer.')
    config['seed'] = int(seed)
    if (
        config['wheel_radius'] <= 0
        or config['sigma_1m'] <= 0
        or config['max_source_gap'] <= 0
        or min(config['left_bias'], config['right_bias']) <= -1
        or (config['left_bias'] == 0 and config['right_bias'] == 0)
    ):
        raise ValueError(
            'Expected an A3 configuration: positive radius/noise/gap, '
            'biases greater than -1, and at least one nonzero wheel bias.'
        )
    return dict(
        node_name=node_name,
        parameters=config,
        source_path=str(path),
        sha256=hashlib.sha256(source).hexdigest(),
        snapshot_text=text,
        snapshot_note=(
            'User-supplied live-node parameter dump; pose messages do not '
            'independently prove that parameters stayed unchanged throughout the run.'
        ),
    )


def command_window(summary, segment_length=2.0, tolerance=.1):
    """Derive this bag's window; never reuse timestamps from an earlier recording."""
    if not all(
        math.isfinite(value) and value > 0
        for value in (segment_length, tolerance)
    ):
        raise ValueError(
            'Segment length and timing tolerance must be finite and positive.'
        )
    baseline.validate_schedule(summary, segment_length, tolerance)
    first_motion = next(
        phase.start_ns for phase in summary.phases if phase.action != 'STOP'
    )
    t0 = first_motion - 1_000_000_000
    t1 = summary.last_header_ns
    if t0 < 0 or t0 < summary.first_header_ns or t1 <= first_motion:
        raise ValueError(
            'Recorded commands do not cover the complete alignment/evaluation window.'
        )
    validation = dict(
        expected_straight_length_m=segment_length,
        expected_turn_angle_deg=90.,
        duration_tolerance_s=tolerance,
        duplicate_headers=summary.duplicates,
        backwards_headers=summary.backwards,
        phases=[
            dict(
                action=p.action,
                start_ns=p.start_ns,
                end_ns=p.end_ns,
                linear_m_s=p.linear,
                angular_rad_s=p.angular,
            )
            for p in summary.phases
        ],
    )
    return int(t0), int(first_motion), int(t1), validation


def read_poses(bag_dir, start_ns, end_ns):
    """Read three pose topics only; crop by source header time, not receive time."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    rows = {topic: [] for topic in TOPIC_TYPES}
    reader = rosbag2_py.SequentialReader()
    try:
        reader.open(
            rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id='mcap'),
            rosbag2_py.ConverterOptions('cdr', 'cdr'),
        )
        metadata = {
            item.name: item for item in reader.get_all_topics_and_types()
        }
        for topic, kind in TOPIC_TYPES.items():
            item = metadata.get(topic)
            if (
                item is None
                or item.type != kind
                or item.serialization_format != 'cdr'
            ):
                raise ValueError(
                    f'{topic} must contain CDR-serialized {kind} messages.'
                )
        classes = {
            topic: get_message(kind) for topic, kind in TOPIC_TYPES.items()
        }
        reader.set_filter(
            rosbag2_py.StorageFilter(topics=list(TOPIC_TYPES))
        )
        while reader.has_next():
            topic, data, bag_ns = reader.read_next()
            if topic not in TOPIC_TYPES:
                raise ValueError('Storage filter returned an unexpected topic.')
            message = deserialize_message(data, classes[topic])
            sec = message.header.stamp.sec
            nano = message.header.stamp.nanosec
            if not 0 <= sec <= 2_147_483_647 or not 0 <= nano < 1_000_000_000:
                raise ValueError(f'{topic}: invalid source timestamp fields.')
            stamp = int(sec) * 1_000_000_000 + int(nano)
            if start_ns <= stamp <= end_ns:
                pose = message.pose if topic == GT else message.pose.pose
                p, q = pose.position, pose.orientation
                rows[topic].append((
                    stamp,
                    int(bag_ns),
                    message.header.frame_id,
                    getattr(message, 'child_frame_id', ''),
                    p.x, p.y, p.z,
                    q.x, q.y, q.z, q.w,
                ))
        return rows
    finally:
        reader.close()


def prepare_online_series(rows, max_gap=.1):
    """Validate all streams, including both odometry child frames."""
    if not math.isfinite(max_gap) or max_gap <= 0:
        raise ValueError('Maximum pose gap must be finite and positive.')
    series = {}
    for topic in TOPIC_TYPES:
        for row in rows[topic]:
            if any(
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Integral)
                or value < 0
                for value in row[:2]
            ):
                raise ValueError(
                    f'{topic}: header and bag timestamps must be nonnegative integers.'
                )
        stamps, poses, checks = baseline.prepare_series(rows[topic], topic)
        if checks['max_header_gap_s'] > max_gap + 1e-12:
            raise ValueError(
                f'{topic}: maximum header gap '
                f'{checks["max_header_gap_s"]:.6f} s '
                f'exceeds {max_gap:.6f} s.'
            )
        if (
            topic in ESTIMATES
            and checks['child_frame_id'] != 'base_footprint'
        ):
            raise ValueError(
                f'{topic}: expected the base_footprint robot reference point.'
            )
        series[topic] = stamps, poses, checks
    return series


def evaluate_online(series, t0, first_motion, t1):
    """Use one shared GT grid and one fixed initial transform per trajectory."""
    gt_t, gt_pose, _ = series[GT]
    initial = {
        topic: baseline.interpolate(stamps, poses, [t0])[0]
        for topic, (stamps, poses, _) in series.items()
    }
    query = gt_t[(gt_t >= first_motion) & (gt_t <= t1)]
    if len(query) < 2:
        raise ValueError('Fewer than two GT evaluation samples.')
    gt = baseline.to_start_frame(
        baseline.interpolate(gt_t, gt_pose, query), initial[GT]
    )
    gt_end = baseline.to_start_frame(
        baseline.interpolate(gt_t, gt_pose, [t1]), initial[GT]
    )[0]
    estimates, errors, results = {}, {}, {}
    for topic in ESTIMATES:
        stamps, poses, _ = series[topic]
        estimate = baseline.to_start_frame(
            baseline.interpolate(stamps, poses, query), initial[topic]
        )
        endpoint = baseline.to_start_frame(
            baseline.interpolate(stamps, poses, [t1]), initial[topic]
        )[0]
        position = np.linalg.norm(estimate[:, :2] - gt[:, :2], axis=1)
        heading = baseline.wrap(estimate[:, 2] - gt[:, 2])
        raw_difference = initial[topic] - initial[GT]
        raw_difference[2] = baseline.wrap(raw_difference[2])
        estimates[topic] = estimate
        errors[topic] = position, heading
        results[topic] = dict(
            condition='baseline' if topic == BASELINE else 'online A3',
            n=1,
            position_rmse_m=float(np.sqrt(np.mean(position ** 2))),
            endpoint_position_error_m=float(
                np.linalg.norm(endpoint[:2] - gt_end[:2])
            ),
            heading_rmse_deg=float(
                np.degrees(np.sqrt(np.mean(heading ** 2)))
            ),
            endpoint_heading_error_deg=float(
                np.degrees(baseline.wrap(endpoint[2] - gt_end[2]))
            ),
            maximum_position_error_m=float(np.max(position)),
            maximum_absolute_heading_error_deg=float(
                np.degrees(np.max(np.abs(heading)))
            ),
            aligned_endpoint_x_m_y_m_yaw_rad=endpoint.tolist(),
            raw_initial_minus_gt_x_m_y_m_yaw_rad=raw_difference.tolist(),
        )
    metrics = dict(
        schema_version=1,
        experiment='paired online baseline and A3',
        alignment_time_ns=int(t0),
        evaluation_start_ns=int(first_motion),
        evaluation_end_ns=int(t1),
        evaluation_samples=len(query),
        read_guard_s=GUARD_NS / 1e9,
        evaluation_sampling=(
            'Same unique GT header timestamps for both estimates; '
            'no receive-time alignment.'
        ),
        alignment=(
            'Each trajectory uses its own inverse initial SE(2) at the same t0; '
            'relative-motion evaluation only, with no full-path fitting.'
        ),
        aligned_gt_endpoint_x_m_y_m_yaw_rad=gt_end.tolist(),
        raw_initial_poses_x_m_y_m_yaw_rad={
            topic: value.tolist() for topic, value in initial.items()
        },
        raw_initial_difference_note=(
            'Raw coordinate diagnostic only; frames need not share an origin.'
        ),
        topic_checks={
            topic: value[2] for topic, value in series.items()
        },
        reference_point_assumption=(
            'Gazebo model origin and both base_footprint references '
            'coincide in the plane.'
        ),
        duplicate_policy=(
            'Raw rows retained; last equivalent planar duplicate used; '
            'conflicts rejected.'
        ),
        sampling_note='Observed header gaps do not prove zero message loss.',
        replication=(
            'One recorded simulation run, paired baseline/A3, n=1 per estimate; '
            'no SD or confidence interval.'
        ),
        online_offline_note=(
            'A fixed seed reproduces a fixed accepted input sequence; '
            'this new online trajectory need not match a previous offline '
            'replay with the same seed.'
        ),
        uncertainty_note=(
            'Ground truth is used only by this offline evaluator. '
            'Coefficients describe an illustrative measurement sensitivity '
            'test, not calibrated hardware specifications.'
        ),
        estimates=results,
    )
    return query, gt, estimates, errors, metrics


def require_new_output(output_dir, bag_dir):
    output_dir = Path(output_dir).resolve()
    bag_dir = Path(bag_dir).resolve()
    if output_dir == bag_dir or bag_dir in output_dir.parents:
        raise ValueError('Output directory must be outside the source bag.')
    if output_dir.exists() and (
        not output_dir.is_dir() or any(output_dir.iterdir())
    ):
        raise ValueError(
            'Output directory already contains files; choose a new empty directory.'
        )


def write_outputs(output_dir, rows, result, params, bag_dir):
    query, gt, estimates, errors, metrics = result
    require_new_output(output_dir, bag_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics['source_bag'] = str(bag_dir)
    metrics['encoder_parameters'] = params
    metrics['source_code'] = {
        name: dict(
            path=str(Path(module.__file__).resolve()),
            sha256=file_hash(module.__file__),
        )
        for name, module in (
            ('analyse_run', baseline), ('inspect_bag', inspect_bag)
        )
    }
    metrics['source_code']['analyse_online'] = dict(
        path=str(Path(__file__).resolve()),
        sha256=file_hash(__file__),
    )
    for topic, name in (
        (GT, 'raw_gt.csv'),
        (BASELINE, 'raw_baseline.csv'),
        (NOISY, 'raw_noisy.csv'),
    ):
        with (output_dir / name).open(
            'w', newline='', encoding='utf-8'
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(baseline.RAW_COLUMNS)
            writer.writerows(rows[topic])
    columns = [
        'header_time_ns', 'time_since_first_motion_s',
        'gt_x_m', 'gt_y_m', 'gt_yaw_rad',
    ]
    for label in ('baseline', 'noisy'):
        columns += [
            f'{label}_{suffix}' for suffix in (
                'x_m', 'y_m', 'yaw_rad',
                'position_error_m', 'heading_error_rad',
            )
        ]
    with (output_dir / 'comparison.csv').open(
        'w', newline='', encoding='utf-8'
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for i, stamp in enumerate(query):
            row = [
                int(stamp),
                (int(stamp) - metrics['evaluation_start_ns']) / 1e9,
                *gt[i],
            ]
            for topic in ESTIMATES:
                row += [
                    *estimates[topic][i],
                    errors[topic][0][i],
                    errors[topic][1][i],
                ]
            writer.writerow(row)
    # Preserve the dump's original line endings and any UTF-8 BOM.
    (output_dir / 'encoder_params_snapshot.yaml').write_bytes(
        params['snapshot_text'].encode('utf-8')
    )
    with (output_dir / 'metrics.json').open(
        'w', encoding='utf-8'
    ) as stream:
        json.dump(metrics, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag_dir', type=Path)
    parser.add_argument(
        '--params-file', type=Path, required=True,
        help='Saved ros2 param dump /encoder_model YAML',
    )
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--segment-length', type=float, default=2.)
    parser.add_argument('--timing-tolerance', type=float, default=.1)
    parser.add_argument(
        '--max-gap', type=float, default=.1,
        help='Maximum accepted pose header gap (s)',
    )
    args = parser.parse_args(argv)
    try:
        bag_dir = args.bag_dir.expanduser().resolve()
        if (
            not bag_dir.is_dir()
            or not (bag_dir / 'metadata.yaml').is_file()
        ):
            raise ValueError(
                'bag_dir must contain metadata.yaml from a completed MCAP recording.'
            )
        output_dir = (
            args.output_dir
            or Path.home() / 'tb3_experiment_data' / 'analysis'
            / bag_dir.name / 'online_comparison'
        ).expanduser().resolve()
        require_new_output(output_dir, bag_dir)
        params = load_params(args.params_file)
        print('Reading recorded command phases...', flush=True)
        summary = inspect_bag.inspect_commands(bag_dir)
        t0, start, end, command_checks = command_window(
            summary, args.segment_length, args.timing_tolerance
        )
        for phase in command_checks['phases']:
            duration = (phase['end_ns'] - phase['start_ns']) / 1e9
            print(f'{phase["action"]:9s} {duration:.6f} s')
        print(
            f'Alignment: {t0 / 1e9:.9f} s; evaluation: '
            f'{start / 1e9:.9f} to {end / 1e9:.9f} s',
            flush=True,
        )
        print(
            'Reading GT, baseline and noisy odometry by source header time...',
            flush=True,
        )
        rows = read_poses(bag_dir, t0 - GUARD_NS, end + GUARD_NS)
        series = prepare_online_series(rows, args.max_gap)
        for topic, (_, _, checks) in series.items():
            print(
                f'{topic}: {checks["raw_samples"]} samples, '
                f'{checks["duplicate_header_timestamps"]} duplicate headers, '
                f'max gap {checks["max_header_gap_s"]:.6f} s, '
                f'frame={checks["frame_id"]!r}, '
                f'child={checks["child_frame_id"]!r}'
            )
        result = evaluate_online(series, t0, start, end)
        result[-1]['command_validation'] = command_checks
        result[-1]['maximum_accepted_pose_gap_s'] = args.max_gap
        write_outputs(output_dir, rows, result, params, bag_dir)
        for topic, metric in result[-1]['estimates'].items():
            print(
                f'{topic}: position RMSE {metric["position_rmse_m"]:.9g} m; '
                f'endpoint {metric["endpoint_position_error_m"]:.9g} m; '
                f'heading RMSE {metric["heading_rmse_deg"]:.9g} deg'
            )
        print(
            'Paired estimates from one recorded simulation run (n=1); '
            'gap checks do not prove zero loss.'
        )
        print(f'Analysis files: {output_dir}')
        return 0
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())