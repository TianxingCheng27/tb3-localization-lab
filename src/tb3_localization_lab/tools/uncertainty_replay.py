"""Replay measured wheels offline; compare baseline before conditional noise trials.

Usage: python3 uncertainty_replay.py BAG_DIR BASELINE_ANALYSIS_DIR --check-only
Omit --check-only to run A0/A2 once and paired A1/A3 seeds 0..19.
Source the existing ROS workspace: tb3_localization_lab.kinematics supplies the
original wheel_motion/integrate_pose functions. Alternatively use
--kinematics-file /path/to/the/existing/kinematics.py. No ROS node is created.
Inputs are read-only; outputs go to BASELINE_ANALYSIS_DIR/uncertainty by default.
"""

import argparse
import csv
import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np


JOINT_TOPIC = '/joint_states'
LEFT, RIGHT = 'wheel_left_joint', 'wheel_right_joint'
GT, ONLINE = '/ground_truth/pose', '/odom_est'
METRIC_NAMES = ('position_rmse_m', 'maximum_position_error_m', 'endpoint_position_error_m',
                'heading_rmse_deg', 'maximum_absolute_heading_error_deg',
                'endpoint_absolute_heading_error_deg')
TRAJECTORY_COLUMNS = ('condition', 'seed', 'header_time_ns', 'time_since_first_motion_s',
                      'gt_x_m', 'gt_y_m', 'gt_yaw_rad', 'estimate_x_m', 'estimate_y_m',
                      'estimate_yaw_rad', 'position_error_m', 'heading_error_rad')


def wrap(value):
    return np.arctan2(np.sin(value), np.cos(value))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_kinematics(module_name='tb3_localization_lab.kinematics', filename=None):
    """Reuse the original pure math module, without importing or starting a node."""
    if filename is None:
        module = importlib.import_module(module_name)
    else:
        filename = Path(filename).expanduser().resolve()
        spec = importlib.util.spec_from_file_location('_replay_original_kinematics', filename)
        if spec is None or spec.loader is None:
            raise ValueError('Cannot load the requested original kinematics file.')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    for name in ('wheel_motion', 'integrate_pose'):
        if not callable(getattr(module, name, None)):
            raise ValueError(f'The original kinematics module must define {name}.')
    source = Path(module.__file__).resolve()
    return module.wheel_motion, module.integrate_pose, {
        'module': module.__name__, 'source_file': str(source), 'source_sha256': sha256(source)}


def load_baseline(folder):
    """Use the saved fixed GT frame and exactly the same evaluation timestamps."""
    metrics_path, csv_path = folder / 'metrics.json', folder / 'comparison.csv'
    with metrics_path.open(encoding='utf-8') as stream:
        metrics = json.load(stream)
    names = ('gt_x_m', 'gt_y_m', 'gt_yaw_rad', 'odom_x_m', 'odom_y_m', 'odom_yaw_rad',
             'position_error_m', 'heading_error_rad', 'time_since_first_motion_s')
    times, values = [], []
    with csv_path.open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        if not set(('header_time_ns', *names)).issubset(reader.fieldnames or []):
            raise ValueError('Baseline comparison.csv has an unsupported schema.')
        for row in reader:
            times.append(int(row['header_time_ns']))
            values.append([float(row[name]) for name in names])
    times, values = np.asarray(times, dtype=np.int64), np.asarray(values, dtype=float)
    if len(times) < 2 or not np.isfinite(values).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Baseline samples must be finite with strictly increasing header times.')
    t0 = int(metrics['alignment_time_ns'])
    start, end = int(metrics['evaluation_start_ns']), int(metrics['evaluation_end_ns'])
    if not t0 < start <= times[0] <= times[-1] <= end:
        raise ValueError('Baseline alignment/evaluation times are inconsistent.')
    if len(times) != metrics['evaluation_samples']:
        raise ValueError('Baseline CSV and JSON sample counts disagree.')
    if not np.allclose(values[:, 8], (times - start).astype(float) / 1e9,
                       rtol=0., atol=1e-9):
        raise ValueError('Baseline relative times do not match its source timestamps.')
    gt, online = values[:, :3], values[:, 3:6]
    if not np.allclose(np.linalg.norm(online[:, :2] - gt[:, :2], axis=1), values[:, 6],
                       rtol=1e-8, atol=1e-10):
        raise ValueError('Baseline position-error columns are inconsistent.')
    if not np.allclose(wrap(online[:, 2] - gt[:, 2]), values[:, 7], rtol=0., atol=1e-9):
        raise ValueError('Baseline heading-error columns are inconsistent.')
    checks = ((np.sqrt(np.mean(values[:, 6] ** 2)), metrics['position_rmse_m']),
              (np.degrees(np.sqrt(np.mean(values[:, 7] ** 2))), metrics['heading_rmse_deg']))
    if not all(np.isclose(a, b, rtol=1e-8, atol=1e-10) for a, b in checks):
        raise ValueError('Baseline CSV and JSON RMSE values disagree.')
    endpoints = metrics['aligned_endpoint_x_m_y_m_yaw_rad']
    gt_end, online_end = np.asarray(endpoints[GT], dtype=float), np.asarray(endpoints[ONLINE], dtype=float)
    if gt_end.shape != (3,) or online_end.shape != (3,) or not np.isfinite([gt_end, online_end]).all():
        raise ValueError('Baseline endpoints must be finite planar poses.')
    endpoint_checks = (
        (np.linalg.norm(online_end[:2] - gt_end[:2]), metrics['endpoint_position_error_m']),
        (np.degrees(wrap(online_end[2] - gt_end[2])), metrics['endpoint_heading_error_deg']),
    )
    if not all(np.isclose(actual, saved, rtol=1e-8, atol=1e-9)
               for actual, saved in endpoint_checks):
        raise ValueError('Baseline endpoint poses and saved endpoint errors disagree.')
    return dict(query_ns=times, gt=gt, online=online, gt_end=gt_end, online_end=online_end,
                t0=t0, start=start, end=end, metrics=metrics,
                input_sha256={'comparison.csv': sha256(csv_path), 'metrics.json': sha256(metrics_path)})


def read_wheels(bag_dir, start_ns, end_ns):
    """Only /joint_states is returned by storage; receipt time is never used for alignment."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    rows, scanned, previous = [], 0, None
    reader = rosbag2_py.SequentialReader()
    try:
        reader.open(rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id='mcap'),
                    rosbag2_py.ConverterOptions('cdr', 'cdr'))
        topics = {item.name: item for item in reader.get_all_topics_and_types()}
        metadata = topics.get(JOINT_TOPIC)
        if metadata is None or metadata.type != 'sensor_msgs/msg/JointState':
            raise ValueError('The source bag must contain sensor_msgs/msg/JointState on /joint_states.')
        if metadata.serialization_format != 'cdr':
            raise ValueError('Joint messages must be CDR serialized.')
        message_class = get_message(metadata.type)
        reader.set_filter(rosbag2_py.StorageFilter(topics=[JOINT_TOPIC]))
        while reader.has_next():
            _, data, bag_ns = reader.read_next()
            message = deserialize_message(data, message_class)
            stamp = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
            scanned += 1
            if previous is not None and stamp < previous:
                raise ValueError('Joint header time moved backwards; replay stopped without sorting.')
            previous = stamp
            if start_ns <= stamp <= end_ns:
                if message.name.count(LEFT) != 1 or message.name.count(RIGHT) != 1:
                    raise ValueError('Each retained JointState must contain both wheel names exactly once.')
                try:
                    left = float(message.position[message.name.index(LEFT)])
                    right = float(message.position[message.name.index(RIGHT)])
                except (IndexError, TypeError, ValueError) as exc:
                    raise ValueError('A retained JointState has invalid wheel positions.') from exc
                if not math.isfinite(left) or not math.isfinite(right):
                    raise ValueError('Non-finite wheel positions; replay stopped.')
                rows.append((int(stamp), int(bag_ns), left, right))
            if scanned % 100_000 == 0:
                print(f'Scanned {scanned} joint messages; retained {len(rows)}.', flush=True)
    finally:
        reader.close()
    return rows, scanned


def prepare_wheel_grid(rows, t0, end, max_gap):
    """Bracket both boundaries, interpolate continuous angles, and trim partial increments."""
    if len(rows) < 2:
        raise ValueError('Too few wheel samples in the guarded replay window.')
    times = np.asarray([row[0] for row in rows], dtype=np.int64)
    angles = np.asarray([row[2:4] for row in rows], dtype=float)
    difference = np.diff(times)
    if not np.isfinite(angles).all() or np.any(difference < 0):
        raise ValueError('Wheel angles must be finite and source timestamps must not go backwards.')
    duplicates = int(np.sum(difference == 0))
    for index in np.flatnonzero(difference == 0):
        if not np.allclose(angles[index], angles[index + 1], rtol=0., atol=1e-12):
            raise ValueError('Conflicting wheel angles share a header timestamp.')
    keep = np.r_[True, difference != 0]  # Match the online node's first-sample duplicate policy.
    times, angles = times[keep], angles[keep]
    if len(times) < 2 or t0 < times[0] or end > times[-1] or end <= t0:
        raise ValueError('Wheel data do not bracket t0 and t1; extrapolation is forbidden.')
    gap = float(np.max(np.diff(times))) / 1e9
    if gap > max_gap + 1e-12:
        raise ValueError(f'Maximum wheel source gap {gap:.6f} s exceeds {max_gap:.6f} s.')
    if np.any(np.abs(np.diff(angles, axis=0)) > math.pi):
        raise ValueError('Wheel angles appear discontinuous or under-sampled; do not silently unwrap/reset.')
    grid = np.r_[np.int64(t0), times[(times > t0) & (times < end)], np.int64(end)]
    source_time = (times - times[0]).astype(float) / 1e9
    query = (grid - times[0]).astype(float) / 1e9
    clipped = np.column_stack([np.interp(query, source_time, angles[:, i]) for i in range(2)])
    checks = dict(raw_guarded_samples=len(rows), unique_guarded_samples=len(times),
                  duplicate_header_timestamps=duplicates, backwards_header_timestamps=0,
                  max_source_gap_s=gap, source_first_header_ns=int(times[0]),
                  source_last_header_ns=int(times[-1]), replay_samples=len(grid),
                  t0_interpolated=not bool(np.any(times == t0)),
                  t1_interpolated=not bool(np.any(times == end)),
                  duplicate_policy='Keep first equivalent duplicate; reject conflicts.',
                  boundary_policy='Linear wheel-angle interpolation at t0/t1; no extrapolation.',
                  angle_assumption='Named wheel angles accumulate continuously; no modulo conversion.')
    return grid, clipped, checks


def replay_wheels(angles, radius, separation, bias, sigma, z, wheel_motion, integrate_pose):
    """Estimator inputs are only wheel measurements, geometry and the specified error model."""
    angles = np.asarray(angles, dtype=float)
    if (angles.ndim != 2 or angles.shape[1] != 2 or len(angles) < 2
            or not all(math.isfinite(value) and value > 0 for value in (radius, separation))
            or not math.isfinite(sigma) or sigma < 0):
        raise ValueError('Invalid wheel samples, geometry or noise standard deviation.')
    nominal = radius * np.diff(angles, axis=0)
    bias = np.asarray(bias, dtype=float)
    if (bias.shape != (2,) or not np.isfinite(bias).all() or np.any(bias <= -1)
            or not np.isfinite(angles).all() or not np.isfinite(nominal).all()):
        raise ValueError('Invalid wheel-noise configuration.')
    measured = nominal * (1.0 + bias)
    if sigma > 0:
        if z is None or np.shape(z) != nominal.shape or not np.isfinite(z).all():
            raise ValueError('Noise requires one independent standard-normal draw per wheel/increment.')
        # |nominal| / (1 metre) is dimensionless; sigma is in metres.
        measured = measured + sigma * np.sqrt(np.abs(nominal)) * z
    poses = np.zeros((len(angles), 3), dtype=float)
    x = y = yaw = continuous_yaw = 0.0
    for index, (left, right) in enumerate(measured, 1):
        # Reuse the original function, whose inputs are angular wheel increments.
        distance, turn = wheel_motion(float(left / radius), float(right / radius),
                                      radius=radius, separation=separation)
        x, y, yaw = integrate_pose(x, y, yaw, distance, turn)
        continuous_yaw += turn
        poses[index] = (x, y, continuous_yaw)
    if not np.isfinite(poses).all():
        raise ValueError('Non-finite replay pose.')
    return poses


def interpolate_pose(times, poses, query_ns):
    query_ns = np.atleast_1d(np.asarray(query_ns, dtype=np.int64))
    if query_ns.min() < times[0] or query_ns.max() > times[-1]:
        raise ValueError('Replay pose interpolation would extrapolate.')
    source = (times - times[0]).astype(float) / 1e9
    query = (query_ns - times[0]).astype(float) / 1e9
    return np.column_stack([np.interp(query, source, poses[:, axis]) for axis in range(3)])


def error_metrics(estimate, reference, endpoint, reference_endpoint):
    position = np.linalg.norm(estimate[:, :2] - reference[:, :2], axis=1)
    heading = wrap(estimate[:, 2] - reference[:, 2])
    end_heading = float(np.degrees(wrap(endpoint[2] - reference_endpoint[2])))
    result = dict(position_rmse_m=float(np.sqrt(np.mean(position ** 2))),
                  maximum_position_error_m=float(np.max(position)),
                  endpoint_position_error_m=float(np.linalg.norm(endpoint[:2] - reference_endpoint[:2])),
                  heading_rmse_deg=float(np.degrees(np.sqrt(np.mean(heading ** 2)))),
                  maximum_absolute_heading_error_deg=float(np.degrees(np.max(np.abs(heading)))),
                  endpoint_absolute_heading_error_deg=abs(end_heading), endpoint_heading_error_deg=end_heading)
    return result, position, heading


def summarize_runs(runs):
    """Summarize per-realization metrics, never the error of an averaged trajectory."""
    result = {}
    for condition in ('A0', 'A1', 'A2', 'A3'):
        selected = [run for run in runs if run['condition'] == condition]
        if selected:
            result[condition] = {'n': len(selected), 'metrics': {
                name: {'mean': float(np.mean([run[name] for run in selected])),
                       'sample_sd': (float(np.std([run[name] for run in selected], ddof=1))
                                     if len(selected) > 1 else None)} for name in METRIC_NAMES}}
    return result


def write_json(path, value):
    with path.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def write_csv(path, columns, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag_dir', type=Path)
    parser.add_argument('baseline_analysis_dir', type=Path)
    parser.add_argument('--check-only', action='store_true', help='Write A0-online diagnostics only; no noise trials')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--wheel-radius', type=float, default=.033)
    parser.add_argument('--wheel-separation', type=float, default=.160)
    parser.add_argument('--sigma-1m', type=float, default=.002)
    parser.add_argument('--bias-left', type=float, default=-.005)
    parser.add_argument('--bias-right', type=float, default=.005)
    parser.add_argument('--seeds', type=int, default=20)
    parser.add_argument('--first-seed', type=int, default=0)
    parser.add_argument('--max-gap', type=float, default=.1)
    parser.add_argument('--kinematics-module', default='tb3_localization_lab.kinematics')
    parser.add_argument('--kinematics-file', type=Path)
    args = parser.parse_args(argv)
    bag = args.bag_dir.expanduser().resolve()
    baseline_dir = args.baseline_analysis_dir.expanduser().resolve()
    output = (args.output_dir or baseline_dir / 'uncertainty').expanduser().resolve()
    try:
        positive = (args.wheel_radius, args.wheel_separation, args.max_gap)
        if (not all(math.isfinite(x) and x > 0 for x in positive)
                or not math.isfinite(args.sigma_1m) or args.sigma_1m < 0
                or not all(math.isfinite(x) and x > -1 for x in (args.bias_left, args.bias_right))
                or args.seeds < 2 or args.first_seed < 0):
            raise ValueError('Invalid geometry, noise, bias, seed or source-gap configuration.')
        if not bag.is_dir() or not (bag / 'metadata.yaml').is_file():
            raise ValueError('bag_dir must be a completed bag directory with metadata.yaml.')
        if output == bag or bag in output.parents or output == baseline_dir:
            raise ValueError('Outputs must be outside the bag and separate from baseline input files.')
        baseline = load_baseline(baseline_dir)
        source_path = baseline['metrics'].get('source_bag', '')
        if not source_path or Path(source_path).expanduser().resolve() != bag:
            raise ValueError(f'The baseline source-bag path ({source_path!r}) does not match '
                             f'the requested bag ({str(bag)!r}). Use the matching input pair.')
        motion, integrate, kinematics = load_kinematics(args.kinematics_module, args.kinematics_file)
        guard = max(args.max_gap, float(baseline['metrics'].get('read_guard_s', .1)))
        print(f'Exact replay: {baseline["t0"] / 1e9:.9f} to {baseline["end"] / 1e9:.9f} s.', flush=True)
        print('Reading only /joint_states, using header timestamps and named wheel positions.', flush=True)
        rows, scanned = read_wheels(bag, baseline['t0'] - round(guard * 1e9),
                                    baseline['end'] + round(guard * 1e9))
        grid, angles, wheel_checks = prepare_wheel_grid(rows, baseline['t0'], baseline['end'], args.max_gap)
        wheel_checks['scanned_joint_messages'] = scanned
        print(f'Wheel grid: {len(grid)} samples; maximum source gap '
              f'{wheel_checks["max_source_gap_s"]:.6f} s.', flush=True)
        radius, separation = args.wheel_radius, args.wheel_separation
        a0_path = replay_wheels(angles, radius, separation, (0., 0.), 0., None, motion, integrate)
        a0 = interpolate_pose(grid, a0_path, baseline['query_ns'])
        a0_end = interpolate_pose(grid, a0_path, [baseline['end']])[0]
        consistency, difference, angle_difference = error_metrics(
            a0, baseline['online'], a0_end, baseline['online_end'])
        print('A0 replay versus saved ONLINE odometry (not ground-truth accuracy):', flush=True)
        for name in METRIC_NAMES:
            print(f'  {name}: {consistency[name]:.9g}', flush=True)
        print('No arbitrary consistency threshold is applied. Review these differences first.', flush=True)
        metadata = dict(source_bag=str(bag), baseline_analysis_dir=str(baseline_dir),
                        baseline_source_bag=source_path, baseline_input_sha256=baseline['input_sha256'],
                        replay_script_sha256=sha256(Path(__file__)), check_only=args.check_only,
                        bag_metadata_sha256=sha256(bag / 'metadata.yaml'), kinematics=kinematics,
                        alignment_time_ns=baseline['t0'], evaluation_start_ns=baseline['start'],
                        evaluation_end_ns=baseline['end'], evaluation_samples=len(baseline['query_ns']),
                        wheel_radius_m=radius, wheel_separation_m=separation, wheel_checks=wheel_checks,
                        initial_state=[0., 0., 0.], a0_online_consistency=consistency,
                        a0_endpoint_x_m_y_m_yaw_rad=a0_end.tolist(),
                        online_endpoint_x_m_y_m_yaw_rad=baseline['online_end'].tolist(),
                        alignment='All replays start at zero at t0. The saved fixed GT initial frame '
                                  'is reused. No per-seed alignment or full-path fitting.',
                        a0_note='Raw-bag and best-effort online callback grids may differ. Accumulated '
                                'wheel positions preserve total rotation across skipped samples; '
                                'piecewise path integration can still differ.',
                        sampling_note='Observed source gaps do not prove zero message loss.')
        output.mkdir(parents=True, exist_ok=True)
        write_csv(output / 'a0_online_consistency.csv',
                  ('header_time_ns', 'time_since_first_motion_s', 'a0_x_m', 'a0_y_m', 'a0_yaw_rad',
                   'online_x_m', 'online_y_m', 'online_yaw_rad', 'position_difference_m', 'heading_difference_rad'),
                  ((int(t), (int(t) - baseline['start']) / 1e9, *a0[i], *baseline['online'][i],
                    difference[i], angle_difference[i]) for i, t in enumerate(baseline['query_ns'])))
        write_json(output / 'a0_check_metrics.json', metadata)
        if args.check_only:
            print(f'A0 check finished. No uncertainty trials were run. Diagnostics: {output}', flush=True)
            return 0

        bias = (args.bias_left, args.bias_right)
        runs = []
        trajectory_path = output / 'trajectories.csv'
        with trajectory_path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(TRAJECTORY_COLUMNS)

            def record(condition, seed, samples, endpoint):
                result, position, heading = error_metrics(samples, baseline['gt'], endpoint, baseline['gt_end'])
                result.update(condition=condition, seed=seed, endpoint_x_m=float(endpoint[0]),
                              endpoint_y_m=float(endpoint[1]), endpoint_yaw_rad=float(endpoint[2]))
                runs.append(result)
                for i, stamp in enumerate(baseline['query_ns']):
                    writer.writerow((condition, '' if seed is None else seed, int(stamp),
                                     (int(stamp) - baseline['start']) / 1e9, *baseline['gt'][i],
                                     *samples[i], position[i], heading[i]))

            record('A0', None, a0, a0_end)
            a2_path = replay_wheels(angles, radius, separation, bias, 0., None, motion, integrate)
            record('A2', None, interpolate_pose(grid, a2_path, baseline['query_ns']), a2_path[-1])
            nominal = radius * np.diff(angles, axis=0)
            bias_heading = np.r_[0., np.cumsum((bias[1] * nominal[:, 1] - bias[0] * nominal[:, 0]) / separation)]
            expected_pair = np.interp((baseline['query_ns'] - grid[0]).astype(float) / 1e9,
                                      (grid - grid[0]).astype(float) / 1e9, bias_heading)
            pair_checks = []
            for seed in range(args.first_seed, args.first_seed + args.seeds):
                # A1 and A3 share each wheel's draws; left and right columns remain independent.
                z = np.random.default_rng(seed).standard_normal(nominal.shape)
                a1_path = replay_wheels(angles, radius, separation, (0., 0.), args.sigma_1m, z, motion, integrate)
                a3_path = replay_wheels(angles, radius, separation, bias, args.sigma_1m, z, motion, integrate)
                a1 = interpolate_pose(grid, a1_path, baseline['query_ns'])
                a3 = interpolate_pose(grid, a3_path, baseline['query_ns'])
                pair_error = float(np.max(np.abs((a3[:, 2] - a1[:, 2]) - expected_pair)))
                endpoint_pair_error = abs((a3_path[-1, 2] - a1_path[-1, 2]) - bias_heading[-1])
                if max(pair_error, endpoint_pair_error) > 1e-8:
                    raise ValueError('Paired-noise heading identity failed; uncertainty outputs are incomplete.')
                pair_checks.append(max(pair_error, float(endpoint_pair_error)))
                record('A1', seed, a1, a1_path[-1])
                record('A3', seed, a3, a3_path[-1])
                print(f'Paired seed {seed}: A1/A3 completed.', flush=True)
        summaries = summarize_runs(runs)
        run_columns = ('condition', 'seed', *METRIC_NAMES, 'endpoint_heading_error_deg',
                       'endpoint_x_m', 'endpoint_y_m', 'endpoint_yaw_rad')
        write_csv(output / 'run_metrics.csv', run_columns,
                  ((run[name] for name in run_columns) for run in runs))
        summary_columns = ('condition', 'n', *(f'{name}_{stat}' for name in METRIC_NAMES for stat in ('mean', 'sample_sd')))
        write_csv(output / 'condition_summary.csv', summary_columns,
                  ((condition, summary['n'], *(summary['metrics'][name][stat]
                    for name in METRIC_NAMES for stat in ('mean', 'sample_sd')))
                   for condition, summary in summaries.items()))
        write_csv(output / 'raw_joint_states.csv',
                  ('header_time_ns', 'bag_time_ns', 'left_angle_rad', 'right_angle_rad'), rows)
        write_csv(output / 'wheel_grid.csv', ('header_time_ns', 'left_angle_rad', 'right_angle_rad'),
                  ((int(stamp), *angles[i]) for i, stamp in enumerate(grid)))
        metadata.update(sigma_1m_m=args.sigma_1m, bias_left=bias[0], bias_right=bias[1],
                        seeds=list(range(args.first_seed, args.first_seed + args.seeds)),
                        numpy_version=np.__version__, random_generator='numpy.default_rng / '
                        + type(np.random.default_rng(0).bit_generator).__name__,
                        noise_model='ds_measured=(1+b)*ds_nominal + sigma_1m*sqrt(abs(ds_nominal)/(1 m))*z; '
                                    'independent wheels/increments, shared A1/A3 z per seed.',
                        condition_definitions={
                            'A0': 'Measured-wheel replay without added noise or scale bias.',
                            'A1': 'Random wheel-travel measurement noise only.',
                            'A2': 'Constant left/right wheel-scale biases only.',
                            'A3': 'Random measurement noise and constant scale biases combined.'},
                        coefficient_status='Illustrative sensitivity-test assumptions, '
                                           'not calibrated hardware specifications.',
                        replication='Conditional measurement-noise realizations of one recorded physical run; '
                                    'not independent physical simulations.',
                        aggregation='Compute each run metric first; report mean and sample SD (ddof=1). '
                                    'Deterministic cases have n=1 and null SD.',
                        condition_summary=summaries, run_metrics=runs,
                        paired_heading_identity_max_error_rad=max(pair_checks),
                        fixed_gt_endpoint_x_m_y_m_yaw_rad=baseline['gt_end'].tolist(),
                        command_validation=baseline['metrics'].get('command_validation'))
        write_json(output / 'uncertainty_metrics.json', metadata)
        print('Condition summaries (position RMSE mean; sample SD):', flush=True)
        for condition, summary in summaries.items():
            values = summary['metrics']['position_rmse_m']
            print(f'  {condition}: n={summary["n"]}, {values["mean"]:.9g} m; SD={values["sample_sd"]}', flush=True)
        print(f'Finished; inputs and original bag unchanged. Numerical outputs: {output}', flush=True)
        return 0
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
