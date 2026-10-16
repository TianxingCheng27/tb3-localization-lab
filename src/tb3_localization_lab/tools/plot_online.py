"""Plot saved online baseline/A3 evaluation without changing its numerical data.

Reads comparison.csv and metrics.json from analyse_online.py. No bag or ROS
installation is needed. Existing figure files require explicit --overwrite.
"""

import argparse
import csv
import hashlib
import json
import math
from numbers import Integral
from pathlib import Path

import numpy as np


GT = '/ground_truth/pose'
BASELINE = '/odom_est'
NOISY = '/odom_noisy'

ESTIMATES = (
    (BASELINE, 'baseline', 'Baseline', '#1976B9', (0, (5, 3))),
    (NOISY, 'noisy', 'Online A3', '#D47721', '-'),
)

COLUMNS = (
    'time_since_first_motion_s',
    'gt_x_m', 'gt_y_m', 'gt_yaw_rad',
    'baseline_x_m', 'baseline_y_m', 'baseline_yaw_rad',
    'baseline_position_error_m', 'baseline_heading_error_rad',
    'noisy_x_m', 'noisy_y_m', 'noisy_yaw_rad',
    'noisy_position_error_m', 'noisy_heading_error_rad',
)


def wrap(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def require_close(actual, saved, description):
    if (
        not np.isfinite(saved).all()
        or not np.allclose(actual, saved, rtol=1e-9, atol=1e-11)
    ):
        raise ValueError(f'CSV/JSON mismatch: {description}.')


def vector3(value, description):
    value = np.asarray(value, dtype=float)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError(
            f'{description} must be a finite three-element pose.'
        )
    return value


def source_ns(value, description):
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or not 0 <= value < 2 ** 63
    ):
        raise ValueError(
            f'{description} must be a nonnegative int64 timestamp.'
        )
    return int(value)


def load_results(folder):
    """Verify timestamps, errors, exact endpoints and parameter provenance."""
    folder = Path(folder)
    with (folder / 'metrics.json').open(encoding='utf-8') as stream:
        metrics = json.load(stream)

    with (folder / 'comparison.csv').open(
        newline='', encoding='utf-8'
    ) as stream:
        reader = csv.DictReader(stream)
        names = reader.fieldnames or []
        if (
            len(names) != len(set(names))
            or not {'header_time_ns', *COLUMNS}.issubset(names)
        ):
            raise ValueError(
                'comparison.csv has missing or duplicate columns.'
            )
        rows, stamps = [], []
        for row in reader:
            stamps.append(
                source_ns(int(row['header_time_ns']), 'CSV header time')
            )
            rows.append([float(row[name]) for name in COLUMNS])

    values = np.asarray(rows, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError(
            'At least two finite comparison samples are required.'
        )

    data = dict(zip(COLUMNS, values.T))
    stamps = np.asarray(stamps, dtype=np.int64)
    data['header_time_ns'] = stamps

    t0 = source_ns(metrics['alignment_time_ns'], 'alignment time')
    start = source_ns(
        metrics['evaluation_start_ns'], 'evaluation start'
    )
    end = source_ns(metrics['evaluation_end_ns'], 'evaluation end')

    if (
        not t0 < start < end
        or np.any(np.diff(stamps) <= 0)
        or stamps[0] < start
        or stamps[-1] > end
    ):
        raise ValueError(
            'Comparison header times are not strictly ordered '
            'within the evaluation window.'
        )

    count = metrics['evaluation_samples']
    if (
        isinstance(count, bool)
        or not isinstance(count, Integral)
        or count != len(stamps)
    ):
        raise ValueError('CSV and JSON sample counts differ.')

    require_close(
        data['time_since_first_motion_s'],
        (stamps - start) / 1e9,
        'relative times',
    )

    if set(metrics['estimates']) != {BASELINE, NOISY}:
        raise ValueError(
            'Expected exactly /odom_est and /odom_noisy estimate metrics.'
        )

    gt_end = vector3(
        metrics['aligned_gt_endpoint_x_m_y_m_yaw_rad'],
        'GT endpoint',
    )

    for topic, prefix, _, _, _ in ESTIMATES:
        metric = metrics['estimates'][topic]

        if (
            isinstance(metric['n'], bool)
            or not isinstance(metric['n'], Integral)
            or metric['n'] != 1
        ):
            raise ValueError(
                'This figure requires one paired run per estimate, '
                'without SD/CI.'
            )

        position = np.hypot(
            data[f'{prefix}_x_m'] - data['gt_x_m'],
            data[f'{prefix}_y_m'] - data['gt_y_m'],
        )
        heading = wrap(
            data[f'{prefix}_yaw_rad'] - data['gt_yaw_rad']
        )

        require_close(
            data[f'{prefix}_position_error_m'],
            position,
            f'{topic} position errors',
        )
        require_close(
            data[f'{prefix}_heading_error_rad'],
            heading,
            f'{topic} wrapped heading errors',
        )

        for actual, name in (
            (np.sqrt(np.mean(position ** 2)), 'position_rmse_m'),
            (np.max(position), 'maximum_position_error_m'),
            (
                np.degrees(np.sqrt(np.mean(heading ** 2))),
                'heading_rmse_deg',
            ),
            (
                np.degrees(np.max(np.abs(heading))),
                'maximum_absolute_heading_error_deg',
            ),
        ):
            require_close(actual, metric[name], f'{topic} {name}')

        endpoint = vector3(
            metric['aligned_endpoint_x_m_y_m_yaw_rad'],
            f'{topic} endpoint',
        )
        require_close(
            np.linalg.norm(endpoint[:2] - gt_end[:2]),
            metric['endpoint_position_error_m'],
            f'{topic} endpoint position error',
        )
        require_close(
            np.degrees(wrap(endpoint[2] - gt_end[2])),
            metric['endpoint_heading_error_deg'],
            f'{topic} signed endpoint heading error',
        )

        # JSON retains the precise endpoint even if the last CSV time is earlier.
        if stamps[-1] == end:
            require_close(
                [
                    data[f'{prefix}_x_m'][-1],
                    data[f'{prefix}_y_m'][-1],
                ],
                endpoint[:2],
                f'{topic} final CSV position',
            )
            require_close(
                wrap(data[f'{prefix}_yaw_rad'][-1] - endpoint[2]),
                0.,
                f'{topic} final CSV yaw',
            )

    if stamps[-1] == end:
        require_close(
            [data['gt_x_m'][-1], data['gt_y_m'][-1]],
            gt_end[:2],
            'GT final CSV position',
        )
        require_close(
            wrap(data['gt_yaw_rad'][-1] - gt_end[2]),
            0.,
            'GT final CSV yaw',
        )

    phases = metrics['command_validation']['phases']
    expected = (
        'STOP', 'STRAIGHT', 'STOP', 'LEFT', 'STOP', 'STRAIGHT',
        'STOP', 'RIGHT', 'STOP', 'STRAIGHT', 'STOP',
    )
    if tuple(phase['action'] for phase in phases) != expected:
        raise ValueError(
            'Expected the complete three-straight, two-turn command schedule.'
        )

    for i, phase in enumerate(phases):
        a = source_ns(phase['start_ns'], 'phase start')
        b = source_ns(phase['end_ns'], 'phase end')
        if b <= a or (i and a != phases[i - 1]['end_ns']):
            raise ValueError(
                'Command phases are empty, reversed or discontinuous.'
            )

    if (
        phases[0]['start_ns'] > t0
        or phases[1]['start_ns'] != start
        or phases[-1]['end_ns'] != end
    ):
        raise ValueError(
            'Command phases and JSON evaluation times differ.'
        )

    provenance = metrics['encoder_parameters']
    snapshot_hash = hashlib.sha256(
        provenance['snapshot_text'].encode('utf-8')
    ).hexdigest()

    if snapshot_hash != provenance['sha256']:
        raise ValueError(
            'Saved parameter snapshot text and SHA256 differ.'
        )

    snapshot_file = folder / 'encoder_params_snapshot.yaml'
    if (
        snapshot_file.exists()
        and hashlib.sha256(snapshot_file.read_bytes()).hexdigest()
        != provenance['sha256']
    ):
        raise ValueError(
            'Parameter snapshot file and JSON SHA256 differ.'
        )

    config = provenance['parameters']
    if config.get('use_sim_time') is not True:
        raise ValueError(
            'Expected a simulation-time parameter snapshot.'
        )

    for name in (
        'wheel_radius', 'sigma_1m', 'left_bias',
        'right_bias', 'max_source_gap',
    ):
        value = config[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f'Invalid saved encoder parameter: {name}.')

    if (
        min(
            config['wheel_radius'],
            config['sigma_1m'],
            config['max_source_gap'],
        ) <= 0
        or min(config['left_bias'], config['right_bias']) <= -1
        or (config['left_bias'] == 0 and config['right_bias'] == 0)
    ):
        raise ValueError(
            'Saved configuration does not describe the combined A3 model.'
        )

    if (
        isinstance(config['seed'], bool)
        or not isinstance(config['seed'], Integral)
        or config['seed'] < 0
    ):
        raise ValueError('Saved seed must be a nonnegative integer.')

    return data, metrics


def draw_results(folder, data, metrics, overwrite=False):
    folder = Path(folder)
    paths = [
        folder / f'online_evaluation.{suffix}'
        for suffix in ('png', 'pdf')
    ]
    if not overwrite and any(path.exists() for path in paths):
        raise ValueError(
            'Figure files already exist; use --overwrite explicitly '
            'to replace them.'
        )

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    style = {
        'font.family': 'DejaVu Sans',
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.edgecolor': '#64748B',
        'axes.linewidth': .8,
        'xtick.color': '#334155',
        'ytick.color': '#334155',
        'text.color': '#172033',
        'axes.labelcolor': '#172033',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
    }

    with plt.rc_context(style):
        fig = plt.figure(figsize=(11.6, 8.2))
        fig.text(
            .075, .955,
            'Online Wheel Measurement Uncertainty',
            fontsize=18, weight='bold',
        )
        fig.text(
            .075, .920,
            'Baseline and combined-error odometry | '
            'Independent simulator ground truth',
            fontsize=10, color='#64748B',
        )

        trajectory = fig.add_axes([.075, .475, .55, .385])
        summary = fig.add_axes([.655, .475, .30, .385])
        position = fig.add_axes([.075, .190, .39, .205])
        heading = fig.add_axes([.565, .190, .39, .205])
        summary.set_axis_off()

        gt_end = np.asarray(
            metrics['aligned_gt_endpoint_x_m_y_m_yaw_rad']
        )
        gt_line, = trajectory.plot(
            np.r_[0., data['gt_x_m'], gt_end[0]],
            np.r_[0., data['gt_y_m'], gt_end[1]],
            color='#172033',
            linewidth=2.2,
            label='Ground truth',
            zorder=2,
        )
        trajectory.scatter(
            *gt_end[:2], marker='s', s=35,
            color='#172033', zorder=5,
        )

        handles, labels = [gt_line], ['Ground truth']

        for topic, prefix, label, color, linestyle in ESTIMATES:
            endpoint = np.asarray(
                metrics['estimates'][topic][
                    'aligned_endpoint_x_m_y_m_yaw_rad'
                ]
            )
            line, = trajectory.plot(
                np.r_[0., data[f'{prefix}_x_m'], endpoint[0]],
                np.r_[0., data[f'{prefix}_y_m'], endpoint[1]],
                color=color,
                linestyle=linestyle,
                linewidth=1.8,
                zorder=3,
            )
            trajectory.scatter(
                *endpoint[:2], marker='s', s=35,
                color=color, zorder=5,
            )
            handles.append(line)
            labels.append(label)

        start_marker = trajectory.scatter(
            0, 0, s=42, color='#24966D',
            edgecolors='white', linewidths=.6, zorder=6,
        )
        endpoint_key = trajectory.scatter(
            [], [], marker='s', s=35, color='#64748B',
        )
        handles += [start_marker, endpoint_key]
        labels += ['Aligned start', 'Exact endpoints (matched colors)']

        trajectory.set(
            xlabel='x in initial frame (m)',
            ylabel='y in initial frame (m)',
        )
        trajectory.set_title(
            '(a) Planar trajectories',
            loc='left', weight='bold', pad=10,
        )
        trajectory.set_aspect('equal', adjustable='box', anchor='W')
        trajectory.margins(x=.06, y=.10)
        trajectory.xaxis.set_major_locator(
            MaxNLocator(nbins=6, steps=[1, 2, 5, 10])
        )
        trajectory.yaxis.set_major_locator(
            MaxNLocator(nbins=5, steps=[1, 2, 5, 10])
        )

        summary.legend(
            handles, labels,
            loc='upper left',
            bbox_to_anchor=(0, 1.02),
            frameon=False,
            borderaxespad=0,
            handlelength=2.7,
            labelspacing=.65,
            fontsize=9,
        )
        summary.plot(
            [0, 1], [.53, .53],
            color='#E2E8F0', linewidth=1,
            transform=summary.transAxes,
        )
        summary.text(
            .66, .46, 'Baseline',
            ha='right', weight='bold',
            color='#1976B9', transform=summary.transAxes,
        )
        summary.text(
            1., .46, 'Online A3',
            ha='right', weight='bold',
            color='#D47721', transform=summary.transAxes,
        )

        for y, label, name, scale, decimals in (
            (
                .34, 'Position RMSE (mm)',
                'position_rmse_m', 1000., 2,
            ),
            (
                .21, 'Endpoint error (mm)',
                'endpoint_position_error_m', 1000., 2,
            ),
            (
                .08, 'Heading RMSE (deg)',
                'heading_rmse_deg', 1., 3,
            ),
        ):
            summary.text(
                0, y, label, fontsize=9,
                color='#64748B', transform=summary.transAxes,
            )
            for x, topic in ((.66, BASELINE), (1., NOISY)):
                value = scale * metrics['estimates'][topic][name]
                summary.text(
                    x, y, f'{value:.{decimals}f}',
                    ha='right', fontsize=10.5,
                    weight='bold', transform=summary.transAxes,
                )

        time = data['time_since_first_motion_s']
        duration = (
            metrics['evaluation_end_ns']
            - metrics['evaluation_start_ns']
        ) / 1e9

        for phase in metrics['command_validation']['phases']:
            if phase['action'] in ('LEFT', 'RIGHT'):
                a = (
                    phase['start_ns']
                    - metrics['evaluation_start_ns']
                ) / 1e9
                b = (
                    phase['end_ns']
                    - metrics['evaluation_start_ns']
                ) / 1e9
                for axis in (position, heading):
                    axis.axvspan(
                        max(0., a), min(duration, b),
                        color='#CBD5E1', alpha=.55,
                        linewidth=0, zorder=0,
                    )

        heading.axhline(
            0, color='#94A3B8', linewidth=.8, zorder=1,
        )

        for topic, prefix, label, color, linestyle in ESTIMATES:
            plot_time = time
            position_error = data[f'{prefix}_position_error_m']
            heading_error = np.degrees(
                data[f'{prefix}_heading_error_rad']
            )

            if data['header_time_ns'][-1] < metrics['evaluation_end_ns']:
                endpoint_metric = metrics['estimates'][topic]
                plot_time = np.r_[time, duration]
                position_error = np.r_[
                    position_error,
                    endpoint_metric['endpoint_position_error_m'],
                ]
                heading_error = np.r_[
                    heading_error,
                    endpoint_metric['endpoint_heading_error_deg'],
                ]

            position.plot(
                plot_time, position_error,
                color=color, linestyle=linestyle,
                linewidth=1.7, label=label, zorder=3,
            )
            heading.plot(
                plot_time, heading_error,
                color=color, linestyle=linestyle,
                linewidth=1.7, zorder=3,
            )

        position.set(
            ylabel='Position error (m)', ylim=(0, None),
        )
        heading.set(ylabel='Signed heading error (deg)')

        position.set_title(
            '(b) Position error',
            loc='left', weight='bold', pad=10,
        )
        heading.set_title(
            '(c) Wrapped heading error',
            loc='left', weight='bold', pad=10,
        )

        for axis in (position, heading):
            axis.set(
                xlabel='Time from first motion (s)',
                xlim=(0, duration),
            )
            axis.xaxis.set_major_locator(
                MaxNLocator(nbins=6, integer=True)
            )
            axis.yaxis.set_major_locator(
                MaxNLocator(nbins=4)
            )

        for axis in (trajectory, position, heading):
            axis.grid(
                True, color='#E2E8F0',
                linewidth=.7, zorder=0,
            )
            axis.set_axisbelow(True)
            axis.tick_params(labelsize=9)

        config = metrics['encoder_parameters']['parameters']
        captions = [
            (
                'One recorded simulation run; paired baseline and A3 '
                '(n = 1 each). Shaded intervals: commanded turns.'
            ),
            (
                f'A3 measurement assumptions: '
                f'sigma_1m = {config["sigma_1m"]:g} m; '
                f'left bias = {100 * config["left_bias"]:+g}%; '
                f'right bias = {100 * config["right_bias"]:+g}%; '
                f'seed = {config["seed"]}. Coefficients are illustrative.'
            ),
            (
                'One fixed initial SE(2) alignment per path; ground truth '
                'is used only for evaluation. Heading errors are wrapped '
                'to [-180 deg, 180 deg].'
            ),
            (
                'A fixed seed reproduces only the same accepted input '
                'sequence; this online path need not match an earlier '
                'offline replay.'
            ),
        ]

        for y, text in zip((.110, .082, .054, .026), captions):
            fig.text(
                .075, y, text,
                fontsize=8.1, color='#64748B',
            )

        for path in paths:
            fig.savefig(path, dpi=300)
            print(f'Saved: {path}')

        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('analysis_dir', type=Path)
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Replace only existing online_evaluation figures',
    )
    args = parser.parse_args(argv)
    folder = args.analysis_dir.expanduser().resolve()

    try:
        data, metrics = load_results(folder)
        draw_results(folder, data, metrics, args.overwrite)
        print(
            'Only figure files were written; CSV, JSON and '
            'parameter snapshots are unchanged.'
        )
        return 0
    except (
        OSError, ValueError, KeyError,
        TypeError, OverflowError,
    ) as error:
        print(f'ERROR: {error}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())