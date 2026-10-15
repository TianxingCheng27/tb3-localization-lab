"""Redraw saved baseline results without opening the bag or changing metrics.

Input: comparison.csv and metrics.json produced by analyse_run.py.
Output: baseline_evaluation.png and baseline_evaluation.pdf in the same folder.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


GT = '/ground_truth/pose'
ODOM = '/odom_est'
COLUMNS = ('time_since_first_motion_s', 'gt_x_m', 'gt_y_m', 'gt_yaw_rad',
           'odom_x_m', 'odom_y_m', 'odom_yaw_rad',
           'position_error_m', 'heading_error_rad')


def load_results(folder):
    """Check that the two saved result files describe the same evaluation."""
    with (folder / 'metrics.json').open(encoding='utf-8') as stream:
        metrics = json.load(stream)
    with (folder / 'comparison.csv').open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        if not set(COLUMNS).issubset(reader.fieldnames or []):
            raise ValueError('comparison.csv is missing required analysis columns.')
        rows = [[float(row[name]) for name in COLUMNS] for row in reader]
    values = np.asarray(rows, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('At least two finite comparison samples are required.')
    data = dict(zip(COLUMNS, values.T))
    time = data['time_since_first_motion_s']
    if time[0] < 0 or np.any(np.diff(time) <= 0):
        raise ValueError('Comparison times must be nonnegative and strictly increasing.')
    if len(time) != metrics['evaluation_samples']:
        raise ValueError('CSV and JSON sample counts do not match.')
    checks = (
        (np.sqrt(np.mean(data['position_error_m'] ** 2)), metrics['position_rmse_m']),
        (np.degrees(np.sqrt(np.mean(data['heading_error_rad'] ** 2))),
         metrics['heading_rmse_deg']),
    )
    if not all(np.isclose(actual, saved, rtol=1e-9, atol=1e-12)
               for actual, saved in checks):
        raise ValueError('CSV errors and JSON metrics do not match.')
    if 'command_validation' not in metrics:
        raise ValueError('metrics.json must include the recorded command phases.')
    return data, metrics


def draw_results(folder, data, metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    style = {
        'font.family': 'DejaVu Sans', 'font.size': 9,
        'axes.labelsize': 10, 'axes.titlesize': 11,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': '#64748B', 'axes.linewidth': 0.8,
        'xtick.color': '#334155', 'ytick.color': '#334155',
        'text.color': '#172033', 'axes.labelcolor': '#172033',
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'figure.facecolor': 'white', 'savefig.facecolor': 'white',
    }
    with plt.rc_context(style):
        fig = plt.figure(figsize=(10.8, 7.4))
        fig.text(0.075, 0.955, 'Baseline Wheel Odometry', fontsize=18, weight='bold')
        fig.text(0.075, 0.917, 'Ground truth comparison | Three straights and two turns',
                 fontsize=10, color='#64748B')

        # A wide upper panel preserves the true spatial aspect ratio.
        trajectory = fig.add_axes([0.075, 0.465, 0.62, 0.405])
        summary = fig.add_axes([0.735, 0.465, 0.225, 0.405])
        position = fig.add_axes([0.075, 0.145, 0.39, 0.225])
        heading = fig.add_axes([0.57, 0.145, 0.39, 0.225])
        summary.set_axis_off()

        gt_end = np.asarray(metrics['aligned_endpoint_x_m_y_m_yaw_rad'][GT])
        od_end = np.asarray(metrics['aligned_endpoint_x_m_y_m_yaw_rad'][ODOM])
        gt_line, = trajectory.plot(
            np.r_[0., data['gt_x_m'], gt_end[0]],
            np.r_[0., data['gt_y_m'], gt_end[1]],
            color='#172033', linewidth=2.2, label='Ground truth', zorder=2)
        od_line, = trajectory.plot(
            np.r_[0., data['odom_x_m'], od_end[0]],
            np.r_[0., data['odom_y_m'], od_end[1]],
            color='#1976B9', linewidth=1.8, linestyle=(0, (5, 3)),
            label='Wheel odometry', zorder=3)
        start = trajectory.scatter(0, 0, s=40, color='#24966D',
                                   edgecolors='white', linewidths=0.6, zorder=5)
        gt_marker = trajectory.scatter(*gt_end[:2], marker='s', s=38,
                                       color='#172033', zorder=4)
        od_marker = trajectory.scatter(*od_end[:2], marker='x', s=55,
                                       color='#1976B9', linewidths=1.8, zorder=5)
        trajectory.set(xlabel='x (m)', ylabel='y (m)')
        trajectory.set_title('(a) Planar trajectory', loc='left', weight='bold', pad=10)
        trajectory.set_aspect('equal', adjustable='box', anchor='W')
        trajectory.margins(x=0.055, y=0.08)
        trajectory.xaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))
        trajectory.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))

        summary.legend([gt_line, od_line, start, gt_marker, od_marker],
                       ['Ground truth', 'Wheel odometry', 'Aligned start',
                        'GT endpoint', 'Odometry endpoint'],
                       loc='upper left', bbox_to_anchor=(0, 1.02),
                       frameon=False, borderaxespad=0, handlelength=2.4,
                       labelspacing=0.65, fontsize=9)
        summary.plot([0, 1], [0.58, 0.58], color='#E2E8F0', linewidth=1,
                     transform=summary.transAxes)
        for y, label, value in (
            (0.49, 'Position RMSE', f"{1000 * metrics['position_rmse_m']:.2f} mm"),
            (0.29, 'Endpoint position error',
             f"{1000 * metrics['endpoint_position_error_m']:.2f} mm"),
            (0.09, 'Heading RMSE', f"{metrics['heading_rmse_deg']:.3f}\N{DEGREE SIGN}"),
        ):
            summary.text(0, y, label, color='#64748B', fontsize=9,
                         transform=summary.transAxes)
            summary.text(0, y - 0.088, value, fontsize=16, weight='bold',
                         transform=summary.transAxes)

        # Append the precise common endpoint rather than using the last GT sample.
        time = data['time_since_first_motion_s']
        position_mm = 1000 * data['position_error_m']
        heading_deg = np.degrees(data['heading_error_rad'])
        duration = (metrics['evaluation_end_ns'] - metrics['evaluation_start_ns']) / 1e9
        if time[-1] > duration + 1e-9:
            raise ValueError('CSV extends beyond the JSON evaluation end time.')
        if time[-1] < duration:
            time = np.r_[time, duration]
            position_mm = np.r_[position_mm, 1000 * metrics['endpoint_position_error_m']]
            heading_deg = np.r_[heading_deg, metrics['endpoint_heading_error_deg']]

        for phase in metrics['command_validation']['phases']:
            if phase['action'] in ('LEFT', 'RIGHT'):
                a = (phase['start_ns'] - metrics['evaluation_start_ns']) / 1e9
                b = (phase['end_ns'] - metrics['evaluation_start_ns']) / 1e9
                for axis in (position, heading):
                    axis.axvspan(max(0., a), min(duration, b),
                                 color='#CBD5E1', alpha=0.55, linewidth=0, zorder=0)
        position.plot(time, position_mm, color='#C8483C', linewidth=1.8, zorder=3)
        heading.axhline(0, color='#94A3B8', linewidth=0.8, zorder=1)
        heading.plot(time, heading_deg, color='#7255A4', linewidth=1.8, zorder=3)
        position.set(ylabel='Position error (mm)', ylim=(0, None))
        heading.set(ylabel='Heading error (\N{DEGREE SIGN})')
        position.set_title('(b) Position error', loc='left', weight='bold', pad=10)
        heading.set_title('(c) Heading error', loc='left', weight='bold', pad=10)
        for axis in (position, heading):
            axis.set(xlabel='Time from first motion (s)', xlim=(0, duration))
            axis.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
            axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
        for axis in (trajectory, position, heading):
            axis.grid(True, color='#E2E8F0', linewidth=0.7, zorder=0)
            axis.set_axisbelow(True)
            axis.tick_params(labelsize=9)
        fig.text(0.075, 0.060, 'Shaded intervals: left and right turn commands.',
                 fontsize=8.5, color='#64748B')
        fig.text(0.075, 0.030,
                 'One fixed initial pose alignment per trajectory; no added wheel noise. '
                 'Heading error is wrapped to [-180\N{DEGREE SIGN}, 180\N{DEGREE SIGN}].',
                 fontsize=8, color='#64748B')
        for suffix in ('png', 'pdf'):
            path = folder / f'baseline_evaluation.{suffix}'
            fig.savefig(path, dpi=300)
            print(f'Saved: {path}')
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('analysis_dir', type=Path,
                        help='Folder containing comparison.csv and metrics.json')
    args = parser.parse_args()
    folder = args.analysis_dir.expanduser().resolve()
    try:
        data, metrics = load_results(folder)
        draw_results(folder, data, metrics)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f'ERROR: {error}\n')
    print('Only new figure files were written; CSV, JSON and the bag are unchanged.')


if __name__ == '__main__':
    main()
