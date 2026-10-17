"""Show baseline residual errors from the current online experiment.

Keep plot_online.py beside this script.
Outputs: baseline_detail.png and baseline_detail.pdf.
"""

import argparse
from pathlib import Path

import numpy as np

import plot_online


BASELINE = '/odom_est'


def draw_results(folder, data, metrics, overwrite=False):
    folder = Path(folder)
    paths = [
        folder / f'baseline_detail.{suffix}'
        for suffix in ('png', 'pdf')
    ]

    if not overwrite and any(path.exists() for path in paths):
        raise ValueError(
            'Baseline detail figures already exist; '
            'use --overwrite to replace them.'
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
        'axes.linewidth': 0.8,
        'text.color': '#172033',
        'axes.labelcolor': '#172033',
        'xtick.color': '#334155',
        'ytick.color': '#334155',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
    }

    with plt.rc_context(style):
        fig = plt.figure(figsize=(10.8, 7.8))

        fig.text(
            0.075, 0.955,
            'Baseline Wheel Odometry: Residual Error',
            fontsize=17,
            weight='bold',
        )

        bag_name = Path(metrics['source_bag']).name
        fig.text(
            0.075, 0.918,
            f'No added wheel measurement error | {bag_name}',
            fontsize=10,
            color='#64748B',
        )

        trajectory = fig.add_axes([0.075, 0.475, 0.62, 0.385])
        summary = fig.add_axes([0.735, 0.475, 0.225, 0.385])
        position = fig.add_axes([0.075, 0.185, 0.39, 0.210])
        heading = fig.add_axes([0.57, 0.185, 0.39, 0.210])
        summary.set_axis_off()

        metric = metrics['estimates'][BASELINE]
        gt_end = np.asarray(
            metrics['aligned_gt_endpoint_x_m_y_m_yaw_rad']
        )
        endpoint = np.asarray(
            metric['aligned_endpoint_x_m_y_m_yaw_rad']
        )

        # Use the exact final poses stored in metrics.json.
        gt_line, = trajectory.plot(
            np.r_[0.0, data['gt_x_m'], gt_end[0]],
            np.r_[0.0, data['gt_y_m'], gt_end[1]],
            color='#172033',
            linewidth=2.2,
            zorder=2,
        )

        baseline_line, = trajectory.plot(
            np.r_[0.0, data['baseline_x_m'], endpoint[0]],
            np.r_[0.0, data['baseline_y_m'], endpoint[1]],
            color='#1976B9',
            linestyle=(0, (5, 3)),
            linewidth=1.8,
            zorder=3,
        )

        start = trajectory.scatter(
            0, 0,
            s=40,
            color='#24966D',
            edgecolors='white',
            linewidths=0.6,
            zorder=5,
        )

        gt_marker = trajectory.scatter(
            *gt_end[:2],
            marker='s',
            s=38,
            color='#172033',
            zorder=4,
        )

        baseline_marker = trajectory.scatter(
            *endpoint[:2],
            marker='x',
            s=55,
            color='#1976B9',
            linewidths=1.8,
            zorder=5,
        )

        trajectory.set(
            xlabel='x in initial frame (m)',
            ylabel='y in initial frame (m)',
        )
        trajectory.set_title(
            '(a) Baseline and ground truth',
            loc='left',
            weight='bold',
            pad=10,
        )
        trajectory.set_aspect('equal', adjustable='box', anchor='W')
        trajectory.margins(x=0.055, y=0.10)
        trajectory.xaxis.set_major_locator(
            MaxNLocator(nbins=6, steps=[1, 2, 5, 10])
        )
        trajectory.yaxis.set_major_locator(
            MaxNLocator(nbins=5, steps=[1, 2, 5, 10])
        )

        summary.legend(
            [
                gt_line,
                baseline_line,
                start,
                gt_marker,
                baseline_marker,
            ],
            [
                'Ground truth',
                'Baseline odometry',
                'Aligned start',
                'GT endpoint',
                'Baseline endpoint',
            ],
            loc='upper left',
            bbox_to_anchor=(0, 1.02),
            frameon=False,
            borderaxespad=0,
            handlelength=2.5,
            labelspacing=0.65,
            fontsize=9,
        )

        summary.plot(
            [0, 1], [0.57, 0.57],
            color='#E2E8F0',
            linewidth=1,
            transform=summary.transAxes,
        )

        metric_rows = (
            (
                0.48,
                'Position RMSE',
                f'{1000 * metric["position_rmse_m"]:.3f} mm',
            ),
            (
                0.28,
                'Endpoint position error',
                f'{1000 * metric["endpoint_position_error_m"]:.3f} mm',
            ),
            (
                0.08,
                'Heading RMSE',
                f'{metric["heading_rmse_deg"]:.4f} deg',
            ),
        )

        for y, label, value in metric_rows:
            summary.text(
                0, y, label,
                fontsize=9,
                color='#64748B',
                transform=summary.transAxes,
            )
            summary.text(
                0, y - 0.085, value,
                fontsize=15,
                weight='bold',
                transform=summary.transAxes,
            )

        # Translate both endpoints by the same GT endpoint.
        # The inset displays the true positional difference in millimetres.
        offset_mm = 1000 * (endpoint[:2] - gt_end[:2])
        bound = max(
            1.0,
            1.45 * float(np.max(np.abs(offset_mm))),
        )

        inset = trajectory.inset_axes([0.075, 0.43, 0.32, 0.48])

        inset.plot(
            [0.0, offset_mm[0]],
            [0.0, offset_mm[1]],
            color='#94A3B8',
            linewidth=0.9,
        )
        inset.scatter(
            0, 0,
            marker='s',
            s=32,
            color='#172033',
            zorder=3,
        )
        inset.scatter(
            *offset_mm,
            marker='x',
            s=45,
            color='#1976B9',
            linewidths=1.6,
            zorder=4,
        )

        inset.set(
            xlim=(-bound, bound),
            ylim=(-bound, bound),
            xlabel='Delta x (mm)',
            ylabel='Delta y (mm)',
        )
        inset.set_title(
            'Endpoint offset from GT',
            fontsize=8.2,
            pad=5,
        )
        inset.set_aspect('equal', adjustable='box', anchor='W')
        inset.xaxis.set_major_locator(MaxNLocator(nbins=3))
        inset.yaxis.set_major_locator(MaxNLocator(nbins=3))
        inset.tick_params(labelsize=7)
        inset.xaxis.label.set_size(7.5)
        inset.yaxis.label.set_size(7.5)
        inset.set_facecolor('#F8FAFC')
        inset.grid(True, color='#E2E8F0', linewidth=0.6)

        time = data['time_since_first_motion_s']
        position_mm = 1000 * data['baseline_position_error_m']
        heading_deg = np.degrees(data['baseline_heading_error_rad'])

        duration = (
            metrics['evaluation_end_ns']
            - metrics['evaluation_start_ns']
        ) / 1e9

        # Include the exact evaluation endpoint when it follows the last CSV row.
        if data['header_time_ns'][-1] < metrics['evaluation_end_ns']:
            time = np.r_[time, duration]
            position_mm = np.r_[
                position_mm,
                1000 * metric['endpoint_position_error_m'],
            ]
            heading_deg = np.r_[
                heading_deg,
                metric['endpoint_heading_error_deg'],
            ]

        # Mark the recorded left-turn and right-turn command intervals.
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
                        max(0.0, a),
                        min(duration, b),
                        color='#CBD5E1',
                        alpha=0.55,
                        linewidth=0,
                        zorder=0,
                    )

        position.plot(
            time, position_mm,
            color='#1976B9',
            linewidth=1.8,
            zorder=3,
        )

        heading.axhline(
            0,
            color='#94A3B8',
            linewidth=0.8,
            zorder=1,
        )
        heading.plot(
            time, heading_deg,
            color='#1976B9',
            linewidth=1.8,
            zorder=3,
        )

        position.set(
            ylabel='Position error (mm)',
            ylim=(0, None),
        )
        heading.set(ylabel='Signed heading error (deg)')

        position.set_title(
            '(b) Baseline position error',
            loc='left',
            weight='bold',
            pad=10,
        )
        heading.set_title(
            '(c) Baseline heading error',
            loc='left',
            weight='bold',
            pad=10,
        )

        for axis in (position, heading):
            axis.set(
                xlabel='Time from first motion (s)',
                xlim=(0, duration),
            )
            axis.xaxis.set_major_locator(
                MaxNLocator(nbins=6, integer=True)
            )
            axis.yaxis.set_major_locator(MaxNLocator(nbins=4))

        for axis in (trajectory, position, heading):
            axis.grid(
                True,
                color='#E2E8F0',
                linewidth=0.7,
                zorder=0,
            )
            axis.set_axisbelow(True)
            axis.tick_params(labelsize=9)

        captions = (
            (
                0.105,
                'Current recorded simulation run; baseline has no added '
                'wheel measurement error. Shaded intervals: commanded turns.',
            ),
            (
                0.077,
                'Inset: exact endpoints translated to the GT endpoint '
                '(origin); true offsets are shown in mm. '
                'Trajectory coordinates are unchanged.',
            ),
            (
                0.049,
                'One fixed initial SE(2) alignment per trajectory; '
                'GT is used only for evaluation. Heading error is '
                'wrapped to [-180 deg, 180 deg].',
            ),
        )

        for y, text in captions:
            fig.text(
                0.075, y, text,
                fontsize=8,
                color='#64748B',
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
        help='Replace only existing baseline_detail figures',
    )
    args = parser.parse_args(argv)
    folder = args.analysis_dir.expanduser().resolve()

    try:
        data, metrics = plot_online.load_results(folder)
        draw_results(folder, data, metrics, args.overwrite)
        print(
            'Only baseline_detail figures were written; '
            'existing overview and numerical inputs are unchanged.'
        )
        return 0
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        OverflowError,
    ) as error:
        print(f'ERROR: {error}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())