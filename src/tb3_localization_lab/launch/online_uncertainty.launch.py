"""Run one simulation with baseline and corrupted-wheel odometry together.

The encoder model changes measurements only. Neither estimator receives ground
truth, and the same existing wheel-odometry executable is used for both paths.
Start the recorder before running the existing trajectory driver separately.
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    lab_share = Path(get_package_share_directory('tb3_localization_lab'))
    baseline_launch = lab_share / 'launch' / 'experiment.launch.py'

    def value(name, value_type):
        return ParameterValue(LaunchConfiguration(name), value_type=value_type)

    return LaunchDescription([
        DeclareLaunchArgument(
            'seed', default_value='0',
            description='Nonnegative noise seed; reproducible for the same wheel input sequence.',
        ),
        DeclareLaunchArgument(
            'sigma_1m', default_value='0.002',
            description='Assumed per-wheel accumulated noise SD after 1 m of travel (m).',
        ),
        DeclareLaunchArgument(
            'left_bias', default_value='-0.005',
            description='Fractional left-wheel measurement scale bias.',
        ),
        DeclareLaunchArgument(
            'right_bias', default_value='0.005',
            description='Fractional right-wheel measurement scale bias.',
        ),

        # Start the extra measurement/estimation chain before starting Gazebo.
        Node(
            package='tb3_localization_lab',
            executable='encoder_model',
            name='encoder_model',
            parameters=[{
                'use_sim_time': True,
                'wheel_radius': 0.033,
                'sigma_1m': value('sigma_1m', float),
                'left_bias': value('left_bias', float),
                'right_bias': value('right_bias', float),
                'seed': value('seed', int),
                'max_source_gap': 0.1,
            }],
            output='screen',
            on_exit=[Shutdown(reason='Encoder model stopped; restart the full experiment.')],
        ),
        Node(
            package='tb3_localization_lab',
            executable='wheel_odometry',
            name='wheel_odometry_noisy',
            parameters=[{
                'use_sim_time': True,
                'wheel_radius': 0.033,
                'wheel_separation': 0.160,
            }],
            remappings=[
                ('/joint_states', '/joint_states_noisy'),
                ('/odom_est', '/odom_noisy'),
            ],
            output='screen',
            on_exit=[Shutdown(reason='Noisy odometry stopped; restart the full experiment.')],
        ),

        # This starts exactly one world, bridge, robot and baseline estimator.
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(baseline_launch)),
        ),
    ])