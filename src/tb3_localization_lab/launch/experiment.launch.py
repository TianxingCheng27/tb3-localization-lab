"""Launch the simulation, ground-truth bridge and wheel odometry."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import AppendEnvironmentVariable
from launch.actions import IncludeLaunchDescription
from launch.actions import SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    # Locate the installed packages.
    lab_share = Path(get_package_share_directory('tb3_localization_lab'))
    tb3_share = Path(get_package_share_directory('turtlebot3_gazebo'))
    ros_gz_share = Path(get_package_share_directory('ros_gz_sim'))

    # Locate our copied world, modified model and bridge configuration.
    world_file = lab_share / 'worlds' / 'odometry_test.world'
    model_file = lab_share / 'models' / 'burger_ground_truth' / 'model.sdf'
    bridge_file = lab_share / 'config' / 'turtlebot3_bridge.yaml'

    # Read the installed robot description.
    urdf_file = tb3_share / 'urdf' / 'turtlebot3_burger.urdf'
    robot_description = urdf_file.read_text(encoding='utf-8')
    gz_launch_file = str(ros_gz_share / 'launch' / 'gz_sim.launch.py')

    return LaunchDescription([
        SetEnvironmentVariable('TURTLEBOT3_MODEL', 'burger'),

        # Allow Gazebo to find the original TurtleBot3 meshes.
        AppendEnvironmentVariable(
            'GZ_SIM_RESOURCE_PATH',
            str(tb3_share / 'models'),
        ),

        # Start the Gazebo simulation server.
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(gz_launch_file),
            launch_arguments={
                'gz_args': ['-r -s -v2 ', str(world_file)],
                'on_exit_shutdown': 'true',
            }.items(),
        ),

        # Start the Gazebo graphical interface.
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(gz_launch_file),
            launch_arguments={'gz_args': '-g -v2 '}.items(),
        ),

        # Spawn one robot using our modified model.
        Node(
            package='ros_gz_sim',
            executable='create',
            arguments=[
                '-name', 'burger',
                '-file', str(model_file),
                '-x', '0.0',
                '-y', '0.0',
                '-z', '0.01',
            ],
            output='screen',
        ),

        # Bridge the original topics and the new ground-truth topic.
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=[
                '--ros-args',
                '-p', f'config_file:={bridge_file}',
            ],
            output='screen',
        ),

        # Publish the robot's link transforms from its joint states.
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[{
                'use_sim_time': True,
                'robot_description': ParameterValue(
                    robot_description, value_type=str,
                ),
            }],
            output='screen',
        ),

        # Start our own wheel-based pose estimator.
        Node(
            package='tb3_localization_lab',
            executable='wheel_odometry',
            name='wheel_odometry',
            parameters=[{
                'use_sim_time': True,
                'wheel_radius': 0.033,
                'wheel_separation': 0.160,
            }],
            output='screen',
        ),
    ])
