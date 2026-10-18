# TurtleBot3 Wheel Odometry Localisation

This project estimates the relative planar pose of a TurtleBot3 Burger from wheel joint angles. It compares baseline odometry with odometry using corrupted wheel measurements. Gazebo ground truth is used only for evaluation.

## Submission contents

- `Assignment 1 Report.pdf`: one-page report.
- `results/online_A3_seed0_run_01/online_evaluation.png` and `.pdf`: paired baseline/A3 comparison.
- `results/online_A3_seed0_run_01/baseline_detail.png` and `.pdf`: baseline residual errors and endpoint offset.
- `results/reference_run_01/uncertainty/`: offline condition summaries and per-run metrics.
- `src/tb3_localization_lab/`: ROS package, simulation configuration and analysis scripts.
- `.git/`: local development history. Inspect it with `git log --oneline`.

## Environment and build

Tested environment: Ubuntu 24.04, ROS 2 Jazzy and Gazebo Harmonic.

Required components include `turtlebot3_gazebo`, `ros_gz_sim`, `ros_gz_bridge`, `robot_state_publisher`, `rclpy`, `rosbag2_py`, MCAP storage support, NumPy, Matplotlib and PyYAML.

The robot uses meshes from the installed TurtleBot3 package. Gazebo Fuel Ground Plane and Sun assets must be cached or accessible during loading.

Commands below assume the project is located at `~/tb3_localization_ws`. Adjust this path if necessary.

```bash
cd ~/tb3_localization_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-select tb3_localization_lab
source install/setup.bash
```

In every additional terminal, change to the project directory and source both ROS and the workspace setup files. Build only once.

## Run a new online experiment

Use a fresh simulation and a new recording name for each run.

### Terminal A: Simulation

```bash
ros2 launch tb3_localization_lab online_uncertainty.launch.py seed:=0 sigma_1m:=0.002 left_bias:=-0.005 right_bias:=0.005
```

Wait until the robot and ROS nodes are ready. This launches one simulation with two odometry branches:

- `/joint_states` -> baseline estimator -> `/odom_est`
- `/joint_states` -> encoder error model -> `/joint_states_noisy` -> second estimator -> `/odom_noisy`

Both estimators use the same odometry algorithm. Measurement corruption does not change the simulated robot motion.

### Terminal B: Recording

Save the encoder parameters and start recording before running the trajectory:

```bash
mkdir -p "$HOME/tb3_experiment_data/bags" "$HOME/tb3_experiment_data/configs"
ros2 param dump /encoder_model > "$HOME/tb3_experiment_data/configs/reviewer_run_01.yaml"
ros2 bag record --use-sim-time -s mcap -o "$HOME/tb3_experiment_data/bags/reviewer_run_01" --topics /odom_est /odom_noisy /ground_truth/pose /joint_states /joint_states_noisy /cmd_vel /clock
```

### Terminal C: Trajectory

```bash
python3 src/tb3_localization_lab/tb3_localization_lab/trajectory_driver.py --ros-args -p use_sim_time:=true
```

The trajectory contains three nominal 2 m straight segments and two nominal 90-degree turns.

After the driver prints "Sequence complete", stop the recorder in Terminal B with Ctrl+C and wait for recording finalisation. Then stop the simulation in Terminal A.

## Analyse the new online recording

Run from the project directory with the ROS and workspace environments sourced. Use a new analysis directory if repeating the analysis.

```bash
python3 src/tb3_localization_lab/tools/inspect_bag.py "$HOME/tb3_experiment_data/bags/reviewer_run_01"
python3 src/tb3_localization_lab/tools/analyse_online.py "$HOME/tb3_experiment_data/bags/reviewer_run_01" --params-file "$HOME/tb3_experiment_data/configs/reviewer_run_01.yaml" --output-dir "$HOME/tb3_experiment_data/analysis/reviewer_run_01/online_comparison"
python3 src/tb3_localization_lab/tools/plot_online.py "$HOME/tb3_experiment_data/analysis/reviewer_run_01/online_comparison"
python3 src/tb3_localization_lab/tools/plot_baseline_detail.py "$HOME/tb3_experiment_data/analysis/reviewer_run_01/online_comparison"
```

## Offline uncertainty replay

Offline replay reads recorded `/joint_states`, perturbs wheel-travel increments, and recomputes the complete odometry trajectory. It does not modify the final estimated positions directly.

Conditions are:

- A0: no additional measurement error.
- A1: random measurement noise only.
- A2: fixed wheel scale biases only.
- A3: random noise and scale biases together.

A1 and A3 each use 20 seeds, numbered 0 to 19, with paired noise draws. These trials share one recorded trajectory.

The following example requires the original `reference_run_01` bag. Regenerating its baseline analysis also updates saved paths when the recordings have been relocated:

```bash
python3 src/tb3_localization_lab/tools/analyse_run.py "$HOME/tb3_experiment_data/bags/reference_run_01" --output-dir "$HOME/tb3_experiment_data/analysis/reference_run_01_review"
python3 src/tb3_localization_lab/tools/uncertainty_replay.py "$HOME/tb3_experiment_data/bags/reference_run_01" "$HOME/tb3_experiment_data/analysis/reference_run_01_review" --seeds 20 --first-seed 0 --sigma-1m 0.002 --bias-left=-0.005 --bias-right=0.005
```

A0 is first checked against the recorded online baseline. This tests replay consistency, not ground-truth accuracy.

## Measurement model and evaluation

Wheel radius is 0.033 m and wheel separation is 0.160 m. The estimator uses wheel-angle differences and constant-curvature pose integration.

For each wheel and travel increment:

```text
ds_measured = (1 + bias) * ds
           + sigma_1m * sqrt(abs(ds) / (1 m)) * z
```

Here, `z` is an independent standard-normal draw for each wheel and increment. Illustrative A3 settings are:

- `sigma_1m = 0.002 m`
- Left scale bias: `-0.005` (-0.5%)
- Right scale bias: `+0.005` (+0.5%)
- Online seed: `0`

`sigma_1m` specifies the accumulated random-noise standard deviation after 1 m of wheel travel under this model. It is not a fixed 2 mm error added at every sample.

Evaluation uses message-header timestamps and one initial planar pose alignment per trajectory, without trajectory fitting. The comparison assumes the same planar reference point. Ground truth never enters either estimator.

## Saved results and reproducibility

The recorded online run gave:

- Baseline: position RMSE 6.84 mm, endpoint error 5.77 mm, heading RMSE 0.098 degrees.
- A3: position RMSE 327.23 mm, endpoint error 710.78 mm, heading RMSE 10.768 degrees.

Offline position RMSEs were A0: 7.30 mm; A1: 55.33 +/- 44.86 mm; A2: 375.87 mm; A3: 365.16 +/- 66.50 mm. For stochastic conditions, values are means +/- sample standard deviations over 20 seeds.

The online and offline results use different recorded simulation runs.

## Main implementation files

- `tb3_localization_lab/kinematics.py`: wheel motion and pose integration.
- `tb3_localization_lab/wheel_odometry.py`: ROS wheel-odometry estimator.
- `tb3_localization_lab/encoder_model.py`: online wheel measurement corruption.
- `tb3_localization_lab/trajectory_driver.py`: predefined motion sequence.
- `launch/`: simulation and online uncertainty launch files.
- `tools/`: bag inspection, numerical evaluation, replay and plotting.

These paths are relative to `src/tb3_localization_lab/`.
