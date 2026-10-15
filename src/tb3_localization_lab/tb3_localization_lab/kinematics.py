"""Wheel odometry math. Distances: metres; angles: radians."""
import math


def wheel_motion(delta_left, delta_right, radius=0.033, separation=0.160):
    # Convert wheel angle changes into wheel travel distances.
    left_distance = radius * delta_left
    right_distance = radius * delta_right
    # Average travel gives translation; the difference gives rotation.
    distance = (left_distance + right_distance) / 2.0
    turn = (right_distance - left_distance) / separation
    return distance, turn


def integrate_pose(x, y, yaw, distance, turn):
    # Assume constant curvature during this sampling interval.
    half_turn = turn / 2.0
    # Handle a near-zero turn without dividing by zero.
    if abs(half_turn) < 1e-6:
        scale = 1.0 - half_turn * half_turn / 6.0
    else:
        scale = math.sin(half_turn) / half_turn
    # Project chord displacement using the middle heading.
    direction = yaw + half_turn
    x += distance * scale * math.cos(direction)
    y += distance * scale * math.sin(direction)
    # Normalize heading to [-pi, pi].
    yaw = math.atan2(math.sin(yaw + turn), math.cos(yaw + turn))
    return x, y, yaw


if __name__ == '__main__':
    # Each example starts independently at the origin.
    for label, left, right in [
        ('straight', 1.0, 1.0),
        ('left_turn', -1.0, 1.0),
    ]:
        distance, turn = wheel_motion(left, right)
        x, y, yaw = integrate_pose(0.0, 0.0, 0.0, distance, turn)
        print(f'{label}: x={x:.6f} m, y={y:.6f} m, yaw={yaw:.6f} rad')
