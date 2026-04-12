"""IMU sensor helpers.

Isaac Lab does not ship a built-in IMU sensor class; IMU readings are typically
computed from rigid body state each step (yaw rate, linear accel). This module
will hold the common post-processing helpers. Stub for now — filled in during M2.
"""

from __future__ import annotations


def simulate_imu(
    root_state_w,
    prev_lin_vel_w,
    dt: float,
    gyro_noise_std: float = 0.01,
    accel_noise_std: float = 0.05,
):
    """Placeholder. Will return (yaw_rate, ax, ay, bias) with noise."""
    raise NotImplementedError("Implemented in M2 (IMU integration for F1Tenth port).")
