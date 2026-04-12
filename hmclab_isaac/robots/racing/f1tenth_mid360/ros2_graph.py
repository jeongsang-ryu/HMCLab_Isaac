"""F1Tenth-Mid360 standard ROS2 OmniGraph (stub).

When implemented (M7), `attach()` will build a bridge graph that publishes:
  - `/tf`                 — TransformTree (chassis + wheels + sensor frames)
  - `/scan`               — Mid-360 as sensor_msgs/PointCloud2
  - `/image_raw`          — front camera RGB as sensor_msgs/Image
  - `/odom`               — base_link odometry
  - `/clock`              — sim clock

And subscribes:
  - `/cmd_vel`            — geometry_msgs/Twist (optional manual control)

Intended use: call from eval envs (`envs/racing/eval/*`), never from
training envs (ROS bridge overhead kills parallel throughput).
"""

from __future__ import annotations


def attach(robot_prim_path: str, ns: str = "") -> None:
    """Build and attach the standard F1Tenth ROS2 graph.

    Args:
        robot_prim_path: Prim path of the spawned robot (not a regex).
        ns: Optional ROS namespace prefix (e.g. "ego", "opp" for MARL eval).
    """
    raise NotImplementedError("M7 deliverable. See hmclab_isaac/utils/ros2.py.")
