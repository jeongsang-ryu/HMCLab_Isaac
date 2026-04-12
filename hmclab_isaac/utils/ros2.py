"""ROS2 bridge helpers (Python-driven OmniGraph).

Usage from an env's `_setup_scene` method:

    from hmclab_isaac.utils.ros2 import attach_ros2_graph
    attach_ros2_graph(
        robot_prim_path="/World/envs/env_0/Ego",
        topics={"lidar": "/scan", "camera": "/image_raw"},
    )

Stub for now — filled in during M7.
"""

from __future__ import annotations


def attach_ros2_graph(robot_prim_path: str, topics: dict[str, str]) -> None:
    """Build a ROS2 bridge OmniGraph attached to the given robot.

    Wraps `omni.graph.core` / `isaacsim.ros2.bridge` node creation so envs
    never touch OmniGraph directly. Implemented in M7.
    """
    raise NotImplementedError("M7 deliverable — ROS2 OmniGraph factory not yet written.")
