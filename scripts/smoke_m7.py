"""M7 smoke: copy of NVIDIA's official clock.py ROS2 bridge sample.

Based on:
  /home/js/isaacsim/standalone_examples/api/isaacsim.ros2.bridge/clock.py

Key pattern differences from a naive AppLauncher approach:
  - Use `SimulationApp` directly (not Isaac Lab's AppLauncher)
  - Call `simulation_app.update()` BEFORE and AFTER building the OG graph so
    the ROS2 bridge gets a chance to register its node types
  - Use `isaacsim.core.api.SimulationContext` to drive the sim loop
  - Use the Omniverse-bundled `rclpy` for subscription (not system `/opt/ros/jazzy`)

External verification from another shell:
  source /opt/ros/jazzy/setup.bash
  ros2 topic list    # should include /sim_time and /manual_time
"""

from __future__ import annotations

import time

from isaacsim import SimulationApp

simulation_app = SimulationApp({"renderer": "RaytracedLighting", "headless": True})

import omni.graph.core as og  # noqa: E402
from isaacsim.core.api import SimulationContext  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

# Note: this imports the Omniverse-bundled rclpy, not the system rclpy.
import rclpy  # noqa: E402
from rosgraph_msgs.msg import Clock  # noqa: E402

rclpy.init()
clock_topic = "sim_time"
manual_clock_topic = "manual_time"

try:
    og.Controller.edit(
        {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
                (
                    "ReadSimTime.outputs:simulationTime",
                    "PublishClock.inputs:timeStamp",
                ),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("PublishClock.inputs:topicName", clock_topic),
            ],
        },
    )
    print(">>> graph built OK", flush=True)
except Exception as e:
    print(f">>> graph build FAILED: {type(e).__name__}: {e}", flush=True)
    raise

simulation_app.update()
simulation_app.update()


received = {"count": 0, "last_sec": None}


def sim_clock_callback(data):
    received["count"] += 1
    received["last_sec"] = data.clock.sec + data.clock.nanosec * 1e-9
    if received["count"] <= 3 or received["count"] % 20 == 0:
        print(f">>> /sim_time msg #{received['count']} t={received['last_sec']:.2f}", flush=True)


node = rclpy.create_node("isaac_sim_clock")
sub = node.create_subscription(Clock, clock_topic, sim_clock_callback, 1)

time.sleep(1.0)
sim_ctx = SimulationContext(physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0, stage_units_in_meters=1.0)
sim_ctx.initialize_physics()
sim_ctx.play()

# Step the sim enough frames that the subscriber has a chance to fire.
for frame in range(60):
    sim_ctx.step(render=False)
    rclpy.spin_once(node, timeout_sec=0.0)
    time.sleep(0.05)

for frame in range(30):
    simulation_app.update()
    rclpy.spin_once(node, timeout_sec=0.0)
    time.sleep(0.05)

print(f">>> received {received['count']} /sim_time messages, last_t={received['last_sec']}", flush=True)
if received["count"] > 0:
    print("M7_SMOKE_OK", flush=True)
else:
    print("M7_SMOKE_FAIL: no clock messages received", flush=True)

rclpy.shutdown()
sim_ctx.stop()
simulation_app.close()
