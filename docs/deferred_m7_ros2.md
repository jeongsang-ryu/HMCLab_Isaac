# M7 ROS2 Bridge — DEFERRED

**Status**: blocked pending environment rework. Not included in `smoke_all.py`.

## Summary

Isaac Sim 5.1 (pip, Python 3.11) can boot its `isaacsim.ros2.bridge` extension
and load OmniGraph ROS2 publisher nodes, but `import rclpy` inside the
SimulationApp process fails with:

```
Attempting to load system rclpy
Could not import system rclpy: No module named 'rclpy._rclpy_pybind11'
Attempting to load internal rclpy for ROS Distro: jazzy
Could not import internal rclpy: No module named 'rclpy._rclpy_pybind11'
```

Without rclpy the sim-to-ROS side can publish (via OmniGraph), but the
Python side can't subscribe/inspect, and our `utils/ros2.py` wrapper can't
build a sensible Python API on top.

## Root cause

`/opt/ros/jazzy` on Ubuntu 24.04 is built for **Python 3.12**:
```
/opt/ros/jazzy/lib/python3.12/site-packages/_rclpy_pybind11.cpython-*.so
```

Isaac Sim requires **Python 3.11**. System rclpy therefore cannot be loaded.

The Isaac-Sim-bundled jazzy rclpy (at
`isaacsim/exts/isaacsim.ros2.bridge/jazzy/rclpy/`) **does** ship a
Python-3.11-compatible `_rclpy_pybind11.so`, and it imports cleanly when
tested **outside** Isaac Sim with:

```bash
PYTHONPATH=<bridge>/jazzy/rclpy:$PYTHONPATH \
LD_LIBRARY_PATH=/opt/ros/jazzy/lib:$LD_LIBRARY_PATH \
python -c "import rclpy; rclpy.init(); print('OK')"
```

But the same environment, when fed into Isaac Sim via its `SimulationApp`,
still fails. The `isaacsim.ros2.bridge` C++ loader appears to rewrite /
override `sys.path` and `LD_LIBRARY_PATH` during extension enable, and
ends up looking for rclpy somewhere it isn't — or loading a conflicting
libfastrtps.

`ldd` confirms the bundled pybind11 `.so` links against
`/opt/ros/jazzy/lib/librcl*.so`, so the bundled rclpy is only a thin
wrapper around system Jazzy native libs — which compounds the Python
3.11/3.12 ABI tangle.

## Recovery options (sorted by effort)

1. **Downgrade to ROS 2 Humble** — Isaac Sim's Humble binding has worked
   in previous projects. But Ubuntu 24.04 doesn't ship Humble and
   back-porting is fragile.

2. **Rebuild ROS 2 Jazzy against Python 3.11** — heavy, half-day work,
   unsupported by NVIDIA.

3. **Skip rclpy entirely and publish via OmniGraph C++ nodes only** — we
   can still publish `/tf`, `/scan`, `/image_raw` from the sim side, and
   consume them from a completely separate ROS 2 process (system Jazzy,
   running its own Python 3.12 interpreter). This decouples Isaac from
   rclpy imports. Requires verifying that Isaac Sim's DDS settings match
   the external subscriber.

4. **Wait for NVIDIA to ship a Python 3.12 Isaac Sim build** — likely
   coming, tracks Omniverse Kit upgrade cadence.

Recommended: **option 3** (OmniGraph publish only, external rclpy
consumer) when picking this up again. It keeps Isaac Sim untouched and
uses the system ROS 2 as-is.

## What's already scaffolded

- `hmclab_isaac/utils/ros2.py` — `attach_ros2_graph` stub, NotImplementedError
- `scripts/smoke_m7.py` — copy of NVIDIA `clock.py` sample (source:
  `/home/js/isaacsim/standalone_examples/api/isaacsim.ros2.bridge/clock.py`)
- No per-robot `ros2_graph.py attach()` implementations yet

When re-opening M7, start by running `smoke_m7.py` in isolation. If it
passes clean, port `attach_ros2_graph` to publish `/tf` + `/odom` +
`/scan` for an existing racing env, then wire it into
`envs/racing/eval/*`.
