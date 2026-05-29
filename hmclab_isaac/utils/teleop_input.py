"""Shared keyboard + Linux joystick input helpers for teleop scripts.

These exist so the various ``scripts/keyboard_teleop_*.py`` entry points
stay short and use the same input model. Two pieces:

* :class:`LinuxJoystick` — non-blocking reader of ``/dev/input/jsX``
  (raw kernel API, no external deps). Background thread decodes 8-byte
  event records and updates a thread-safe state.

* :class:`CarKeyboard` — small wrapper over Carb's keyboard event API
  that exposes ``throttle`` / ``steer`` properties for ↑↓←→ + Space.
  Carb is imported lazily inside ``__init__`` so this module is safe
  to import before AppLauncher has started Kit.

Default Linux-js axis mapping (Xbox-style):
    axis 0 = left-stick X
    axis 1 = left-stick Y
    axis 2 = LT
    axis 3 = right-stick X
    axis 4 = right-stick Y
    axis 5 = RT
"""
from __future__ import annotations

import os


# ────────────────────────────────────────────────────────────────────
# helpers
# ────────────────────────────────────────────────────────────────────
def apply_deadzone(v: float, dz: float) -> float:
    """Clamp small values to 0 and rescale outside the deadzone for a
    continuous response."""
    if abs(v) < dz:
        return 0.0
    s = 1.0 if v > 0 else -1.0
    return s * (abs(v) - dz) / (1.0 - dz)


# ────────────────────────────────────────────────────────────────────
# Linux joystick reader (/dev/input/jsX)
# ────────────────────────────────────────────────────────────────────
# Event record format: struct js_event { uint32 time; int16 value;
#                                        uint8 type; uint8 number; }
#   type & 0x01 = button event
#   type & 0x02 = axis event
#   type & 0x80 = init/sync (initial state burst on open)
class LinuxJoystick:
    """Background-thread reader for /dev/input/jsX. Thread-safe state."""

    _EVENT_FMT = "IhBB"
    _EVENT_SIZE = 8

    def __init__(self, path: str = "/dev/input/js0"):
        import threading
        self.path = path
        self.connected = False
        self._fd = None
        self._axes: dict[int, float] = {}
        self._buttons: dict[int, int] = {}
        self._lock = threading.Lock()
        self._stop = False
        self._name = "?"

        try:
            self._fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        except OSError as e:
            print(f"[joystick] could not open {path}: {e}  "
                  f"(continuing with keyboard only)", flush=True)
            return

        # JSIOCGNAME(128) = _IOC(_IOC_READ, 'j', 0x13, 128) = 0x80806a13.
        try:
            import ctypes
            import fcntl
            buf = ctypes.create_string_buffer(128)
            try:
                fcntl.ioctl(self._fd, 0x80806a13, buf, True)
                self._name = buf.value.decode(errors="replace").strip() or "?"
            except OSError:
                pass
        except Exception:
            pass

        self.connected = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        print(f"[joystick] opened {path}  name={self._name!r}", flush=True)

    def _reader_loop(self):
        import select
        import struct
        buf = b""
        while not self._stop:
            try:
                r, _, _ = select.select([self._fd], [], [], 0.1)
                if not r:
                    continue
                buf += os.read(self._fd, 4096)
                while len(buf) >= self._EVENT_SIZE:
                    chunk = buf[: self._EVENT_SIZE]
                    buf = buf[self._EVENT_SIZE :]
                    _, value, typ, num = struct.unpack(self._EVENT_FMT, chunk)
                    base = typ & 0x7F
                    with self._lock:
                        if base == 1:
                            self._buttons[num] = int(value)
                        elif base == 2:
                            # Linux js: axis range = -32767 .. +32767
                            self._axes[num] = max(-1.0, min(1.0, value / 32767.0))
            except (BlockingIOError, OSError):
                pass
            except Exception:
                pass  # don't kill thread on transient decode errors

    def axis(self, num: int) -> float:
        with self._lock:
            return float(self._axes.get(num, 0.0))

    def button(self, num: int) -> int:
        with self._lock:
            return int(self._buttons.get(num, 0))

    def is_active(self, axis_threshold: float = 0.05) -> bool:
        """Any non-deadzone axis pressed or any button held."""
        with self._lock:
            for v in self._axes.values():
                if abs(v) > axis_threshold:
                    return True
            for v in self._buttons.values():
                if v:
                    return True
        return False

    def close(self):
        self._stop = True
        try:
            if self._fd is not None:
                os.close(self._fd)
        except Exception:
            pass


# ────────────────────────────────────────────────────────────────────
# Carb keyboard wrapper
# ────────────────────────────────────────────────────────────────────
class CarKeyboard:
    """↑↓ throttle, ←→ steer, Space brake, L reset.

    Must be constructed AFTER AppLauncher has started Kit (carb.input
    needs the running app). Throttle/steer return values in [-1, +1]:
    +throttle=forward, +steer=left.
    """

    def __init__(self):
        import weakref
        import carb.input
        import omni.appwindow

        self._held: set[str] = set()
        self._app = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._kb = self._app.get_keyboard()
        self._sub = self._input.subscribe_to_keyboard_events(
            self._kb,
            lambda e, *a, obj=weakref.proxy(self): obj._on_event(e, *a),
        )
        self._carb_input = carb.input  # for KeyboardEventType lookup later

    def __del__(self):
        try:
            self._input.unsubscribe_to_keyboard_events(self._kb, self._sub)
        except Exception:
            pass

    def _on_event(self, event, *args, **kwargs):
        # Newer carb releases hand `event.input` as a plain string;
        # older ones expose an enum with `.name`. Handle both.
        inp = event.input
        name = inp.name if hasattr(inp, "name") else str(inp)
        if event.type == self._carb_input.KeyboardEventType.KEY_PRESS:
            if name == "L":
                self._held.clear()
            else:
                self._held.add(name)
        elif event.type == self._carb_input.KeyboardEventType.KEY_RELEASE:
            self._held.discard(name)
        return True

    @property
    def throttle(self) -> float:
        if "SPACE" in self._held:
            return 0.0
        return ((1.0 if "UP" in self._held else 0.0)
                - (1.0 if "DOWN" in self._held else 0.0))

    @property
    def steer(self) -> float:
        return ((1.0 if "LEFT" in self._held else 0.0)
                - (1.0 if "RIGHT" in self._held else 0.0))


# ────────────────────────────────────────────────────────────────────
# High-level controller: keyboard + joystick fused
# ────────────────────────────────────────────────────────────────────
class CarController:
    """Keyboard + (optional) joystick fused into (throttle, steer) ∈ [-1, 1].

    Joystick wins when active (any axis past deadzone or any button down);
    otherwise keyboard provides input. ``last_source`` reports which
    one was last used (``"joystick"`` or ``"keyboard"``).
    """

    def __init__(
        self,
        joystick_path: str = "/dev/input/js0",
        *,
        # Defaults match Xbox-style mapping: left-stick Y throttle,
        # right-stick X steer, A button brake.
        axis_throttle: int = 1,
        axis_steer: int = 3,
        axis_throttle_fwd: int = -1,   # if >=0, trigger mode (RT axis)
        button_brake: int = 0,
        deadzone: float = 0.20,
        invert_throttle: bool = True,  # stick-up → +throttle
        invert_steer: bool = True,     # stick-left → +steer (=turn left)
    ):
        self.kb = CarKeyboard()
        self.js = LinuxJoystick(joystick_path) if joystick_path else None
        if self.js is not None and not self.js.connected:
            self.js = None
        self.axis_throttle = axis_throttle
        self.axis_steer = axis_steer
        self.axis_throttle_fwd = axis_throttle_fwd
        self.button_brake = button_brake
        self.deadzone = deadzone
        self.invert_throttle = invert_throttle
        self.invert_steer = invert_steer
        self.last_source = "keyboard"

    def _read_joystick(self) -> tuple[float, float, bool]:
        """Compute (throttle, steer, brake_held) from the joystick. All
        zero if no joystick is connected. Deadzone is applied; small
        stick drift below the deadzone returns exactly 0 (no creep)."""
        if self.js is None:
            return 0.0, 0.0, False
        sx = apply_deadzone(self.js.axis(self.axis_steer), self.deadzone)
        steer_v = -sx if self.invert_steer else sx
        if self.axis_throttle_fwd >= 0:
            # Trigger mode: idle trigger axis = -1. Map to (val+1)/2 ∈ [0,1].
            rt = (self.js.axis(self.axis_throttle_fwd) + 1.0) * 0.5
            lt = (self.js.axis(self.axis_throttle) + 1.0) * 0.5
            rt = apply_deadzone(rt, self.deadzone)
            lt = apply_deadzone(lt, self.deadzone)
            thr_v = rt - lt
        else:
            ty = apply_deadzone(self.js.axis(self.axis_throttle), self.deadzone)
            thr_v = -ty if self.invert_throttle else ty
        brake = bool(self.js.button(self.button_brake))
        return (
            float(max(-1.0, min(1.0, thr_v))),
            float(max(-1.0, min(1.0, steer_v))),
            brake,
        )

    def read(self) -> tuple[float, float]:
        """Return (throttle, steer) ∈ [-1, +1].

        Sum of keyboard + joystick (clamped). Either source idle = 0,
        so a stationary stick + no keys = exactly 0 (no creep). When
        both contribute, they add — useful for fine joystick steering
        with keyboard throttle, or vice versa.
        """
        j_thr, j_steer, j_brake = self._read_joystick()
        k_thr, k_steer = self.kb.throttle, self.kb.steer
        thr = j_thr + k_thr
        steer = j_steer + k_steer
        if j_brake:
            thr = 0.0
        # Diagnostic source label: whichever side dominates.
        if abs(j_thr) > abs(k_thr) or abs(j_steer) > abs(k_steer) or j_brake:
            self.last_source = "joystick"
        else:
            self.last_source = "keyboard"
        return (
            float(max(-1.0, min(1.0, thr))),
            float(max(-1.0, min(1.0, steer))),
        )

    def close(self):
        if self.js is not None:
            self.js.close()


def add_input_args(parser):
    """Argparse helper — adds --joystick/--js-* flags consistently across
    teleop scripts. Returns the parser for chaining."""
    import argparse
    parser.add_argument("--joystick", default="/dev/input/js0",
                        help="Linux joystick device path. '' to disable.")
    parser.add_argument("--js-axis-steer", type=int, default=3,
                        help="Joystick axis index for steering "
                             "(default 3 = right stick X on Xbox-style pads).")
    parser.add_argument("--js-axis-throttle", type=int, default=1,
                        help="Joystick axis index for throttle "
                             "(default 1 = left stick Y).")
    parser.add_argument("--js-axis-throttle-fwd", type=int, default=-1,
                        help="If >=0, this axis becomes forward throttle "
                             "(e.g. 5 = RT). --js-axis-throttle then = LT.")
    parser.add_argument("--js-button-brake", type=int, default=0,
                        help="Joystick button index that acts as brake "
                             "(default 0 = A on Xbox).")
    parser.add_argument("--js-deadzone", type=float, default=0.10,
                        help="Stick deadzone (axis values |v|<deadzone → 0).")
    parser.add_argument("--js-invert-steer", action=argparse.BooleanOptionalAction,
                        default=True, help="Invert steer axis sign.")
    parser.add_argument("--js-invert-throttle",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Invert throttle axis sign.")
    return parser


def make_controller(args) -> CarController:
    """Build a CarController from parsed args (output of add_input_args)."""
    return CarController(
        joystick_path=args.joystick or "",
        axis_throttle=args.js_axis_throttle,
        axis_steer=args.js_axis_steer,
        axis_throttle_fwd=args.js_axis_throttle_fwd,
        button_brake=args.js_button_brake,
        deadzone=args.js_deadzone,
        invert_throttle=args.js_invert_throttle,
        invert_steer=args.js_invert_steer,
    )
