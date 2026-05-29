"""Quick joystick test — opens /dev/input/jsX and prints axis/button events.

Use this to verify your joystick is recognized and to find the right axis /
button indices for keyboard_teleop_chassis.py.

Run:
    python scripts/joystick_probe.py            # /dev/input/js0
    python scripts/joystick_probe.py --device /dev/input/js1
"""
from __future__ import annotations

import argparse
import os
import select
import struct
import sys
import time


def read_device_name(fd: int) -> str:
    try:
        import fcntl, ctypes
        buf = ctypes.create_string_buffer(128)
        try:
            JSIOCGNAME = 0x80006a13 | (128 << 16)
            fcntl.ioctl(fd, JSIOCGNAME, buf, True)
            return buf.value.decode(errors="replace").strip() or "?"
        except OSError:
            return "?"
    except Exception:
        return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="/dev/input/js0")
    ap.add_argument("--show-init", action="store_true",
                    help="Also print the initial-state burst that the kernel "
                         "sends right after open (events with the 0x80 bit).")
    args = ap.parse_args()

    try:
        fd = os.open(args.device, os.O_RDONLY | os.O_NONBLOCK)
    except OSError as e:
        print(f"[probe] could not open {args.device}: {e}")
        sys.exit(1)

    name = read_device_name(fd)
    print(f"[probe] {args.device}  name = {name!r}")
    print(f"[probe] Move sticks / press buttons. Ctrl-C to exit.\n")

    EVENT_FMT = "IhBB"
    EVENT_SIZE = 8
    buf = b""
    last_axes: dict[int, float] = {}
    last_btns: dict[int, int] = {}
    t0 = time.time()
    try:
        while True:
            r, _, _ = select.select([fd], [], [], 0.5)
            if not r:
                continue
            buf += os.read(fd, 4096)
            while len(buf) >= EVENT_SIZE:
                chunk = buf[:EVENT_SIZE]
                buf = buf[EVENT_SIZE:]
                _, value, typ, num = struct.unpack(EVENT_FMT, chunk)
                is_init = (typ & 0x80) != 0
                base = typ & 0x7F
                if is_init and not args.show_init:
                    # capture initial state silently for the live snapshot
                    if base == 1:
                        last_btns[num] = value
                    elif base == 2:
                        last_axes[num] = value / 32767.0
                    continue
                ts = time.time() - t0
                if base == 1:
                    last_btns[num] = value
                    state = "↓" if value else "↑"
                    print(f"  [{ts:6.2f}s] BUTTON  num={num:<2d}  state={state}",
                          flush=True)
                elif base == 2:
                    v = value / 32767.0
                    last_axes[num] = v
                    bar_n = int((v + 1.0) * 10)  # 0..20
                    bar = "[" + "=" * bar_n + " " * (20 - bar_n) + "]"
                    print(f"  [{ts:6.2f}s] AXIS    num={num:<2d}  v={v:+.3f}  {bar}",
                          flush=True)
                else:
                    print(f"  [{ts:6.2f}s] (unknown type 0x{typ:02x})", flush=True)
    except KeyboardInterrupt:
        print("\n[probe] axes summary:")
        for k in sorted(last_axes):
            print(f"   axis {k:<2d} = {last_axes[k]:+.3f}")
        print("[probe] buttons last state:")
        for k in sorted(last_btns):
            print(f"   button {k:<2d} = {last_btns[k]}")
    finally:
        os.close(fd)


if __name__ == "__main__":
    main()
