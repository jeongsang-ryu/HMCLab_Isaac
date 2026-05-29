"""Track authoring GUI — Tkinter + embedded matplotlib canvas.

Draw a centerline (and optionally left/right walls) on a 1 m grid, smooth
with a B-spline, save as the 8-column RacingTrack .txt that
``hmclab_isaac.worlds.racing`` consumes.

Uses Tkinter (Python standard library) for the control panel so buttons
and sliders are native and reliably clickable. The drawing area is a
matplotlib canvas embedded in the Tk window.

Usage:
    python scripts/author_track.py
"""
from __future__ import annotations

import argparse
import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox

# Make `hmclab_isaac` importable from anywhere
_PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk
)

try:
    from scipy.interpolate import splprep, splev
except ImportError:
    print("scipy is required: pip install scipy", file=sys.stderr)
    sys.exit(1)

from hmclab_isaac.worlds.racing import RacingTrack


_DEFAULT_TRACKS_DIR = os.path.abspath(os.path.join(
    _PROJ_ROOT, "hmclab_isaac/worlds/racing/_tracks_data"
))


class TrackAuthorApp:
    LABELS = {"C": "Centerline", "L": "Left wall", "R": "Right wall"}
    COLORS = {"C": "#1f77b4", "L": "#2ca02c", "R": "#d62728"}

    def __init__(self, root: tk.Tk, default_output: str, default_closed: bool):
        self.root = root
        self.output_path = tk.StringVar(value=default_output)
        self.mode = "C"
        self.points: dict[str, list[tuple[float, float]]] = {"C": [], "L": [], "R": []}
        self.smoothing = tk.DoubleVar(value=5.0)
        self.n_samples = tk.IntVar(value=200)
        self.default_width = tk.DoubleVar(value=1.0)
        self.drag_min_step = tk.DoubleVar(value=0.20)
        self.closed = tk.BooleanVar(value=default_closed)
        self.show_raw = tk.BooleanVar(value=True)
        self.drag_enabled = tk.BooleanVar(value=True)
        self._dragging = False

        root.title("Track Authoring Tool")
        # Pane: left (plot) + right (controls)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        plot_frame = ttk.Frame(root)
        plot_frame.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        ctrl_frame = ttk.Frame(root, padding=8)
        ctrl_frame.grid(row=0, column=1, sticky="ns")

        # -------- matplotlib canvas --------
        self.fig = Figure(figsize=(10, 9))
        self.ax = self.fig.add_subplot(111)
        self._setup_drawing_area()
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        # matplotlib toolbar (pan/zoom) at the bottom
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()
        # Connect mouse events to draw on the plot
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        # Bind 'C/L/R/D/X/S' key shortcuts on the Tk root
        root.bind("<Key>", self._on_key)

        # -------- Tk controls --------
        self._build_controls(ctrl_frame)
        self._refresh()

    # =====================================================
    # Drawing area setup
    # =====================================================
    def _setup_drawing_area(self):
        self.ax.set_aspect("equal")
        self.ax.set_xlim(-15, 15)
        self.ax.set_ylim(-15, 15)
        self.ax.set_xlabel("x (m)")
        self.ax.set_ylabel("y (m)")
        # 1m major + 0.2m minor grid
        self.ax.xaxis.set_major_locator(MultipleLocator(1.0))
        self.ax.yaxis.set_major_locator(MultipleLocator(1.0))
        self.ax.xaxis.set_minor_locator(MultipleLocator(0.2))
        self.ax.yaxis.set_minor_locator(MultipleLocator(0.2))
        self.ax.grid(True, which="major", linewidth=0.7, color="#888", alpha=0.6)
        self.ax.grid(True, which="minor", linewidth=0.3, color="#888", alpha=0.25)
        self.ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
        self.ax.axvline(0, color="black", linewidth=0.5, alpha=0.5)
        self.ax.text(
            0.02, 0.98, "1 m grid (minor: 0.2 m)",
            transform=self.ax.transAxes, va="top", ha="left",
            fontsize=9, color="#555",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", lw=0.5, alpha=0.8),
        )
        self.raw_lines = {}
        self.scatters = {}
        self.smooth_lines = {}
        for m, c in self.COLORS.items():
            (rl,) = self.ax.plot([], [], color=c, linestyle=":", alpha=0.5, linewidth=1.0)
            self.raw_lines[m] = rl
            self.scatters[m] = self.ax.scatter(
                [], [], c=c, s=40, edgecolors="black", linewidths=0.5, zorder=5
            )
            (sl,) = self.ax.plot([], [], color=c, linewidth=2.5,
                                 label=self.LABELS[m], zorder=4)
            self.smooth_lines[m] = sl
        self.ax.legend(loc="upper right", framealpha=0.9)

    # =====================================================
    # Tk controls
    # =====================================================
    def _build_controls(self, parent: ttk.Frame):
        big_font = ("Helvetica", 11, "bold")
        normal_font = ("Helvetica", 10)

        ttk.Label(parent, text="Controls", font=("Helvetica", 13, "bold")).pack(
            anchor="w", pady=(0, 6))

        # -------- Mode buttons --------
        ttk.Label(parent, text="Mode", font=big_font).pack(anchor="w")
        self.mode_btns: dict[str, tk.Button] = {}
        for m in ("C", "L", "R"):
            b = tk.Button(parent, text=self.LABELS[m], font=normal_font,
                          width=22, anchor="w",
                          command=lambda mm=m: self._set_mode(mm))
            b.pack(fill=tk.X, pady=2)
            self.mode_btns[m] = b
        self._refresh_mode_buttons()

        ttk.Separator(parent, orient="horizontal").pack(fill=tk.X, pady=8)

        # -------- Sliders --------
        for label, var, frm, to, res in [
            ("Smoothing factor", self.smoothing, 0.0, 30.0, 0.5),
            ("Output samples",   self.n_samples, 20, 1000, 10),
            ("Default width (m)", self.default_width, 0.1, 5.0, 0.05),
            ("Drag step (m)",    self.drag_min_step, 0.05, 2.0, 0.05),
        ]:
            ttk.Label(parent, text=label, font=normal_font).pack(anchor="w")
            tk.Scale(parent, variable=var, from_=frm, to=to, resolution=res,
                     orient="horizontal", length=220,
                     command=lambda _v: self._refresh()).pack(fill=tk.X)

        ttk.Separator(parent, orient="horizontal").pack(fill=tk.X, pady=8)

        # -------- Checkbuttons --------
        for label, var in [
            ("Closed loop",       self.closed),
            ("Show raw clicks",   self.show_raw),
            ("Drag to draw",      self.drag_enabled),
        ]:
            tk.Checkbutton(parent, text=label, variable=var, font=normal_font,
                           command=self._refresh).pack(anchor="w")

        ttk.Separator(parent, orient="horizontal").pack(fill=tk.X, pady=8)

        # -------- Action buttons --------
        row1 = ttk.Frame(parent)
        row1.pack(fill=tk.X, pady=2)
        tk.Button(row1, text="Delete last", bg="#fff0c0", font=normal_font,
                  command=self._delete_last).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        tk.Button(row1, text="Clear MODE", bg="#ffd0d0", font=normal_font,
                  command=self._clear_mode).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))

        row2 = ttk.Frame(parent)
        row2.pack(fill=tk.X, pady=2)
        tk.Button(row2, text="Clear ALL", bg="#ffb0b0", font=normal_font,
                  command=self._clear_all).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        tk.Button(row2, text="Reset view", bg="#e0e0ff", font=normal_font,
                  command=self._reset_view).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))

        # -------- Auto wall generation --------
        ttk.Separator(parent, orient="horizontal").pack(fill=tk.X, pady=8)
        ttk.Label(parent, text="Auto-generate walls", font=big_font).pack(anchor="w")
        ttk.Label(parent,
                  text="Offsets centerline by Def width (m).\nReplaces existing L/R points.",
                  font=("Helvetica", 8), foreground="#666").pack(anchor="w")
        autorow = ttk.Frame(parent)
        autorow.pack(fill=tk.X, pady=2)
        tk.Button(autorow, text="Auto LEFT", bg="#c8f0c8", font=normal_font,
                  command=lambda: self._auto_wall("L")).pack(
                      side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        tk.Button(autorow, text="Auto RIGHT", bg="#f0c8c8", font=normal_font,
                  command=lambda: self._auto_wall("R")).pack(
                      side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))
        tk.Button(parent, text="Auto BOTH walls", bg="#d8e8d8", font=normal_font,
                  command=lambda: (self._auto_wall("L"), self._auto_wall("R"))
                  ).pack(fill=tk.X, pady=2)

        ttk.Separator(parent, orient="horizontal").pack(fill=tk.X, pady=8)

        # -------- Output path --------
        ttk.Label(parent, text="Output path", font=big_font).pack(anchor="w")
        tk.Entry(parent, textvariable=self.output_path, width=30,
                 font=("Helvetica", 9)).pack(fill=tk.X, pady=2)

        # -------- Save (big), Quit --------
        tk.Button(parent, text="SAVE TRACK", bg="#a8e0a8", font=("Helvetica", 13, "bold"),
                  height=2, command=self._save).pack(fill=tk.X, pady=(8, 4))
        tk.Button(parent, text="Quit", bg="#e0e0e0", font=normal_font,
                  command=self.root.destroy).pack(fill=tk.X)

        ttk.Separator(parent, orient="horizontal").pack(fill=tk.X, pady=8)

        # -------- Status --------
        self.status_lbl = tk.Label(parent, text="", justify="left", anchor="w",
                                   font=("Courier", 9), bg="#f8f8f8",
                                   relief="solid", borderwidth=1, padx=4, pady=4)
        self.status_lbl.pack(fill=tk.X)

    # =====================================================
    # Mode handling
    # =====================================================
    def _set_mode(self, m: str):
        self.mode = m
        self._refresh_mode_buttons()
        self._refresh()

    def _refresh_mode_buttons(self):
        for m, btn in self.mode_btns.items():
            if m == self.mode:
                bg = {"C": "#a3c9f5", "L": "#a8e0a8", "R": "#f5a8a8"}[m]
                btn.configure(bg=bg, fg=self.COLORS[m],
                              font=("Helvetica", 10, "bold"), relief="sunken")
            else:
                btn.configure(bg="#e8e8e8", fg=self.COLORS[m],
                              font=("Helvetica", 10), relief="raised")

    # =====================================================
    # Mouse events on the plot
    # =====================================================
    def _is_drawable(self, event) -> bool:
        if event.inaxes != self.ax or event.button != 1:
            return False
        if event.xdata is None or event.ydata is None:
            return False
        # If pan/zoom is active, skip
        tb = self.canvas.toolbar
        if getattr(tb, "mode", "") not in ("", None):
            return False
        return True

    def _on_press(self, event):
        if not self._is_drawable(event):
            return
        self.points[self.mode].append((float(event.xdata), float(event.ydata)))
        self._dragging = True
        self._refresh()

    def _on_motion(self, event):
        if not self._dragging or not self.drag_enabled.get():
            return
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        if not self.points[self.mode]:
            return
        last = self.points[self.mode][-1]
        dx = event.xdata - last[0]
        dy = event.ydata - last[1]
        min_step = self.drag_min_step.get()
        if (dx * dx + dy * dy) ** 0.5 < min_step:
            return
        self.points[self.mode].append((float(event.xdata), float(event.ydata)))
        self._refresh()

    def _on_release(self, event):
        if event.button == 1:
            self._dragging = False

    def _on_key(self, event):
        k = (event.keysym or "").lower()
        if k in ("c", "l", "r"):
            self._set_mode(k.upper())
        elif k == "d":
            self._delete_last()
        elif k == "x":
            self._clear_mode()
        elif k == "s":
            self._save()
        elif k == "q":
            self.root.destroy()

    # =====================================================
    # Edit ops
    # =====================================================
    def _delete_last(self):
        if self.points[self.mode]:
            self.points[self.mode].pop()
            self._refresh()

    def _clear_mode(self):
        self.points[self.mode] = []
        self._refresh()

    def _clear_all(self):
        for m in self.points:
            self.points[m] = []
        self._refresh()

    def _auto_wall(self, side: str):
        """Generate L or R wall points by offsetting the centerline.

        Uses the smoothed centerline's local tangent at each point and
        the current 'Default width' slider to compute a parallel curve.
        Replaces existing points on that side.
        """
        if side not in ("L", "R"):
            return
        c = self._smooth(self.points["C"])
        if c is None or len(c) < 4:
            messagebox.showwarning(
                "Auto wall",
                "Draw at least 4 centerline points first.",
            )
            return
        width = float(self.default_width.get())
        closed = bool(self.closed.get())
        # Centered finite-difference tangents
        if closed:
            nxt = np.roll(c, -1, axis=0)
            prv = np.roll(c, 1, axis=0)
        else:
            nxt = np.concatenate([c[1:], c[-1:]], axis=0)
            prv = np.concatenate([c[:1], c[:-1]], axis=0)
        tangent = nxt - prv
        norms = np.linalg.norm(tangent, axis=1, keepdims=True)
        tangent = tangent / np.maximum(norms, 1e-8)
        # In 2D, the left normal (rotate tangent +90° CCW) is (-ty, tx).
        left_normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
        if side == "L":
            offset_pts = c + width * left_normal
        else:
            offset_pts = c - width * left_normal     # right = -left
        self.points[side] = [tuple(p) for p in offset_pts]
        # Switch UI to the side we just generated so user sees the result
        self._set_mode(side)

    def _reset_view(self):
        all_pts = [p for pts in self.points.values() for p in pts]
        if all_pts:
            arr = np.asarray(all_pts)
            m = 2.0
            self.ax.set_xlim(arr[:, 0].min() - m, arr[:, 0].max() + m)
            self.ax.set_ylim(arr[:, 1].min() - m, arr[:, 1].max() + m)
        else:
            self.ax.set_xlim(-15, 15)
            self.ax.set_ylim(-15, 15)
        self.canvas.draw_idle()

    # =====================================================
    # Smoothing
    # =====================================================
    def _smooth(self, pts: list[tuple[float, float]]) -> np.ndarray | None:
        if len(pts) < 4:
            return None
        arr = np.asarray(pts, dtype=float)
        x, y = arr[:, 0], arr[:, 1]
        try:
            tck, _ = splprep([x, y], s=self.smoothing.get(),
                             per=self.closed.get())
            u = np.linspace(0.0, 1.0, int(self.n_samples.get()),
                            endpoint=not self.closed.get())
            sx, sy = splev(u, tck)
            return np.stack([np.asarray(sx), np.asarray(sy)], axis=1)
        except Exception:
            return None

    def _curve_length(self, pts) -> float:
        sm = self._smooth(pts)
        if sm is None:
            return 0.0
        diffs = np.diff(sm, axis=0)
        L = float(np.linalg.norm(diffs, axis=1).sum())
        if self.closed.get() and len(sm) > 1:
            L += float(np.linalg.norm(sm[0] - sm[-1]))
        return L

    # =====================================================
    # Refresh / draw
    # =====================================================
    def _refresh(self):
        show_raw = self.show_raw.get()
        for m, pts in self.points.items():
            arr = np.asarray(pts, dtype=float) if pts else np.zeros((0, 2))
            self.scatters[m].set_offsets(arr if len(arr) else np.zeros((0, 2)))
            if show_raw and len(arr):
                if self.closed.get() and len(arr) >= 2:
                    closed_arr = np.vstack([arr, arr[:1]])
                    self.raw_lines[m].set_data(closed_arr[:, 0], closed_arr[:, 1])
                else:
                    self.raw_lines[m].set_data(arr[:, 0], arr[:, 1])
            else:
                self.raw_lines[m].set_data([], [])
            sm = self._smooth(pts)
            if sm is not None:
                self.smooth_lines[m].set_data(sm[:, 0], sm[:, 1])
            else:
                self.smooth_lines[m].set_data([], [])

        length_c = self._curve_length(self.points["C"])
        status = (
            f"Mode:   {self.LABELS[self.mode]}\n"
            f"Points: C={len(self.points['C'])}  "
            f"L={len(self.points['L'])}  R={len(self.points['R'])}\n"
            f"Length: {length_c:.2f} m\n"
            f"Closed: {self.closed.get()}   "
            f"Drag: {self.drag_enabled.get()}\n\n"
            "Keys:  C/L/R = mode    D = del last\n"
            "       X = clear mode  S = save   Q = quit"
        )
        self.status_lbl.configure(text=status)
        self.canvas.draw_idle()

    # =====================================================
    # Save
    # =====================================================
    def _save(self):
        out = self.output_path.get().strip()
        if not out:
            messagebox.showerror("Save", "Output path is empty.")
            return
        if len(self.points["C"]) < 4:
            messagebox.showwarning("Save", "Need at least 4 centerline points.")
            return
        c = self._smooth(self.points["C"])
        l = self._smooth(self.points["L"])
        r = self._smooth(self.points["R"])
        if c is None:
            messagebox.showerror("Save", "Centerline smoothing failed.")
            return
        n = len(c)
        dw = self.default_width.get()
        d_left = np.full(n, dw)
        d_right = np.full(n, dw)
        if l is not None:
            for i, p in enumerate(c):
                d_left[i] = float(np.linalg.norm(l - p, axis=1).min())
        if r is not None:
            for i, p in enumerate(c):
                d_right[i] = float(np.linalg.norm(r - p, axis=1).min())
        name = os.path.splitext(os.path.basename(out))[0]
        track = RacingTrack.from_2d_centerline(
            name, c, d_left, d_right, closed=self.closed.get()
        )
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        track.save(out)
        msg = (f"Saved: {out}\n"
               f"{n} pts  length={track.length():.2f} m\n"
               f"L={d_left.mean():.2f}±{d_left.std():.2f}  "
               f"R={d_right.mean():.2f}±{d_right.std():.2f}")
        print(msg)
        messagebox.showinfo("Saved", msg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=os.path.join(_DEFAULT_TRACKS_DIR, "my_track.txt"),
                        help="Initial output path (editable in GUI).")
    parser.add_argument("--open", action="store_true",
                        help="Initial 'closed loop' = off.")
    args = parser.parse_args()

    root = tk.Tk()
    TrackAuthorApp(root, default_output=args.output, default_closed=not args.open)
    root.geometry("1400x900")
    root.mainloop()


if __name__ == "__main__":
    main()
