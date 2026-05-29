"""Batched Frenet projection for RL training.

Given a :class:`RacingTrack` (centerline + per-row half-widths), build a GPU
tensor representation and expose a fast batched ``query(xy, yaw)`` that returns
the Frenet coordinates ``(s, d_signed, psi_err)`` plus per-env curvature
lookahead for the privileged critic observation.

Pure torch — importable without AppLauncher.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch

from ._schema import RacingTrack


def _wrap_pi(x: torch.Tensor) -> torch.Tensor:
    """Wrap angles into [-pi, +pi]."""
    return (x + math.pi) % (2.0 * math.pi) - math.pi


class FrenetField:
    """Batched closest-point + Frenet/curvature lookup against one track.

    All precomputed quantities live on the given device. ``query`` takes batch
    XY/yaw tensors and returns per-env Frenet coordinates plus 5-element
    curvature lookahead.
    """

    LOOKAHEAD_METERS: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0)

    def __init__(
        self,
        track: RacingTrack,
        device: str | torch.device = "cuda:0",
        lookaheads: Sequence[float] | None = None,
    ):
        self.device = torch.device(device)
        self.closed = bool(track.closed)
        if lookaheads is None:
            lookaheads = self.LOOKAHEAD_METERS
        self.lookaheads = tuple(float(x) for x in lookaheads)

        xy = track.positions[:, :2].astype(np.float64)
        n = xy.shape[0]
        tang = track.tangents()
        normal = np.stack([-tang[:, 1], tang[:, 0]], axis=1)  # left-normal
        arclen = track.arclen()
        kappa = track.curvatures()
        widths = track.widths.astype(np.float64)

        # Per-segment length (segment i is from point i to point i+1)
        if self.closed:
            seg_vec = np.roll(xy, -1, axis=0) - xy
        else:
            seg_vec = np.zeros_like(xy)
            seg_vec[:-1] = xy[1:] - xy[:-1]
        seg_len = np.linalg.norm(seg_vec, axis=1)
        seg_len = np.maximum(seg_len, 1e-9)
        self.total_length = float(seg_len.sum()) if self.closed else float(arclen[-1])

        # Lookahead index table: for each point i and each Δ, the index whose
        # cumulative arclen is closest to arclen[i] + Δ.
        la_idx = np.zeros((n, len(self.lookaheads)), dtype=np.int64)
        total = float(seg_len.sum())
        for i in range(n):
            for j, dist in enumerate(self.lookaheads):
                target = arclen[i] + dist
                if self.closed:
                    target = target % total
                else:
                    target = min(target, arclen[-1])
                la_idx[i, j] = int(np.argmin(np.abs(arclen - target)))

        d = self.device
        self.cl_xy = torch.as_tensor(xy, dtype=torch.float32, device=d)
        self.cl_tang = torch.as_tensor(tang, dtype=torch.float32, device=d)
        self.cl_normal = torch.as_tensor(normal, dtype=torch.float32, device=d)
        self.cl_arclen = torch.as_tensor(arclen, dtype=torch.float32, device=d)
        self.cl_kappa = torch.as_tensor(kappa, dtype=torch.float32, device=d)
        self.cl_widths = torch.as_tensor(widths, dtype=torch.float32, device=d)
        self.seg_vec = torch.as_tensor(seg_vec, dtype=torch.float32, device=d)
        self.seg_len = torch.as_tensor(seg_len, dtype=torch.float32, device=d)
        self.la_idx = torch.as_tensor(la_idx, dtype=torch.long, device=d)
        self.tang_angle = torch.atan2(self.cl_tang[:, 1], self.cl_tang[:, 0])
        self.num_points = n

    # ------------------------------------------------------------------
    # Main query
    # ------------------------------------------------------------------
    @torch.no_grad()
    def query(self, xy: torch.Tensor, yaw: torch.Tensor):
        """Project (E, 2) XY and (E,) yaw onto the centerline.

        Returns a dict with keys (all shape (E,) unless noted):
            s              — arc-length along centerline (meters)
            s_norm         — s / total_length (in [0, 1) for closed tracks)
            d_signed       — signed lateral offset (+ = left of centerline)
            psi_err        — wrapped heading error (rad)
            tangent_angle  — yaw of nearest centerline tangent (rad)
            d_left, d_right — half-widths at projection (meters)
            kappa          — curvature at projection (1/m, signed)
            kappa_lookahead — curvatures at LOOKAHEAD_METERS (E, K)
            idx            — nearest centerline index
        """
        assert xy.dim() == 2 and xy.shape[1] == 2
        assert yaw.dim() == 1 and yaw.shape[0] == xy.shape[0]

        # (E, N) squared distances — N is centerline point count (~2000)
        diff = xy.unsqueeze(1) - self.cl_xy.unsqueeze(0)
        d2 = (diff * diff).sum(dim=-1)
        idx = torch.argmin(d2, dim=1)  # (E,)

        # Project onto segment [idx, idx+1]
        base = self.cl_xy[idx]                  # (E, 2)
        seg_v = self.seg_vec[idx]               # (E, 2)
        seg_l = self.seg_len[idx]               # (E,)
        rel = xy - base
        t = (rel * seg_v).sum(dim=-1) / (seg_l * seg_l)
        t = t.clamp(0.0, 1.0)
        foot = base + t.unsqueeze(-1) * seg_v

        s = self.cl_arclen[idx] + t * seg_l
        s_norm = s / max(self.total_length, 1e-9)
        normal = self.cl_normal[idx]            # (E, 2)
        d_signed = ((xy - foot) * normal).sum(dim=-1)
        tangent_angle = self.tang_angle[idx]
        psi_err = _wrap_pi(yaw - tangent_angle)

        d_left = self.cl_widths[idx, 0]
        d_right = self.cl_widths[idx, 1]
        kappa = self.cl_kappa[idx]

        # Curvature lookahead — (E, K)
        la = self.la_idx[idx]                   # (E, K)
        kappa_la = self.cl_kappa[la]

        return {
            "s": s,
            "s_norm": s_norm,
            "d_signed": d_signed,
            "psi_err": psi_err,
            "tangent_angle": tangent_angle,
            "d_left": d_left,
            "d_right": d_right,
            "kappa": kappa,
            "kappa_lookahead": kappa_la,
            "idx": idx,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def lap_delta(self, s_curr: torch.Tensor, s_prev: torch.Tensor) -> torch.Tensor:
        """Forward arc-length progress since last step, handling wrap-around.

        For a closed track, a step that crosses the start/finish line shows up
        as a large negative bare difference (e.g. s went from 59 m to 1 m on a
        60 m loop). This adds back ``total_length`` if the raw diff is more
        negative than -total_length/2.
        """
        d = s_curr - s_prev
        if self.closed:
            half = 0.5 * self.total_length
            d = torch.where(d < -half, d + self.total_length, d)
            d = torch.where(d > half, d - self.total_length, d)
        return d
