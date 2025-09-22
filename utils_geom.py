from __future__ import annotations

from typing import Optional
import numpy as np


def calculate_angle(a, b, c) -> float:
    """Szög a-b-c pontokból (fok, 0..180)."""
    ax, ay = a; bx, by = b; cx, cy = c
    radians = np.arctan2(cy - by, cx - bx) - np.arctan2(ay - by, ax - bx)
    angle = abs(np.degrees(radians))
    return float(360.0 - angle if angle > 180.0 else angle)


def trunk_angle_deg(shoulder_xy, hip_xy) -> float:
    """Törzs dőlése a függőlegestől: 0°=függőleges, 90°=vízszintes (irányfüggetlen)."""
    v = np.array(shoulder_xy, dtype=float) - np.array(hip_xy, dtype=float)
    vx, vy = float(v[0]), float(v[1])
    norm = np.hypot(vx, vy) + 1e-9
    cosang = np.clip(abs(-vy) / norm, 0.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def get_xy(lm, idx, min_vis: float = 0.5):
    pt = lm[idx]
    vis = getattr(pt, "visibility", 1.0)
    if vis is not None and vis < min_vis:
        return None
    if pt.x is None or pt.y is None:
        return None
    return [float(pt.x), float(pt.y)]
