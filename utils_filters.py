from __future__ import annotations

from collections import deque
from typing import Optional


class EMASmoother:
    def __init__(self, alpha: float = 0.25):
        self.alpha = float(alpha)
        self.prev: Optional[float] = None

    def update(self, x: float) -> float:
        x = float(x)
        self.prev = x if self.prev is None else self.alpha * x + (1 - self.alpha) * self.prev
        return self.prev


class WindowDerivative:
    def __init__(self, window: int = 3, dt: float = 1 / 30.0):
        self.buf = deque(maxlen=window)
        self.dt = float(dt)

    def update(self, x: float) -> float:
        self.buf.append(float(x))
        n = len(self.buf)
        if n < 2:
            return 0.0
        return (self.buf[-1] - self.buf[0]) / ((n - 1) * self.dt)


class WindowPeak:
    def __init__(self, window: int = 7, mode: str = "max", tol: float = 2.0):
        self.buf = deque(maxlen=window)
        self.mode = mode
        self.tol = float(tol)

    def update(self, x: float) -> bool:
        x = float(x)
        self.buf.append(x)
        if len(self.buf) < self.buf.maxlen:
            return False
        return x >= (max(self.buf) - self.tol) if self.mode == "max" else x <= (min(self.buf) + self.tol)
