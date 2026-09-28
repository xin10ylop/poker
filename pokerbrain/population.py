"""Population model fitted to real hand histories (see experiments/fit_population.py).

Holds the average behaviour of the player pool: prior means for every tracked stat, and how often
players fold / raise facing a bet, as a function of street and bet size.  The tracker uses the means as
its priors for unknown players; the villain model uses the curves and shifts them per player.

The default file is fitted on real online hands; POKERBRAIN_POPULATION=<path> points elsewhere
(e.g. a file fitted on your own hand histories) and POKERBRAIN_POPULATION=none disables it.
"""
from __future__ import annotations

import json
import math
import os
from typing import Optional

DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "data", "population.json")
_DATA: Optional[dict] = None


def path() -> Optional[str]:
    p = os.environ.get("POKERBRAIN_POPULATION", DEFAULT_PATH)
    return None if p.lower() == "none" else p


def data() -> dict:
    global _DATA
    if _DATA is None:
        p = path()
        try:
            with open(p) as f:
                _DATA = json.load(f)
        except (OSError, TypeError, ValueError):
            _DATA = {}
    return _DATA


def curve(kind: str, street: str, facing_raise: bool, multiway: bool) -> Optional[list]:
    """[[size, p, n], ...] for kind 'fold' or 'raise'; falls back to the heads-up curve."""
    table = data().get(f"{kind}_curve") or {}
    key = f"{street}|{'raise' if facing_raise else 'bet'}|"
    return table.get(key + ("mw" if multiway else "hu")) or table.get(key + "hu")


def interp(points: list, size: float) -> float:
    """Piecewise-linear in log(size), flat beyond the ends."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x = max(size, 1e-3)
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    lx = math.log(x)
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        if x0 <= x <= x1:
            t = (lx - math.log(x0)) / max(1e-9, math.log(x1) - math.log(x0))
            return y0 + t * (y1 - y0)
    return ys[-1]


def logit(q: float) -> float:
    q = min(1 - 1e-4, max(1e-4, q))
    return math.log(q / (1 - q))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, x))))
