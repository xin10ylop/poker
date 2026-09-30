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


def curve(kind: str, street: str, facing_raise: bool, multiway: bool, vs_cbet: bool = False) -> Optional[list]:
    """[[size, p, n], ...] for kind 'fold' or 'raise'.

    Facing a flop continuation bet has its own curve when fitted (real players fold less to c-bets than
    to other bets); otherwise the generic bet curve; multiway falls back to heads-up."""
    table = data().get(f"{kind}_curve") or {}
    kinds = (["cbet", "bet"] if (vs_cbet and not facing_raise) else ["raise" if facing_raise else "bet"])
    for k in kinds:
        for seats in (["mw", "hu"] if multiway else ["hu"]):
            c = table.get(f"{street}|{k}|{seats}")
            if c:
                return c
    return None


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


def prior_mean(stat: str, seats=None) -> Optional[float]:
    """Pool mean of a stat for the table size band (6-max vs full ring), when fitted."""
    if not seats:
        return None
    band = "6" if seats <= 6 else "9"
    v = (data().get("prior_means_by_seats") or {}).get(band, {}).get(stat)
    return float(v) if v is not None else None


def commit_shift(commit: float) -> float:
    """Logit shift of the fold probability when calling costs this share of the villain's stack.

    Fitted on real hands: players who must call off most of their stack fold a little LESS at a given
    bet size (they are committed), not more."""
    cs = data().get("commit_shift") or {}
    if not cs or commit <= 0.35:
        return 0.0
    pts = [(x, cs.get(k)) for x, k in ((0.475, "0.35-0.6"), (0.75, "0.6-0.9"), (0.95, "0.9+"))]
    pts = [(x, float(y)) for x, y in pts if y is not None]
    if not pts:
        return 0.0
    if commit <= pts[0][0]:
        return pts[0][1] * (commit - 0.35) / max(1e-9, pts[0][0] - 0.35)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= commit <= x1:
            return y0 + (y1 - y0) * (commit - x0) / max(1e-9, x1 - x0)
    return pts[-1][1]


def bluff_curve(street: str, kind: str = "bet") -> Optional[list]:
    """[[size, bluff share, n], ...]: share of the pool's bets of this size that were bluffs at showdown."""
    return (data().get("bluff_by_size") or {}).get(f"{street}|{kind}")


def bluff_mult(street: str) -> float:
    """Correction for flop/turn showdown selection (bluffs that give up are never shown)."""
    return float((data().get("bluff_street_mult") or {}).get(street, 1.0))


def curve_mean(points: list) -> float:
    n = sum(p[2] for p in points) or 1.0
    return sum(p[1] * p[2] for p in points) / n
