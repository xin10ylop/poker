"""Weighted hand ranges over the 1326 two-card combos.

A Range maps combo -> weight in [0, 1].  Weights are *relative frequencies*
("this combo is in the range 40% of the time"), which is what Bayesian range
narrowing needs: every observed action multiplies each combo's weight by the
likelihood that the villain would have taken that action with that combo.

Range strings follow the usual poker notation:
    "22+, A2s+, KTs+, QJs, ATo+, KQo, 76s:0.5, K9s-KJs, 22-55, AK"
A ":w" suffix sets the weight of that token.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Callable, Iterable

from .cards import (ALL_CLASSES, ALL_COMBOS, RANK_VALUE, RANKS, Combo,
                    combos_of_class, hand_class)


def _pair_classes(lo: int, hi: int) -> list[str]:
    return [RANKS[r] * 2 for r in range(lo, hi + 1)]


def _expand_token(tok: str) -> list[str]:
    """Expand one range token (no weight) into hand classes."""
    tok = tok.strip()
    if not tok:
        return []
    if "-" in tok:  # ranges like 22-55 or K9s-KJs
        a, b = [t.strip() for t in tok.split("-")]
        if len(a) == 2 and a[0] == a[1]:  # pairs
            lo, hi = sorted((RANK_VALUE[a[0]], RANK_VALUE[b[0]]))
            return _pair_classes(lo, hi)
        hi_card = a[0]
        if b[0] != hi_card:
            raise ValueError(f"bad range token {tok!r}")
        suffix = a[2:] if len(a) > 2 else ""
        lo, hi = sorted((RANK_VALUE[a[1]], RANK_VALUE[b[1]]))
        out = []
        for r in range(lo, hi + 1):
            out.extend(_with_suffix(hi_card + RANKS[r], suffix))
        return out
    plus = tok.endswith("+")
    base = tok[:-1] if plus else tok
    if len(base) == 2 and base[0] == base[1]:  # pair
        r = RANK_VALUE[base[0]]
        return _pair_classes(r, 12) if plus else [base]
    if len(base) not in (2, 3):
        raise ValueError(f"bad range token {tok!r}")
    hi_card, lo_card = base[0], base[1]
    if RANK_VALUE[lo_card] > RANK_VALUE[hi_card]:
        hi_card, lo_card = lo_card, hi_card
    suffix = base[2:] if len(base) == 3 else ""
    if not plus:
        return _with_suffix(hi_card + lo_card, suffix)
    out = []
    for r in range(RANK_VALUE[lo_card], RANK_VALUE[hi_card]):
        out.extend(_with_suffix(hi_card + RANKS[r], suffix))
    return out


def _with_suffix(two: str, suffix: str) -> list[str]:
    if two[0] == two[1]:
        return [two]
    if suffix in ("s", "o"):
        return [two + suffix]
    return [two + "s", two + "o"]


def parse_range_classes(text: str) -> dict[str, float]:
    """Parse a range string into {hand_class: weight}."""
    out: dict[str, float] = {}
    for raw in text.replace(";", ",").split(","):
        raw = raw.strip()
        if not raw:
            continue
        weight = 1.0
        if ":" in raw:
            raw, w = raw.split(":")
            weight = float(w)
        for cls in _expand_token(raw):
            out[cls] = max(out.get(cls, 0.0), weight)
    return out


class Range:
    __slots__ = ("w",)

    def __init__(self, weights: dict[Combo, float] | None = None):
        self.w: dict[Combo, float] = {k: v for k, v in (weights or {}).items() if v > 0}

    # ------------------------------------------------------------------ build
    @classmethod
    def parse(cls, text: str) -> "Range":
        w: dict[Combo, float] = {}
        for hc, weight in parse_range_classes(text).items():
            for combo in combos_of_class(hc):
                w[combo] = weight
        return cls(w)

    @classmethod
    def full(cls) -> "Range":
        return cls({c: 1.0 for c in ALL_COMBOS})

    @classmethod
    def from_classes(cls, class_weights: dict[str, float]) -> "Range":
        w: dict[Combo, float] = {}
        for hc, weight in class_weights.items():
            for combo in combos_of_class(hc):
                w[combo] = weight
        return cls(w)

    @classmethod
    def top_fraction(cls, frac: float, ranking: list[str]) -> "Range":
        """Top `frac` of all combos by a class ranking (partial last class)."""
        frac = max(0.0, min(1.0, frac))
        target = frac * 1326
        w: dict[Combo, float] = {}
        acc = 0.0
        for hc in ranking:
            combos = combos_of_class(hc)
            n = len(combos)
            if acc >= target:
                break
            take = min(1.0, (target - acc) / n)
            for c in combos:
                w[c] = take
            acc += n * take
        return cls(w)

    @classmethod
    def band(cls, lo_frac: float, hi_frac: float, ranking: list[str]) -> "Range":
        """Hands between two percentiles of a ranking (e.g. flatting range)."""
        top = cls.top_fraction(hi_frac, ranking)
        cut = cls.top_fraction(lo_frac, ranking)
        return cls({c: max(0.0, w - cut.w.get(c, 0.0)) for c, w in top.w.items()})

    # --------------------------------------------------------------- queries
    def copy(self) -> "Range":
        return Range(dict(self.w))

    def __len__(self) -> int:
        return len(self.w)

    def __iter__(self):
        return iter(self.w.items())

    def total(self) -> float:
        return sum(self.w.values())

    def fraction_of_all(self) -> float:
        return self.total() / 1326.0

    def weight(self, combo: Combo) -> float:
        return self.w.get(combo, 0.0)

    def remove_dead(self, dead: Iterable[str]) -> "Range":
        d = set(dead)
        return Range({c: v for c, v in self.w.items() if c[0] not in d and c[1] not in d})

    def scaled(self, fn: Callable[[Combo], float]) -> "Range":
        return Range({c: v * fn(c) for c, v in self.w.items()})

    def union_max(self, other: "Range") -> "Range":
        keys = set(self.w) | set(other.w)
        return Range({k: max(self.w.get(k, 0), other.w.get(k, 0)) for k in keys})

    def minus(self, other: "Range") -> "Range":
        return Range({c: max(0.0, v - other.w.get(c, 0.0)) for c, v in self.w.items()})

    def sample(self, rng: random.Random, dead: Iterable[str] = ()) -> Combo | None:
        d = set(dead)
        items = [(c, v) for c, v in self.w.items() if c[0] not in d and c[1] not in d]
        if not items:
            return None
        tot = sum(v for _, v in items)
        x = rng.random() * tot
        for c, v in items:
            x -= v
            if x <= 0:
                return c
        return items[-1][0]

    def class_weights(self) -> dict[str, float]:
        acc: dict[str, float] = defaultdict(float)
        for c, v in self.w.items():
            acc[hand_class(c)] += v
        return dict(acc)

    def to_eval7_list(self, resolution: int = 20) -> list:
        """Replicate combos in proportion to weight (eval7 MC ignores weights)."""
        from .cards import _E7
        out = []
        for (a, b), v in self.w.items():
            k = int(round(v * resolution))
            if k <= 0 and v > 0:
                k = 1 if v * resolution > 0.25 else 0
            if k:
                out.extend([((_E7[a], _E7[b]), 1.0)] * k)
        return out

    def describe(self, max_classes: int = 40) -> str:
        """Compact human-readable summary (for prompts)."""
        cw = self.class_weights()
        if not cw:
            return "(empty)"
        from .cards import class_size
        items = sorted(cw.items(), key=lambda kv: -kv[1])
        parts = []
        for hc, tot in items[:max_classes]:
            frac = tot / class_size(hc)
            parts.append(hc if frac > 0.95 else f"{hc}:{frac:.1f}")
        more = len(items) - max_classes
        return ", ".join(parts) + (f" (+{more} more classes)" if more > 0 else "")


def ordered_classes(ranking_scores: dict[str, float]) -> list[str]:
    return sorted(ALL_CLASSES, key=lambda c: -ranking_scores[c])
