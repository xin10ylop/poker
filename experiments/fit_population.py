"""Fit PokerBrain's population model to real hand histories.

Writes a population file the villain model and tracker load (pokerbrain/data/population.json by default):
  prior_means  - population averages of every tracked stat (the tracker's priors for unknown players)
  fold_curve   - P(fold) facing a bet, by street x bet-or-raise x heads-up/multiway, as a function of size
  raise_curve  - P(raise) in the same situations
Hands are ordered by hand id (time); --time-slice picks the part used, so a later part can be held out.

  python experiments/fit_population.py --files 'real/ps25/*.phhs' --time-slice 0:0.5 --out pop_h1.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.real_players import label_of, situation_of  # noqa: E402
from pokerbrain.adapters.phh import load_phhs, replay  # noqa: E402
from pokerbrain.opponents import PRIORS, OpponentDB  # noqa: E402

EDGES = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.05, 1.3, 1.7, 2.5, 99.0]
MIN_N = 40


def pava(xs, ws):
    """Weighted isotonic (non-decreasing) regression."""
    blocks = [[x, w, 1] for x, w in zip(xs, ws)]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0]:
            a, b = blocks[i], blocks[i + 1]
            w = a[1] + b[1]
            blocks[i] = [(a[0] * a[1] + b[0] * b[1]) / w, w, a[2] + b[2]]
            del blocks[i + 1]
            i = max(0, i - 1)
        else:
            i += 1
    out = []
    for v, _, k in blocks:
        out += [v] * k
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", required=True)
    ap.add_argument("--time-slice", default="0:1")
    ap.add_argument("--out", default="pokerbrain/data/population.json")
    ap.add_argument("--source", default="HandHQ PokerStars 25NL, July 2009")
    ap.add_argument("--temper", type=float, default=0.35, help="postflop likelihood tempering (experiments/real_cards.py)")
    ap.add_argument("--pf-temper", type=float, default=0.45, help="preflop likelihood tempering")
    a = ap.parse_args()
    t0 = time.time()
    hands = []
    for fn in sorted(glob.glob(a.files)):
        hands += load_phhs(fn)
    hands.sort(key=lambda h: int(h.get("hand", 0)))
    lo, hi = (float(x) for x in a.time_slice.split(":"))
    hands = hands[int(lo * len(hands)): int(hi * len(hands))]
    print(f"{len(hands)} hands in slice {a.time_slice} ({time.time() - t0:.0f}s)", flush=True)
    db = OpponentDB(platform="real")
    buckets = defaultdict(lambda: defaultdict(lambda: [0, 0.0, 0, 0]))    # key -> bucket -> [n, sum size, fold, raise]
    used = 0
    for h in hands:
        r = replay(h)
        if r is None:
            continue
        used += 1
        for dp in r.points:
            v = dp.view
            if v.street == "preflop":
                continue
            f = situation_of(v, dp.seat)
            if not f["facing"]:
                continue
            y = label_of(dp.kind, True)
            key = f"{v.street}|{'raise' if f['facing_raise'] else 'bet'}|{'mw' if f['multiway'] else 'hu'}"
            b = next(i for i in range(len(EDGES) - 1) if f["size"] < EDGES[i + 1])
            c = buckets[key][b]
            c[0] += 1; c[1] += f["size"]; c[2] += y == "fold"; c[3] += y == "raise"
        db.update(r.history)
    fold_curve, raise_curve = {}, {}
    for key, bs in buckets.items():
        pts = [(c[1] / c[0], c[2] / c[0], c[3] / c[0], c[0]) for _, c in sorted(bs.items()) if c[0] >= MIN_N]
        if len(pts) < 2:
            continue
        folds = pava([p[1] for p in pts], [p[3] for p in pts])
        fold_curve[key] = [[round(p[0], 3), round(fv, 4), p[3]] for p, fv in zip(pts, folds)]
        raise_curve[key] = [[round(p[0], 3), round(p[2], 4), p[3]] for p in pts]
    prior_means = {}
    for name in PRIORS:
        k = sum(p.counts.get(name, [0, 0])[0] for p in db.profiles.values())
        n = sum(p.counts.get(name, [0, 0])[1] for p in db.profiles.values())
        if n >= 200:
            prior_means[name] = round(k / n, 4)
    out = {"source": a.source, "time_slice": a.time_slice, "hands": used, "prior_means": prior_means,
           "temper": {"postflop": a.temper, "preflop": a.pf_temper},
           "fold_curve": fold_curve, "raise_curve": raise_curve}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"fitted on {used} hands -> {a.out} ({time.time() - t0:.0f}s)")
    for k in sorted(fold_curve):
        print(f"  {k:16s} " + " ".join(f"{x:.2f}:{p:.2f}" for x, p, _ in fold_curve[k]))


if __name__ == "__main__":
    main()
