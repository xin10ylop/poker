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
from pokerbrain.texture import hand_strength  # noqa: E402

COMMIT_EDGES = [0.0, 0.35, 0.6, 0.9, 1.01]
COMMIT_KEYS = ["0-0.35", "0.35-0.6", "0.6-0.9", "0.9+"]


def _logit(q):
    import math
    q = min(1 - 1e-4, max(1e-4, q))
    return math.log(q / (1 - q))

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
    ap.add_argument("--save-db", default="", help="also save the tracker's profiles (JSON) for further fitting")
    ap.add_argument("--bluff-mult", default="flop=1.0,turn=1.0,river=1.0",
                    help="showdown-selection correction for the bluff share by street (fit with experiments/real_cards.py)")
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
    db_band = {"6": OpponentDB(platform="real"), "9": OpponentDB(platform="real")}
    buckets = defaultdict(lambda: defaultdict(lambda: [0, 0.0, 0, 0]))    # key -> bucket -> [n, sum size, fold, raise]
    cbuckets = defaultdict(lambda: defaultdict(lambda: [0, 0]))            # (key, size b) -> commit b -> [n, folds]
    shown = defaultdict(lambda: defaultdict(lambda: [0, 0, 0.0]))          # street|kind -> size b -> [n, bluffs, sum size]
    used = 0
    for h in hands:
        r = replay(h)
        if r is None:
            continue
        used += 1
        hist = r.history
        for st, nb in (("flop", 3), ("turn", 4), ("river", 5)):        # what the street's last bettor showed down
            acts = [x for x in hist.actions if x.street == st]
            bets = [x for x in acts if x.kind in ("bet", "raise")]
            if not bets or len(hist.board) < nb or bets[-1].seat not in r.shown:
                continue
            last = bets[-1]
            size = last.added / max(1, last.pot_before)
            b = next(i for i in range(len(EDGES) - 1) if size < EDGES[i + 1])
            c = shown[f"{st}|{'raise' if last.kind == 'raise' else 'bet'}"][b]
            c[0] += 1; c[1] += hand_strength(tuple(r.shown[last.seat]), hist.board[:nb]) < 0.45; c[2] += size
        for dp in r.points:
            v = dp.view
            if v.street == "preflop":
                continue
            f = situation_of(v, dp.seat)
            if not f["facing"]:
                continue
            y = label_of(dp.kind, True)
            cb = next(i for i in range(len(COMMIT_EDGES) - 1) if f["commit"] < COMMIT_EDGES[i + 1])
            kind = "raise" if f["facing_raise"] else ("cbet" if f.get("vs_cbet") else "bet")
            key = f"{v.street}|{kind}|{'mw' if f['multiway'] else 'hu'}"
            keys = [key] if kind != "cbet" else [key, key.replace("|cbet|", "|bet|")]   # c-bets also count as bets
            b = next(i for i in range(len(EDGES) - 1) if f["size"] < EDGES[i + 1])
            for kk in keys:
                c = buckets[kk][b]
                c[0] += 1; c[1] += f["size"]; c[2] += y == "fold"; c[3] += y == "raise"
            cc = cbuckets[(key, b)][cb]
            cc[0] += 1; cc[1] += y == "fold"
        db.update(r.history)
        db_band["6" if len(hist.names) <= 6 else "9"].update(r.history)
    fold_curve, raise_curve = {}, {}
    for key, bs in buckets.items():
        pts = [(c[1] / c[0], c[2] / c[0], c[3] / c[0], c[0]) for _, c in sorted(bs.items()) if c[0] >= MIN_N]
        if len(pts) < 2:
            continue
        folds = pava([p[1] for p in pts], [p[3] for p in pts])
        fold_curve[key] = [[round(p[0], 3), round(fv, 4), p[3]] for p, fv in zip(pts, folds)]
        raise_curve[key] = [[round(p[0], 3), round(p[2], 4), p[3]] for p in pts]
    def pool_means(d):
        out = {}
        for name in PRIORS:
            k = sum(p.counts.get(name, [0, 0])[0] for p in d.profiles.values())
            n = sum(p.counts.get(name, [0, 0])[1] for p in d.profiles.values())
            if n >= 200:
                out[name] = round(k / n, 4)
        return out
    prior_means = pool_means(db)
    by_seats = {band: pool_means(d) for band, d in db_band.items()}
    # bluff share of shown bets by size (river is unbiased: a called bettor must show); size-tell priors
    bluff_by_size = {}
    for key, bs in shown.items():
        pts = [[round(c[2] / c[0], 3), round(c[1] / c[0], 4), c[0]] for _, c in sorted(bs.items()) if c[0] >= 30]
        if len(pts) >= 2:
            bluff_by_size[key] = pts
    big = [c for key, bs in shown.items() if key in ("turn|bet", "river|bet") for b, c in bs.items() if EDGES[b] >= 0.7]
    small = [c for key, bs in shown.items() if key in ("turn|bet", "river|bet") for b, c in bs.items() if EDGES[b] < 0.7]
    for name, rows in (("bigbet_bluff", big), ("smallbet_bluff", small)):
        n = sum(c[0] for c in rows)
        if n >= 200:
            prior_means[name] = round(sum(c[1] for c in rows) / n, 4)
    # commitment: fold logit shift per commitment bucket relative to the same street/kind/size cell
    cshift = {}
    for ci, ck in enumerate(COMMIT_KEYS[1:], start=1):
        num = den = 0.0
        for (key, b), cbs in cbuckets.items():
            pooled_n = sum(c[0] for c in cbs.values()); pooled_f = sum(c[1] for c in cbs.values())
            c = cbs.get(ci)
            if not c or c[0] < 30 or pooled_n < 60:
                continue
            num += c[0] * (_logit(c[1] / c[0]) - _logit(pooled_f / pooled_n)); den += c[0]
        if den >= 200:
            cshift[ck] = round(num / den, 4)
    mults = {k: float(v) for k, v in (kv.split("=") for kv in a.bluff_mult.split(","))}
    from experiments.fit_prior_counts import prior_counts_from_db
    counts = prior_counts_from_db(db)
    out = {"source": a.source, "time_slice": a.time_slice, "hands": used, "prior_means": prior_means,
           "prior_means_by_seats": by_seats, "prior_counts": {k: v["count"] for k, v in counts.items()},
           "commit_shift": cshift, "bluff_by_size": bluff_by_size, "bluff_street_mult": mults,
           "temper": {"postflop": a.temper, "preflop": a.pf_temper},
           "fold_curve": fold_curve, "raise_curve": raise_curve}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    if a.save_db:
        db.path = a.save_db
        db.save()
    print(f"fitted on {used} hands -> {a.out} ({time.time() - t0:.0f}s)")
    print("prior pseudo-counts:", {k: v["count"] for k, v in counts.items()})
    print("commit shift (logit):", cshift)
    print("bluff by size:", {k: [(x[0], x[1]) for x in v] for k, v in bluff_by_size.items()})
    print("seat bands:", {b: {k: m[k] for k in ("vpip", "pfr", "threebet") if k in m} for b, m in by_seats.items()})
    for k in sorted(fold_curve):
        print(f"  {k:16s} " + " ".join(f"{x:.2f}:{p:.2f}" for x, p, _ in fold_curve[k]))


if __name__ == "__main__":
    main()
