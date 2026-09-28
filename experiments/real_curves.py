"""Population behaviour curves measured on real hands (the targets the villain model should match).

  python experiments/real_curves.py --files 'real/ps25/*.phhs' --max-files 100 --out results/real/ps25_curves.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.real_players import label_of, situation_of  # noqa: E402
from pokerbrain.adapters.phh import load_phhs, replay  # noqa: E402
from pokerbrain.texture import hand_features  # noqa: E402

SIZE_EDGES = [0.0, 0.3, 0.45, 0.6, 0.8, 1.05, 1.5, 99]


def size_bucket(x):
    for lo, hi in zip(SIZE_EDGES, SIZE_EDGES[1:]):
        if lo <= x < hi:
            return f"{lo:.2f}-{hi:.2f}"
    return "big"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", required=True)
    ap.add_argument("--max-files", type=int, default=100)
    ap.add_argument("--out", default="results/real/ps25_curves.json")
    a = ap.parse_args()
    files = sorted(glob.glob(a.files))[: a.max_files]
    facing = defaultdict(lambda: defaultdict(int))     # (street, facing_raise, size bucket, multiway) -> action counts
    opening = defaultdict(lambda: defaultdict(int))    # (street, situation, multiway) -> check/bet counts
    sizes = defaultdict(list)                          # street -> bet sizes (fraction of pot) of first bets
    sd_strength = defaultdict(lambda: defaultdict(int))  # (street, size bucket) -> bettor's shown hand class
    hands = 0
    for fn in files:
        for h in load_phhs(fn):
            r = replay(h)
            if r is None:
                continue
            hands += 1
            hist = r.history
            for dp in r.points:
                v = dp.view
                if v.street == "preflop":
                    continue
                f = situation_of(v, dp.seat)
                y = label_of(dp.kind, f["facing"])
                if f["facing"]:
                    facing[(v.street, f["facing_raise"], size_bucket(f["size"]), f["multiway"])][y] += 1
                else:
                    opening[(v.street, f["situation"], f["multiway"])][y] += 1
                    if y == "raise":
                        sizes[v.street].append(round(dp.amount / max(1, v.pot), 3))
            # what did bettors show down?  (last bet/raise of each street by a player who showed)
            board = hist.board
            for st, nboard in (("flop", 3), ("turn", 4), ("river", 5)):
                acts = [x for x in hist.actions if x.street == st]
                bets = [x for x in acts if x.kind in ("bet", "raise")]
                if not bets:
                    continue
                last = bets[-1]
                if last.seat not in r.shown or len(board) < nboard:
                    continue
                size = last.added / max(1, last.pot_before)
                cls = hand_features(tuple(r.shown[last.seat]), board[:nboard]).strength_class
                sd_strength[(st, size_bucket(size))][cls] += 1
    def fmt(d):
        return {"|".join(map(str, k)): dict(v) for k, v in sorted(d.items(), key=lambda kv: str(kv[0]))}
    out = {"hands": hands, "facing": fmt(facing), "opening": fmt(opening),
           "bet_size_quantiles": {st: [sorted(v)[int(q * (len(v) - 1))] for q in (0.1, 0.25, 0.5, 0.75, 0.9)]
                                  for st, v in sizes.items() if v},
           "showdown_strength": fmt(sd_strength)}
    json.dump(out, open(a.out, "w"), indent=1)
    print("hands", hands)
    print("\nfold/call/raise facing a bet (heads-up), by size (bet/pot):")
    for st in ("flop", "turn", "river"):
        for fr in (False, True):
            for sb in [f"{lo:.2f}-{hi:.2f}" for lo, hi in zip(SIZE_EDGES, SIZE_EDGES[1:])]:
                c = facing.get((st, fr, sb, False))
                if not c or sum(c.values()) < 50:
                    continue
                n = sum(c.values())
                print(f"  {st:5s} {'vs raise' if fr else 'vs bet  '} size {sb:10s} n={n:6d}  fold {c['fold'] / n:.2f}  "
                      f"call {c['call'] / n:.2f}  raise {c['raise'] / n:.2f}")
    print("\nbet frequency when not facing a bet (heads-up):")
    for k, c in sorted(opening.items(), key=lambda kv: str(kv[0])):
        if k[2]:
            continue
        n = sum(c.values())
        print(f"  {k[0]:5s} {k[1]:6s} n={n:6d}  bet {c['raise'] / n:.2f}")
    print("\nfirst-bet sizes (fraction of pot) quantiles 10/25/50/75/90:", out["bet_size_quantiles"])
    print("\nhands shown by the last bettor of the street, by size:")
    for k, c in sorted(sd_strength.items(), key=lambda kv: str(kv[0])):
        n = sum(c.values())
        if n < 30:
            continue
        print(f"  {k[0]:5s} size {k[1]:10s} n={n:5d}  " + "  ".join(f"{cl} {c[cl] / n:.2f}" for cl in sorted(c)))


if __name__ == "__main__":
    main()
