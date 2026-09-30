"""Empirical-Bayes pseudo-counts for the tracker's priors, from the real player pool.

For each stat, the between-player variance of the true rates is the observed variance of the players'
raw rates minus the binomial sampling noise.  A Beta prior with mean m and pseudo-count s has variance
m(1-m)/(s+1), so s = m(1-m)/var_between - 1: how many observed hands it takes before a player's own
sample outweighs the pool.  Real players vary a lot in how many hands they play (VPIP: trust the player
after a few hands) and very little in 3-bet / raise / showdown rates (trust the pool for a long time).

  python experiments/fit_prior_counts.py --db <tracker db json> [--population pokerbrain/data/population.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from pokerbrain.opponents import PRIORS, OpponentDB  # noqa: E402


def prior_counts_from_db(db: OpponentDB, min_n: int = 60, min_players: int = 30) -> dict:
    out = {}
    for stat in PRIORS:
        rows = [(p.counts[stat][0], p.counts[stat][1]) for p in db.profiles.values()
                if stat in p.counts and p.counts[stat][1] >= min_n]
        if len(rows) < min_players:
            continue
        k = np.array([r[0] for r in rows], float)
        n = np.array([r[1] for r in rows], float)
        r = k / n
        m = float(k.sum() / n.sum())
        noise = float(np.mean(m * (1 - m) / n))
        var_between = max(1e-6, float(r.var()) - noise)
        s = float(np.clip(m * (1 - m) / var_between - 1.0, 3.0, 400.0))
        out[stat] = {"mean": round(m, 4), "count": round(s, 1), "players": len(rows),
                     "sd_between": round(var_between ** 0.5, 4), "old_count": PRIORS[stat][1]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--population", default="")
    ap.add_argument("--min-n", type=int, default=60)
    ap.add_argument("--out", default="results/real/prior_counts.json")
    a = ap.parse_args()
    out = prior_counts_from_db(OpponentDB(a.db), a.min_n)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    for stat, v in out.items():
        print(f"  {stat:18s} mean {v['mean']:.3f}  sd between players {v['sd_between']:.3f}  "
              f"pseudo-count {v['count']:6.1f}  (was {v['old_count']})  players {v['players']}")
    if a.population:
        pop = json.load(open(a.population))
        pop["prior_counts"] = {k: v["count"] for k, v in out.items()}
        json.dump(pop, open(a.population, "w"), indent=1)
        print("prior_counts written to", a.population)


if __name__ == "__main__":
    main()
