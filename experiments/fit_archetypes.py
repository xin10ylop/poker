"""Refit the player-type prototypes on real regulars (k-means in HUD-stat space).

  python experiments/fit_archetypes.py --db <tracker db json> --min-hands 200 --k 5 --out results/real/archetypes.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from pokerbrain.opponents import OpponentDB  # noqa: E402

FEATS = ["vpip", "pfr", "afq", "fcb", "wtsd", "tb"]


def features(p):
    return [p.stat("vpip"), p.stat("pfr"), p.afq(), p.stat("fold_to_cbet"), p.stat("wtsd"), p.stat("threebet")]


def kmeans(X, k, seed, iters=100):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        lab = d.argmin(1)
        newC = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
        if np.allclose(newC, C):
            break
        C = newC
    d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
    lab = d.argmin(1)
    return C, lab, float(d.min(1).sum())


def name_clusters(cent):
    """Assign the existing archetype names by the cluster's position (vpip, pfr, wtsd)."""
    names = {}
    order = sorted(range(len(cent)), key=lambda j: cent[j]["vpip"])
    remaining = set(range(len(cent)))
    names[order[0]] = "nit"; remaining.discard(order[0])
    loosest = sorted(remaining, key=lambda j: -cent[j]["vpip"])
    for j in loosest:
        c = cent[j]
        if c["pfr"] / max(1e-6, c["vpip"]) < 0.45 and "calling_station" not in names.values() and c["wtsd"] >= 0.24:
            names[j] = "calling_station"
        elif c["pfr"] / max(1e-6, c["vpip"]) < 0.45 and "weak_passive" not in names.values():
            names[j] = "weak_passive"
        elif c["pfr"] / max(1e-6, c["vpip"]) >= 0.45 and c["vpip"] >= 0.27 and "lag" not in names.values():
            names[j] = "lag"
        elif "tag" not in names.values():
            names[j] = "tag"
        else:
            names[j] = f"cluster{j}"
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--min-hands", type=int, default=200)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default="results/real/archetypes.json")
    a = ap.parse_args()
    db = OpponentDB(a.db)
    regs = [p for p in db.profiles.values() if p.hands >= a.min_hands]
    X = np.array([features(p) for p in regs])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    best = None
    for seed in range(20):
        C, lab, inertia = kmeans(Z, a.k, seed)
        if best is None or inertia < best[2]:
            best = (C, lab, inertia)
    C, lab, _ = best
    cent = [{f: round(float(v), 4) for f, v in zip(FEATS, C[j] * sd + mu)} for j in range(a.k)]
    sizes = [int((lab == j).sum()) for j in range(a.k)]
    names = name_clusters(cent)
    out = {"regulars": len(regs), "min_hands": a.min_hands, "feature_std": {f: round(float(s), 4) for f, s in zip(FEATS, sd)},
           "prototypes": {names[j]: cent[j] for j in range(a.k)}, "sizes": {names[j]: sizes[j] for j in range(a.k)}}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"{len(regs)} regulars with >= {a.min_hands} hands; feature std {out['feature_std']}")
    for j in range(a.k):
        print(f"  {names[j]:16s} n={sizes[j]:4d}  " + "  ".join(f"{f} {cent[j][f]:.3f}" for f in FEATS))


if __name__ == "__main__":
    main()
