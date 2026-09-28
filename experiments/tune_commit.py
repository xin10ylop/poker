"""Tune the villain model's stack-commitment threshold (villain.COMMIT_STRENGTH) on the benchmark.

Candidate menus depend only on legal actions and pot size, so stored oracle EVs stay valid.
Tune on 'dev', confirm on 'test'.   python experiments/tune_commit.py
"""
from __future__ import annotations

import json
import os
import random
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.bench import interesting, load, paired, quant_strategies, score, split  # noqa: E402

VALUES = [None, 0.55, 0.60, 0.65, 0.70, 0.75]


def _init(val):
    import pokerbrain.villain as vil
    vil.COMMIT_STRENGTH = val


def _pick(sd):
    from pokerbrain.quant import QuantEngine
    from pokerbrain.spots import Spot
    sp = Spot.from_json({k: v for k, v in sd.items() if k in Spot.__dataclass_fields__})
    return sd["spot_id"], QuantEngine(sp.db(), rng=random.Random(1)).analyze(sp.view()).best.id


if __name__ == "__main__":
    parts = {k: [s for s in v if interesting(s)] for k, v in split(load()).items()}
    out = {}
    for val in VALUES:
        with Pool(4, initializer=_init, initargs=(val,)) as pool:
            picks = dict(pool.map(_pick, parts["dev"] + parts["test"]))
        row = {}
        for k, L in parts.items():
            st = {s["spot_id"]: {picks[s["spot_id"]]: 1.0} for s in L}
            sc = score(L, st)
            m, se = paired(L, st, quant_strategies(L))
            row[k] = {"ev_loss": sc["ev_loss_bb"], "vs_old_engine": round(m, 3), "se": round(se, 3)}
        out[str(val)] = row
        print(f"COMMIT_STRENGTH={val}: " + "  ".join(f"{k} {r['ev_loss']:.2f} ({r['vs_old_engine']:+.2f}±{r['se']:.2f} vs old)"
                                                    for k, r in row.items()), flush=True)
    json.dump(out, open("results/bench/tune_commit.json", "w"), indent=1)
