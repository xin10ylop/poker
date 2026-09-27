"""How accurate are the 'psychology' reads?  Jev vs the statistical model vs truth.

For every benchmark spot where hero faces a postflop bet, the oracle posterior
over the villain's actual strategy gives the TRUE probability that he is
bluffing (holding a weak hand: effective strength < 0.45 on this board).
We compare:
  * Jev's P(bluff)                      (calibrated intuition model)
  * the engine's range-model P(bluff)   (Bayesian stats model)
  * blends of the two
by Brier score and correlation.

  python experiments/calibration.py
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.bench import get_reads, load, split  # noqa: E402

OUT = "results/bench/calibration.json"


def truth_and_model(sd: dict):
    from pokerbrain.quant import QuantEngine
    from pokerbrain.spots import Spot, villain_posterior
    from pokerbrain.villain import IDX, strength_vec
    sp = Spot.from_json({k: v for k, v in sd.items() if k in Spot.__dataclass_fields__})
    v = sp.view()
    if v.street == "preflop" or v.legal.call_amount <= 0:
        return None
    s = strength_vec(tuple(v.board))
    cands, post = villain_posterior(sp)
    true_bluff = float(sum(p for c, p in zip(cands, post) if s[IDX[c]] < 0.45))
    rep = QuantEngine(sp.db(), rng=random.Random(1)).analyze(v, with_ev=False)
    w = rep.villains[0].w
    model_bluff = float((w * (s < 0.45)).sum() / w.sum())
    return {"spot_id": sd["spot_id"], "style": sd["villain_style"], "street": sd["street"],
            "true": true_bluff, "model": model_bluff}


def brier(xs, ys):
    return sum((x - y) ** 2 for x, y in zip(xs, ys)) / len(xs)


def corr(xs, ys):
    return float(np.corrcoef(xs, ys)[0, 1])


if __name__ == "__main__":
    spots = split(load())["all"]
    with Pool(4) as pool:
        rows = [r for r in pool.map(truth_and_model, spots) if r]
    reads = get_reads([s for s in spots if s["spot_id"] in {r["spot_id"] for r in rows}])
    for r in rows:
        rd = reads.get(r["spot_id"], {})
        vals = [x.get("villain_bluffing") for x in rd.values() if "villain_bluffing" in x]
        r["jev"] = vals[0] if vals else None
    rows = [r for r in rows if r["jev"] is not None]
    t = [r["true"] for r in rows]
    res = {"n": len(rows), "mean_true": sum(t) / len(t)}
    for name, xs in (("model", [r["model"] for r in rows]), ("jev", [r["jev"] for r in rows])):
        res[name] = {"brier": round(brier(xs, t), 4), "corr": round(corr(xs, t), 3), "mean": round(sum(xs) / len(xs), 3)}
    for wgt in (0.25, 0.35, 0.5, 0.65):
        xs = [(1 - wgt) * r["model"] + wgt * r["jev"] for r in rows]
        res[f"blend_{wgt}"] = {"brier": round(brier(xs, t), 4), "corr": round(corr(xs, t), 3)}
    base = [res["mean_true"]] * len(t)
    res["constant_baseline_brier"] = round(brier(base, t), 4)
    by = {}
    for r in rows:
        by.setdefault(r["style"], []).append(r)
    res["by_style"] = {k: {"n": len(v), "true": round(sum(x["true"] for x in v) / len(v), 2),
                           "model": round(sum(x["model"] for x in v) / len(v), 2),
                           "jev": round(sum(x["jev"] for x in v) / len(v), 2)} for k, v in by.items()}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"summary": res, "rows": rows}, open(OUT, "w"), indent=1)
    print(json.dumps(res, indent=1))
