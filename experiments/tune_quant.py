"""Tune structural knobs of the quant engine against the oracle benchmark.

Candidate menus depend only on legal actions and pot size, so the stored oracle
EVs stay valid for any engine setting: re-running the engine is enough.
Tune on 'dev', confirm on 'test'.

  python experiments/tune_quant.py --set dev
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.bench import interesting, load, score, split  # noqa: E402

CONFIGS = {
    "baseline": {},
    "realize_oop_low": {"REALIZATION": {("flop", False): 0.80, ("turn", False): 0.86}},
    "realize_high": {"REALIZATION": {("flop", True): 1.0, ("flop", False): 0.92, ("turn", True): 1.0,
                                     ("turn", False): 0.96}},
    "fast_learning": {"PRIOR_SCALE": 0.5},
    "slow_learning": {"PRIOR_SCALE": 1.6},
    "less_flop_bluff": {"STREET_BETA": {"flop": 1.4, "turn": 1.2}},
    "more_flop_bluff": {"STREET_BETA": {"flop": 2.4, "turn": 1.7}},
    "low_bluff_prior": {"PRIORS": {"river_bluff": (0.15, 6), "bigbet_bluff": (0.18, 6), "smallbet_bluff": (0.14, 6)}},
    "low_bluff_all": {"PRIORS": {"river_bluff": (0.15, 6), "bigbet_bluff": (0.18, 6), "smallbet_bluff": (0.14, 6)},
                      "STREET_BETA": {"flop": 1.5, "turn": 1.2}},
    "low_bluff_all_strong": {"PRIORS": {"river_bluff": (0.12, 8), "bigbet_bluff": (0.15, 8),
                                        "smallbet_bluff": (0.12, 8)}, "STREET_BETA": {"flop": 1.3, "turn": 1.1}},
    "low_bluff_realize": {"PRIORS": {"river_bluff": (0.15, 6), "bigbet_bluff": (0.18, 6), "smallbet_bluff": (0.14, 6)},
                          "STREET_BETA": {"flop": 1.5, "turn": 1.2},
                          "REALIZATION": {("flop", False): 0.80, ("turn", False): 0.86}},
}


_APPLIED = set()


def apply_config(cfg: dict) -> None:
    key = repr(sorted(cfg.items(), key=lambda kv: kv[0]))
    if key in _APPLIED:        # worker processes run many tasks: patch once, never compound
        return
    _APPLIED.add(key)
    import pokerbrain.opponents as opp
    import pokerbrain.quant as q
    import pokerbrain.villain as vil
    if "REALIZATION" in cfg:
        q.REALIZATION.update(cfg["REALIZATION"])
    if "PRIOR_SCALE" in cfg:
        for k, (m, s) in list(opp.PRIORS.items()):
            opp.PRIORS[k] = (m, s * cfg["PRIOR_SCALE"])
    if "STREET_BETA" in cfg:
        vil.STREET_BETA.update(cfg["STREET_BETA"])
    if "PRIORS" in cfg:
        opp.PRIORS.update(cfg["PRIORS"])


def choose(args):
    cfg_name, sd = args
    from pokerbrain.quant import QuantEngine
    from pokerbrain.spots import Spot
    apply_config(CONFIGS[cfg_name])
    sp = Spot.from_json({k: v for k, v in sd.items() if k in Spot.__dataclass_fields__})
    rep = QuantEngine(sp.db(), rng=random.Random(1)).analyze(sp.view())
    return sd["spot_id"], rep.best.id


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="dev")
    ap.add_argument("--configs", default=",".join(CONFIGS))
    a = ap.parse_args()
    spots = [s for s in split(load())[a.set] if interesting(s)]
    results = {}
    for name in a.configs.split(","):
        with Pool(4) as pool:     # fresh processes so module patches don't leak between configs
            picks = dict(pool.map(choose, [(name, s) for s in spots]))
        st = {sid: {oid: 1.0} for sid, oid in picks.items()}
        results[name] = score(spots, st)
        print(f"{name:18s} {json.dumps({k: results[name][k] for k in ('n', 'ev_loss_bb', 'se', 'best_pct', 'blunders_gt10bb')})}",
              flush=True)
