"""One-pass comparison of opponent-model variants on REAL players (held-out later half of the hands).

Every variant predicts the same sampled decisions from the same tracked profiles (snapshotted before the
hand), so differences are purely the model.  Log-loss, lower is better.

  python experiments/real_eval.py --files 'real/ps25/*.phhs' --population results/real/population_h1.json \
      --temper 0.35 --pf-temper 0.1 --rate 0.025 --out results/real/ps25_eval.json
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import math
import os
import sys
import time
from collections import defaultdict
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.real_players import (CLASSES_FACING, CLASSES_OPEN, hud_probs, label_of, logloss,  # noqa: E402
                                      model_probs, pop_key, situation_of)
import pokerbrain.opponents as opp  # noqa: E402
import pokerbrain.population as popm  # noqa: E402
import pokerbrain.villain as vil  # noqa: E402
from pokerbrain.adapters.phh import load_phhs, replay  # noqa: E402
from pokerbrain.cards import stable_hash  # noqa: E402
from pokerbrain.opponents import OpponentDB, PlayerProfile  # noqa: E402
from pokerbrain.villain import VillainParams  # noqa: E402

VARIANTS: dict = {}
POP: dict = {}


_ORIG = {"commit_shift": popm.commit_shift, "bluff_curve": popm.bluff_curve, "prior_mean": popm.prior_mean}


def configure(v: dict) -> None:
    popm._DATA = dict(POP) if v["population"] else {}
    if v["population"] and not v.get("new_pieces", True):
        popm._DATA.pop("prior_counts", None)              # the earlier build: curves + tempering only
        popm.commit_shift = lambda c: 0.0
        popm.bluff_curve = lambda st, kind="bet": None
        popm.prior_mean = lambda st, seats=None: None
    else:
        popm.commit_shift, popm.bluff_curve, popm.prior_mean = _ORIG["commit_shift"], _ORIG["bluff_curve"], _ORIG["prior_mean"]
    opp.apply_population_priors()
    vil.USE_POPULATION = v["population"]
    vil.POP_THRESHOLDS = v.get("thresholds", "current")
    vil.TEMPER = v.get("temper", 0.0)
    vil.PF_TEMPER = v.get("pf_temper", 0.0)


def _init(variants, pop):
    VARIANTS.update(variants)
    POP.update(pop)


def _predict(item):
    out = {}
    for name, v in VARIANTS.items():
        configure(v)
        prof = item["profile"]
        try:
            out[name] = model_probs(item["view"], item["seat"], VillainParams.from_profile(prof), item["f"])
            if v.get("unknown"):
                out[name + "|unknown"] = model_probs(item["view"], item["seat"],
                                                     VillainParams.from_profile(PlayerProfile(name="?")), item["f"])
            if v.get("hud"):
                out[name + "|hud"] = hud_probs(prof, item["f"])
        except Exception as e:  # noqa: BLE001
            out[name] = {"error": repr(e)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", required=True)
    ap.add_argument("--population", required=True)
    ap.add_argument("--temper", type=float, default=0.35)
    ap.add_argument("--pf-temper", type=float, default=0.1)
    ap.add_argument("--rate", type=float, default=0.025)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--out", default="results/real/ps25_eval.json")
    a = ap.parse_args()
    pop = json.load(open(a.population))
    variants = {
        "old (research priors, formula)": {"population": False, "unknown": True, "hud": True},
        "curves + tempering (previous build)": {"population": True, "temper": a.temper, "pf_temper": a.pf_temper,
                                                 "new_pieces": False},
        "full real-data model (audit build)": {"population": True, "temper": a.temper, "pf_temper": a.pf_temper,
                                                "unknown": True, "hud": True},
    }
    t0 = time.time()
    hands = []
    for fn in sorted(glob.glob(a.files)):
        hands += load_phhs(fn)
    hands.sort(key=lambda h: int(h.get("hand", 0)))
    start = len(hands) // 2
    db = OpponentDB(platform="real")
    thr = int(a.rate * 2 ** 32)
    items = []
    for k, h in enumerate(hands):
        r = replay(h)
        if r is None:
            continue
        if k >= start:
            for dp in r.points:
                if dp.view.street == "preflop" or stable_hash(r.history.hand_id, dp.seat, len(dp.view.actions)) >= thr:
                    continue
                prof = db.get(dp.name)
                f = situation_of(dp.view, dp.seat)
                items.append({"view": dp.view, "seat": dp.seat, "f": f, "y": label_of(dp.kind, f["facing"]),
                              "n": prof.samples("vpip"), "profile": copy.deepcopy(prof)})
        db.update(r.history)
    print(f"{len(items)} sampled decisions from the later half ({time.time() - t0:.0f}s)", flush=True)
    with Pool(a.procs, initializer=_init, initargs=(variants, pop)) as pool:
        preds = pool.map(_predict, items, chunksize=32)
    half = len(items) // 2
    popc = defaultdict(lambda: defaultdict(float))
    for it in items[:half]:
        popc[pop_key(it["f"])][it["y"]] += 1
    res = defaultdict(lambda: defaultdict(list))
    for it, pr in zip(items[half:], preds[half:]):
        classes = CLASSES_FACING if it["f"]["facing"] else CLASSES_OPEN
        if any(isinstance(P, dict) and "error" in P or P is None for P in pr.values()):
            continue
        c = popc.get(pop_key(it["f"]))
        tot = sum(c.values()) if c else 0
        pr = dict(pr)
        pr["population frequency table"] = ({k: (c.get(k, 0) + 1) / (tot + len(classes)) for k in classes} if c
                                            else {k: 1 / len(classes) for k in classes})
        n = it["n"]
        nb = "0-19" if n < 20 else "20-99" if n < 100 else "100-499" if n < 500 else "500+"
        grp = "facing_bet" if it["f"]["facing"] else "not_facing"
        for m, P in pr.items():
            ll = logloss(P, it["y"])
            for g in ("ALL", grp, f"hands {nb}", f"street {it['f']['street']}"):
                res[g][m].append(ll)
    table = {g: {m: [round(sum(v) / len(v), 4), len(v)] for m, v in d.items()} for g, d in res.items()}
    # paired differences vs the old model (all decisions)
    base = "old (research priors, formula)"
    paired = {}
    for m in res["ALL"]:
        dd = [x - y for x, y in zip(res["ALL"][m], res["ALL"][base])]
        mu = sum(dd) / len(dd)
        se = math.sqrt(sum((x - mu) ** 2 for x in dd) / max(1, len(dd) - 1) / len(dd))
        paired[m] = [round(mu, 4), round(se, 4)]
    json.dump({"items": len(items), "scored": len(res["ALL"][base]), "table": table, "paired_vs_old": paired,
               "temper": a.temper, "pf_temper": a.pf_temper}, open(a.out, "w"), indent=1)
    print(f"scored {len(res['ALL'][base])} held-out decisions")
    for g in ("ALL", "facing_bet", "not_facing", "hands 0-19", "hands 20-99", "hands 100-499", "hands 500+"):
        print(f"\n{g} (n={table[g][base][1]})")
        for m, (ll, _) in sorted(table[g].items(), key=lambda kv: kv[1][0]):
            extra = f"   paired vs old {paired[m][0]:+.4f} ± {paired[m][1]:.4f}" if g == "ALL" else ""
            print(f"  {ll:.4f}  {m}{extra}")


if __name__ == "__main__":
    main()
