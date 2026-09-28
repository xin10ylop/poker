"""Does PokerBrain's player study work on REAL humans?  (HandHQ PokerStars 25NL, July 2009)

Hands are replayed in time order and the opponent tracker learns exactly as it would live.  At sampled
postflop decision points (taken BEFORE the tracker sees that hand) PokerBrain predicts the acting
player's action from public information only, and the prediction is scored against what he did:
  studied   - villain model with the tracker's profile of this player (what PokerBrain uses live)
  unknown   - villain model with a blank profile (population priors only)
  hud       - the player's own HUD frequencies for the situation (no range reasoning)
  pop_freq  - population frequency per situation class, fitted on the first half of the sample
All models are scored on the second (later) half of the sample.

  python experiments/real_players.py --files 'real/ps25/*.phhs' --rate 0.05 --out results/real/ps25
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import pickle
import sys
import time
from collections import defaultdict
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from pokerbrain.adapters.phh import load_phhs, replay  # noqa: E402
from pokerbrain.cards import stable_hash  # noqa: E402
from pokerbrain.opponents import PRIORS, OpponentDB, PlayerProfile  # noqa: E402
from pokerbrain.villain import (VillainModel, VillainParams, bet_situation, estimate_range,  # noqa: E402
                                strength_vec)

CLASSES_FACING = ("fold", "call", "raise")
CLASSES_OPEN = ("check", "raise")


def situation_of(view, seat):
    """Public features of a decision point."""
    acts = view.street_actions()
    la = view.legal
    call = la.call_amount
    facing = call > 0
    order = ["preflop", "flop", "turn", "river"]
    prev = order[order.index(view.street) - 1]
    prev_aggr = None
    for a in view.actions:
        if a.street == prev and a.kind in ("bet", "raise"):
            prev_aggr = a.seat
    f = {"street": view.street, "facing": facing, "multiway": sum(p.in_hand for p in view.players) > 2,
         "can_raise": la.can_raise}
    if facing:
        f["size"] = call / max(1, view.pot - call)
        f["facing_raise"] = any(a.seat == seat and a.kind in ("bet", "raise") for a in acts)
        bettors = [a for a in acts if a.kind in ("bet", "raise")]
        f["vs_cbet"] = (view.street == "flop" and len(bettors) == 1 and bettors[0].seat == prev_aggr)
        f["commit"] = call / max(1, view.players[seat].stack)
    else:
        f["situation"] = bet_situation(view.street, seat, prev_aggr, any(a.kind == "check" for a in acts))
    return f


def label_of(kind, facing):
    if facing:
        return {"fold": "fold", "call": "call", "check": "call", "raise": "raise", "bet": "raise"}[kind]
    return "check" if kind in ("check", "fold") else "raise"


def hud_probs(prof: PlayerProfile, f) -> dict:
    st = f["street"]
    if f["facing"]:
        key = "fold_vs_raise" if f["facing_raise"] else ("fold_to_cbet" if f["vs_cbet"] else f"fold_vs_bet_{st}")
        pf = prof.stat(key)
        pr = prof.stat("raise_vs_bet") if f["can_raise"] else 0.0
        pr = min(pr, 1 - pf)
        return {"fold": pf, "call": 1 - pf - pr, "raise": pr}
    key = {"cbet": "cbet", "barrel": "barrel", "donk": "donk", "stab": "bet_checked_to"}[f["situation"]]
    pb = prof.stat(key)
    return {"check": 1 - pb, "raise": pb}


def model_probs(view, seat, params: VillainParams, f) -> dict:
    model = VillainModel(params)
    w, w_ref = estimate_range(view, seat, model, with_ref=True)
    W = float(w.sum())
    if W <= 0:
        return None
    s = strength_vec(tuple(view.board))
    if f["facing"]:
        pf, pc, pr = model.response_probs(w, s, view.street, f["size"], f["vs_cbet"], None, w_ref,
                                          facing_raise=f["facing_raise"], commit=f["commit"], multiway=f["multiway"])
        P = {"fold": float((w * pf).sum() / W), "call": float((w * pc).sum() / W), "raise": float((w * pr).sum() / W)}
        if not f["can_raise"]:
            P["call"] += P["raise"]; P["raise"] = 0.0
        return P
    pb = model.bet_probs(w, s, view.street, f["situation"], None, tuple(view.board), w_ref)
    p = float((w * pb).sum() / W)
    return {"check": 1 - p, "raise": p}


def _predict(item):
    view, seat, params1, f = item["view"], item["seat"], item["params"], item["f"]
    try:
        out = {"studied": model_probs(view, seat, params1, f),
               "unknown": model_probs(view, seat, VillainParams.from_profile(PlayerProfile(name="?")), f)}
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}
    return out


def pop_key(f):
    if f["facing"]:
        b = min(4, int(f["size"] / 0.34))            # size buckets ~1/3, 2/3, 1, 4/3, bigger
        return (f["street"], "F", f["facing_raise"], f["vs_cbet"], b, f["multiway"])
    return (f["street"], "O", f["situation"], f["multiway"])


def logloss(P: dict, y: str) -> float:
    return -math.log(min(0.99, max(0.01, P.get(y, 0.0))))


def brier(P: dict, y: str, classes) -> float:
    return sum((P.get(c, 0.0) - (1.0 if c == y else 0.0)) ** 2 for c in classes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", required=True)
    ap.add_argument("--rate", type=float, default=0.05)
    ap.add_argument("--max-files", type=int, default=10 ** 9)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--out", default="results/real/ps25")
    ap.add_argument("--eval-from", type=float, default=0.0,
                    help="only sample decision points after this fraction of the (time-ordered) hands")
    ap.add_argument("--priors", default="", help="JSON {stat: mean} replacing the tracker's prior means")
    ap.add_argument("--pop-thresholds", default="", help="villain.POP_THRESHOLDS override: current | ref")
    a = ap.parse_args()
    if a.pop_thresholds:
        import pokerbrain.villain as vil
        vil.POP_THRESHOLDS = a.pop_thresholds
    if a.priors:
        import pokerbrain.opponents as opp
        for k, m in json.load(open(a.priors)).items():
            if k in opp.PRIORS:
                opp.PRIORS[k] = (float(m), opp.PRIORS[k][1])
        print("priors replaced from", a.priors, flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    files = sorted(glob.glob(a.files))[: a.max_files]
    t0 = time.time()
    hands = []
    for fn in files:
        hands += load_phhs(fn)
    hands.sort(key=lambda h: int(h.get("hand", 0)))
    print(f"{len(hands)} hands from {len(files)} files ({time.time() - t0:.0f}s)", flush=True)

    db = OpponentDB(platform="real")
    items, replayed, points_total = [], 0, 0
    thr = int(a.rate * 2 ** 32)
    start = int(a.eval_from * len(hands))
    pop_before = {}
    for k, h in enumerate(hands):
        if k == start and start > 0:
            for name in PRIORS:
                kk = sum(p.counts.get(name, [0, 0])[0] for p in db.profiles.values())
                nn = sum(p.counts.get(name, [0, 0])[1] for p in db.profiles.values())
                if nn:
                    pop_before[name] = round(kk / nn, 4)
        r = replay(h)
        if r is None:
            continue
        replayed += 1
        for dp in r.points:
            if dp.view.street == "preflop" or k < start:
                continue
            points_total += 1
            if stable_hash(r.history.hand_id, dp.seat, len(dp.view.actions)) >= thr:
                continue
            prof = db.get(dp.name)
            f = situation_of(dp.view, dp.seat)
            items.append({"view": dp.view, "seat": dp.seat, "f": f, "y": label_of(dp.kind, f["facing"]),
                          "n": prof.samples("vpip"), "hud": hud_probs(prof, f),
                          "params": VillainParams.from_profile(prof), "seats": r.seat_count})
        db.update(r.history)
        if (k + 1) % 50000 == 0:
            print(f"  {k + 1} hands, {len(items)} sampled points ({time.time() - t0:.0f}s)", flush=True)
    print(f"replayed {replayed}/{len(hands)}; postflop decision points {points_total}; sampled {len(items)}",
          flush=True)

    # ---- population frequencies from the tracker's counts (same definitions as the priors)
    pop = {}
    for name in PRIORS:
        k = sum(p.counts.get(name, [0, 0])[0] for p in db.profiles.values())
        n = sum(p.counts.get(name, [0, 0])[1] for p in db.profiles.values())
        if n:
            pop[name] = {"real": round(k / n, 3), "prior": PRIORS[name][0], "n": n}
    regs = [p for p in db.profiles.values() if p.samples("vpip") >= 200]
    arch = defaultdict(int)
    for p in regs:
        arch[p.archetype()[0]] += 1
    dist = {s: [round(float(np.percentile([p.raw(s) or 0 for p in regs], q)), 3) for q in (10, 25, 50, 75, 90)]
            for s in ("vpip", "pfr", "threebet", "fold_to_cbet", "wtsd")} if regs else {}

    # ---- predictions (parallel)
    with Pool(a.procs) as pool:
        preds = pool.map(_predict, items, chunksize=64)
    half = len(items) // 2
    popc = defaultdict(lambda: defaultdict(float))
    for it in items[:half]:
        popc[pop_key(it["f"])][it["y"]] += 1
    res = defaultdict(lambda: defaultdict(list))
    errors = 0
    for it, pr in zip(items[half:], preds[half:]):
        if "error" in pr or pr["studied"] is None or pr["unknown"] is None:
            errors += 1
            continue
        classes = CLASSES_FACING if it["f"]["facing"] else CLASSES_OPEN
        c = popc.get(pop_key(it["f"]))
        tot = sum(c.values()) if c else 0
        pf = {k: (c.get(k, 0) + 1) / (tot + len(classes)) for k in classes} if c else {k: 1 / len(classes) for k in classes}
        models = {"studied": pr["studied"], "unknown": pr["unknown"], "hud": it["hud"], "pop_freq": pf}
        n = it["n"]
        nb = "0-19" if n < 20 else "20-99" if n < 100 else "100-499" if n < 500 else "500+"
        grp = ("facing_bet" if it["f"]["facing"] else "not_facing")
        for m, P in models.items():
            ll, br = logloss(P, it["y"]), brier(P, it["y"], classes)
            for g in ("ALL", grp, f"hands {nb}", f"{grp} | hands {nb}", f"street {it['f']['street']}"):
                res[g][m].append((ll, br))
    table = {g: {m: {"n": len(v), "logloss": round(sum(x[0] for x in v) / len(v), 4),
                     "brier": round(sum(x[1] for x in v) / len(v), 4)} for m, v in d.items()}
             for g, d in res.items()}
    out = {"hands": len(hands), "replayed": replayed, "sampled_points": len(items), "scored": len(items) - half - errors,
           "errors": errors, "population_vs_priors": pop, "population_before_eval": pop_before, "regulars_200plus": len(regs), "archetypes": dict(arch),
           "stat_percentiles_10_25_50_75_90": dist, "prediction": table,
           "actual_mix": {g: dict(v) for g, v in {"facing": defaultdict(int), "open": defaultdict(int)}.items()}}
    for it in items[half:]:
        out["actual_mix"]["facing" if it["f"]["facing"] else "open"][it["y"]] = \
            out["actual_mix"]["facing" if it["f"]["facing"] else "open"].get(it["y"], 0) + 1
    json.dump(out, open(a.out + ".json", "w"), indent=1)
    db.path = a.out + "_db.json"
    try:
        db.save()
    except Exception as e:  # noqa: BLE001
        print("db save failed:", e)
    print(json.dumps({k: out[k] for k in ("hands", "replayed", "sampled_points", "scored", "errors")}))
    print("population vs priors:")
    for k, v in pop.items():
        print(f"  {k:18s} real {v['real']:.3f}  prior {v['prior']:.3f}  (n={v['n']})")
    print("archetypes of regulars (200+ hands):", dict(arch), "stat percentiles:", dist)
    for g in sorted(table):
        row = "  ".join(f"{m} {v['logloss']:.3f}" for m, v in sorted(table[g].items(), key=lambda kv: kv[1]["logloss"]))
        print(f"  {g:32s} n={next(iter(table[g].values()))['n']:6d}  logloss: {row}")


if __name__ == "__main__":
    main()
