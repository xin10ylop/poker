"""Card-level test of the villain model on real humans whose hole cards are all known (Pluribus hands:
professionals playing 6-max, Brown & Sandholm 2019).

For every postflop decision of a human (tracker learning in hand order, from public info only):
  range score  - log P(his actual hand) under PokerBrain's estimated range vs a uniform range over all
                 unblocked hands (how much the range narrowing knew about his cards)
  action score - log-loss of his actual action (a) using the model's per-hand probabilities for his real
                 hand, vs (b) the model's range-average probabilities (what the engine uses for EV)
  python experiments/real_cards.py --files 'real/pluribus/*.phh' --out results/real/pluribus_cards.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from experiments.real_players import label_of, situation_of  # noqa: E402
from pokerbrain.adapters.phh import load_phhs, replay  # noqa: E402
from pokerbrain.opponents import OpponentDB  # noqa: E402
from pokerbrain.villain import IDX, VillainModel, VillainParams, estimate_range, strength_vec  # noqa: E402


def _score(item):
    view, seat, params, f, hole, y = item["view"], item["seat"], item["params"], item["f"], item["hole"], item["y"]
    model = VillainModel(params)
    w, w_ref = estimate_range(view, seat, model, with_ref=True)
    W = float(w.sum())
    k = IDX.get(hole) if hole in IDX else IDX.get((hole[1], hole[0]))
    if W <= 0 or k is None:
        return None
    live = float(((w_ref > 0) | (w > 0)).sum()) or 1.0
    out = {"range_logp": math.log(max(1e-9, w[k] / W)), "uniform_logp": -math.log(live),
           "preflop_logp": math.log(max(1e-9, w_ref[k] / max(1e-12, float(w_ref.sum()))))}
    s = strength_vec(tuple(view.board))
    if f["facing"]:
        pf, pc, pr = model.response_probs(w, s, view.street, f["size"], f["vs_cbet"], None, w_ref,
                                          facing_raise=f["facing_raise"], commit=f["commit"], multiway=f["multiway"])
        if not f["can_raise"]:
            pc, pr = pc + pr, pr * 0.0
        per = {"fold": pf[k], "call": pc[k], "raise": pr[k]}
        avg = {"fold": float((w * pf).sum() / W), "call": float((w * pc).sum() / W), "raise": float((w * pr).sum() / W)}
    else:
        pb = model.bet_probs(w, s, view.street, f["situation"], None, tuple(view.board), w_ref)
        per = {"check": 1 - pb[k], "raise": pb[k]}
        p = float((w * pb).sum() / W)
        avg = {"check": 1 - p, "raise": p}
    ll = lambda P: -math.log(min(0.99, max(0.01, float(P.get(y, 0.0)))))
    out.update(card_ll=ll(per), avg_ll=ll(avg), facing=f["facing"], street=view.street, y=y)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", required=True)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--exclude", default="Pluribus")
    ap.add_argument("--max-points", type=int, default=12000)
    ap.add_argument("--out", default="results/real/pluribus_cards.json")
    ap.add_argument("--temper", type=float, default=None)
    ap.add_argument("--pf-temper", type=float, default=None)
    ap.add_argument("--shown-only", action="store_true", help="only points of players whose cards were shown")
    a = ap.parse_args()
    import pokerbrain.villain as vil
    if a.temper is not None:
        vil.TEMPER = a.temper
    if a.pf_temper is not None:
        vil.PF_TEMPER = a.pf_temper
    hands = []
    for fn in sorted(glob.glob(a.files), key=lambda p: [(int(x), "") if x.isdigit() else (-1, x) for x in
                                                         os.path.basename(p).replace(".", "_").split("_")]):
        hands += load_phhs(fn)
    db = OpponentDB(platform="real")
    items = []
    for h in hands:
        r = replay(h)
        if r is None:
            continue
        for dp in r.points:
            if dp.view.street == "preflop" or dp.name == a.exclude or dp.seat not in r.holes:
                continue
            if a.shown_only and dp.seat not in r.shown:
                continue
            f = situation_of(dp.view, dp.seat)
            items.append({"view": dp.view, "seat": dp.seat, "f": f, "hole": tuple(r.holes[dp.seat]),
                          "y": label_of(dp.kind, f["facing"]), "n": db.get(dp.name).samples("vpip"),
                          "params": VillainParams.from_profile(db.get(dp.name))})
        db.update(r.history)
    step = max(1, len(items) // a.max_points)
    items = items[::step]
    with Pool(a.procs) as pool:
        res = [x for x in pool.map(_score, items, chunksize=32) if x]
    groups = defaultdict(list)
    for x in res:
        for g in ("ALL", "facing_bet" if x["facing"] else "not_facing", f"street {x['street']}"):
            groups[g].append(x)
    table = {}
    for g, L in sorted(groups.items()):
        m = lambda k: sum(x[k] for x in L) / len(L)
        table[g] = {"n": len(L), "range_bits_vs_uniform": round((m("range_logp") - m("uniform_logp")) / math.log(2), 3),
                    "range_bits_vs_preflop": round((m("range_logp") - m("preflop_logp")) / math.log(2), 3),
                    "action_logloss_card_aware": round(m("card_ll"), 4), "action_logloss_range_avg": round(m("avg_ll"), 4)}
    json.dump({"points": len(res), "table": table}, open(a.out, "w"), indent=1)
    print(f"scored {len(res)} human decision points")
    for g, t in table.items():
        print(f"  {g:14s} n={t['n']:6d}  range: {t['range_bits_vs_uniform']:+.2f} bits vs uniform, "
              f"{t['range_bits_vs_preflop']:+.2f} vs preflop range | action log-loss: card-aware "
              f"{t['action_logloss_card_aware']:.3f} vs range-average {t['action_logloss_range_avg']:.3f}")


if __name__ == "__main__":
    main()
