"""Full-match test of preflop policy: static charts vs the engine's opponent-aware EV.

The oracle benchmark showed the charts losing ~1.9 bb/decision to the engine on non-trivial preflop
spots.  Single-decision EV can mislead (the engine's preflop EV uses realization factors), so this
plays complete paired matches (same decks, same bot RNG) and compares bb/100.

  python experiments/preflop_test.py --hands 500 --seeds 4 --decks 200 --procs 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pokerbrain.agents.quant_agent import PreflopSpot, QuantAgent  # noqa: E402
from pokerbrain.arena import duplicate_hu, paired_diff, ring_session  # noqa: E402
from pokerbrain.bots import StyleBot  # noqa: E402

FACING = frozenset({"vs_open", "squeeze", "vs_3bet", "vs_4bet", "vs_5bet", "cold_3bet", "cold_4bet"})
POLICIES = {"chart": frozenset(), "engine_facing_raise": FACING,
            "engine_all": FACING | {"unopened", "limped"}}
FIELDS = {"A": "nit,station,maniac,fish,tilter", "B": "tag,lag,station,fish,sizer"}
HU = ["lag", "maniac", "station", "nit", "tag", "tilter", "sizer"]


class PFAgent(QuantAgent):
    """QuantAgent whose preflop chart is bypassed (engine EV decides) for the given spot kinds."""

    def __init__(self, engine_kinds=frozenset(), **kw):
        super().__init__(**kw)
        self.engine_kinds = engine_kinds

    def preflop(self, view):
        if PreflopSpot(view).kind in self.engine_kinds:
            return None
        return super().preflop(view)


def job(args):
    kind, key, seed, n, policy = args
    hero = lambda: PFAgent(POLICIES[policy], name="Hero", seed=5)
    if kind == "ring":
        styles = FIELDS[key].split(",")
        field = lambda: [StyleBot(f"{s.capitalize()}{i + 1}", s, 300 + i) for i, s in enumerate(styles)]
        st = ring_session(hero, field, n, seed=seed)
    else:
        st = duplicate_hu(hero, lambda: StyleBot("Villain", key, 77), n, seed=seed)
    return (kind, key, seed, policy), st


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=500)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--decks", type=int, default=200)
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--out", default="results/preflop_test.json")
    a = ap.parse_args()
    jobs = [("ring", f, s, a.hands, p) for f in FIELDS for s in range(1, a.seeds + 1) for p in POLICIES]
    jobs += [("hu", h, 1, a.decks, p) for h in HU for p in POLICIES]
    with Pool(a.procs) as pool:
        res = dict(pool.imap_unordered(job, jobs))
    out = {}
    for pol in POLICIES:
        if pol == "chart":
            continue
        rows = {}
        for grp in list(FIELDS) + HU:
            ks = [k for k in res if k[1] == grp and k[3] == pol]
            a_ev, b_ev = [], []
            for k in sorted(ks):
                base = res[(k[0], k[1], k[2], "chart")]
                a_ev += res[k].ev_bb
                b_ev += base.ev_bb
            A = type(res[ks[0]])(name=pol); A.ev_bb = a_ev; A.results_bb = a_ev
            B = type(res[ks[0]])(name="chart"); B.ev_bb = b_ev; B.results_bb = b_ev
            d, ci = paired_diff(A, B)
            rows[grp] = {"diff_bb100": round(d, 1), "ci95": round(ci, 1), "hands": len(a_ev),
                         "policy_bb100": round(100 * sum(a_ev) / len(a_ev), 1),
                         "chart_bb100": round(100 * sum(b_ev) / len(b_ev), 1)}
        allA = [x for k in res if k[3] == pol for x in res[k].ev_bb]
        allB = [x for k in res if k[3] == pol for x in res[(k[0], k[1], k[2], "chart")].ev_bb]
        A = type(next(iter(res.values())))(name=pol); A.ev_bb = allA
        B = type(next(iter(res.values())))(name="chart"); B.ev_bb = allB
        d, ci = paired_diff(A, B)
        rows["ALL"] = {"diff_bb100": round(d, 1), "ci95": round(ci, 1), "hands": len(allA)}
        out[pol] = rows
        print(pol, json.dumps(rows, indent=1))
    json.dump(out, open(a.out, "w"), indent=1)
