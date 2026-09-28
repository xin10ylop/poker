"""Paired full-match A/B of two engine builds (e.g. a git worktree of an older commit vs the current tree).

  python experiments/engine_ab.py --root <path-to-repo> --tag old  [--commit 0.6|none] --out <file.json>
Every run uses the same decks and bot seeds, so per-hand results from two runs pair up exactly;
compare them with --compare a.json b.json.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from multiprocessing import Pool

FIELDS = {"A": "nit,station,maniac,fish,tilter", "B": "tag,lag,station,fish,sizer"}
HU = ["lag", "maniac", "station", "nit", "tag", "tilter", "sizer"]
CFG = {}


def _init(root, commit, population="default", temper=None, pf_temper=None):
    import os
    sys.path.insert(0, root)
    if population != "default":
        os.environ["POKERBRAIN_POPULATION"] = population       # before pokerbrain is imported
    CFG.update(root=root, commit=commit)
    import pokerbrain.villain as vil
    if commit != "keep":
        vil.COMMIT_STRENGTH = None if commit == "none" else float(commit)
    if temper is not None:
        vil.TEMPER = temper
    if pf_temper is not None:
        vil.PF_TEMPER = pf_temper


def job(args):
    from pokerbrain.agents.quant_agent import QuantAgent
    from pokerbrain.arena import duplicate_hu, ring_session
    from pokerbrain.bots import StyleBot
    kind, key, seed, n = args
    hero = lambda: QuantAgent("Hero", seed=5)
    if kind == "ring":
        styles = FIELDS[key].split(",")
        st = ring_session(hero, lambda: [StyleBot(f"{s.capitalize()}{i + 1}", s, 300 + i) for i, s in enumerate(styles)],
                          n, seed=seed)
    else:
        st = duplicate_hu(hero, lambda: StyleBot("Villain", key, 77), n, seed=seed)
    return f"{kind}|{key}|{seed}", st.ev_bb


def compare(a_path, b_path):
    a, b = json.load(open(a_path)), json.load(open(b_path))
    groups = {}
    for k in a:
        g = k.split("|")[1]
        groups.setdefault(g, []).extend(x - y for x, y in zip(a[k], b[k]))
    groups["ALL"] = [d for k in a for d in (x - y for x, y in zip(a[k], b[k]))]
    out = {}
    for g, d in groups.items():
        n = len(d); m = sum(d) / n
        ci = 1.96 * 100 * math.sqrt(sum((x - m) ** 2 for x in d) / max(1, n - 1) / n)
        out[g] = (round(100 * m, 1), round(ci, 1), n)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--commit", default="keep")
    ap.add_argument("--out", default="")
    ap.add_argument("--hands", type=int, default=500)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--decks", type=int, default=300)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--seed-start", type=int, default=1)
    ap.add_argument("--no-hu", action="store_true")
    ap.add_argument("--population", default="default", help="population file path, or 'none'")
    ap.add_argument("--temper", type=float, default=None)
    ap.add_argument("--pf-temper", type=float, default=None)
    ap.add_argument("--compare", nargs=2, default=None)
    a = ap.parse_args()
    if a.compare:
        for g, (m, ci, n) in compare(*a.compare).items():
            print(f"  {g:8s} {m:+7.1f} ± {ci:5.1f} bb/100  ({n} hands)")
        sys.exit()
    jobs = [("ring", f, s, a.hands) for f in FIELDS for s in range(a.seed_start, a.seed_start + a.seeds)]
    if not a.no_hu:
        jobs += [("hu", h, 1, a.decks) for h in HU]
    with Pool(a.procs, initializer=_init, initargs=(a.root, a.commit, a.population, a.temper, a.pf_temper)) as pool:
        res = dict(pool.map(job, jobs))
    json.dump(res, open(a.out, "w"))
    print("saved", a.out, sum(len(v) for v in res.values()), "hands")
