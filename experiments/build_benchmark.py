"""Build the oracle-scored decision benchmark used to compare agents / prompts.

python experiments/build_benchmark.py --out results/bench/spots.json
"""
import argparse, json, os, random, sys, time
from collections import Counter, defaultdict
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pokerbrain.bots import StyleBot
from pokerbrain.agents.quant_agent import QuantAgent
from pokerbrain.spots import record_spots, oracle_evaluate, save_spots, Spot
from pokerbrain.quant import QuantEngine

def six(styles, base):
    return lambda: [StyleBot(f"{s.capitalize()}{i+1}", s, base + i) for i, s in enumerate(styles)]
def hu(style, base):
    return lambda: [StyleBot(style.capitalize(), style, base)]

SESSIONS = [
    ("6max_A", six(["nit", "station", "maniac", "fish", "tilter"], 100), 900, 150),
    ("6max_B", six(["tag", "lag", "sizer", "station", "nit"], 200), 900, 150),
    ("hu_sizer", hu("sizer", 300), 500, 80),
    ("hu_tilter", hu("tilter", 400), 500, 80),
    ("hu_lag", hu("lag", 500), 400, 80),
    ("hu_station", hu("station", 600), 350, 60),
]

def evaluate(args):
    sp_json, n_samples = args
    sp = Spot.from_json(sp_json)
    v = sp.view()
    rep = QuantEngine(sp.db(), rng=random.Random(1)).analyze(v)
    opts = [o.decision for o in rep.options]
    t = time.time()
    res = oracle_evaluate(sp, opts, n_samples=n_samples)
    sp.options = [{"id": o.id, "label": o.label, "kind": o.decision.kind, "amount": o.decision.amount,
                   "quant_ev_bb": round(o.ev_bb, 3)} for o in rep.options]
    sp.oracle = {o.id: round(res["evs"][j]["ev_bb"], 3) for j, o in enumerate(rep.options)}
    sp.oracle_se = {o.id: round(res["evs"][j]["se_bb"], 3) for j, o in enumerate(rep.options)}
    sp.quant_choice = rep.best.id
    d = sp.to_json(); d["oracle_se"] = sp.oracle_se; d["posterior_top"] = res["posterior_top"]; d["oracle_secs"] = round(time.time()-t, 1)
    return d

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/bench/spots.json")
    ap.add_argument("--per-session", type=int, default=60)
    ap.add_argument("--samples", type=int, default=400)
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    rng = random.Random(2026)
    chosen = []
    for name, fieldf, n, warm in SESSIONS:
        t = time.time()
        spots = record_spots(lambda db: QuantAgent("Hero", db=db, seed=7), fieldf, n, seed=abs(hash(name)) % 997 if False else sum(map(ord, name)),
                             warmup=warm, session_name=name)
        spots = [s for s in spots if s.pot_bb >= 6]
        by = defaultdict(list)
        for s in spots:
            by[(s.villain_style, s.street)].append(s)
        # stratified sample: round-robin over strata
        pick = []
        keys = sorted(by)
        for k in keys: rng.shuffle(by[k])
        while len(pick) < a.per_session and any(by[k] for k in keys):
            for k in keys:
                if by[k] and len(pick) < a.per_session:
                    pick.append(by[k].pop())
        print(f"{name}: recorded {len(spots)} spots (pot>=6bb) in {time.time()-t:.0f}s -> picked {len(pick)}", flush=True)
        chosen += pick
    print("total", len(chosen), Counter(s.villain_style for s in chosen), Counter(s.street for s in chosen), flush=True)
    t = time.time()
    with Pool(a.procs) as pool:
        out = []
        for i, d in enumerate(pool.imap_unordered(evaluate, [(s.to_json(), a.samples) for s in chosen])):
            out.append(d)
            if (i + 1) % 20 == 0:
                print(f"  oracle {i+1}/{len(chosen)}  ({time.time()-t:.0f}s)", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"))
    print("saved", a.out, len(out), f"{time.time()-t:.0f}s")
