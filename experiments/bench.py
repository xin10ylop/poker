"""Benchmark harness: score agents / prompt variants against the oracle.

  python experiments/bench.py summary                       # benchmark stats + quant baseline
  python experiments/bench.py jev --build decide|reads      # live Jev builds (OpenRouter)
  python experiments/bench.py export --variant v3_elite --set dev --out bridge/v3_elite
  python experiments/bench.py score-answers --dir bridge/v3_elite --set dev

Metric: EV loss per decision in bb = max_j oracle_EV_j - sum_j p_j * oracle_EV_j,
where p is the agent's (possibly mixed) strategy over the engine's action menu.
Same spots for every agent -> paired comparison.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pokerbrain.spots import Spot  # noqa: E402

BENCH = "results/bench/spots.json"
READS_CACHE = "results/bench/jev_reads.json"
BANKROLL_CTX = {"stakes": "$1/$2 NLHE cash (6-max)", "bb_value": "$2", "buy_in": "$200", "bankroll": "$12,000.00",
                "bankroll_buyins": 60.0, "risk_mode": "normal", "session_pnl": "+$0.00", "session_hands": 0,
                "session_status": "ok", "rake": "5% cap 3bb", "risk_aversion": 1.0}


def load(path: str = BENCH) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def split(spots: list[dict]) -> dict[str, list[dict]]:
    """Deterministic dev/test split (by hash of spot id), stratified implicitly."""
    import zlib
    dev, test = [], []
    for s in sorted(spots, key=lambda d: d["spot_id"]):
        (dev if zlib.crc32(s["spot_id"].encode()) % 2 == 0 else test).append(s)
    return {"dev": dev, "test": test, "all": dev + test}


def interesting(s: dict, min_gap: float = 0.75) -> bool:
    """Spots where the choice matters: best option beats the median option by min_gap bb."""
    evs = sorted(s["oracle"].values(), reverse=True)
    return len(evs) >= 2 and evs[0] - evs[len(evs) // 2] >= min_gap


def ev_loss(s: dict, strategy: dict) -> float:
    best = max(s["oracle"].values())
    tot = sum(strategy.values()) or 1.0
    got = sum(p * s["oracle"].get(oid, min(s["oracle"].values())) for oid, p in strategy.items()) / tot
    return best - got


def score(spots: list[dict], strategies: dict[str, dict]) -> dict:
    losses, best_hits, cats = [], 0, 0
    by_style: dict[str, list] = defaultdict(list)
    by_street: dict[str, list] = defaultdict(list)
    for s in spots:
        st = strategies.get(s["spot_id"])
        if st is None:
            continue
        l = ev_loss(s, st)
        losses.append(l)
        best_id = max(s["oracle"], key=s["oracle"].get)
        best_hits += st.get(best_id, 0) / (sum(st.values()) or 1)
        cats += l > 10
        by_style[s["villain_style"]].append(l)
        by_street[s["street"]].append(l)
    n = len(losses)
    if n == 0:
        return {"n": 0}
    m = sum(losses) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in losses) / max(1, n - 1))
    return {"n": n, "ev_loss_bb": round(m, 3), "se": round(sd / math.sqrt(n), 3),
            "best_pct": round(100 * best_hits / n, 1), "blunders_gt10bb": cats,
            "by_style": {k: round(sum(v) / len(v), 2) for k, v in sorted(by_style.items())},
            "by_street": {k: round(sum(v) / len(v), 2) for k, v in sorted(by_street.items())}}


def paired(spots: list[dict], a: dict, b: dict) -> tuple[float, float]:
    d = [ev_loss(s, a[s["spot_id"]]) - ev_loss(s, b[s["spot_id"]]) for s in spots
         if s["spot_id"] in a and s["spot_id"] in b]
    n = len(d)
    m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / max(1, n - 1))
    return m, sd / math.sqrt(n)


def quant_strategies(spots) -> dict:
    return {s["spot_id"]: {s["quant_choice"]: 1.0} for s in spots}


def oracle_strategies(spots) -> dict:
    return {s["spot_id"]: {max(s["oracle"], key=s["oracle"].get): 1.0} for s in spots}


# ---------------------------------------------------------------------------
def _report(spot: Spot, reads_params=None):
    from pokerbrain.quant import QuantEngine
    db = spot.db()
    eng = QuantEngine(db, rng=random.Random(1))
    return db, eng.analyze(spot.view(), reads=reads_params)


def get_reads(spots: list[dict], refresh: bool = False) -> dict:
    """Jev reads per spot (cached on disk: one cheap call per spot)."""
    from pokerbrain.agents.llm_agents import jev_reads
    from pokerbrain.llm.jev import JevClient
    cache = {}
    if os.path.exists(READS_CACHE) and not refresh:
        cache = json.load(open(READS_CACHE))
    todo = [s for s in spots if s["spot_id"] not in cache]
    if todo:
        jev = JevClient()
        for i, sd in enumerate(todo):
            sp = Spot.from_json({k: v for k, v in sd.items() if k in Spot.__dataclass_fields__})
            db, rep = _report(sp)
            try:
                cache[sd["spot_id"]] = jev_reads(jev, sp.view(), rep, db, BANKROLL_CTX)
            except Exception as exc:  # noqa: BLE001
                print("jev error", exc)
            if (i + 1) % 25 == 0:
                print(f"  reads {i + 1}/{len(todo)}  spent ${jev.usd:.4f}", flush=True)
                json.dump(cache, open(READS_CACHE, "w"))
        json.dump(cache, open(READS_CACHE, "w"))
        print(f"jev reads: {jev.calls} calls, ${jev.usd:.4f}")
    return cache


def run_jev(spots: list[dict], build: str, weight: float = 0.5) -> dict:
    from pokerbrain.agents.llm_agents import reads_to_params
    from pokerbrain.llm.jev import JevClient, decision_question
    from pokerbrain.llm.render import jev_state
    reads = get_reads(spots)
    out = {}
    jev = JevClient() if build == "decide" else None
    for sd in spots:
        sp = Spot.from_json({k: v for k, v in sd.items() if k in Spot.__dataclass_fields__})
        if build == "reads":
            r = reads.get(sd["spot_id"])
            db, rep = _report(sp, reads_to_params(r, weight) if r else None)
            out[sd["spot_id"]] = {rep.best.id: 1.0}
        elif build == "decide":
            db, rep = _report(sp)
            st = jev_state(sp.view(), rep, db, BANKROLL_CTX)
            try:
                ans = jev.ask(st, decision_question([o.brief() for o in rep.options]), tag="bench-decide")
                out[sd["spot_id"]] = {k: float(v) for k, v in ans["best_action"]["probabilities"].items()}
            except Exception as exc:  # noqa: BLE001
                print("jev error", exc)
    if jev:
        print(f"jev decide: {jev.calls} calls ${jev.usd:.4f}")
    return out


def export(spots: list[dict], variant: str, outdir: str, with_reads: bool = True) -> None:
    from pokerbrain.llm.prompts import VARIANTS
    from pokerbrain.llm.render import render_dashboard
    v = VARIANTS[variant]
    reads = get_reads(spots) if (with_reads and "reads" in v["sections"]) else {}
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "SYSTEM_PROMPT.txt"), "w") as f:
        f.write(v["system"])
    index = []
    for sd in spots:
        sp = Spot.from_json({k: v2 for k, v2 in sd.items() if k in Spot.__dataclass_fields__})
        db, rep = _report(sp)
        text = render_dashboard(sp.view(), rep, db, BANKROLL_CTX, reads.get(sd["spot_id"]),
                                {"hand_number": sd.get("session_context", {}).get("hand_number", "?")}, v["sections"])
        fn = f"{sd['spot_id']}.txt"
        with open(os.path.join(outdir, fn), "w") as f:
            f.write(text)
        index.append({"spot_id": sd["spot_id"], "file": fn, "ids": [o.id for o in rep.options]})
    json.dump(index, open(os.path.join(outdir, "index.json"), "w"), indent=1)
    print(f"exported {len(index)} prompts to {outdir}")


def merge_answers(d: str) -> dict:
    """Merge answers_*.json files written by parallel subagents into answers.json."""
    merged = {}
    for fn in sorted(os.listdir(d)):
        if fn.startswith("answers_") and fn.endswith(".json"):
            try:
                merged.update(json.load(open(os.path.join(d, fn))))
            except json.JSONDecodeError as exc:
                print("bad json in", fn, exc)
    json.dump(merged, open(os.path.join(d, "answers.json"), "w"), indent=1)
    return merged


def load_answers(d: str) -> dict:
    if not os.path.exists(os.path.join(d, "answers.json")) or any(
            f.startswith("answers_") for f in os.listdir(d)):
        merge_answers(d)
    ans = json.load(open(os.path.join(d, "answers.json")))
    out = {}
    for sid, a in ans.items():
        mix = a.get("mix") or []
        st = {m["id"]: float(m["p"]) for m in mix if float(m.get("p", 0)) > 0}
        if not st:
            st = {a.get("action_id"): 1.0}
        out[sid] = st
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--set", default="dev")
    ap.add_argument("--variant", default="v3_elite")
    ap.add_argument("--out", default="")
    ap.add_argument("--dir", default="")
    ap.add_argument("--build", default="reads")
    ap.add_argument("--weight", type=float, default=0.5)
    ap.add_argument("--bench", default=BENCH)
    ap.add_argument("--all-spots", action="store_true", help="don't filter to interesting spots")
    a = ap.parse_args()
    sets = split(load(a.bench))
    spots = sets[a.set]
    if not a.all_spots:
        spots = [s for s in spots if interesting(s)]
    if a.cmd == "summary":
        print(f"{a.set}: {len(spots)} spots")
        print("quant :", score(spots, quant_strategies(spots)))
        print("oracle:", score(spots, oracle_strategies(spots)))
    elif a.cmd == "jev":
        st = run_jev(spots, a.build, a.weight)
        print(f"jev-{a.build}:", score(spots, st))
        m, se = paired(spots, st, quant_strategies(spots))
        print(f"paired vs quant: {m:+.3f} ± {se:.3f} bb (negative = better than quant)")
    elif a.cmd == "export":
        export(spots, a.variant, a.out or f"bridge/{a.variant}_{a.set}")
    elif a.cmd == "score-answers":
        st = load_answers(a.dir)
        print(os.path.basename(a.dir), score(spots, st))
        m, se = paired(spots, st, quant_strategies(spots))
        print(f"paired vs quant: {m:+.3f} ± {se:.3f} bb (negative = better than quant)")
