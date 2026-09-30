"""Stress test: run the full decision engine at real decision points where the actor's cards are known.

Catches crashes, NaN / absurd EVs, illegal decisions and slow spots - the things that matter at a live
table.  Uses Pluribus hands (all cards known) and 25NL hands whose actor showed down.

  python experiments/stress_real.py --files 'real/pluribus/*.phh' --rate 0.1 --out results/real/stress_pluribus.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
import traceback
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pokerbrain.adapters.phh import load_phhs, replay  # noqa: E402
from pokerbrain.agents.quant_agent import QuantAgent  # noqa: E402
from pokerbrain.cards import stable_hash  # noqa: E402
from pokerbrain.opponents import OpponentDB  # noqa: E402


def legal(dec, view) -> str:
    la = view.legal
    if dec.kind == "fold":
        return "" if la.can_fold else "fold when check is free"
    if dec.kind == "check":
        return "" if la.can_check else "check facing a bet"
    if dec.kind == "call":
        return "" if la.call_amount > 0 else "call with nothing to call"
    if dec.kind == "raise":
        if not la.can_raise:
            return "raise not allowed"
        if not (la.min_raise_to <= dec.amount <= la.max_raise_to):
            return f"raise {dec.amount} outside [{la.min_raise_to}, {la.max_raise_to}]"
        return ""
    return f"unknown kind {dec.kind}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", required=True)
    ap.add_argument("--max-files", type=int, default=10 ** 9)
    ap.add_argument("--rate", type=float, default=0.1)
    ap.add_argument("--out", default="results/real/stress.json")
    a = ap.parse_args()
    files = sorted(glob.glob(a.files), key=lambda p: [(int(x), "") if x.isdigit() else (-1, x)
                                                     for x in os.path.basename(p).replace(".", "_").split("_")])[: a.max_files]
    db = OpponentDB(platform="real")
    agent = QuantAgent("Hero", db=db, seed=3)
    thr = int(a.rate * 2 ** 32)
    n_pts = 0
    times, slow = [], []
    errors: Counter = Counter()
    first_tb: dict = {}
    bad: list = []
    illegal: list = []
    by_street = defaultdict(int)
    t0 = time.time()
    for fn in files:
        for h in load_phhs(fn):
            r = replay(h, keep_holes=True)
            if r is None:
                continue
            for i, dp in enumerate(r.points):
                v = dp.view
                if not v.hole or stable_hash(r.history.hand_id, dp.seat, len(v.actions)) >= thr:
                    continue
                n_pts += 1
                by_street[v.street] += 1
                t = time.time()
                try:
                    agent.new_hand(n_pts)
                    dec = agent.act(v)
                    rep = agent.last_report
                    dt = time.time() - t
                    times.append(dt)
                    if dt > 2.0:
                        slow.append({"hand": r.history.hand_id, "street": v.street, "players": len(v.players),
                                     "in_hand": sum(p.in_hand for p in v.players), "secs": round(dt, 2)})
                    why = legal(dec, v)
                    if why:
                        illegal.append({"hand": r.history.hand_id, "street": v.street, "decision": (dec.kind, dec.amount), "why": why})
                    if rep is not None and rep.options:
                        cap = v.pot + v.hero.stack + v.hero.bet + 1
                        for o in rep.options:
                            if not math.isfinite(o.ev) or abs(o.ev) > cap or not math.isfinite(o.variance) or \
                                    (o.fold_prob is not None and not 0 <= o.fold_prob <= 1.0001) or \
                                    (o.eq_called is not None and not 0 <= o.eq_called <= 1.0001):
                                bad.append({"hand": r.history.hand_id, "street": v.street, "option": o.label,
                                            "ev": o.ev, "fold_prob": o.fold_prob, "eq_called": o.eq_called, "pot": v.pot})
                        if not math.isfinite(rep.equity) or not 0 <= rep.equity <= 1.0001:
                            bad.append({"hand": r.history.hand_id, "street": v.street, "equity": rep.equity})
                except Exception as exc:  # noqa: BLE001
                    tb = traceback.format_exc().strip().splitlines()
                    key = f"{type(exc).__name__}: {str(exc)[:80]} @ {tb[-3].strip()[:80] if len(tb) >= 3 else ''}"
                    errors[key] += 1
                    if key not in first_tb:
                        first_tb[key] = {"hand": r.history.hand_id, "street": v.street, "traceback": tb[-12:]}
            db.update(r.history)
            if n_pts and n_pts % 500 == 0 and len(times) and len(times) % 500 == 0:
                print(f"  {n_pts} points, errors {sum(errors.values())}, p95 {sorted(times)[int(0.95 * len(times))]:.2f}s ({time.time() - t0:.0f}s)", flush=True)
    times.sort()
    q = lambda p: round(times[min(len(times) - 1, int(p * len(times)))], 3) if times else None
    out = {"points": n_pts, "by_street": dict(by_street), "errors": dict(errors), "first_tracebacks": first_tb,
           "latency": {"p50": q(0.5), "p90": q(0.9), "p95": q(0.95), "p99": q(0.99), "max": q(1.0), "mean": round(sum(times) / max(1, len(times)), 3)},
           "slow_over_2s": slow[:30], "bad_numbers": bad[:50], "bad_count": len(bad), "illegal": illegal[:30], "illegal_count": len(illegal)}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("points", "by_street", "latency", "bad_count", "illegal_count")}))
    print("errors:", sum(errors.values()))
    for k, v in errors.most_common(10):
        print(f"  {v:5d}  {k}")
    for k, v in list(first_tb.items())[:5]:
        print("\n".join("    " + ln for ln in v["traceback"]))
    print(f"done in {time.time() - t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
