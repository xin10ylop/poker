"""Jev as a blunder-check verifier: would vetoing Opus's overrides (P(blunder) >= 0.85) have helped?

Uses every Opus override of the engine recorded in the prompt rounds (answers from subagents) and
asks Jev the verifier question once per unique (spot, override).  Results cached in
results/bench/jev_verifier.json.
  python experiments/jev_verifier.py <override_rows.json>
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.bench import _report, load  # noqa: E402
from pokerbrain.llm.budget import load_dotenv  # noqa: E402
from pokerbrain.llm.jev import JevClient, verify_question  # noqa: E402
from pokerbrain.llm.render import jev_state  # noqa: E402
from pokerbrain.spots import Spot  # noqa: E402

CACHE = "results/bench/jev_verifier.json"


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


if __name__ == "__main__":
    load_dotenv()
    rows = [r for r in json.load(open(sys.argv[1])) if r["v"] not in ("v1_raw", "v2_quant") and r["pick"] != r["q"]]
    allsp = {s["spot_id"]: s for s in load()}
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    jev = JevClient()
    for key in sorted({f"{r['sid']}|{r['pick']}" for r in rows}):
        if key in cache:
            continue
        sid, pick = key.split("|")
        sp = Spot.from_json({k: v for k, v in allsp[sid].items() if k in Spot.__dataclass_fields__})
        db, rep = _report(sp)
        ans = jev.ask(jev_state(sp.view(), rep, db), verify_question(rep.option(pick).label), tag="verify-bench")
        cache[key] = float(ans["is_blunder"]["noul"])
        json.dump(cache, open(CACHE, "w"), indent=1)
    p = [cache[f"{r['sid']}|{r['pick']}"] for r in rows]
    bad = [x for x, r in zip(p, rows) if r["gain"] < -0.01]
    good = [x for x, r in zip(p, rows) if r["gain"] > 0.01]
    print(f"overrides: {len(rows)} ({len(set(cache))} unique asked); AUC of P(blunder) for losing overrides: {auc(bad, good):.3f}")
    for thr in (0.5, 0.7, 0.85):
        veto = [r for x, r in zip(p, rows) if x >= thr]
        print(f"  veto at P>={thr}: vetoes {len(veto)} overrides, EV change {-sum(r['gain'] for r in veto):+.1f}bb "
              f"(vetoed wins {sum(r['gain'] > 0.01 for r in veto)}, losses {sum(r['gain'] < -0.01 for r in veto)})")
    gated = [(x, r) for x, r in zip(p, rows) if r["qp"] <= 0.2 + 1e-9]
    veto = [r for x, r in gated if x >= 0.85]
    print(f"  on top of the 0.2 gate: {len(gated)} overrides survive the gate; Jev would veto {len(veto)}, "
          f"EV change {-sum(r['gain'] for r in veto):+.1f}bb")
