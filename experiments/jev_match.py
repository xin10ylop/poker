"""Full matches: Jev-reads agent vs the pure quant agent on identical seeds.

  python experiments/jev_match.py --hands 600 --field hu:tilter
  python experiments/jev_match.py --hands 800 --field nit,station,maniac,fish,tilter
Costs one Jev call per postflop hero decision in pots >= min-pot (about $0.00005 each).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pokerbrain.agents.llm_agents import JevReadsAgent  # noqa: E402
from pokerbrain.agents.quant_agent import QuantAgent  # noqa: E402
from pokerbrain.arena import paired_diff, ring_session  # noqa: E402
from pokerbrain.bots import StyleBot  # noqa: E402
from pokerbrain.llm.jev import JevClient  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=600)
    ap.add_argument("--field", default="hu:tilter")
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--weight", type=float, default=0.35)
    ap.add_argument("--min-pot", type=float, default=6.0)
    a = ap.parse_args()
    if a.field.startswith("hu:"):
        st = a.field[3:]
        field = lambda: [StyleBot(st.capitalize(), st, 700)]
    else:
        styles = a.field.split(",")
        field = lambda: [StyleBot(f"{s.capitalize()}{i + 1}", s, 700 + i) for i, s in enumerate(styles)]
    jev = JevClient()
    j = ring_session(lambda: JevReadsAgent(jev, weight=a.weight, min_pot_bb=a.min_pot, name="Hero", seed=5),
                     field, a.hands, seed=a.seed)
    q = ring_session(lambda: QuantAgent("Hero", seed=5), field, a.hands, seed=a.seed)
    d, ci = paired_diff(j, q)
    out = {"field": a.field, "hands": a.hands, "jev_reads": j.summary(), "quant": q.summary(),
           "paired_diff_bb100": round(d, 1), "ci95": round(ci, 1), "jev_calls": jev.calls,
           "jev_usd": round(jev.usd, 5)}
    print(json.dumps(out, indent=1))
    os.makedirs("results", exist_ok=True)
    with open(f"results/jev_match_{a.field.replace(':', '_').replace(',', '-')}_{a.hands}.json", "w") as f:
        json.dump(out, f, indent=1)
