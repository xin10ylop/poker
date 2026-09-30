"""Full match with Opus making the escalated decisions live, through the file bridge.

  python experiments/live_match.py --qdir bridge/live --hands 120 [--variant v13_final] [--jev]
Then an answering process (e.g. a Claude Code session) loops:
  python -m pokerbrain.bridge next bridge/live
  python -m pokerbrain.bridge answer bridge/live <id> '<json>'
The same seeds are replayed with the pure QuantAgent for a paired comparison.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pokerbrain.agents.llm_agents import EscalationPolicy, OpusAgent  # noqa: E402
from pokerbrain.agents.quant_agent import QuantAgent  # noqa: E402
from pokerbrain.arena import paired_diff, ring_session  # noqa: E402
from pokerbrain.bots import StyleBot  # noqa: E402
from pokerbrain.bridge import FileBridgeDecider  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdir", default="bridge/live")
    ap.add_argument("--hands", type=int, default=120)
    ap.add_argument("--variant", default=None, help="prompt variant (default: config.ULTIMATE)")
    ap.add_argument("--field", default="nit,station,maniac,fish,tilter")
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("--jev", action="store_true")
    ap.add_argument("--min-pot", type=float, default=12.0)
    a = ap.parse_args()
    styles = a.field.split(",")
    field = lambda: [StyleBot(f"{s.capitalize()}{i + 1}", s, 900 + i) for i, s in enumerate(styles)]
    jev = None
    if a.jev:
        from pokerbrain.llm.jev import JevClient
        jev = JevClient()
    log: list = []
    from pokerbrain.config import ULTIMATE
    from pokerbrain.llm.prompts import VARIANTS
    a.variant = a.variant or ULTIMATE["variant"]
    e = ULTIMATE["escalation"]
    esc = EscalationPolicy(mode="jev" if jev is not None else e["mode"], min_pot_bb=a.min_pot,
                           tricky_threshold=e["tricky_threshold"], always_pot_bb=e["always_pot_bb"])
    os.makedirs(a.qdir, exist_ok=True)
    with open(os.path.join(a.qdir, "SYSTEM_PROMPT.txt"), "w") as f:
        f.write(VARIANTS[a.variant]["system"])
    opus = ring_session(lambda: OpusAgent(FileBridgeDecider(a.qdir), variant=a.variant, jev=jev, escalation=esc,
                                          log=log, mix=ULTIMATE["mix"], override_gate=ULTIMATE.get("override_gate"),
                                          require_stakes=False, name="Hero", seed=5),
                        field, a.hands, seed=a.seed)
    quant = ring_session(lambda: QuantAgent("Hero", seed=5), field, a.hands, seed=a.seed)
    d, ci = paired_diff(opus, quant)
    summary = {"opus": opus.summary(), "quant": quant.summary(), "paired_diff_bb100": round(d, 1),
               "ci95": round(ci, 1), "opus_decisions": sum(1 for x in log if "choice" in x),
               "changed_vs_engine": sum(1 for x in log if x.get("choice") and x["choice"] != x.get("engine")),
               "gated_overrides": sum(1 for x in log if x.get("gated")),
               "errors": [x for x in log if "error" in x][:5]}
    os.makedirs(a.qdir, exist_ok=True)
    json.dump({"summary": summary, "log": log}, open(os.path.join(a.qdir, "result.json"), "w"), indent=1)
    with open(os.path.join(a.qdir, "MATCH_DONE"), "w") as f:
        f.write(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
