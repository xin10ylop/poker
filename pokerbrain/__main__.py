"""pokerbrain command line.

  python -m pokerbrain sim      --agent ultimate --hands 500          # vs simulated field
  python -m pokerbrain slumbot  --agent quant --hands 200             # vs slumbot.com benchmark bot
  python -m pokerbrain acpc     --agent quant --host H --port P       # ACPC dealer (bot competitions)
  python -m pokerbrain serve    --agent ultimate --port 8765          # local HTTP decision service
  python -m pokerbrain analyze  --hole AhKh --board Qh7h2c --actions "r2.5 c | x" --button

Agents: quant | jev-reads | jev-decide | opus | ultimate
Keys come from the environment or a .env file (see .env.example); spending is
capped by POKERBRAIN_MAX_SPEND_USD.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from .bankroll import BankrollManager, Stakes
from .llm.budget import load_dotenv
from .opponents import OpponentDB


def make_agent(kind: str, db: OpponentDB, bankroll: BankrollManager | None = None, seed: int = 0,
               variant: str | None = None, escalation: str = "key"):
    from .agents.quant_agent import QuantAgent
    if kind == "quant":
        return QuantAgent("PokerBrain", db=db, bankroll=bankroll, seed=seed)
    from .llm.jev import JevClient
    if kind == "jev-reads":
        from .agents.llm_agents import JevReadsAgent
        return JevReadsAgent(JevClient(), name="PokerBrain", db=db, bankroll=bankroll, seed=seed)
    if kind == "jev-decide":
        from .agents.llm_agents import JevDeciderAgent
        return JevDeciderAgent(JevClient(), name="PokerBrain", db=db, bankroll=bankroll, seed=seed)
    if kind in ("opus", "ultimate"):
        from .agents.llm_agents import EscalationPolicy, OpusAgent, api_decider
        from .config import ULTIMATE
        from .llm.claude import OpusClient
        cfg = ULTIMATE
        client = OpusClient(effort=cfg["effort"])
        jev = JevClient() if (kind == "ultimate" and cfg["use_jev"]) else None
        e = cfg["escalation"]
        mode = escalation if kind == "opus" else e["mode"]
        if mode == "jev" and jev is None:
            mode = "key"
        esc = EscalationPolicy(mode=mode, min_pot_bb=e["min_pot_bb"], close_ev_bb=e["close_ev_bb"],
                               preflop=e["preflop"], tricky_threshold=e.get("tricky_threshold", 1.6),
                               always_pot_bb=e.get("always_pot_bb", 40.0))
        return OpusAgent(api_decider(client), variant=variant or cfg["variant"], jev=jev, escalation=esc,
                         verifier=cfg["verifier"] and jev is not None, jev_weight=cfg["jev_weight"],
                         reads_in_dashboard=cfg.get("reads_in_dashboard", False),
                         name="PokerBrain", db=db, bankroll=bankroll, seed=seed)
    raise SystemExit(f"unknown agent {kind!r}")


def main(argv=None) -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(prog="pokerbrain")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("sim", "slumbot", "acpc", "serve"):
        p = sub.add_parser(name)
        p.add_argument("--agent", default="quant")
        p.add_argument("--variant", default=None)
        p.add_argument("--db", default=None, help="opponent notes file (JSON); persisted across sessions")
        p.add_argument("--bankroll", type=float, default=None, help="bankroll in currency")
        p.add_argument("--stakes", default="1/2", help="small/big blind in currency, e.g. 0.5/1")
        p.add_argument("--seed", type=int, default=0)
        if name in ("sim", "slumbot", "acpc"):
            p.add_argument("--hands", type=int, default=200)
        if name == "sim":
            p.add_argument("--field", default="nit,station,maniac,fish,tilter",
                           help="comma list of bot styles (1 = heads-up)")
        if name == "acpc":
            p.add_argument("--host", required=True)
            p.add_argument("--port", type=int, required=True)
        if name == "serve":
            p.add_argument("--port", type=int, default=8765)
            p.add_argument("--host", default="127.0.0.1")
    p = sub.add_parser("analyze")
    p.add_argument("--hole", required=True)
    p.add_argument("--board", default="")
    p.add_argument("--actions", default="")
    p.add_argument("--button", action="store_true", help="hero is the button (heads-up)")
    p.add_argument("--stack", type=float, default=100)
    p.add_argument("--db", default=None)
    p.add_argument("--villain", default="Villain")
    a = ap.parse_args(argv)

    if a.cmd == "analyze":
        from .adapters.manual import analyze
        db = OpponentDB(a.db) if a.db else None
        print(analyze(a.hole, a.board, a.actions, a.button, a.stack, db, a.villain))
        return

    platform = {"sim": "sim", "slumbot": "slumbot", "acpc": "acpc", "serve": "api"}[a.cmd]
    db = OpponentDB(a.db, platform=platform) if a.db else OpponentDB(platform=platform)
    bankroll = None
    if a.bankroll:
        sb, bb = (float(x) for x in a.stakes.split("/"))
        bankroll = BankrollManager(bankroll=a.bankroll, stakes=Stakes(sb=sb, bb=bb))
    agent = make_agent(a.agent, db, bankroll, a.seed, a.variant)
    try:
        if a.cmd == "sim":
            from .arena import ring_session
            from .bots import StyleBot
            styles = a.field.split(",")
            field = lambda: [StyleBot(f"{s.capitalize()}{i + 1}", s, 100 + i) for i, s in enumerate(styles)]
            st = ring_session(lambda: agent, field, a.hands, seed=a.seed, progress=True)
            print(json.dumps(st.summary(), indent=1))
        elif a.cmd == "slumbot":
            from .adapters.slumbot import run
            s = run(agent, a.hands)
            print(json.dumps({"hands": s.hands, "bb100_raw": round(s.bb100(), 2),
                              "bb100_baseline_adjusted": round(s.bb100_baseline_adjusted(), 2)}, indent=1))
        elif a.cmd == "acpc":
            from .adapters.acpc import play
            print(play(agent, a.host, a.port, max_hands=a.hands))
        elif a.cmd == "serve":
            from .adapters.server import serve
            serve(agent, a.host, a.port)
    finally:
        if a.db:
            db.save()
        if bankroll:
            print("bankroll:", json.dumps(bankroll.context(), indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
