"""pokerbrain command line.

  python -m pokerbrain sim      --agent ultimate --hands 500          # vs simulated field
  python -m pokerbrain slumbot  --agent quant --hands 200             # vs slumbot.com benchmark bot
  python -m pokerbrain acpc     --agent quant --host H --port P       # ACPC dealer (plays until it hangs up)
  python -m pokerbrain serve    --agent ultimate --port 8765          # local HTTP decision service
  python -m pokerbrain analyze  --hole AhKh --board Qh7h2c --actions "r2.5 c | x" --button

Agents: quant | jev-reads | jev-decide | opus | ultimate   (--router postflop|jev|key picks when Opus is called)
Keys come from the environment or a .env file (see .env.example); spending is
capped by POKERBRAIN_MAX_SPEND_USD.  Give --stakes for opus/ultimate: Opus is only
consulted where its fee is small relative to the pot.
"""
from __future__ import annotations

import argparse
import json
import math
import signal
import sys

from .bankroll import BankrollManager, Stakes
from .llm.budget import load_dotenv
from .opponents import OpponentDB

AGENTS = ("quant", "jev-reads", "jev-decide", "opus", "ultimate")
OPUS_KINDS = ("opus", "ultimate")
DEFAULT_BANKROLL_STAKES = (1.0, 2.0)
SAVE_EVERY = 25          # hands between opponent-DB saves in sim / slumbot / acpc
NO_STAKES_WARNING = ("no --stakes given: Opus is only consulted where its fee is small relative to the pot; "
                     "without stakes the engine plays alone")


# ------------------------------------------------------------------ argument validators
def stakes_arg(text: str) -> tuple[float, float]:
    """'sb/bb' -> (sb, bb): two positive finite numbers with sb <= bb."""
    parts = str(text).split("/")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"expected small/big blind like 0.5/1, got {text!r}")
    try:
        sb, bb = float(parts[0]), float(parts[1])
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected two numbers like 0.5/1, got {text!r}") from None
    if not (math.isfinite(sb) and math.isfinite(bb)) or sb <= 0 or bb <= 0:
        raise argparse.ArgumentTypeError(f"blinds must be positive numbers, got {text!r}")
    if sb > bb:
        raise argparse.ArgumentTypeError(f"small blind must not exceed the big blind, got {text!r}")
    return sb, bb


def bankroll_arg(text: str) -> float:
    try:
        v = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number, got {text!r}") from None
    if not math.isfinite(v) or v < 0:
        raise argparse.ArgumentTypeError(f"bankroll must be a non-negative number, got {text!r}")
    return v


def variant_arg(text: str) -> str:
    from .llm.prompts import VARIANTS
    if text not in VARIANTS:
        raise argparse.ArgumentTypeError(f"unknown variant {text!r}; valid: {', '.join(sorted(VARIANTS))}")
    return text


def _warn(msg: str) -> None:
    print(f"pokerbrain: warning: {msg}", file=sys.stderr, flush=True)


# ------------------------------------------------------------------ agents
def make_agent(kind: str, db: OpponentDB, bankroll: BankrollManager | None = None, seed: int = 0,
               variant: str | None = None, escalation: str | None = None, stakes: Stakes | None = None):
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
    if kind in OPUS_KINDS:
        from .agents.llm_agents import EscalationPolicy, OpusAgent, api_decider
        from .config import ULTIMATE
        from .llm.claude import OpusClient
        cfg = ULTIMATE
        client = OpusClient(effort=cfg["effort"])
        e = cfg["escalation"]
        mode = escalation or e["mode"]
        jev = JevClient() if (kind == "ultimate" and (cfg["use_jev"] or mode == "jev")) else None
        if mode == "jev" and jev is None:
            mode = "key"
        esc = EscalationPolicy(mode=mode, min_pot_bb=e["min_pot_bb"], close_ev_bb=e["close_ev_bb"],
                               preflop=e["preflop"], tricky_threshold=e.get("tricky_threshold", 1.6),
                               always_pot_bb=e.get("always_pot_bb", 40.0))
        return OpusAgent(api_decider(client), variant=variant or cfg["variant"], jev=jev, escalation=esc,
                         verifier=cfg["verifier"] and jev is not None, jev_weight=cfg["jev_weight"],
                         reads_in_dashboard=cfg.get("reads_in_dashboard", False), mix=cfg.get("mix", False),
                         override_gate=cfg.get("override_gate"), name="PokerBrain", db=db, bankroll=bankroll,
                         stakes=stakes, seed=seed)
    raise SystemExit(f"unknown agent {kind!r}")


class _Autosave:
    """Transparent agent wrapper that saves the opponent DB every `every` observed hands."""

    def __init__(self, agent, save, every: int = SAVE_EVERY):
        self._agent, self._save, self._every, self._seen = agent, save, every, 0

    def __getattr__(self, name):
        agent = self.__dict__.get("_agent")
        if agent is None:
            raise AttributeError(name)
        return getattr(agent, name)

    def observe(self, history, my_seat):
        self._agent.observe(history, my_seat)
        self._seen += 1
        if self._every and self._seen % self._every == 0:
            try:
                self._save()
            except OSError as exc:
                _warn(f"could not save the opponent DB: {exc}")


# ------------------------------------------------------------------ signals
_SIGNALS = tuple(s for s in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGHUP", None)) if s is not None)


def _raise_interrupt(signum, frame):
    raise KeyboardInterrupt(f"signal {signum}")


def _set_handlers(handler) -> dict:
    """Install `handler` for SIGTERM/SIGHUP (main thread only); returns the previous handlers."""
    old = {}
    for s in _SIGNALS:
        try:
            old[s] = signal.signal(s, handler)
        except (ValueError, OSError):      # not the main thread / unsupported
            pass
    return old


def _restore_handlers(old: dict) -> None:
    for s, h in old.items():
        try:
            signal.signal(s, h if h is not None else signal.SIG_DFL)
        except (ValueError, OSError):
            pass


# ------------------------------------------------------------------ CLI
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="pokerbrain")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("sim", "slumbot", "acpc", "serve"):
        p = sub.add_parser(name)
        p.add_argument("--agent", default="quant", choices=AGENTS)
        p.add_argument("--variant", default=None, type=variant_arg, help="Opus prompt variant (opus/ultimate)")
        p.add_argument("--db", default=None, help="opponent notes file (JSON); persisted across sessions")
        p.add_argument("--bankroll", type=bankroll_arg, default=None, help="bankroll in currency")
        p.add_argument("--stakes", type=stakes_arg, default=None,
                       help="small/big blind in currency, e.g. 0.5/1 (needed for opus/ultimate to weigh "
                            "Opus fees; --bankroll without it assumes 1/2)")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--router", default=None, choices=["postflop", "jev", "key", "all", "never"],
                       help="when Opus is consulted, opus/ultimate only (default: config.ULTIMATE); "
                            "jev = cheaper budget router")
        if name in ("sim", "slumbot"):
            p.add_argument("--hands", type=int, default=200)
        if name == "sim":
            p.add_argument("--field", default="nit,station,maniac,fish,tilter",
                           help="comma list of bot styles (1 = heads-up)")
        if name == "acpc":
            p.add_argument("--host", required=True)
            p.add_argument("--port", type=int, required=True)
            p.add_argument("--hands", type=int, default=None,
                           help="stop after this many hands (default: play until the dealer ends the match)")
            p.add_argument("--stack", type=int, default=20000, help="starting stack in chips (dealer's game)")
            p.add_argument("--sb", type=int, default=50, help="small blind in chips (dealer's game)")
            p.add_argument("--bb", type=int, default=100, help="big blind in chips (dealer's game)")
        if name == "serve":
            p.add_argument("--port", type=int, default=8765)
            p.add_argument("--host", default="127.0.0.1")
            p.add_argument("--allow-remote", action="store_true",
                           help="accept requests from other hosts (only with --host other than 127.0.0.1)")
    p = sub.add_parser("analyze")
    p.add_argument("--hole", required=True)
    p.add_argument("--board", default="")
    p.add_argument("--actions", default="")
    p.add_argument("--button", action="store_true", help="hero is the button (heads-up)")
    p.add_argument("--stack", type=float, default=100)
    p.add_argument("--db", default=None)
    p.add_argument("--villain", default="Villain")
    return ap


def main(argv=None) -> None:
    load_dotenv()
    ap = build_parser()
    a = ap.parse_args(argv)

    if a.cmd == "analyze":
        from .adapters.manual import analyze
        db = OpponentDB(a.db) if a.db else None
        try:
            out = analyze(a.hole, a.board, a.actions, a.button, a.stack, db, a.villain)
        except ValueError as exc:
            print(f"pokerbrain analyze: error: {exc}", file=sys.stderr)
            raise SystemExit(2) from None
        print(out)
        return

    if a.cmd == "acpc" and not (a.stack > 0 and 0 < a.sb <= a.bb):
        ap.error(f"acpc needs --stack > 0 and 0 < --sb <= --bb (got {a.stack}, {a.sb}, {a.bb})")
    if a.router is not None and a.agent not in OPUS_KINDS:
        _warn(f"--router only applies to opus/ultimate; ignored for --agent {a.agent}")
    if a.variant is not None and a.agent not in OPUS_KINDS:
        _warn(f"--variant only applies to opus/ultimate; ignored for --agent {a.agent}")
    stakes = Stakes(sb=a.stakes[0], bb=a.stakes[1]) if a.stakes is not None else None
    if a.agent in OPUS_KINDS and stakes is None:
        _warn(NO_STAKES_WARNING)

    platform = {"sim": "sim", "slumbot": "slumbot", "acpc": "acpc", "serve": "api"}[a.cmd]
    db = OpponentDB(a.db, platform=platform) if a.db else OpponentDB(platform=platform)
    bankroll = None
    if a.bankroll is not None:
        if stakes is None:
            _warn("--bankroll without --stakes: assuming stakes %g/%g" % DEFAULT_BANKROLL_STAKES)
        bankroll = BankrollManager(bankroll=a.bankroll,
                                   stakes=stakes or Stakes(sb=DEFAULT_BANKROLL_STAKES[0], bb=DEFAULT_BANKROLL_STAKES[1]))
    agent = make_agent(a.agent, db, bankroll, a.seed, a.variant, a.router, stakes=stakes)
    if a.db and a.cmd != "serve":
        agent = _Autosave(agent, db.save)

    old_handlers = _set_handlers(_raise_interrupt)   # SIGTERM / SIGHUP -> KeyboardInterrupt -> finally saves
    code = 0
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
            print(json.dumps(s.summary(), indent=1))
        elif a.cmd == "acpc":
            from .adapters.acpc import play
            print(json.dumps(play(agent, a.host, a.port, stack=a.stack, sb=a.sb, bb=a.bb, max_hands=a.hands),
                             indent=1))
        elif a.cmd == "serve":
            from .adapters.server import serve
            serve(agent, a.host, a.port, allow_remote=a.allow_remote)
    except KeyboardInterrupt:
        print("pokerbrain: interrupted, saving and exiting", file=sys.stderr, flush=True)
        code = 130
    finally:
        _set_handlers(signal.SIG_IGN)      # a second signal must not abort the save
        try:
            if a.db:
                db.save()
            if bankroll is not None:
                print("bankroll:", json.dumps(bankroll.context(), indent=1))
        finally:
            _restore_handlers(old_handlers)
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    main(sys.argv[1:])
