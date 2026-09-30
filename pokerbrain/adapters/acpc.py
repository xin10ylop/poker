"""ACPC dealer protocol v2 client (Annual Computer Poker Competition servers).

  -> VERSION:2:0:0
  <- MATCHSTATE:<position>:<handNumber>:<betting>:<cards>
  -> MATCHSTATE:<position>:<handNumber>:<betting>:<cards>:<action>
betting: f / c / r<X> where rX = raise TO X *total chips in the pot for the hand*;
'/' separates rounds.  cards: 'hole0|hole1/flop/turn/river' (only ours visible).
Heads-up no-limit reverse blinds: position 0 = big blind, position 1 = small blind
(button, first to act preflop).  Note the dealer's default time budget (~7 s average
per hand): keep the model escalation policy tight when playing ACPC matches.

Robustness: malformed lines are logged and skipped, a hand whose betting string cannot
be replayed is logged and skipped (we answer 'f' so the dealer never waits on us; it turns
an invalid fold into a check/call itself), agent exceptions fall back to a safe action,
and the socket has a timeout and is always closed, returning the partial result.
"""
from __future__ import annotations

import logging
import random
import re
import socket
from typing import Optional

from ..agents.base import Agent
from ..cards import ALL_CARDS
from ..engine import HandState, IllegalAction
from ..view import Decision, GameView, HandHistory

log = logging.getLogger("pokerbrain.acpc")

TIMEOUT = 30.0                 # seconds: connect and per-read socket timeout
FALLBACK_CALL_FRACTION = 0.05  # on an agent error, still call when the price is <= 5% of hero's stack
_CARDS = frozenset(ALL_CARDS)
_BETTING_RE = re.compile(r"(?:[fc/]|r\d+)*")


class ACPCError(ValueError):
    """Malformed or unsupported dealer input (the offending line or hand is skipped)."""


# ------------------------------------------------------------------ agent guard (shared with slumbot)
def fallback_decision(view: GameView) -> Decision:
    """Safe action when the agent fails: check if free, call if cheap (<= 5% of hero's stack), else fold."""
    la = view.legal
    cheap = 0 < la.call_amount <= FALLBACK_CALL_FRACTION * max(0, view.hero.stack)
    d = Decision("call" if (not la.can_check and cheap) else "check", source="fallback",
                 reason="agent error")
    return d.normalized(la)


def guarded_act(agent: Agent, view: GameView) -> tuple[Decision, bool]:
    """agent.act(view) normalized to a legal action; (fallback, True) if the agent raised or returned junk."""
    try:
        return agent.act(view).normalized(view.legal), False
    except Exception as exc:  # noqa: BLE001 - a broken agent must never end a match
        d = fallback_decision(view)
        log.warning("agent %s raised %s: %s; playing fallback %s", getattr(agent, "name", "?"),
                    type(exc).__name__, exc, d.kind)
        return d, True


# ------------------------------------------------------------------ parsing
def _split_cards(s: str, what: str) -> list:
    if len(s) % 2:
        raise ACPCError(f"bad {what} cards {s!r}")
    out = [s[i:i + 2] for i in range(0, len(s), 2)]
    for c in out:
        if c not in _CARDS:
            raise ACPCError(f"bad card {c!r} in {what} cards {s!r}")
    return out


def parse_cards(cards: str, position: int) -> tuple[list, list, list]:
    """'AhKh|/Qh7h2c/3d' -> (my hole, [revealed opponent holes], board).  Raises ACPCError."""
    parts = cards.split("/")
    holes = parts[0].split("|")
    if len(holes) != 2 or position not in (0, 1):
        raise ACPCError(f"expected two hole-card slots for a heads-up game, got {cards!r}")
    mine = _split_cards(holes[position], "hole")
    if len(mine) != 2:
        raise ACPCError(f"our hole cards are missing in {cards!r}")
    others = [_split_cards(h, "opponent hole") for i, h in enumerate(holes) if i != position and h]
    if any(len(o) != 2 for o in others):
        raise ACPCError(f"bad opponent hole cards in {cards!r}")
    rounds = [_split_cards(p, "board") for p in parts[1:]]
    if [len(r) for r in rounds] not in ([], [3], [3, 1], [3, 1, 1]):
        raise ACPCError(f"bad board layout in {cards!r}")
    board = [c for r in rounds for c in r]
    seen = mine + [c for o in others for c in o] + board
    if len(set(seen)) != len(seen):
        raise ACPCError(f"duplicate cards in {cards!r}")
    return mine, others, board


def parse_matchstate(line: str) -> tuple[int, int, str, str]:
    """'MATCHSTATE:<pos>:<hand>:<betting>:<cards>' -> (position, hand_no, betting, cards).  Raises ACPCError."""
    parts = line.strip().split(":")
    if len(parts) != 5 or parts[0] != "MATCHSTATE":
        raise ACPCError(f"expected MATCHSTATE:<position>:<hand>:<betting>:<cards>, got {line.strip()!r}")
    _, pos, hand_no, betting, cards = parts
    if pos not in ("0", "1"):
        raise ACPCError(f"bad position {pos!r} (heads-up only)")
    if not hand_no.isdigit():
        raise ACPCError(f"bad hand number {hand_no!r}")
    return int(pos), int(hand_no), betting, cards


def rebuild(position: int, betting: str, hole: list, board: list, stack: int, sb: int, bb: int,
            hand_id: str, opp_hole: Optional[list] = None) -> HandState:
    """Replay an ACPC betting string (hero = seat 0) through our engine.

    Raises ACPCError for unknown characters (e.g. limit-style 'r' without a size), actions on the
    wrong round, or actions our engine rejects (usually a --stack/--sb/--bb mismatch with the dealer)."""
    if not _BETTING_RE.fullmatch(betting):
        bad = next((i for i, ch in enumerate(betting) if ch not in "fcr/0123456789"), None)
        where = (f"unknown betting character {betting[bad]!r} at {bad}" if bad is not None
                 else "raise without a size (limit games are not supported)")
        raise ACPCError(f"{where} in {betting!r}")
    button = 0 if position == 1 else 1
    dead = set(hole) | set(board) | set(opp_hole or [])
    filler = [c for c in ALL_CARDS if c not in dead]
    random.Random(hand_id).shuffle(filler)
    opp = list(opp_hole) if opp_hole else filler[:2]
    rest = [c for c in filler if c not in opp]
    deck = list(hole) + opp + list(board) + rest[: 5 - len(board)]
    deck += [c for c in rest if c not in deck]
    h = HandState([stack, stack], button=button, sb=sb, bb=bb, deck=deck, names=["Hero", "Opponent"],
                  hand_id=hand_id)
    rnd, i = 0, 0
    try:
        while i < len(betting):
            ch = betting[i]
            if ch == "/":
                rnd += 1
                i += 1
                continue
            if h.finished:
                raise ACPCError(f"action {ch!r} at {i} after the hand ended in {betting!r}")
            if h.street_idx != rnd:
                raise ACPCError(f"action at {i} is on round {rnd} but the betting is on round {h.street_idx} "
                                f"in {betting!r}")
            if ch == "f":
                h.apply(Decision("fold"))
                i += 1
            elif ch == "c":
                la = h.legal_actions()
                h.apply(Decision("check" if la.can_check else "call"))
                i += 1
            else:  # 'r' + digits (guaranteed by _BETTING_RE)
                j = i + 1
                while j < len(betting) and betting[j].isdigit():
                    j += 1
                total = int(betting[i + 1:j])
                seat = h.to_act
                before_street = h.total_in[seat] - h.street_bets[seat]
                h.apply(Decision("raise", total - before_street))   # ACPC: hand total -> street level
                i = j
    except IllegalAction as exc:
        raise ACPCError(f"cannot replay {betting!r} with stack {stack}, blinds {sb}/{bb}: {exc} "
                        f"(do --stack/--sb/--bb match the dealer's game?)") from exc
    if not h.finished and rnd != h.street_idx:
        raise ACPCError(f"betting {betting!r} ends on round {rnd} but the betting is on round {h.street_idx}")
    return h


def to_acpc(d: Decision, h: HandState) -> str:
    if d.kind == "fold":
        return "f"
    if d.kind in ("check", "call"):
        return "c"
    seat = h.to_act
    before_street = h.total_in[seat] - h.street_bets[seat]
    return f"r{int(d.amount) + before_street}"


# ------------------------------------------------------------------ match loop
def play(agent: Agent, host: str, port: int, stack: int = 20000, sb: int = 50, bb: int = 100,
         max_hands: Optional[int] = None, timeout: float = TIMEOUT) -> dict:
    """Play until the dealer closes the connection (or max_hands finished hands).

    Always returns the (partial) result: hands, net_chips, bb100, plus skipped_hands, bad_lines,
    agent_errors and, when the match ended abnormally, 'error' / 'interrupted'."""
    if not (stack > 0 and 0 < sb <= bb):
        raise ValueError(f"need stack > 0 and 0 < sb <= bb, got stack={stack} sb={sb} bb={bb}")
    total, hands = 0, 0
    stats = {"skipped_hands": 0, "bad_lines": 0, "agent_errors": 0}
    last_hand: Optional[int] = None
    broken_hand: Optional[int] = None
    counted_hand: Optional[int] = None
    extra: dict = {}
    sock, f = None, None
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.settimeout(timeout)
        f = sock.makefile("rw", newline="", encoding="ascii", errors="replace")
        f.write("VERSION:2:0:0\r\n")
        f.flush()
        for raw in f:
            line = raw.strip()
            if not line or line[0] in "#;":
                continue
            try:
                position, hand_no, betting, cards = parse_matchstate(line)
            except ACPCError as exc:
                stats["bad_lines"] += 1
                log.warning("acpc: skipping malformed line: %s", exc)
                continue
            if hand_no != last_hand:
                last_hand = hand_no
                try:
                    agent.new_hand(hand_no)
                except Exception as exc:  # noqa: BLE001
                    stats["agent_errors"] += 1
                    log.warning("acpc: agent.new_hand raised %s: %s", type(exc).__name__, exc)
            if hand_no == broken_hand:
                if "f" not in betting:     # keep the dealer from waiting on us; ignored if it is not our turn
                    f.write(f"{line}:f\r\n")
                    f.flush()
                continue
            hand_id = f"acpc-{hand_no}"
            try:
                hole, others, board = parse_cards(cards, position)
                h = rebuild(position, betting, hole, board, stack, sb, bb, hand_id,
                            others[0] if others else None)
            except ACPCError as exc:
                broken_hand = hand_no
                stats["skipped_hands"] += 1
                log.error("acpc: skipping hand %d: %s", hand_no, exc)
                if "f" not in betting:
                    f.write(f"{line}:f\r\n")
                    f.flush()
                continue
            if h.finished:
                if counted_hand == hand_no:
                    continue
                counted_hand = hand_no
                hands += 1
                total += h.result.net[0]
                shown = {0: tuple(hole)}
                if others:
                    shown[1] = tuple(others[0])
                hist = HandHistory(hand_id=hand_id, sb=sb, bb=bb, button_seat=h.button, names=["Hero", "Opponent"],
                                   positions=list(h.positions), start_stacks=[stack, stack], actions=list(h.log),
                                   board=list(board), shown=shown if others else {},
                                   net={0: h.result.net[0], 1: h.result.net[1]}, hero_seat=0, platform="acpc")
                try:
                    agent.observe(hist, 0)
                except Exception as exc:  # noqa: BLE001
                    stats["agent_errors"] += 1
                    log.warning("acpc: agent.observe raised %s: %s", type(exc).__name__, exc)
                if agent.should_stop():      # stop-loss / circuit breaker / broke: leave the match
                    extra["stopped"] = "bankroll manager"
                    break
                if max_hands and hands >= max_hands:
                    break
                continue
            if h.to_act != 0:
                continue
            view = h.view_for(0, platform="acpc")
            d, failed = guarded_act(agent, view)
            stats["agent_errors"] += failed
            f.write(f"{line}:{to_acpc(d, h)}\r\n")
            f.flush()
    except KeyboardInterrupt:
        extra["interrupted"] = True
        log.warning("acpc: interrupted after %d hands", hands)
    except OSError as exc:   # refused / timeout / reset: the match is over, keep what we have
        extra["error"] = f"{type(exc).__name__}: {exc}"
        log.error("acpc: connection ended after %d hands: %s", hands, extra["error"])
    finally:
        for closer in (f, sock):
            try:
                if closer is not None:
                    closer.close()
            except OSError:
                pass
    return {"hands": hands, "net_chips": total, "bb100": 100.0 * total / bb / max(1, hands), **stats, **extra}
