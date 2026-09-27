"""ACPC dealer protocol v2 client (Annual Computer Poker Competition servers).

  -> VERSION:2:0:0
  <- MATCHSTATE:<position>:<handNumber>:<betting>:<cards>
  -> MATCHSTATE:<position>:<handNumber>:<betting>:<cards>:<action>
betting: f / c / r<X> where rX = raise TO X *total chips in the pot for the hand*;
'/' separates rounds.  cards: 'hole0|hole1/flop/turn/river' (only ours visible).
Heads-up no-limit reverse blinds: position 0 = big blind, position 1 = small blind
(button, first to act preflop).  Note the dealer's default time budget (~7 s average
per hand): keep the model escalation policy tight when playing ACPC matches.
"""
from __future__ import annotations

import random
import socket
from typing import Optional

from ..agents.base import Agent
from ..cards import ALL_CARDS
from ..engine import HandState
from ..view import Decision, HandHistory


def parse_cards(cards: str, position: int) -> tuple[list, list, list]:
    parts = cards.split("/")
    holes = parts[0].split("|")
    mine = holes[position] if position < len(holes) else ""
    others = [h for i, h in enumerate(holes) if i != position and h]
    board = "".join(parts[1:])
    split = lambda s: [s[i:i + 2] for i in range(0, len(s), 2)]
    return split(mine), [split(o) for o in others], split(board)


def rebuild(position: int, betting: str, hole: list, board: list, stack: int, sb: int, bb: int,
            hand_id: str, opp_hole: Optional[list] = None) -> HandState:
    """Replay an ACPC betting string (hero = seat 0) through our engine."""
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
    i = 0
    while i < len(betting) and not h.finished:
        ch = betting[i]
        if ch == "/":
            i += 1
            continue
        if ch == "f":
            h.apply(Decision("fold"))
            i += 1
        elif ch == "c":
            la = h.legal_actions()
            h.apply(Decision("check" if la.can_check else "call"))
            i += 1
        elif ch == "r":
            j = i + 1
            while j < len(betting) and betting[j].isdigit():
                j += 1
            total = int(betting[i + 1:j])
            seat = h.to_act
            before_street = h.total_in[seat] - h.street_bets[seat]
            h.apply(Decision("raise", total - before_street))   # ACPC: hand total -> street level
            i = j
        else:
            i += 1
    return h


def to_acpc(d: Decision, h: HandState) -> str:
    if d.kind == "fold":
        return "f"
    if d.kind in ("check", "call"):
        return "c"
    seat = h.to_act
    before_street = h.total_in[seat] - h.street_bets[seat]
    return f"r{int(d.amount) + before_street}"


def play(agent: Agent, host: str, port: int, stack: int = 20000, sb: int = 50, bb: int = 100,
         max_hands: Optional[int] = None) -> dict:
    sock = socket.create_connection((host, port))
    f = sock.makefile("rw", newline="")
    f.write("VERSION:2:0:0\r\n")
    f.flush()
    total, hands, last_hand = 0, 0, None
    for line in f:
        line = line.strip()
        if not line.startswith("MATCHSTATE"):
            continue
        _, pos, hand_no, betting, cards = line.split(":", 4)
        position = int(pos)
        hole, others, board = parse_cards(cards, position)
        hand_id = f"acpc-{hand_no}"
        if hand_no != last_hand:
            agent.new_hand(int(hand_no))
            last_hand = hand_no
        h = rebuild(position, betting, hole, board, stack, sb, bb, hand_id, others[0] if others else None)
        if h.finished:
            hands += 1
            total += h.result.net[0]
            shown = {0: tuple(hole)}
            if others:
                shown[1] = tuple(others[0])
            hist = HandHistory(hand_id=hand_id, sb=sb, bb=bb, button_seat=h.button, names=["Hero", "Opponent"],
                               positions=list(h.positions), start_stacks=[stack, stack], actions=list(h.log),
                               board=list(board), shown=shown if others else {},
                               net={0: h.result.net[0], 1: h.result.net[1]}, hero_seat=0, platform="acpc")
            agent.observe(hist, 0)
            if max_hands and hands >= max_hands:
                break
            continue
        if h.to_act != 0:
            continue
        view = h.view_for(0, platform="acpc")
        d = agent.act(view).normalized(view.legal)
        f.write(f"{line}:{to_acpc(d, h)}\r\n")
        f.flush()
    sock.close()
    return {"hands": hands, "net_chips": total, "bb100": 100.0 * total / bb / max(1, hands)}
