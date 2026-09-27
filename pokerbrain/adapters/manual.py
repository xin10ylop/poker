"""Manual hand analysis (study / hand review / home games).

Describe a heads-up spot in a compact line and get the full dashboard:
engine hand reading, villain range estimate, pot odds, EV of every action,
plus the villain's stored profile and notes if you have tracked him.

  actions: "r2.5 c | x b3.5 c | x x | b10"   (streets separated by |, amounts in bb:
            r/b = raise/bet to that street total, c = call, x = check, f = fold)
  The first action preflop belongs to the button (heads-up: button = small blind).
"""
from __future__ import annotations

import random
from typing import Optional

from ..cards import ALL_CARDS, parse_cards
from ..engine import HandState
from ..opponents import OpponentDB
from ..quant import QuantEngine
from ..view import Decision


def build_spot(hero_cards: str, board: str, actions: str, hero_is_button: bool, stack_bb: float = 100,
               bb: int = 100, villain_name: str = "Villain") -> HandState:
    hole = parse_cards(hero_cards)
    bd = parse_cards(board) if board else []
    dead = set(hole) | set(bd)
    filler = [c for c in ALL_CARDS if c not in dead]
    random.Random(0).shuffle(filler)
    deck = hole + filler[:2] + bd + filler[2:2 + 5 - len(bd)]
    deck += [c for c in filler[2:] if c not in deck]
    button = 0 if hero_is_button else 1
    h = HandState([int(stack_bb * bb)] * 2, button=button, sb=bb // 2, bb=bb, deck=deck,
                  names=["Hero", villain_name], hand_id="manual")
    for street_txt in actions.split("|"):
        for tok in street_txt.split():
            t = tok.lower()
            if t in ("x", "k", "check"):
                h.apply(Decision("check"))
            elif t in ("c", "call"):
                h.apply(Decision("call"))
            elif t in ("f", "fold"):
                h.apply(Decision("fold"))
            elif t[0] in "rb":
                h.apply(Decision("raise", int(round(float(t[1:]) * bb))))
            else:
                raise ValueError(f"bad action token {tok!r}")
    return h


def analyze(hero_cards: str, board: str, actions: str, hero_is_button: bool, stack_bb: float = 100,
            db: Optional[OpponentDB] = None, villain_name: str = "Villain") -> str:
    from ..llm.render import render_dashboard
    h = build_spot(hero_cards, board, actions, hero_is_button, stack_bb, villain_name=villain_name)
    if h.finished or h.to_act != 0:
        return "It is not hero's turn at the end of that action sequence."
    db = db or OpponentDB()
    view = h.view_for(0, platform="manual")
    rep = QuantEngine(db).analyze(view)
    return render_dashboard(view, rep, db, None, None, None)
