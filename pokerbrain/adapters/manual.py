"""Manual hand analysis (study / hand review / home games).

Describe a heads-up spot in a compact line and get the full dashboard:
engine hand reading, villain range estimate, pot odds, EV of every action,
plus the villain's stored profile and notes if you have tracked him.

  actions: "r2.5 c | x b3.5 c | x x | b10"   (streets separated by |, amounts in bb:
            r/b = raise/bet to that street total, c = call, x = check, f = fold)
  also accepted: "raise 3", "bet 10", "r 3", "check", "call", "fold", "allin" / "shove" / "jam"
  The first action preflop belongs to the button (heads-up: button = small blind).
  Every '|' must close exactly one betting round, and the board must hold exactly the cards
  of the street the actions reach (0 / 3 / 4 / 5): no card is ever invented.

All input problems raise ValueError with a one-line explanation.
"""
from __future__ import annotations

import random
import re
from typing import Optional

from ..cards import ALL_CARDS, parse_cards
from ..engine import HandState, IllegalAction
from ..opponents import OpponentDB
from ..quant import QuantEngine
from ..view import Decision

STREETS = ("preflop", "flop", "turn", "river")
BOARD_CARDS = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}
_SIMPLE = {"x": "check", "k": "check", "check": "check", "c": "call", "call": "call", "f": "fold", "fold": "fold"}
_ALLIN = {"allin", "all-in", "all_in", "shove", "jam"}
_SIZED = re.compile(r"(r|b|raise|bet)(\d*\.?\d*)(?:bb)?")
_NUMBER = re.compile(r"(\d+\.?\d*|\.\d+)(?:bb)?")


def _fmt_bb(chips: int, bb: int) -> str:
    return f"{chips / bb:g}bb"


def _cards(text: str, what: str) -> list:
    try:
        return parse_cards(text) if text else []
    except ValueError as exc:
        raise ValueError(f"{what} {text!r}: {exc} (use e.g. AhKd, Ts9s)") from None


def _tokens(street_txt: str, street: str) -> list[tuple[str, str, Optional[float]]]:
    """'raise 3 c' -> [('raise 3', 'raise', 3.0), ('c', 'call', None)]  ('allin' has amount None)."""
    words, out, i = street_txt.split(), [], 0
    while i < len(words):
        tok, t = words[i], words[i].lower()
        i += 1
        if t in _SIMPLE:
            out.append((tok, _SIMPLE[t], None))
            continue
        if t in _ALLIN:
            out.append((tok, "allin", None))
            continue
        m = _SIZED.fullmatch(t)
        if not m:
            hint = " (use a dot for decimals, e.g. r2.5)" if "," in t else ""
            raise ValueError(f"unknown action {tok!r} on the {street}{hint}; use x/c/f, r<bb>/b<bb> "
                             f"(e.g. r2.5, 'raise 3', 'bet 10') or allin")
        num = m.group(2)
        if not num:                              # 'raise 3' / 'r 3' / 'bet 10'
            if i < len(words) and _NUMBER.fullmatch(words[i].lower()):
                num = _NUMBER.fullmatch(words[i].lower()).group(1)
                tok = f"{tok} {words[i]}"
                i += 1
            else:
                raise ValueError(f"{tok!r} on the {street} needs a size in bb, e.g. r2.5 or 'raise 3'")
        try:
            amount = float(num)
        except ValueError:
            raise ValueError(f"bad size in {tok!r} on the {street}") from None
        if not amount > 0:
            raise ValueError(f"size must be positive in {tok!r} on the {street}")
        out.append((tok, "raise", amount))
    return out


def _apply(h: HandState, tok: str, kind: str, amount_bb: Optional[float], bb: int) -> None:
    street, actor = h.street, h.names[h.to_act]
    where = f"{tok!r} on the {street} ({actor} to act)"
    la = h.legal_actions()
    if kind == "allin":
        if la.can_raise:
            d = Decision("raise", la.max_raise_to)
        elif la.call_amount > 0:
            d = Decision("call")                  # facing a bet that covers us: all-in = call
        else:
            raise ValueError(f"{where}: cannot go all-in here (nothing to call and raising is closed)")
    elif kind == "raise":
        if not la.can_raise:
            raise ValueError(f"{where}: raising is not possible here")
        to = int(round(amount_bb * bb))
        if to > la.max_raise_to:
            raise ValueError(f"{where}: {_fmt_bb(to, bb)} exceeds the all-in amount of "
                             f"{_fmt_bb(la.max_raise_to, bb)} (use allin)")
        if to < la.min_raise_to and to != la.max_raise_to:
            raise ValueError(f"{where}: {_fmt_bb(to, bb)} is below the minimum raise to "
                             f"{_fmt_bb(la.min_raise_to, bb)} (sizes are the street total, not the increment)")
        d = Decision("raise", to)
    else:
        d = Decision(kind)
    try:
        h.apply(d)
    except IllegalAction as exc:
        raise ValueError(f"{where}: {exc}") from None


def build_spot(hero_cards: str, board: str, actions: str, hero_is_button: bool, stack_bb: float = 100,
               bb: int = 100, villain_name: str = "Villain") -> HandState:
    """Replay the described spot.  Raises ValueError (one line) on any inconsistent input."""
    hole = _cards(hero_cards, "hole cards")
    if len(hole) != 2:
        raise ValueError(f"hole cards {hero_cards!r}: need exactly 2 cards, got {len(hole)}")
    bd = _cards(board, "board")
    if len(bd) not in (0, 3, 4, 5):
        raise ValueError(f"board {board!r}: need 0, 3, 4 or 5 cards, got {len(bd)}")
    seen: set = set()
    for c in hole + bd:
        if c in seen:
            raise ValueError(f"card {c} appears twice in hole cards {hero_cards!r} / board {board!r}")
        seen.add(c)
    try:
        chips = int(round(float(stack_bb) * bb))
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"bad stack {stack_bb!r}") from None
    if chips <= 0:
        raise ValueError(f"stack must be positive, got {stack_bb!r}bb")
    filler = [c for c in ALL_CARDS if c not in seen]
    random.Random(0).shuffle(filler)
    deck = hole + filler[:2] + bd + filler[2:2 + 5 - len(bd)]
    deck += [c for c in filler[2:] if c not in deck]
    button = 0 if hero_is_button else 1
    h = HandState([chips] * 2, button=button, sb=bb // 2, bb=bb, deck=deck,
                  names=["Hero", villain_name], hand_id="manual")
    for k, street_txt in enumerate((actions or "").split("|")):
        street = STREETS[min(k, 3)]
        if k > 0:                                  # the '|' before this segment closes street k-1
            prev = STREETS[min(k - 1, 3)]
            if k > 3:
                raise ValueError(f"too many '|' in {actions!r}: at most 3 (preflop | flop | turn | river)")
            if h.finished:
                raise ValueError(f"the hand is already over before '|' number {k} in {actions!r}")
            if h.street_idx != k:
                raise ValueError(f"'|' number {k} in {actions!r} closes the {prev}, but the {prev} betting is "
                                 f"not complete ({h.names[h.to_act]} still to act)")
        for tok, kind, amount in _tokens(street_txt, street):
            if h.finished:
                raise ValueError(f"{tok!r}: the hand is already over; remove the extra actions")
            if h.street_idx != k:
                raise ValueError(f"{tok!r}: the {street} betting is already complete; start the "
                                 f"{h.street} with '|'")
            _apply(h, tok, kind, amount, bb)
    if not h.finished:
        need = BOARD_CARDS[h.street]
        if len(bd) != need:
            raise ValueError(f"the actions end on the {h.street}, so the board needs {need} cards, "
                             f"but it has {len(bd)} ({board!r})")
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
