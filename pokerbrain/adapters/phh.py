"""Real hand histories in the PHH format (https://phh.readthedocs.io) -> PokerBrain.

Used to study real human players: every hand is replayed through the rules engine, so the
opponent tracker, range estimation and decision views work exactly as in live play.

  hands = load_phhs("ps NLH handhq_1-OBFUSCATED.phhs")        # list of dicts
  rep = replay(hands[0])                                     # Replayed(history, points) or None

Hidden hole cards ("????") are filled with random stand-ins that never leave this module: only
cards actually shown at showdown appear in the HandHistory, and decision views are built with
no hole cards, i.e. from an outside observer's seat.
"""
from __future__ import annotations

import dataclasses
import random
import tomllib
from dataclasses import dataclass, field
from typing import Optional

from ..cards import ALL_CARDS, stable_hash
from ..engine import HandState, IllegalAction
from ..view import Decision, GameView, HandHistory

BB_CHIPS = 100


@dataclass
class DecisionPoint:
    view: GameView            # public information only (no hole cards), before the action
    seat: int
    name: str
    kind: str                 # fold | check | call | raise
    amount: int = 0           # raise-to amount (chips)


@dataclass
class Replayed:
    history: HandHistory
    points: list = field(default_factory=list)
    shown: dict = field(default_factory=dict)     # seat -> (card, card)  revealed at showdown
    seat_count: int = 0
    holes: dict = field(default_factory=dict)     # seat -> (card, card)  every hole card the record knows


def load_phhs(path: str) -> list[dict]:
    """All hands of a .phhs (or single-hand .phh) file, in file order (non-hand entries skipped)."""
    with open(path, "rb") as f:
        data = tomllib.load(f)
    if "variant" in data:                       # single-hand .phh
        return [data]
    out = []
    for k, v in data.items():
        if isinstance(v, dict):
            v.setdefault("_key", k)
            out.append(v)
    return out


def _cards(s: str) -> list[str]:
    return [s[i:i + 2] for i in range(0, len(s), 2)]


def replay(h: dict, keep_points: bool = True, platform: str = "phh", keep_holes: bool = False) -> Optional[Replayed]:
    """Replay one no-limit hand through the engine; None if unsupported (antes, straddles, malformed records).
    A showdown the record cannot settle keeps its actions and shown cards but has result_known=False.

    keep_holes: decision views keep the actor's hole cards when the record knows them (never stand-ins)."""
    try:
        return _replay(h, keep_points, platform, keep_holes)
    except Exception:  # noqa: BLE001 - a corrupt record must never abort an import
        return None


def _replay(h: dict, keep_points: bool, platform: str, keep_holes: bool) -> Optional[Replayed]:
    if not isinstance(h, dict) or h.get("variant") != "NT" or any(h.get("antes", [])):
        return None
    blinds = list(h.get("blinds_or_straddles", []))
    n = len(h["starting_stacks"])
    if n < 2 or len(blinds) < 2 or any(blinds[2:]) or blinds[1] <= 0:
        return None
    unit = BB_CHIPS / float(blinds[1])
    chips = lambda x: int(round(float(x) * unit))
    stacks = [chips(x) for x in h["starting_stacks"]]
    if min(stacks) <= 0:
        return None
    names = [str(p) for p in h.get("players", [f"p{i + 1}" for i in range(n)])]
    if len(names) != n:
        return None
    win = h.get("winnings")
    wins = None
    if isinstance(win, list) and len(win) == n:
        try:
            wins = [int(round(float(x) * unit)) for x in win]
        except (TypeError, ValueError):
            wins = None
    button = 1 if n == 2 else n - 1          # PHH p1 = small blind; heads-up (reversed blinds) p1 = big blind

    shown: dict[int, tuple] = {}
    holes: dict[int, tuple] = {}
    sd_seats: set = set()
    board: list[str] = []
    for a in h["actions"]:
        t = a.split()
        if t[0] == "d" and t[1] == "db":
            board += _cards(t[2])
        elif t[0] == "d" and t[1] == "dh" and len(t) >= 4 and "?" not in t[3]:
            holes[int(t[2][1:]) - 1] = tuple(_cards(t[3]))
        elif len(t) >= 3 and t[1] == "sm":
            sd_seats.add(int(t[0][1:]) - 1)                # reached showdown (shown or mucked)
            if "?" not in t[2]:
                shown[int(t[0][1:]) - 1] = tuple(_cards(t[2]))
    holes.update(shown)
    known = set(board) | {c for hc in holes.values() for c in hc}
    if len(known) != len(board) + 2 * len(holes):
        return None                               # duplicated cards: corrupt record
    rest = [c for c in ALL_CARDS if c not in known]
    random.Random(stable_hash(h.get("hand", 0), n)).shuffle(rest)
    deck = []
    for i in range(n):
        deck += list(holes[i]) if i in holes else [rest.pop(), rest.pop()]
    deck += board + [rest.pop() for _ in range(5 - len(board))]
    try:
        st = HandState(stacks, button=button, sb=chips(min(blinds[:2])), bb=BB_CHIPS, deck=deck, names=names,
                       hand_id=str(h.get("hand", h.get("_key", "?"))))
    except (ValueError, IllegalAction):
        return None

    points = []
    try:
        for a in h["actions"]:
            t = a.split()
            if t[0] == "d" or t[1] == "sm":
                continue
            if st.finished:
                break
            seat = int(t[0][1:]) - 1
            if st.to_act != seat:
                return None                        # out-of-order record
            la = st.legal_actions(seat)
            if t[1] == "f":
                d = Decision("fold") if la.can_fold else Decision("check")
            elif t[1] == "cc":
                d = Decision("check") if la.can_check else Decision("call")
            elif t[1] == "cbr":
                to = chips(t[2])
                d = Decision("raise", la.clamp_raise(to)) if la.can_raise else Decision("call")
            else:
                return None
            if keep_points:
                v = st.view_for(seat)
                hole = v.hole if (keep_holes and seat in holes) else ()
                players = [dataclasses.replace(pv, hole=(hole if (pv.seat == seat and hole) else None))
                           for pv in v.players]                    # stand-in cards never leave this module
                points.append(DecisionPoint(dataclasses.replace(v, hole=hole, players=players), seat, names[seat],
                                            d.kind, int(d.amount or 0)))
            st.apply(d)
    except (IllegalAction, ValueError, KeyError):
        return None
    if not st.finished:
        return None
    hist = st.result.to_history(platform=platform, table_id=str(h.get("table", "t1")))
    hist.shown = dict(shown)                       # only what really was shown
    hist.showdown = sorted(set(hist.showdown or []) | sd_seats)
    if wins is not None:
        # the record's result (what each player collected, net of rake, excluding his own uncalled bet) beats
        # the engine's settlement, which may rest on stand-in cards or on the first of two run-outs
        tin = list(st.total_in)
        top = sorted(tin, reverse=True)
        refund = [top[0] - top[1] if tin[i] == top[0] and tin.count(top[0]) == 1 else 0 for i in range(n)]
        hist.net = {i: wins[i] - (tin[i] - refund[i]) for i in range(n)}
        hist.ev_net = None
    elif len(board) > 5 or any(sd not in holes for sd in hist.showdown):
        # a showdown the record cannot settle (a muck, no recorded winnings): the shown cards and the
        # actions are still real, only the money result is unknown
        hist.net = {i: 0 for i in range(n)}
        hist.ev_net = None
        hist.result_known = False
    return Replayed(hist, points, shown, int(h.get("seat_count", n)), holes)
