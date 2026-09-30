"""Preflop knowledge: hand ranking + approximate GTO charts + exploit adjustments.

Charts are solver-approximate (6-max 100bb, 2.5bb opens; heads-up up to 200bb)
and are *baselines*.  The agent layers exploitative adjustments on top using the
opponent's HUD profile (fold-to-steal, fold-to-3bet, looseness...).
"""
from __future__ import annotations

import json
import os
from functools import lru_cache

from .cards import ALL_CLASSES
from .ranges import Range, parse_range_classes

_DATA = os.path.join(os.path.dirname(__file__), "data", "preflop_equity.json")


@lru_cache(maxsize=1)
def preflop_equity() -> dict[str, dict[str, float]]:
    with open(_DATA) as f:
        return json.load(f)


@lru_cache(maxsize=1)
def class_ranking() -> list[str]:
    """169 classes ordered strongest -> weakest (HU equity + 3-way equity blend)."""
    eq = preflop_equity()
    return sorted(ALL_CLASSES, key=lambda c: -(eq[c]["eq1"] + eq[c]["eq2"]))


@lru_cache(maxsize=1)
def class_percentile() -> dict[str, float]:
    """Percentile (0 = best, 1 = worst) of each class, combo-weighted."""
    from .cards import class_size
    out, acc = {}, 0
    for cls in class_ranking():
        n = class_size(cls)
        out[cls] = (acc + n / 2) / 1326
        acc += n
    return out


def top_range(frac: float) -> Range:
    return Range.top_fraction(frac, class_ranking())


# --------------------------------------------------------------------------
# 6-max 100bb charts (approximate solver output, simplified mixes)
# --------------------------------------------------------------------------
RFI_6MAX = {
    "UTG": "33+, 22:0.5, A2s+, K9s+, Q9s+, J9s+, T9s, 98s, 87s:0.5, 76s:0.5, 65s:0.5, "
           "AJo+, ATo:0.75, KQo, KJo:0.5",
    "HJ": "22+, A2s+, K7s+, Q9s+, J9s+, T8s+, 97s+, 87s, 76s, 65s, 54s:0.5, ATo+, KJo+, QJo, "
          "A9o:0.5, KTo:0.5",
    "CO": "22+, A2s+, K4s+, Q7s+, J8s+, T7s+, 97s+, 86s+, 75s+, 65s, 54s, A8o+, A5o:0.5, KTo+, "
          "QTo+, JTo, K9o:0.5",
    "BTN": "22+, A2s+, K2s+, Q3s+, J5s+, T6s+, 96s+, 85s+, 74s+, 64s+, 53s+, 43s, A2o+, K8o+, "
           "Q9o+, J9o+, T8o+, 98o, 87o:0.5, K7o:0.5, Q8o:0.5",
    "SB": "22+, A2s+, K2s+, Q4s+, J6s+, T6s+, 96s+, 85s+, 75s+, 64s+, 54s, A2o+, K8o+, Q9o+, "
          "J9o+, T9o, 98o:0.5",
}

# Facing a single open raise: {hero_pos: {opener_group: {"3bet": str, "call": str}}}
# opener groups: "EP" (UTG/HJ), "CO", "BTN", "SB"
VS_OPEN_6MAX = {
    "IP": {  # HJ / CO / BTN cold-calling or 3-betting in position
        "EP": {"3bet": "QQ+, AKs, AKo, AQs:0.5, A5s:0.5, A4s:0.5, KQs:0.3",
               "call": "JJ-22, AQs:0.5, AJs, ATs, KQs:0.7, KJs, QJs, JTs, T9s, 98s, AQo:0.5"},
        "CO": {"3bet": "TT+, AJs+, KQs, AQo+, A5s, A4s, KJs:0.5, QJs:0.5, JTs:0.3, 76s:0.3, 65s:0.3",
               "call": "99-22, ATs-A6s, KTs-K9s, QTs, J9s+, T8s+, 97s+, 86s+, 75s+, 65s:0.7, 54s:0.5, "
                       "AJo, KQo, KJo:0.5, QJo:0.5"},
        "BTN": {"3bet": "TT+, AJs+, KQs, AQo+, A5s, A4s", "call": "99-22, ATs-A8s, KJs, QJs, JTs"},
        "SB": {"3bet": "TT+, AJs+, KQs, AQo+", "call": "99-22, ATs-A8s, KJs, QJs, JTs"},
    },
    "SB": {  # SB is 3-bet-or-fold (mostly)
        "EP": {"3bet": "JJ+, AKs, AKo, AQs, A5s, KQs:0.5", "call": "TT:0.3, 99:0.3"},
        "CO": {"3bet": "99+, ATs+, A5s-A4s, KJs+, QJs, JTs:0.5, AJo+, KQo", "call": ""},
        "BTN": {"3bet": "88+, A9s+, A5s-A2s, KTs+, QTs+, JTs, T9s, 98s:0.5, AJo+, KQo, KJo:0.5",
                "call": ""},
        "SB": {"3bet": "", "call": ""},
    },
    "BB": {
        "EP": {"3bet": "QQ+, AKs, AKo, A5s:0.5, A4s:0.5",
               "call": "JJ-22, AQs-A2s, KJs-K6s, Q8s+, J8s+, T7s+, 96s+, 86s+, 75s+, 64s+, 53s+, "
                       "AQo-A9o, KTo+, QTo+, JTo"},
        "CO": {"3bet": "JJ+, AQs+, AKo, A5s-A4s, KQs:0.5, 76s:0.3, 65s:0.3",
               "call": "TT-22, AJs-A2s, KJs-K2s, Q5s+, J7s+, T7s+, 96s+, 85s+, 74s+, 63s+, 53s+, 43s, "
                       "AQo-A7o, A5o, KTo+, K9o:0.5, QTo+, Q9o:0.5, JTo, J9o:0.5, T9o, 98o:0.5"},
        "BTN": {"3bet": "TT+, AJs+, A5s-A2s:0.5, KQs, K5s:0.3, K4s:0.3, QJs:0.5, J9s:0.3, T8s:0.3, "
                        "76s:0.3, 65s:0.3, AQo+, A5o:0.3",
                "call": "99-22, ATs-A6s, KJs-K2s, Q2s+, J4s+, T6s+, 96s+, 85s+, 74s+, 63s+, 53s+, 42s+, "
                        "AJo-A2o, KJo-K5o, Q7o+, J7o+, T7o+, 97o+, 86o+, 76o, 65o"},
        "SB": {"3bet": "99+, ATs+, A5s-A2s, KTs+, QTs+, JTs, T9s:0.5, AJo+, KQo, K9s:0.3, 87s:0.3",
               "call": "88-22, A9s-A6s, K9s-K2s, Q2s+, J4s+, T5s+, 95s+, 85s+, 74s+, 63s+, 52s+, 42s+, "
                       "ATo-A2o, KJo-K4o, Q6o+, J7o+, T7o+, 97o+, 86o+, 75o+, 65o"},
    },
}

# Opener facing a 3-bet.
VS_3BET_6MAX = {
    "IP": {"4bet": "QQ+, AKs, AKo, A5s:0.5, A4s:0.3",
           "call": "JJ-77, AQs-ATs, KQs, KJs, QJs, JTs, T9s, 98s:0.5, AQo"},
    "OOP": {"4bet": "QQ+, AKs, AKo, A5s:0.5",
            "call": "JJ-99, AQs-AJs, KQs, AQo:0.5, JTs:0.5"},
}
VS_4BET = {"5bet": "KK+, AKs, QQ:0.5, AKo:0.5", "call": "QQ:0.5, JJ:0.3, AKo:0.5, AQs:0.3"}
VS_5BET = {"call": "QQ+, AKs, AKo"}

# Iso-raising limpers / overlimping.
ISO_RANGE = "22+, A2s+, K8s+, Q9s+, J9s+, T9s, 98s, A9o+, KTo+, QJo"
OVERLIMP_RANGE = "22-66, A2s-A9s, K9s, Q9s, J9s, T8s, 97s, 87s, 76s, 65s, 54s"
BB_RAISE_VS_LIMP = "88+, ATs+, A5s, KJs+, AJo+, KQo"

# --------------------------------------------------------------------------
# Heads-up charts (percentiles of the ranking; 100-200bb)
# --------------------------------------------------------------------------
HU = {
    "sb_open": 0.82,       # SB/BTN raises ~82%, folds the rest
    "bb_3bet": 0.14,       # BB vs open: 3-bet top 14% (plus some bluffs below)
    "bb_3bet_bluff": (0.62, 0.70),
    "bb_call": 0.68,       # BB continues (call) up to 68% (after 3-bets removed)
    "sb_4bet": 0.05,
    "sb_4bet_bluff": (0.30, 0.33),
    "sb_call_3bet": 0.42,
    "bb_call_4bet": 0.08,
    "bb_5bet": 0.035,
    "bb_raise_vs_limp": 0.30,
}

_POS_GROUP = {"UTG": "EP", "UTG1": "EP", "UTG2": "EP", "MP": "EP", "LJ": "EP", "HJ": "EP", "CO": "CO",
              "BTN": "BTN", "SB": "SB"}
# full-ring seats use the nearest (tighter) 6-max chart; anything unknown gets the tightest one
_RFI_ALIAS = {"UTG1": "UTG", "UTG2": "UTG", "MP": "UTG", "LJ": "HJ"}


@lru_cache(maxsize=256)
def chart_range(text: str) -> Range:
    return Range.from_classes(parse_range_classes(text)) if text else Range()


def rfi_range(position: str) -> Range:
    pos = _RFI_ALIAS.get(position, position)
    return chart_range(RFI_6MAX.get(pos, RFI_6MAX["UTG"]))


def vs_open_ranges(hero_pos: str, opener_pos: str) -> tuple[Range, Range]:
    if hero_pos in ("SB", "BB"):
        table = VS_OPEN_6MAX[hero_pos]
    else:
        table = VS_OPEN_6MAX["IP"]
    group = _POS_GROUP.get(opener_pos, "EP")
    entry = table.get(group) or table["CO"]
    return chart_range(entry["3bet"]), chart_range(entry["call"])


def hu_range(key: str) -> Range:
    v = HU[key]
    if isinstance(v, tuple):
        return Range.band(v[0], v[1], class_ranking())
    return Range.top_fraction(v, class_ranking())


def stack_depth_adjust_note(eff_bb: float) -> str:
    if eff_bb < 20:
        return "short-stack: push/fold territory; avoid flatting raises"
    if eff_bb < 40:
        return "medium-short: 3-bet/shove more, fewer speculative calls"
    if eff_bb > 150:
        return "deep: implied odds up for suited connectors/small pairs; 4-bet bluffs riskier"
    return ""
