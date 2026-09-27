"""Board texture, hand features and fast strength estimates.

`hand_strength(combo, board)` = raw strength percentile vs a uniform random
hand on this board (1.0 = nuts).  `effective_strength` adds draw potential
(EHS = HS + (1-HS) * PPot).  Both are cached per board so that range
narrowing over ~1000 combos stays cheap.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from functools import lru_cache

from .cards import ALL_COMBOS, RANK_VALUE, RANKS, evaluate, hand_type

HAND_TYPES = ["High Card", "Pair", "Two Pair", "Trips", "Straight", "Flush", "Full House", "Quads",
              "Straight Flush"]
SUIT_SYMBOL = {"c": "♣", "d": "♦", "h": "♥", "s": "♠"}


def pretty(cards) -> str:
    return " ".join(c[0] + SUIT_SYMBOL[c[1]] for c in cards)


def _rank_set(cards) -> set[int]:
    return {RANK_VALUE[c[0]] for c in cards}


def _windows(ranks: set[int]):
    rs = set(ranks)
    if 12 in rs:
        rs.add(-1)
    for low in range(-1, 9):
        yield low, set(range(low, low + 5)), rs


def straight_completers(ranks: set[int]) -> set[int]:
    """Ranks that would complete a 5-card straight given these ranks."""
    need = set()
    for _, window, rs in _windows(ranks):
        missing = window - rs
        if len(missing) == 1:
            m = missing.pop()
            need.add(12 if m == -1 else m)
    return need


def has_straight(ranks: set[int]) -> bool:
    return any(window <= rs for _, window, rs in _windows(ranks))


@dataclass
class BoardTexture:
    cards: list
    paired: bool = False
    trips: bool = False
    max_suit: int = 0
    monotone: bool = False
    two_tone: bool = False
    flush_possible: bool = False      # >= 3 of a suit
    flush_draw_possible: bool = False  # exactly 2 of a suit with cards to come
    straight_possible: bool = False   # 3 ranks within a 5-window
    four_straight: bool = False
    high_rank: str = ""
    wetness: float = 0.0              # 0 dry .. 1 very wet
    label: str = ""


def board_texture(board) -> BoardTexture:
    t = BoardTexture(cards=list(board))
    if not board:
        t.label = "preflop"
        return t
    ranks = [RANK_VALUE[c[0]] for c in board]
    suits = [c[1] for c in board]
    counts = {r: ranks.count(r) for r in set(ranks)}
    t.paired = any(v >= 2 for v in counts.values())
    t.trips = any(v >= 3 for v in counts.values())
    t.max_suit = max(suits.count(s) for s in set(suits))
    t.monotone = len(board) >= 3 and t.max_suit == len(board) == 3
    t.flush_possible = t.max_suit >= 3
    t.two_tone = t.max_suit == 2
    t.flush_draw_possible = t.max_suit == 2 and len(board) < 5
    rs = set(ranks)
    best = 0
    for _, window, rset in _windows(rs):
        best = max(best, len(window & rset))
    t.straight_possible = best >= 3
    t.four_straight = best >= 4
    t.high_rank = RANKS[max(ranks)]
    wet = 0.0
    wet += 0.35 if t.flush_possible else (0.2 if t.flush_draw_possible else 0.0)
    wet += 0.35 if t.four_straight else (0.25 if t.straight_possible else 0.0)
    close = sum(1 for a in rs for b in rs if a < b and b - a <= 2)
    wet += min(0.3, 0.1 * close)
    if t.paired:
        wet -= 0.1
    t.wetness = max(0.0, min(1.0, wet))
    parts = []
    parts.append("wet" if t.wetness >= 0.55 else ("medium" if t.wetness >= 0.3 else "dry"))
    if t.monotone:
        parts.append("monotone")
    elif t.flush_possible:
        parts.append(f"{t.max_suit}-flush")
    elif t.two_tone:
        parts.append("two-tone")
    else:
        parts.append("rainbow")
    if t.trips:
        parts.append("trips on board")
    elif t.paired:
        parts.append("paired")
    if t.four_straight:
        parts.append("4-straight")
    elif t.straight_possible:
        parts.append("straighty")
    parts.append(f"{t.high_rank}-high")
    t.label = ", ".join(parts)
    return t


@dataclass
class HandFeatures:
    made: str = "high_card"          # fine-grained class (see classify below)
    category: str = "High Card"      # evaluator category
    strength_class: str = "air"      # nuts / strong / medium / weak / draw / air
    flush_draw: bool = False
    nut_flush_draw: bool = False
    oesd: bool = False
    gutshot: bool = False
    backdoor_fd: bool = False
    overcards: int = 0
    outs: int = 0
    notes: list = field(default_factory=list)

    def describe(self) -> str:
        d = self.made.replace("_", " ")
        extras = []
        if self.nut_flush_draw:
            extras.append("nut flush draw")
        elif self.flush_draw:
            extras.append("flush draw")
        if self.oesd:
            extras.append("open-ended straight draw")
        elif self.gutshot:
            extras.append("gutshot")
        if self.backdoor_fd:
            extras.append("backdoor flush draw")
        if self.overcards and self.category == "High Card":
            extras.append(f"{self.overcards} overcard(s)")
        s = d + (" + " + ", ".join(extras) if extras else "")
        if self.outs:
            s += f" (~{self.outs} outs)"
        return s


def hand_features(hole, board) -> HandFeatures:
    f = HandFeatures()
    if len(board) < 3:
        return f
    cards = list(hole) + list(board)
    val = evaluate(cards)
    f.category = hand_type(val)
    board_val = evaluate(list(board)) if len(board) >= 5 else None
    hr = sorted((RANK_VALUE[c[0]] for c in hole), reverse=True)
    br = sorted((RANK_VALUE[c[0]] for c in board), reverse=True)
    board_ranks = set(br)
    pocket_pair = hr[0] == hr[1]
    cat = f.category

    # --- made hand classification
    if cat == "Pair":
        if pocket_pair:
            if hr[0] > br[0]:
                f.made = "overpair"
            elif hr[0] > br[-1]:
                f.made = "middle_pocket_pair"
            else:
                f.made = "underpair"
        else:
            paired = [r for r in hr if r in board_ranks]
            if not paired:
                f.made = "board_pair_only"
            else:
                p = paired[0]
                kicker = hr[1] if hr[0] == p else hr[0]
                if p == br[0]:
                    top_kick = 12 if br[0] != 12 else 11
                    good = kicker >= 9 or kicker == top_kick
                    f.made = "top_pair_top_kicker" if kicker == top_kick else (
                        "top_pair_good_kicker" if good else "top_pair_weak_kicker")
                elif len(br) > 1 and p == sorted(board_ranks, reverse=True)[1]:
                    f.made = "second_pair"
                else:
                    f.made = "bottom_pair"
    elif cat == "Two Pair":
        used = [r for r in hr if r in board_ranks]
        if len(set(used)) == 2 and not pocket_pair:
            f.made = "two_pair_both_cards"
        elif pocket_pair and hr[0] > br[0]:
            f.made = "overpair_on_paired_board"
        else:
            f.made = "one_card_two_pair" if used else "board_two_pair"
    elif cat == "Trips":
        if pocket_pair and hr[0] in board_ranks:
            f.made = "set"
        elif any(r in board_ranks for r in hr):
            f.made = "trips"
        else:
            f.made = "board_trips"
    elif cat in ("Straight", "Flush", "Full House", "Quads", "Straight Flush"):
        f.made = cat.lower().replace(" ", "_")
        if board_val is not None and board_val == val:
            f.made = "board_plays_" + f.made
    else:
        f.made = "high_card"
        f.overcards = sum(1 for r in hr if r > br[0])

    # --- draws (only with cards to come)
    if len(board) < 5:
        for s in "cdhs":
            n_all = sum(1 for c in cards if c[1] == s)
            n_hole = sum(1 for c in hole if c[1] == s)
            if n_all == 4 and n_hole >= 1 and cat not in ("Flush", "Straight Flush"):
                f.flush_draw = True
                hole_suited = sorted((RANK_VALUE[c[0]] for c in hole if c[1] == s), reverse=True)
                missing_high = [r for r in range(12, -1, -1)
                                if not any(RANK_VALUE[c[0]] == r and c[1] == s for c in board)]
                f.nut_flush_draw = bool(hole_suited) and hole_suited[0] == missing_high[0]
            if len(board) == 3 and n_all == 3 and n_hole >= 1 and sum(1 for c in board if c[1] == s) <= 2:
                f.backdoor_fd = True
        if cat not in ("Straight", "Flush", "Full House", "Quads", "Straight Flush"):
            all_r = _rank_set(cards)
            mine = straight_completers(all_r) - straight_completers(board_ranks) - all_r
            if not has_straight(all_r):
                if len(mine) >= 2:
                    f.oesd = True
                elif len(mine) == 1:
                    f.gutshot = True
        outs = 0
        if f.flush_draw:
            outs += 9
        if f.oesd:
            outs += 8 if not f.flush_draw else 6
        elif f.gutshot:
            outs += 4 if not f.flush_draw else 3
        if f.made == "high_card":
            outs += 3 * f.overcards
        elif f.made in ("bottom_pair", "second_pair", "underpair", "middle_pocket_pair"):
            outs += 2 if f.made in ("underpair", "middle_pocket_pair") else 5
        f.outs = outs

    # --- coarse strength class
    strong_made = {"set", "straight", "flush", "full_house", "quads", "straight_flush", "two_pair_both_cards"}
    medium_made = {"overpair", "top_pair_top_kicker", "top_pair_good_kicker", "overpair_on_paired_board", "trips"}
    weak_made = {"top_pair_weak_kicker", "second_pair", "middle_pocket_pair", "one_card_two_pair",
                 "bottom_pair", "underpair"}
    if f.made in strong_made:
        f.strength_class = "strong"
    elif f.made in medium_made:
        f.strength_class = "medium"
    elif f.made in weak_made:
        f.strength_class = "weak"
    elif f.flush_draw or f.oesd or (f.gutshot and f.overcards):
        f.strength_class = "draw"
    else:
        f.strength_class = "air"
    return f


# ---------------------------------------------------------------------------
# Strength percentiles per board (cached)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=512)
def _board_table(board: tuple) -> tuple[dict, list]:
    dead = set(board)
    vals = {}
    for combo in ALL_COMBOS:
        if combo[0] in dead or combo[1] in dead:
            continue
        vals[combo] = evaluate(list(combo) + list(board))
    return vals, sorted(vals.values())


def hand_strength(combo, board) -> float:
    """Percentile of the combo's current hand vs all live combos (0..1)."""
    if len(board) < 3:
        from .preflop import class_percentile
        from .cards import hand_class
        return 1.0 - class_percentile()[hand_class(combo)]
    vals, sorted_vals = _board_table(tuple(board))
    v = vals.get(tuple(combo))
    if v is None:
        v = evaluate(list(combo) + list(board))
    lo = bisect.bisect_left(sorted_vals, v)
    hi = bisect.bisect_right(sorted_vals, v)
    return (lo + 0.5 * (hi - lo)) / len(sorted_vals)


_POT = {  # probability of improving to a strong hand by the river
    3: {"fd": 0.35, "nfd_bonus": 0.04, "oesd": 0.31, "combo": 0.52, "gut": 0.16, "over": 0.10, "bdfd": 0.04},
    4: {"fd": 0.19, "nfd_bonus": 0.02, "oesd": 0.17, "combo": 0.32, "gut": 0.09, "over": 0.05, "bdfd": 0.0},
}


@lru_cache(maxsize=512)
def _board_ehs(board: tuple) -> dict:
    vals, _ = _board_table(board)
    out = {}
    for combo in vals:
        hs = hand_strength(combo, board)
        if len(board) >= 5:
            out[combo] = hs
            continue
        p = _POT[len(board)]
        f = hand_features(combo, board)
        pot = 0.0
        if f.flush_draw and (f.oesd or f.gutshot):
            pot = p["combo"]
        elif f.flush_draw:
            pot = p["fd"] + (p["nfd_bonus"] if f.nut_flush_draw else 0.0)
        elif f.oesd:
            pot = p["oesd"]
        elif f.gutshot:
            pot = p["gut"]
        if f.made == "high_card" and f.overcards:
            pot += p["over"] * f.overcards / 2
        if f.backdoor_fd:
            pot += p["bdfd"]
        out[combo] = hs + (1 - hs) * min(pot, 0.6)
    return out


def effective_strength(combo, board) -> float:
    if len(board) < 3:
        return hand_strength(combo, board)
    table = _board_ehs(tuple(board))
    v = table.get(tuple(combo))
    return v if v is not None else hand_strength(combo, board)


def strengths_for(combos, board) -> dict:
    """EHS for many combos at once (uses the per-board cache)."""
    if len(board) < 3:
        return {c: hand_strength(c, board) for c in combos}
    table = _board_ehs(tuple(board))
    return {c: table.get(tuple(c), 0.0) for c in combos}


def nut_rank(hole, board) -> tuple[int, int]:
    """(number of live combos that beat hero, number that tie) on this board."""
    if len(board) < 3:
        return (0, 0)
    vals, sorted_vals = _board_table(tuple(board))
    hv = evaluate(list(hole) + list(board))
    dead = set(hole)
    better = sum(1 for c, v in vals.items() if v > hv and c[0] not in dead and c[1] not in dead)
    ties = sum(1 for c, v in vals.items() if v == hv and c[0] not in dead and c[1] not in dead)
    return better, ties
