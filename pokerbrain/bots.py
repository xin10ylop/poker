"""Simulated opponent field: parametric player styles incl. psychological types.

These bots are the sparring partners used to test and tune the brain.  Each
style is a caricature of a real online player type, and some carry
*exploitable psychology* on purpose:
  * TILTER   - solid TAG until he loses a big pot, then plays like a maniac
  * SIZER    - bet size gives away hand strength (big = value, small = bluff)
  * STATION  - calls too much, never bluffs
  * MANIAC   - over-bluffs and over-raises
  * NIT      - folds too much, only bets strong hands
A good player must *discover* these tendencies from the table (the bots never
announce them) and exploit them.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, replace

from .agents.base import Agent
from .preflop import class_percentile
from .cards import hand_class, stable_hash
from .texture import effective_strength, hand_features, hand_strength
from .view import Decision, GameView, HandHistory

POS_MULT = {"UTG": 0.8, "UTG1": 0.8, "UTG2": 0.85, "MP": 0.9, "HJ": 1.0, "CO": 1.3, "BTN": 2.0, "SB": 1.6,
            "BB": 1.0}


@dataclass(frozen=True)
class Style:
    name: str
    vpip: float
    pfr: float
    threebet: float
    fold_to_3bet: float
    limp: float            # share of non-raising VPIP hands that limp (vs fold)
    cbet: float
    barrel: float
    stab: float            # bet when checked to
    value_thr: float       # strength needed to bet for value
    bluff: float           # probability to bluff a weak hand when a bet is "available"
    call_base: float       # strength needed to call a ~half-pot bet
    size_sens: float       # how much bigger bets tighten calling (0 = station, 1 = normal)
    raise_thr: float       # strength needed to raise a bet
    bet_size: float        # typical bet (pot fraction)
    big_size: float        # value-heavy size
    trap: float = 0.1      # slowplay probability with monsters
    sizing_tell: bool = False
    tilt_prone: bool = False


STYLES = {
    "nit": Style("nit", 0.13, 0.10, 0.03, 0.72, 0.1, 0.55, 0.35, 0.25, 0.74, 0.05, 0.62, 1.0, 0.93, 0.55, 0.7),
    "tag": Style("tag", 0.22, 0.18, 0.07, 0.55, 0.05, 0.66, 0.50, 0.40, 0.64, 0.18, 0.50, 1.0, 0.88, 0.6, 0.8),
    "lag": Style("lag", 0.32, 0.26, 0.11, 0.45, 0.05, 0.76, 0.60, 0.55, 0.58, 0.32, 0.44, 0.9, 0.84, 0.65, 0.9),
    "station": Style("station", 0.48, 0.07, 0.02, 0.30, 0.8, 0.35, 0.25, 0.25, 0.80, 0.03, 0.22, 0.3, 0.96, 0.5,
                     0.6, trap=0.3),
    "maniac": Style("maniac", 0.62, 0.45, 0.22, 0.25, 0.1, 0.90, 0.78, 0.75, 0.50, 0.55, 0.34, 0.6, 0.72, 0.9,
                    1.2),
    "fish": Style("fish", 0.42, 0.08, 0.02, 0.60, 0.85, 0.30, 0.20, 0.20, 0.80, 0.05, 0.56, 1.1, 0.95, 0.5, 0.6,
                  trap=0.3),
    "tilter": Style("tilter", 0.22, 0.18, 0.07, 0.55, 0.05, 0.66, 0.50, 0.40, 0.64, 0.18, 0.50, 1.0, 0.88, 0.6,
                    0.8, tilt_prone=True),
    "sizer": Style("sizer", 0.28, 0.22, 0.09, 0.50, 0.05, 0.72, 0.58, 0.50, 0.62, 0.30, 0.46, 0.9, 0.86, 0.33,
                   1.15, sizing_tell=True),
}

TILTED = Style("tilted", 0.58, 0.44, 0.20, 0.25, 0.1, 0.90, 0.80, 0.75, 0.50, 0.55, 0.30, 0.5, 0.70, 0.9, 1.2)


class StyleBot(Agent):
    def __init__(self, name: str, style: Style | str, seed: int = 0):
        self.name = name
        self.base_style = STYLES[style] if isinstance(style, str) else style
        self.style = self.base_style
        self.seed = seed
        self.rng = random.Random(seed)
        self.tilt_hands_left = 0
        self.hands = 0
        self.tilt_events = 0

    # ------------------------------------------------------------------ hooks
    def new_hand(self, hand_index: int) -> None:
        self.rng = random.Random(stable_hash(self.seed, hand_index, self.name))
        if self.tilt_hands_left > 0:
            self.tilt_hands_left -= 1
            self.style = TILTED
        else:
            self.style = self.base_style

    def observe(self, history: HandHistory, my_seat: int) -> None:
        self.hands += 1
        if self.base_style.tilt_prone:
            net_bb = history.net.get(my_seat, 0) / history.bb
            if net_bb <= -45:
                self.tilt_hands_left = 18
                self.tilt_events += 1

    def stats(self) -> dict:
        return {"tilt_events": self.tilt_events}

    # ------------------------------------------------------------------ acting
    def act(self, view: GameView) -> Decision:
        if view.street == "preflop":
            return self._preflop(view)
        return self._postflop(view)

    def _jit(self, x: float, sd: float = 0.025) -> float:
        return x + self.rng.gauss(0, sd)

    def _preflop(self, view: GameView) -> Decision:
        st, la, bb = self.style, view.legal, view.bb
        pct = class_percentile()[hand_class(view.hole)]
        pos = view.hero.position
        mult = POS_MULT.get(pos, 1.0)
        pre = [a for a in view.actions if a.street == "preflop" and not a.kind.startswith("post")]
        raises = [a for a in pre if a.kind == "raise"]
        limpers = [a for a in pre if a.kind == "call" and not raises]
        open_thr = min(0.95, st.pfr * mult)
        play_thr = min(0.98, max(st.vpip * mult, open_thr + 0.02))
        if not raises:
            if pct < self._jit(open_thr):
                size = (3.0 if pos == "SB" else 2.5) * bb + len(limpers) * bb
                if limpers:
                    size = (3.5 + len(limpers)) * bb
                return Decision("raise", size)
            if la.can_check:
                return Decision("check")
            if pct < self._jit(play_thr) and (self.rng.random() < st.limp or pos == "SB"):
                return Decision("call")
            return Decision("fold")
        last = raises[-1].to
        n_raises = len(raises)
        i_opened = any(a.seat == view.hero_seat for a in raises)
        ip = pos in ("BTN", "CO") or (pos == "HJ" and raises[-1].seat != view.hero_seat)
        if n_raises == 1:
            tb = st.threebet
            cont = min(0.85, st.vpip * (1.25 if pos == "BB" else 0.6) + (0.08 if pos == "BB" else 0.0))
            if pct < self._jit(tb * 0.65, 0.01) or (cont < pct < cont + tb * 0.6 and self.rng.random() < 0.5 * (tb / 0.07)):
                mult3 = 3.0 if ip else 4.0
                return Decision("raise", last * mult3)
            if pct < self._jit(cont):
                return Decision("call")
            return Decision("fold")
        facing_shove = la.call_amount >= 0.6 * (view.hero.stack + view.hero.bet)
        if facing_shove:
            # preflop all-in: call with a tight, style-dependent range
            call_thr = min(0.25, 0.035 + 0.12 * max(0.0, st.vpip - 0.2) + 0.05 * (1 - st.fold_to_3bet))
            return Decision("call" if pct < call_thr else "fold")
        if n_raises == 2:
            if i_opened or any(a.seat == view.hero_seat for a in pre):
                cont = open_thr * (1 - st.fold_to_3bet)
                fourbet = 0.025 + 0.08 * max(0.0, st.threebet - 0.08)
                if pct < fourbet:
                    return Decision("raise", last * 2.3)
                if pct < self._jit(cont, 0.01):
                    return Decision("call")
                return Decision("fold")
            # cold facing a 3-bet
            if pct < 0.025:
                return Decision("raise", last * 2.3)
            if pct < 0.04:
                return Decision("call")
            return Decision("fold")
        # 4-bet and beyond
        jam = 0.02 + 0.06 * max(0.0, st.threebet - 0.1)
        if pct < jam:
            return Decision("raise", la.max_raise_to)
        if pct < jam + 0.015:
            return Decision("call")
        return Decision("fold")

    def _postflop(self, view: GameView) -> Decision:
        st, la, bb = self.style, view.legal, view.bb
        board = view.board
        s = effective_strength(view.hole, board)
        hs = hand_strength(view.hole, board)
        f = hand_features(view.hole, board)
        pot = view.pot
        drawish = f.flush_draw or f.oesd
        # was I the aggressor on the previous street?
        streets = ["preflop", "flop", "turn", "river"]
        prev = streets[streets.index(view.street) - 1]
        prev_aggr = None
        for a in view.actions:
            if a.street == prev and a.kind in ("bet", "raise"):
                prev_aggr = a.seat
        if la.call_amount == 0:
            if prev_aggr == view.hero_seat:
                freq = st.cbet if view.street == "flop" else st.barrel
            else:
                freq = st.stab
            if s >= self._jit(st.value_thr):
                if s > 0.95 and self.rng.random() < st.trap:
                    return Decision("check")
                if self.rng.random() < min(1.0, freq + 0.35):
                    return self._bet(view, s, value=True)
                return Decision("check")
            bluff_p = st.bluff * freq * (1.6 if drawish and view.street != "river" else 1.0)
            if s < 0.45 and self.rng.random() < bluff_p:
                return self._bet(view, s, value=False)
            if drawish and view.street != "river" and self.rng.random() < st.bluff * 0.8:
                return self._bet(view, s, value=False)
            return Decision("check")
        # facing a bet
        to_call = la.call_amount
        x = to_call / max(1, pot - to_call)
        # bigger bets tighten calling, but the effect saturates: whoever calls 2x pot calls a shove
        thr = st.call_base + 0.22 * st.size_sens * (min(x, 2.0) - 0.5)
        if view.street == "river":
            thr += 0.04
        if to_call >= 40 * bb or to_call >= 0.9 * view.hero.stack:
            # calling off a big stack: need a genuinely good hand (stations less so)
            thr = max(thr, 0.70 + 0.5 * (st.call_base - 0.5))
        # pot odds sanity for draws
        if drawish and view.street != "river":
            outs = f.outs
            eq_draw = min(0.6, outs * (0.04 if view.street == "flop" else 0.02))
            if eq_draw > to_call / (pot + to_call) * 0.9:
                thr = min(thr, s - 0.01)
        if s >= self._jit(st.raise_thr) and la.can_raise:
            if not (s > 0.95 and self.rng.random() < st.trap):
                return Decision("raise", view.current_bet_level() + (pot + to_call) * 0.9)
        if la.can_raise and s < 0.35 and self.rng.random() < st.bluff * 0.15:
            return Decision("raise", view.current_bet_level() + (pot + to_call) * 0.9)
        if s >= self._jit(thr):
            return Decision("call")
        return Decision("fold")

    def _bet(self, view: GameView, s: float, value: bool) -> Decision:
        st, la = self.style, view.legal
        pot = view.pot
        if st.sizing_tell:
            frac = st.big_size if s >= 0.8 else (0.5 if s >= 0.55 else st.bet_size)
        else:
            frac = st.big_size if (value and s > 0.85) else st.bet_size
            frac *= self.rng.uniform(0.85, 1.15)
        amt = int(frac * pot)
        if not la.can_raise:
            return Decision("check")
        return Decision("raise", amt)


def make_field(styles: list[str], seed: int = 0, prefix: str = "") -> list[StyleBot]:
    return [StyleBot(f"{prefix}{s.capitalize()}{i}", s, seed=seed * 1000 + i) for i, s in enumerate(styles)]
