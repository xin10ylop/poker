"""QuantAgent: the non-LLM brain.

Preflop: solver-approximate charts + exploitative adjustments driven by the
opponents' HUD profiles (steal vs over-folders, 3-bet bluff vs high
fold-to-3bet, tighten vs nits, value-heavy vs stations...).  Short stacks and
odd spots fall back to the EV engine.

Postflop: maximize risk-adjusted EV from the QuantEngine (Bayesian ranges,
calibrated villain response model), mixing between near-equal options.
"""
from __future__ import annotations

import random
from typing import Callable, Optional

from ..bankroll import BankrollManager
from ..cards import hand_class, make_combo, stable_hash
from ..opponents import OpponentDB
from ..preflop import (BB_RAISE_VS_LIMP, ISO_RANGE, OVERLIMP_RANGE, VS_3BET_6MAX, VS_4BET, VS_5BET,
                       chart_range, class_percentile, hu_range, rfi_range, vs_open_ranges)
from ..quant import QuantEngine, QuantReport
from ..view import Decision, GameView, HandHistory
from .base import Agent


class PreflopSpot:
    def __init__(self, view: GameView):
        pre = [a for a in view.actions if a.street == "preflop" and not a.kind.startswith("post")]
        self.raises = [a for a in pre if a.kind == "raise"]
        self.limpers = [a for a in pre if a.kind == "call" and not any(r for r in pre[:pre.index(a)]
                                                                         if r.kind == "raise")]
        self.callers_after_raise = []
        if self.raises:
            first_raise_idx = pre.index(self.raises[-1])
            self.callers_after_raise = [a for a in pre[first_raise_idx + 1:] if a.kind == "call"]
        self.hero_acted = any(a.seat == view.hero_seat for a in pre)
        self.hero_raised = any(a.seat == view.hero_seat for a in self.raises)
        self.n_players = len(view.players)
        if not self.raises:
            self.kind = "unopened" if not self.limpers else "limped"
        elif len(self.raises) == 1:
            self.kind = "squeeze" if self.callers_after_raise else "vs_open"
        elif len(self.raises) == 2:
            self.kind = "vs_3bet" if self.hero_raised else "cold_3bet"
        elif len(self.raises) == 3:
            self.kind = "vs_4bet" if self.hero_raised else "cold_4bet"
        else:
            self.kind = "vs_5bet"
        self.last_raiser = self.raises[-1].seat if self.raises else None


class QuantAgent(Agent):
    def __init__(self, name: str = "QuantBrain", db: Optional[OpponentDB] = None,
                 bankroll: Optional[BankrollManager] = None, seed: int = 0, exploit: bool = True,
                 mix_temperature_bb: float = 0.12, reads_provider: Optional[Callable] = None,
                 iters: int = 400):
        self.name = name
        self.db = db or OpponentDB()
        self.rng = random.Random(seed)
        self.seed = seed
        self.bankroll = bankroll
        self.engine = QuantEngine(self.db, bankroll, rng=random.Random(seed + 7), iters=iters,
                                  mix_temperature_bb=mix_temperature_bb)
        self.exploit = exploit
        self.reads_provider = reads_provider
        self.last_report: Optional[QuantReport] = None
        self.decisions = 0

    def new_hand(self, hand_index: int) -> None:
        self.rng = random.Random(stable_hash(self.seed, hand_index, "q"))
        self.engine.rng = random.Random(stable_hash(self.seed, hand_index, "e"))

    def observe(self, history: HandHistory, my_seat: int) -> None:
        self.db.update(history)
        if self.bankroll is not None:
            self.bankroll.record_hand(history.net.get(my_seat, 0))

    # ---------------------------------------------------------------- acting
    def act(self, view: GameView) -> Decision:
        self.decisions += 1
        eff_bb = view.effective_stack() / view.bb
        if view.street == "preflop" and eff_bb >= 25:
            d = self.preflop(view)
            if d is not None:
                return d
        reads = self.reads_provider(view) if self.reads_provider else None
        rep = self.engine.analyze(view, reads=reads)
        self.last_report = rep
        choice = self.engine.choose(rep.options)
        d = choice.decision
        return Decision(d.kind, d.amount, source="quant", reason=choice.label)

    # ---------------------------------------------------------------- preflop
    def _profile(self, seat: int, view: GameView):
        return self.db.get(view.players[seat].name)

    def preflop(self, view: GameView) -> Optional[Decision]:
        spot = PreflopSpot(view)
        combo = make_combo(*view.hole)
        hc = hand_class(combo)
        pct = class_percentile()[hc]
        pos = view.hero.position
        heads_up = len(view.players) == 2
        sizes = self.engine._preflop_sizes(view)
        la = view.legal

        def raise_to(i=0):
            return Decision("raise", sizes[min(i, len(sizes) - 1)], source="chart")

        def mix(p: float) -> bool:
            return self.rng.random() < p

        if heads_up:
            return self._preflop_hu(view, spot, combo, pct, raise_to, mix)

        if spot.kind == "unopened":
            p_raise = rfi_range(pos).weight(combo)
            if self.exploit and pos in ("CO", "BTN", "SB"):
                # widen steals against blinds that over-fold; tighten against sticky blinds
                blinds = [p for p in view.players if p.position in ("SB", "BB") and p.seat != view.hero_seat]
                if blinds:
                    fts = sum(self._profile(p.seat, view).stat("fold_to_steal") for p in blinds) / len(blinds)
                    base_frac = rfi_range(pos).fraction_of_all()
                    extra = max(-0.15, min(0.25, (fts - 0.62) * 1.2))
                    if extra > 0 and p_raise < 1 and pct < base_frac + extra:
                        p_raise = max(p_raise, 0.85)
                    if extra < 0 and pct > base_frac + extra:
                        p_raise *= 0.3
            if p_raise > 0 and mix(p_raise):
                return raise_to(0)
            return Decision("check" if la.can_check else "fold", source="chart")

        if spot.kind == "limped":
            if pos == "BB":
                if chart_range(BB_RAISE_VS_LIMP).weight(combo) > 0:
                    return raise_to(0)
                return Decision("check", source="chart")
            fishy = sum(1 for a in spot.limpers
                        if self._profile(a.seat, view).stat("vpip") > 0.33) if self.exploit else 0
            iso = chart_range(ISO_RANGE).weight(combo)
            if iso > 0 and (fishy or pct < 0.2):
                return raise_to(0)
            if chart_range(OVERLIMP_RANGE).weight(combo) > 0 and pos in ("CO", "BTN", "SB"):
                return Decision("call", source="chart")
            return Decision("check" if la.can_check else "fold", source="chart")

        if spot.kind in ("vs_open", "squeeze"):
            opener = spot.raises[0].seat
            opos = view.players[opener].position
            tb, call = vs_open_ranges(pos, opos)
            p3 = tb.weight(combo)
            pc = call.weight(combo)
            if self.exploit:
                prof = self._profile(opener, view)
                f3 = prof.stat("fold_to_3bet")
                pfr = prof.stat("pfr")
                # 3-bet bluffs vs over-folders; value-only vs stations
                if f3 > 0.62 and pc > 0 and pct > 0.08 and hc.endswith("s"):
                    p3 = max(p3, 0.6)
                if f3 < 0.4 and p3 > 0 and pct > 0.06:
                    pc, p3 = max(pc, p3), 0.0
                if pfr > 0.3 and pct < 0.09:        # loose opener: widen value 3-bets
                    p3 = max(p3, 0.9)
                if pfr < 0.11:                       # nit opener: tighten continuing range
                    if pct > 0.07:
                        p3 = 0.0
                    if pct > 0.10:
                        pc *= 0.4
            if spot.kind == "squeeze":
                p3 = p3 if pct < 0.05 else p3 * 0.3
                pc = pc * 0.6
            if p3 > 0 and mix(p3):
                return raise_to(0)
            if pc > 0 and mix(min(1.0, pc / max(1e-9, 1 - p3))):
                return Decision("call", source="chart")
            return Decision("check" if la.can_check else "fold", source="chart")

        if spot.kind == "vs_3bet":
            ip = self.engine._in_position(view)
            table = VS_3BET_6MAX["IP" if ip else "OOP"]
            p4 = chart_range(table["4bet"]).weight(combo)
            pc = chart_range(table["call"]).weight(combo)
            if self.exploit:
                prof = self._profile(spot.last_raiser, view)
                tbr = prof.stat("threebet")
                if tbr < 0.045:                    # tight 3-bettor: their range is QQ+/AK
                    p4 = 1.0 if pct < 0.012 else 0.0
                    pc = pc if pct < 0.05 else pc * 0.2
                elif tbr > 0.12:                   # light 3-bettor: 4-bet value wider, defend more
                    p4 = max(p4, 1.0 if pct < 0.035 else 0.0)
                    pc = max(pc, 1.0 if pct < 0.16 else 0.0)
            if p4 > 0 and mix(p4):
                return raise_to(0)
            if pc > 0 and mix(min(1.0, pc / max(1e-9, 1 - p4))):
                return Decision("call", source="chart")
            return Decision("fold", source="chart")

        if spot.kind == "vs_4bet":
            p5 = chart_range(VS_4BET["5bet"]).weight(combo)
            pc = chart_range(VS_4BET["call"]).weight(combo)
            if p5 > 0 and mix(p5):
                return Decision("raise", la.max_raise_to, source="chart")
            if pc > 0 and mix(pc):
                return Decision("call", source="chart")
            return Decision("fold", source="chart")

        if spot.kind == "vs_5bet":
            if chart_range(VS_5BET["call"]).weight(combo) > 0:
                return Decision("call", source="chart")
            return Decision("fold", source="chart")

        # cold 3-bet / cold 4-bet: premium only
        if spot.kind in ("cold_3bet", "cold_4bet"):
            if pct < 0.012:
                return raise_to(0) if spot.kind == "cold_3bet" else Decision("raise", la.max_raise_to)
            if pct < 0.03 and spot.kind == "cold_3bet":
                return Decision("call", source="chart")
            return Decision("fold", source="chart")
        return None

    def _preflop_hu(self, view, spot, combo, pct, raise_to, mix) -> Optional[Decision]:
        la = view.legal
        pos = view.hero.position   # "BTN" (= SB) or "BB"
        if spot.kind == "unopened":           # button first to act
            if hu_range("sb_open").weight(combo) > 0 and mix(hu_range("sb_open").weight(combo)):
                return raise_to(0)
            return Decision("fold" if la.can_fold else "check", source="chart")
        if spot.kind == "limped":              # BB vs button limp
            if hu_range("bb_raise_vs_limp").weight(combo) > 0:
                return raise_to(0)
            return Decision("check", source="chart")
        if spot.kind == "vs_open":             # BB vs open
            opener = spot.raises[0].seat
            prof = self._profile(opener, view)
            p3 = max(hu_range("bb_3bet").weight(combo), hu_range("bb_3bet_bluff").weight(combo) * 0.5)
            pc = hu_range("bb_call").weight(combo)
            if self.exploit:
                if prof.stat("fold_to_3bet") > 0.62 and pct < 0.75:
                    p3 = max(p3, 0.5 if pct > 0.3 else p3)
                if prof.stat("pfr") > 0.9 and pct > 0.72:
                    pc = max(pc, 0.6)
            to = view.current_bet_level()
            if la.call_amount > 0 and to > 4 * view.bb:  # big open: defend tighter
                pc *= 0.6
            if p3 > 0 and mix(p3):
                return raise_to(0)
            if pc > 0 and mix(min(1.0, pc / max(1e-9, 1 - p3))):
                return Decision("call", source="chart")
            return Decision("fold", source="chart")
        if spot.kind == "vs_3bet":             # button facing BB 3-bet
            p4 = max(hu_range("sb_4bet").weight(combo), hu_range("sb_4bet_bluff").weight(combo))
            pc = hu_range("sb_call_3bet").weight(combo)
            if p4 > 0 and mix(p4):
                return raise_to(0)
            if pc > 0 and mix(min(1.0, pc / max(1e-9, 1 - p4))):
                return Decision("call", source="chart")
            return Decision("fold", source="chart")
        if spot.kind == "vs_4bet":
            if hu_range("bb_5bet").weight(combo) > 0:
                return Decision("raise", la.max_raise_to, source="chart")
            if hu_range("bb_call_4bet").weight(combo) > 0:
                return Decision("call", source="chart")
            return Decision("fold", source="chart")
        if spot.kind == "vs_5bet":
            if pct < 0.03:
                return Decision("call", source="chart")
            return Decision("fold", source="chart")
        return None
