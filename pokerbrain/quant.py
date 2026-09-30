"""Quant engine: equity vs Bayesian villain ranges and EV of every candidate action.

For each decision:
  1. estimate each active villain's range (villain.estimate_range)
  2. hero equity per villain combo (exact on the river, eval7 Monte Carlo otherwise)
  3. build a menu of legal candidate actions (standard sizes, no "leave 0.2 pot behind")
  4. EV of each candidate with a one-street look-ahead:
       - villain response per combo (fold / call / raise) from the calibrated model
       - equity vs the continuing sub-range, realization factor for future streets
  5. risk adjustment from the bankroll (log-utility / fractional Kelly)
Everything is in chips; *_bb fields are for display.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

import eval7
import numpy as np

from .bankroll import BankrollManager
from .cards import ALL_CARDS, ALL_COMBOS, _E7, evaluate, hand_class
from .opponents import OpponentDB
from .preflop import class_percentile
from .texture import board_texture, hand_features, nut_rank, pretty
from .view import Decision, GameView
from .villain import (N, VillainModel, VillainParams, dead_mask, estimate_range, preflop_pct,
                      range_from_vec, range_summary, strength_vec)

REALIZATION = {("preflop", True): 0.92, ("preflop", False): 0.75, ("flop", True): 0.96, ("flop", False): 0.86,
               ("turn", True): 0.98, ("turn", False): 0.92, ("river", True): 1.0, ("river", False): 1.0}


@lru_cache(maxsize=256)
def river_values(board: tuple) -> np.ndarray:
    from .texture import _board_table
    vals, _ = _board_table(board)
    v = np.full(N, -1, dtype=np.int64)
    idx = {c: i for i, c in enumerate(ALL_COMBOS)}
    for c, val in vals.items():
        v[idx[c]] = val
    return v


def per_combo_equity(hole, board, w: np.ndarray, iters: int = 400) -> np.ndarray:
    """Hero's equity against each villain combo (0 where the combo has no weight)."""
    e = np.zeros(N)
    idx = np.nonzero(w > 0)[0]
    if len(idx) == 0:
        return e
    if len(board) == 5:
        hv = evaluate(list(hole) + list(board))
        vv = river_values(tuple(board))[idx]
        e[idx] = (hv > vv) * 1.0 + (hv == vv) * 0.5
        return e
    villain = [((_E7[ALL_COMBOS[i][0]], _E7[ALL_COMBOS[i][1]]), 1.0) for i in idx]
    res = eval7.py_all_hands_vs_range(villain, [((_E7[hole[0]], _E7[hole[1]]), 1.0)],
                                      [_E7[c] for c in board], iters)
    lookup = {(str(a), str(b)): v for (a, b), v in res.items()}
    for i in idx:
        a, b = ALL_COMBOS[i]
        v = lookup.get((a, b), lookup.get((b, a)))
        e[i] = 1.0 - v if v is not None else 0.5
    return e


def multiway_equity(hole, board, ws: list, iters: int = 1500, rng: Optional[random.Random] = None) -> float:
    rng = rng or random.Random(0)
    nprng = np.random.default_rng(rng.randrange(1 << 30))
    probs = []
    for w in ws:
        t = w.sum()
        probs.append(w / t if t > 0 else dead_mask(list(hole) + list(board)) / dead_mask(list(hole) + list(board)).sum())
    draws = [nprng.choice(N, size=iters * 4, p=p) for p in probs]
    ptr = [0] * len(ws)
    base_dead = set(hole) | set(board)
    need = 5 - len(board)
    total, n_ok = 0.0, 0
    deck_all = [c for c in ALL_CARDS if c not in base_dead]
    for _ in range(iters):
        used = set(base_dead)
        hands = []
        ok = True
        for k in range(len(ws)):
            got = None
            while ptr[k] < len(draws[k]):
                c = ALL_COMBOS[draws[k][ptr[k]]]
                ptr[k] += 1
                if c[0] not in used and c[1] not in used:
                    got = c
                    break
            if got is None:
                ok = False
                break
            used.update(got)
            hands.append(got)
        if not ok:
            break
        rest = [c for c in deck_all if c not in used]
        fill = rng.sample(rest, need)
        full = list(board) + fill
        hv = evaluate(list(hole) + full)
        vs = [evaluate(list(h) + full) for h in hands]
        best = max(vs + [hv])
        if hv == best:
            total += 1.0 / (1 + sum(1 for v in vs if v == best))
        n_ok += 1
    return total / n_ok if n_ok else 0.0


@dataclass
class ActionOption:
    id: str
    label: str
    decision: Decision
    ev: float = 0.0
    ev_bb: float = 0.0
    risk_adj_bb: float = 0.0
    variance: float = 0.0
    fold_prob: Optional[float] = None
    eq_called: Optional[float] = None
    raise_prob: Optional[float] = None
    breakeven_fold: Optional[float] = None
    detail: str = ""

    def brief(self) -> dict:
        d = {"id": self.id, "action": self.label, "ev_bb": round(self.ev_bb, 2),
             "risk_adj_ev_bb": round(self.risk_adj_bb, 2)}
        if self.fold_prob is not None:
            d["villain_fold_prob"] = round(self.fold_prob, 2)
        if self.breakeven_fold is not None:
            d["breakeven_fold_prob"] = round(self.breakeven_fold, 2)
        if self.eq_called is not None:
            d["equity_when_called"] = round(self.eq_called, 3)
        if self.raise_prob is not None and self.raise_prob > 0.005:
            d["villain_raise_prob"] = round(self.raise_prob, 2)
        return d


@dataclass
class VillainInfo:
    seat: int
    name: str
    position: str
    stack_bb: float
    params: VillainParams
    w: np.ndarray
    w_ref: Optional[np.ndarray] = None
    equity_vs: float = 0.0
    composition: dict = field(default_factory=dict)
    range_text: str = ""
    combos: float = 0.0


@dataclass
class QuantReport:
    street: str
    pot_bb: float
    to_call_bb: float
    pot_odds: Optional[float]
    mdf: Optional[float]
    spr: float
    eff_stack_bb: float
    in_position: bool
    hero_hand: str
    hand_desc: str
    hand_class_pct: Optional[float]
    board: str
    texture: str
    equity: float
    nut_info: str
    geometric_bet: Optional[float]
    villains: list
    options: list
    best: ActionOption
    notes: list = field(default_factory=list)

    def option(self, oid: str) -> Optional[ActionOption]:
        return next((o for o in self.options if o.id == oid), None)


class QuantEngine:
    def __init__(self, db: OpponentDB, bankroll: Optional[BankrollManager] = None,
                 rng: Optional[random.Random] = None, iters: int = 400, mix_temperature_bb: float = 0.12):
        self.db = db
        self.bankroll = bankroll
        self.rng = rng or random.Random(1)
        self.iters = iters
        self.mix_temperature_bb = mix_temperature_bb

    # ------------------------------------------------------------------ public
    def analyze(self, view: GameView, reads: Optional[dict] = None, with_ev: bool = True) -> QuantReport:
        bb = view.bb
        hero = view.hero
        street = view.street
        villains = [p for p in view.opponents() if p.in_hand]
        infos: list[VillainInfo] = []
        for p in villains:
            prof = self.db.get(p.name)
            params = VillainParams.from_profile(prof, (reads or {}).get(p.name), seats=len(view.players))
            w, w_ref = estimate_range(view, p.seat, VillainModel(params), with_ref=True)
            vi = VillainInfo(seat=p.seat, name=p.name, position=p.position, stack_bb=(p.stack + p.bet) / bb,
                             params=params, w=w, w_ref=w_ref, combos=float(w.sum()))
            infos.append(vi)
        e_hu = None
        if len(infos) == 1:
            e_hu = per_combo_equity(view.hole, view.board, infos[0].w, self.iters)
            tot = infos[0].w.sum()
            eq = float((infos[0].w * e_hu).sum() / tot) if tot > 0 else 0.5
            infos[0].equity_vs = eq
        else:
            eq = multiway_equity(view.hole, view.board, [vi.w for vi in infos], rng=self.rng) if infos else 1.0
            for vi in infos:
                ee = per_combo_equity(view.hole, view.board, vi.w, 150)
                t = vi.w.sum()
                vi.equity_vs = float((vi.w * ee).sum() / t) if t > 0 else 0.5
        for vi in infos:
            vi.composition = range_summary(vi.w, view.board)
            vi.range_text = range_from_vec(vi.w).describe(max_classes=30)

        pot = view.pot
        to_call = view.legal.call_amount
        in_pos = self._in_position(view)
        eff = view.effective_stack()
        spr = (eff - hero.bet) / max(1, pot) if pot else 0.0
        feats = hand_features(view.hole, view.board)
        tex = board_texture(view.board)
        if len(view.board) >= 3:
            better, ties = nut_rank(view.hole, view.board)
            nut_info = "the NUTS" if better == 0 else f"{better} combos beat you, {ties} tie (vs all possible hands)"
        else:
            nut_info = "preflop"
        streets_left = {"preflop": 4, "flop": 3, "turn": 2, "river": 1}[street]
        geo = None
        if street != "preflop" and pot > 0:
            n = streets_left
            geo = ((1 + 2 * (eff - hero.bet) / pot) ** (1.0 / n) - 1) / 2
        hcls = hand_class(view.hole)
        report = QuantReport(
            street=street, pot_bb=pot / bb, to_call_bb=to_call / bb,
            pot_odds=(to_call / (pot + to_call)) if to_call else None,
            mdf=(1 - to_call / pot) if to_call and pot else None, spr=spr, eff_stack_bb=eff / bb,
            in_position=in_pos, hero_hand=pretty(view.hole) + f" ({hcls})",
            hand_desc=feats.describe() if view.board else f"{hcls} (top {class_percentile()[hcls] * 100:.0f}% of hands)",
            hand_class_pct=class_percentile()[hcls], board=pretty(view.board) if view.board else "(preflop)",
            texture=tex.label, equity=eq, nut_info=nut_info, geometric_bet=geo, villains=infos,
            options=[], best=None)  # type: ignore[arg-type]
        if with_ev:
            opts = self.candidates(view)
            for o in opts:
                self._evaluate(view, o, infos, e_hu, eq, in_pos)
            report.options = opts
            report.best = self.choose(opts, deterministic=True)
        return report

    # ---------------------------------------------------------------- menu
    def candidates(self, view: GameView) -> list[ActionOption]:
        la = view.legal
        bb, pot = view.bb, view.pot
        opts: list[ActionOption] = []
        if la.can_fold:
            opts.append(ActionOption("", "fold", Decision("fold")))
        if la.can_check:
            opts.append(ActionOption("", "check", Decision("check")))
        if la.call_amount > 0:
            allin = view.hero.stack <= la.call_amount
            opts.append(ActionOption("", f"call {la.call_amount / bb:g}bb" + (" (all-in)" if allin else ""),
                                     Decision("call")))
        if la.can_raise:
            level = view.current_bet_level()
            tos: list[int] = []
            if view.street == "preflop":
                tos = self._preflop_sizes(view)
            elif level == 0:
                tos = [int(round(f * pot)) for f in (0.33, 0.5, 0.75, 1.0, 1.5)]
            else:
                after_call = pot + la.call_amount
                tos = [int(round(level + f * after_call)) for f in (0.5, 0.8, 1.2)]
            clean: list[int] = []
            for x in tos:
                x = la.clamp_raise(x)
                if x >= 0.8 * la.max_raise_to:   # don't leave a sliver behind: just shove
                    x = la.max_raise_to
                if all(abs(x - y) > 0.06 * max(x, y) for y in clean):
                    clean.append(x)
            if la.max_raise_to not in clean:
                clean.append(la.max_raise_to)
            for x in sorted(clean):
                opts.append(ActionOption("", self._raise_label(view, x), Decision("raise", x)))
        for i, o in enumerate(opts):
            o.id = f"A{i + 1}"
        return opts

    def _preflop_sizes(self, view: GameView) -> list[int]:
        bb = view.bb
        pre = [a for a in view.actions if a.street == "preflop" and not a.kind.startswith("post")]
        raises = [a for a in pre if a.kind == "raise"]
        limpers = [a for a in pre if a.kind == "call" and not raises]
        pos = view.hero.position
        if not raises:
            base = (3.0 if pos == "SB" else 2.5) * bb if not limpers else (3.5 + len(limpers)) * bb
            return [int(base), int(base * 1.4)]
        last = raises[-1].to
        ip = self._in_position(view)
        if len(raises) == 1:
            mult = 3.0 if ip else 4.0
            return [int(last * mult), int(last * (mult + 1))]
        return [int(last * 2.3), int(last * 2.8)]

    def _raise_label(self, view: GameView, x: int) -> str:
        la, bb, pot = view.legal, view.bb, view.pot
        level = view.current_bet_level()
        if x == la.max_raise_to:
            return f"all-in {x / bb:g}bb"
        if level == 0:
            return f"bet {x / bb:g}bb ({x / max(1, pot):.0%} pot)"
        if view.street == "preflop" and level <= view.bb:
            return f"raise to {x / bb:g}bb"
        return f"raise to {x / bb:g}bb ({x / max(1, level):.1f}x)"

    # ---------------------------------------------------------------- EV
    def _in_position(self, view: GameView) -> bool:
        n = len(view.players)
        order = sorted((p for p in view.players if p.in_hand),
                       key=lambda p: (p.seat - view.button_seat - 1) % n)
        return bool(order) and order[-1].seat == view.hero_seat

    def _realization(self, street: str, in_pos: bool, eq: float) -> float:
        r = REALIZATION[(street, in_pos)]
        return r + (1 - r) * float(np.clip((eq - 0.5) / 0.4, 0, 1))

    def _preflop_fold_base(self, view: GameView, vp: VillainParams) -> float:
        raises = [a for a in view.actions if a.street == "preflop" and a.kind == "raise"]
        if not raises:
            return float(np.clip(1.0 - vp.vpip * 1.1, 0.3, 0.95))
        if len(raises) == 1:
            return vp.fold_to_3bet
        return float(np.clip(0.45 + 0.5 * (vp.fold_to_3bet - 0.55), 0.2, 0.85))

    def _evaluate(self, view: GameView, o: ActionOption, infos: list, e_hu, eq: float, in_pos: bool) -> None:
        bb = view.bb
        pot = view.pot
        hero = view.hero
        street = view.street
        k = o.decision.kind
        if k == "fold":
            o.ev, o.variance = 0.0, 0.0
        elif not infos:
            o.ev, o.variance = float(pot), 0.0
        elif len(infos) == 1:
            self._ev_heads_up(view, o, infos[0], e_hu, in_pos)
        else:
            self._ev_multiway(view, o, infos, eq, in_pos)
        o.ev_bb = o.ev / bb
        pen = self.bankroll.risk_penalty(o.variance, bb) if self.bankroll else 0.0
        o.risk_adj_bb = (o.ev - pen) / bb
        if k == "raise":
            add = o.decision.amount - hero.bet
            o.breakeven_fold = add / (pot + add)

    def _ev_heads_up(self, view: GameView, o: ActionOption, vi: VillainInfo, e: np.ndarray, in_pos: bool) -> None:
        pot, hero, street = view.pot, view.hero, view.street
        w = vi.w
        W = float(w.sum())
        if W <= 0:
            o.ev = float(pot)
            return
        vp = next(p for p in view.players if p.seat == vi.seat)
        vil_in, vil_max = vp.bet, vp.bet + vp.stack
        hero_in, hero_max = hero.bet, hero.bet + hero.stack
        model = VillainModel(vi.params)
        s = strength_vec(tuple(view.board)) if len(view.board) >= 3 else 1.0 - preflop_pct()
        eqw = lambda ww: float((ww * e).sum() / ww.sum()) if ww.sum() > 1e-12 else 0.0
        eq_all = eqw(w)
        R = self._realization(street, in_pos, eq_all)
        k = o.decision.kind
        if k == "call":
            c = view.legal.call_amount
            allin = hero.stack <= c or vil_max <= hero_in + c
            r = 1.0 if allin else R
            o.ev = r * eq_all * (pot + c) - c
            o.variance = eq_all * (1 - eq_all) * (pot + c) ** 2
            o.eq_called = eq_all
            return
        if k == "check":
            villain_to_act = self._villain_acts_after(view, vi.seat)
            if not villain_to_act or vil_max <= vil_in:
                o.ev = R * eq_all * pot
                o.variance = eq_all * (1 - eq_all) * pot ** 2
                return
            sit = self._villain_situation_after_check(view, vi.seat)
            pb = model.bet_probs(w, s, street if street != "preflop" else "flop", sit, None,
                                 tuple(view.board) if len(view.board) >= 3 else None, vi.w_ref)
            wb, wc = w * pb, w * (1 - pb)
            p_bet = float(wb.sum() / W)
            B = min(vil_max - vil_in, max(view.bb, vi.params.avg_bet_size * pot))
            eb = eqw(wb)
            call_ev = R * eb * (pot + 2 * B) - B
            ev_bet_branch = max(0.0, call_ev)
            ec = eqw(wc)
            o.ev = (1 - p_bet) * R * ec * pot + p_bet * ev_bet_branch
            o.variance = (1 - p_bet) * ec * (1 - ec) * pot ** 2 + p_bet * (eb * (1 - eb) * (pot + 2 * B) ** 2
                                                                           if call_ev > 0 else 0.0)
            o.fold_prob = None
            o.detail = f"villain bets ~{p_bet:.0%} when checked to"
            return
        # bet / raise to X
        X = o.decision.amount
        A = X - hero_in
        c = max(0, min(X, vil_max) - vil_in)
        if c <= 0:
            o.ev = eq_all * (pot + A) - A
            return
        x_frac = c / max(1, pot + A - c)
        if street == "preflop":
            pf, pc, pr = model.response_probs(w, s, "flop", x_frac,
                                              base_fold=self._preflop_fold_base(view, vi.params))
        else:
            vs_cbet = street == "flop" and self._last_aggressor(view, "preflop") == view.hero_seat and \
                view.current_bet_level() == 0
            villain_bet_here = any(a.seat == vi.seat and a.kind in ("bet", "raise")
                                   for a in view.street_actions())
            pf, pc, pr = model.response_probs(w, s, street, x_frac, vs_cbet, None,
                                              vi.w_ref if vi.w_ref is not None else None,
                                              facing_raise=villain_bet_here,
                                              commit=c / max(1, vil_max - vil_in))
        allin_now = X >= hero_max or min(X, vil_max) >= vil_max
        if allin_now:                     # an all-in can't be re-raised: would-be raises just call
            pc, pr = pc + pr, pr * 0.0
        Pf = float((w * pf).sum() / W)
        Pc = float((w * pc).sum() / W)
        Pr = float((w * pr).sum() / W)
        ec = eqw(w * pc)
        r1 = 1.0 if allin_now else self._realization(street, in_pos, ec)
        final_pot = pot + A + c
        ev_call = r1 * ec * final_pot - A
        # villain re-raise branch: model as shove (or 3x) and hero best-responds
        ev_raise = -A
        er = eqw(w * pr)
        if Pr > 1e-4 and not allin_now:
            X2 = min(vil_max, max(3 * X, X + final_pot))
            extra = min(X2, hero_max) - X
            pot2 = final_pot + (min(X2, vil_max) - X) + extra
            r2 = 1.0 if X2 >= vil_max or min(X2, hero_max) >= hero_max else self._realization(street, in_pos, er)
            ev_raise = max(-A, r2 * er * pot2 - A - extra)
        o.ev = Pf * pot + Pc * ev_call + Pr * ev_raise
        outcomes = [(Pf, pot), (Pc * ec, final_pot - A), (Pc * (1 - ec), -A)]
        if Pr > 0:
            outcomes.append((Pr, ev_raise))
        m = sum(p * v for p, v in outcomes)
        o.variance = max(0.0, sum(p * v * v for p, v in outcomes) - m * m)
        o.fold_prob, o.eq_called, o.raise_prob = Pf, ec, Pr

    def _ev_multiway(self, view: GameView, o: ActionOption, infos: list, eq: float, in_pos: bool) -> None:
        pot, hero, street = view.pot, view.hero, view.street
        R = self._realization(street, in_pos, eq) * (0.95 if len(infos) > 1 else 1.0)
        k = o.decision.kind
        if k == "call":
            c = view.legal.call_amount
            o.ev = R * eq * (pot + c) - c
            o.variance = eq * (1 - eq) * (pot + c) ** 2
            o.eq_called = eq
            return
        if k == "check":
            o.ev = R * eq * pot
            o.variance = eq * (1 - eq) * pot ** 2
            return
        X = o.decision.amount
        A = X - hero.bet
        folds, cont_ws, calls = [], [], []
        for vi in infos:
            vp = next(p for p in view.players if p.seat == vi.seat)
            c = max(0, min(X, vp.bet + vp.stack) - vp.bet)
            x_frac = c / max(1, pot + A - c)
            s = strength_vec(tuple(view.board)) if len(view.board) >= 3 else 1.0 - preflop_pct()
            model = VillainModel(vi.params)
            if street == "preflop":
                pf, pc, pr = model.response_probs(vi.w, s, "flop", x_frac,
                                                  base_fold=self._preflop_fold_base(view, vi.params))
            else:
                pf, pc, pr = model.response_probs(vi.w, s, street, x_frac, False, None, vi.w_ref, multiway=True)
            W = vi.w.sum()
            folds.append(float((vi.w * pf).sum() / W) if W > 0 else 1.0)
            cont_ws.append(vi.w * (1 - pf))
            calls.append(c)
        all_fold = float(np.prod(folds))
        eq_cont = multiway_equity(view.hole, view.board, cont_ws, iters=600, rng=self.rng) if all_fold < 0.999 else eq
        exp_callers = sum((1 - f) * c for f, c in zip(folds, calls)) / max(1e-6, 1 - all_fold)
        final_pot = pot + A + exp_callers
        ev_called = R * eq_cont * final_pot - A
        o.ev = all_fold * pot + (1 - all_fold) * ev_called
        o.variance = all_fold * pot ** 2 + (1 - all_fold) * (eq_cont * (final_pot - A) ** 2 + (1 - eq_cont) * A ** 2) - o.ev ** 2
        o.variance = max(0.0, o.variance)
        o.fold_prob, o.eq_called = all_fold, eq_cont

    def _villain_acts_after(self, view: GameView, seat: int) -> bool:
        """Does this villain still get to act on this street after hero checks?"""
        n = len(view.players)
        order = sorted((p for p in view.players if p.in_hand and not p.all_in),
                       key=lambda p: (p.seat - view.button_seat - 1) % n)
        seats = [p.seat for p in order]
        if seat not in seats or view.hero_seat not in seats:
            return False
        acted = {a.seat for a in view.street_actions()}
        if view.street == "preflop":
            return seat not in acted or seats.index(seat) > seats.index(view.hero_seat)
        return seats.index(seat) > seats.index(view.hero_seat) or seat not in acted

    def _villain_situation_after_check(self, view: GameView, seat: int) -> str:
        """Villain's betting situation if hero checks now (cbet / barrel / stab)."""
        order = ["preflop", "flop", "turn", "river"]
        prev = order[order.index(view.street) - 1] if view.street != "preflop" else None
        prev_aggr = self._last_aggressor(view, prev) if prev else None
        if prev_aggr == seat:
            return "cbet" if view.street == "flop" else "barrel"
        return "stab"

    def _last_aggressor(self, view: GameView, street: Optional[str] = None) -> Optional[int]:
        last = None
        for a in view.actions:
            if street and a.street != street:
                continue
            if a.kind in ("bet", "raise"):
                last = a.seat
        return last

    # ---------------------------------------------------------------- choice
    def choose(self, opts: list[ActionOption], deterministic: bool = False) -> ActionOption:
        best = max(opts, key=lambda o: o.risk_adj_bb)
        if deterministic or self.mix_temperature_bb <= 0:
            return best
        close = [o for o in opts if best.risk_adj_bb - o.risk_adj_bb <= 3 * self.mix_temperature_bb]
        if len(close) == 1:
            return best
        ws = [math.exp((o.risk_adj_bb - best.risk_adj_bb) / self.mix_temperature_bb) for o in close]
        x = self.rng.random() * sum(ws)
        for o, wgt in zip(close, ws):
            x -= wgt
            if x <= 0:
                return o
        return best
