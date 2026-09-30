"""Villain model: Bayesian range narrowing + response prediction.

Ranges are numpy vectors over the 1326 combos (index = ALL_COMBOS order).
Each observed villain action multiplies every combo's weight by the
probability the villain would take that action holding that combo.  The
action probabilities come from a parametric strategy whose free thresholds
are *calibrated* so that the villain's overall frequencies (bet %, fold %,
raise %) match his HUD statistics (shrunk to population priors).  This is
how "study the player" turns into math: a calling station's model folds
little, a nit's range after a raise is narrow, a sizing-tell player's big
bets are value-heavy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

import numpy as np

from . import population
from .cards import ALL_COMBOS, hand_class
from .opponents import PRIORS, PlayerProfile
from .view import GameView

N = len(ALL_COMBOS)
IDX = {c: i for i, c in enumerate(ALL_COMBOS)}
STREET_BETA = {"flop": 1.9, "turn": 1.4}   # bluff share multiplier vs river (semi-bluffs early)
# Stack-off decisions: calling off most of the remaining stack takes a genuinely strong hand (effective
# strength >= this, shifted by the player's folding tendency).  Without it the size effect saturates and
# huge overbet shoves look called by far too wide a range.  None disables.
COMMIT_STRENGTH = 0.60
# Response curves fitted to real hands (pokerbrain/population.py) replace the MDF-style size formula
# when available.  POP_THRESHOLDS: fold thresholds solved on the villain's current range ("current") or on
# his preflop reference range ("ref").
USE_POPULATION = True
POP_THRESHOLDS = "current"
# Tempering (robustness to model error): every per-hand action probability is blended with the range's
# average probability for that action, so no hand a real player might hold is ever ruled out.  Real
# hands with known hole cards showed the untempered model is over-confident.  Fitted on real data.
# The values live in the population file (fitted with it on real hands); without a population file the
# engine runs untempered, which suits deterministic simulated opponents.
_TEMPER_FIT = population.data().get("temper") or {}
TEMPER = float(_TEMPER_FIT.get("postflop", 0.0))      # postflop bet / fold / call / raise probabilities
PF_TEMPER = float(_TEMPER_FIT.get("preflop", 0.0))    # preflop range updates


def _temper(v: np.ndarray, w: np.ndarray, lam: float) -> np.ndarray:
    if lam <= 0:
        return v
    W = float(w.sum())
    if W <= 0:
        return v
    return (1 - lam) * v + lam * float((w * v).sum() / W)
POS_OPEN_MULT = {"UTG": 0.85, "UTG1": 0.85, "UTG2": 0.9, "MP": 0.95, "HJ": 1.05, "CO": 1.4, "BTN": 2.2,
                 "SB": 1.9, "BB": 1.0}


def _card_masks() -> dict:
    masks = {}
    for i, (a, b) in enumerate(ALL_COMBOS):
        for c in (a, b):
            masks.setdefault(c, []).append(i)
    return {c: np.array(v) for c, v in masks.items()}


CARD_MASK = _card_masks()


def dead_mask(cards) -> np.ndarray:
    m = np.ones(N)
    for c in cards:
        m[CARD_MASK[c]] = 0.0
    return m


@lru_cache(maxsize=1)
def preflop_pct() -> np.ndarray:
    from .preflop import class_percentile
    cp = class_percentile()
    return np.array([cp[hand_class(c)] for c in ALL_COMBOS])


@lru_cache(maxsize=256)
def strength_vec(board: tuple) -> np.ndarray:
    """Effective hand strength of every combo on this board (0 where blocked)."""
    from .texture import _board_ehs
    table = _board_ehs(board)
    v = np.zeros(N)
    for c, s in table.items():
        v[IDX[c]] = s
    return v


@lru_cache(maxsize=256)
def raw_strength_vec(board: tuple) -> np.ndarray:
    from .texture import hand_strength, _board_table
    vals, _ = _board_table(board)
    v = np.zeros(N)
    for c in vals:
        v[IDX[c]] = hand_strength(c, board)
    return v


def sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def solve_top(values: np.ndarray, weights: np.ndarray, target: float, tau: float,
              cap: float = 1.0) -> np.ndarray:
    """p = cap * sig((values - t)/tau) with sum(w*p) == target (clipped to [0, cap*sum w])."""
    total = float(weights.sum())
    if total <= 0 or target <= 0:
        return np.zeros_like(values)
    if target >= cap * total:
        return np.full_like(values, cap)
    lo, hi = -3.0, 4.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        mass = cap * float((weights * sig((values - mid) / tau)).sum())
        if mass > target:
            lo = mid
        else:
            hi = mid
    return cap * sig((values - 0.5 * (lo + hi)) / tau)


@dataclass
class VillainParams:
    name: str = "villain"
    vpip: float = 0.27
    pfr: float = 0.18
    limp: float = 0.10
    threebet: float = 0.07
    fold_to_3bet: float = 0.55
    cbet: float = 0.60
    bet_checked_to: float = 0.40
    donk: float = 0.10
    barrel: float = 0.50
    fold_vs_bet: dict = field(default_factory=lambda: {"flop": 0.42, "turn": 0.45, "river": 0.48})
    fold_to_cbet: float = 0.45
    raise_vs_bet: float = 0.10
    check_raise: float = 0.08
    fold_vs_raise: float = 0.50
    bluff_share: float = 0.30          # fraction of turn/river bets that are bluffs
    bigbet_bluff: float = 0.25
    smallbet_bluff: float = 0.35
    size_sensitivity: float = 1.0      # how much bigger bets increase folds (stations ~0.4)
    avg_bet_size: float = 0.6
    afq: float = 0.40
    tilt: float = 0.0
    archetype: str = "unknown"

    @classmethod
    def from_profile(cls, p: PlayerProfile, reads: Optional[dict] = None, seats: Optional[int] = None) -> "VillainParams":
        tells = p.sizing_tells()
        arch, _ = p.archetype()
        st = lambda k: p.stat(k, seats)          # priors depend on the table size (6-max vs full ring)
        vp = cls(
            name=p.name, vpip=st("vpip"), pfr=st("pfr"), limp=st("limp"),
            threebet=st("threebet"), fold_to_3bet=st("fold_to_3bet"), cbet=st("cbet"),
            bet_checked_to=st("bet_checked_to"), donk=st("donk"), barrel=st("barrel"),
            fold_vs_bet={"flop": st("fold_vs_bet_flop"), "turn": st("fold_vs_bet_turn"),
                         "river": st("fold_vs_bet_river")},
            fold_to_cbet=st("fold_to_cbet"), raise_vs_bet=st("raise_vs_bet"),
            fold_vs_raise=st("fold_vs_raise"),
            check_raise=st("check_raise"), bluff_share=st("river_bluff"),
            bigbet_bluff=tells["bigbet_bluff"], smallbet_bluff=tells["smallbet_bluff"],
            avg_bet_size=p.avg_bet_size(), afq=p.afq(), tilt=p.tilt_signals()["score"], archetype=arch)
        # loose-passive players barely react to sizing
        vp.size_sensitivity = float(np.clip(0.4 + 1.6 * (vp.fold_vs_bet["river"] - 0.2), 0.3, 1.3))
        if reads:
            vp.apply_reads(reads)
        vp.apply_tilt()
        return vp

    def apply_reads(self, reads: dict) -> None:
        """Blend in 'System 1' reads (e.g. Jev probabilities) with the stat model."""
        w = float(reads.get("weight", 0.5))
        if "bluff_share" in reads:
            b = float(reads["bluff_share"])
            self.bluff_share = (1 - w) * self.bluff_share + w * b
            self.bigbet_bluff = (1 - w) * self.bigbet_bluff + w * b
            self.smallbet_bluff = (1 - w) * self.smallbet_bluff + w * b
        if "fold_prob" in reads:
            fp = float(np.clip(reads["fold_prob"], 0.02, 0.95))
            self.fold_vs_bet = {k: (1 - w) * v + w * fp for k, v in self.fold_vs_bet.items()}
            self.fold_to_cbet = (1 - w) * self.fold_to_cbet + w * fp
        if "fold_scale" in reads:
            s = float(reads["fold_scale"])
            self.fold_vs_bet = {k: float(np.clip(v * s, 0.02, 0.95)) for k, v in self.fold_vs_bet.items()}
            self.fold_to_cbet = float(np.clip(self.fold_to_cbet * s, 0.02, 0.95))
        if "tilt" in reads:
            self.tilt = max(self.tilt, float(reads["tilt"]))

    def apply_tilt(self) -> None:
        # Real players (286k hands): behaviour barely changes in the 12 hands after a 30bb+ loss (VPIP +0.5
        # points, fold-to-bet unchanged).  The score is driven by a MEASURED change in this player's own
        # recent play; its effect on the model is kept small.
        t = self.tilt
        if t <= 0.05:
            return
        self.vpip = min(0.9, self.vpip * (1 + 0.3 * t))
        self.pfr = min(0.8, self.pfr * (1 + 0.3 * t))
        self.bluff_share = min(0.6, self.bluff_share + 0.08 * t)
        self.bigbet_bluff = min(0.6, self.bigbet_bluff + 0.08 * t)
        self.smallbet_bluff = min(0.6, self.smallbet_bluff + 0.06 * t)
        self.fold_vs_bet = {k: v * (1 - 0.15 * t) for k, v in self.fold_vs_bet.items()}
        self.bet_checked_to = min(0.85, self.bet_checked_to * (1 + 0.2 * t))
        self.raise_vs_bet = min(0.4, self.raise_vs_bet * (1 + 0.3 * t))


class VillainModel:
    TAU = 0.035

    def __init__(self, params: VillainParams):
        self.p = params

    # ------------------------------------------------------------- preflop
    def preflop_likelihood(self, situation: str, action: str, pos: str) -> np.ndarray:
        p, pct = self.p, preflop_pct()
        mult = POS_OPEN_MULT.get(pos, 1.0)
        open_thr = float(np.clip(p.pfr * mult, 0.02, 0.95))
        vpip_thr = float(np.clip(max(p.vpip * mult, open_thr + 0.02), 0.03, 0.98))
        t = 0.025
        if situation == "unopened":
            raise_p = sig((open_thr - pct) / t)
            play = sig((vpip_thr - pct) / 0.03)
            if action == "raise":
                return raise_p
            if action == "call":  # limp: middle band + occasional trap
                return np.maximum(play - raise_p, 0.0) + 0.08 * raise_p * min(1.0, p.limp * 4)
            if action == "check":  # BB option in a limped pot
                return 1.0 - 0.9 * sig((open_thr * 0.8 - pct) / t)
            return 1.0 - play
        if situation == "limped":
            iso_thr = open_thr * 0.8
            raise_p = sig((iso_thr - pct) / t)
            play = sig((vpip_thr - pct) / 0.03)
            if action == "raise":
                return raise_p
            if action == "check":
                return 1.0 - 0.9 * raise_p
            if action == "call":
                return np.maximum(play - raise_p, 0.0) + 0.05
            return 1.0 - play
        if situation == "vs_raise":
            bb = pos == "BB"
            # real pool facing a single open: BB continues ~24%, SB ~17%, others ~13% (24% VPIP pool)
            cont = float(np.clip(p.vpip * 0.6 + 0.10 if bb else p.vpip * (0.7 if pos == "SB" else 0.55), 0.02, 0.85))
            tb = float(np.clip(p.threebet, 0.01, 0.4))
            value_thr = tb * (1.0 - min(0.5, 0.15 + 2.0 * max(0.0, p.threebet - 0.06)))
            bluff_mass_frac = tb - value_thr
            value = sig((value_thr - pct) / 0.012)
            bluff_band = sig((cont + 0.12 - pct) / 0.03) - sig((cont - pct) / 0.03)
            bluff_band = np.clip(bluff_band, 0, 1)
            bluff = bluff_band * min(1.0, bluff_mass_frac / 0.12)
            if action == "raise":
                return np.clip(value + bluff, 0, 1)
            if action == "call":
                return np.clip(sig((cont - pct) / 0.03) - value, 0, 1)
            return np.clip(1.0 - sig((cont - pct) / 0.03) - bluff, 0.0, 1.0) + 1e-3
        if situation == "vs_3bet":
            cont = float(np.clip(open_thr * (1.0 - p.fold_to_3bet), 0.01, 0.6))
            fb = float(np.clip(0.02 + 0.1 * max(0.0, p.threebet - 0.07), 0.01, 0.1))
            value = sig((fb - pct) / 0.008)
            if action == "raise":
                return np.clip(value + 0.05 * sig((open_thr - pct) / t) * (p.afq > 0.5), 0, 1)
            if action == "call":
                return np.clip(sig((cont - pct) / 0.02) - value, 0, 1)
            return np.clip(1.0 - sig((cont - pct) / 0.02), 0, 1) + 1e-3
        # vs 4-bet and beyond
        if action == "raise":
            return sig((0.018 - pct) / 0.005)
        if action == "call":
            return np.clip(sig((0.045 - pct) / 0.008) - sig((0.018 - pct) / 0.005), 0, 1)
        return np.clip(1.0 - sig((0.045 - pct) / 0.008), 0, 1) + 1e-3

    # ------------------------------------------------------------ postflop
    def bet_probs(self, w: np.ndarray, s: np.ndarray, street: str, situation: str = "stab",
                  size_frac: Optional[float] = None, board: Optional[tuple] = None,
                  w_ref: Optional[np.ndarray] = None) -> np.ndarray:
        """P(villain bets | combo) when not facing a bet.

        situation: "cbet" (flop, preflop aggressor), "barrel" (aggressor on the previous street),
        "donk" (out of position into the aggressor), "stab" (checked to / no aggressor).
        Value thresholds are *absolute*: calibrated so the villain's reference range (his preflop
        range on this board) bets at his observed frequency.  Bluffs scale with the value bets he
        actually has, at his bluff ratio.  So a range weakened by earlier checks rarely bets, and
        when it does, the bet means something.
        """
        p = self.p
        f = {"cbet": p.cbet, "barrel": p.barrel, "donk": p.donk}.get(situation, p.bet_checked_to)
        W = float(w.sum())
        if W <= 0:
            return np.zeros(N)
        ref = w if w_ref is None else w_ref
        Wref = float(ref.sum()) or W
        curve = population.bluff_curve(street, "bet") if USE_POPULATION else None
        if curve:
            # real pool: bluff share of bets of this size at showdown, scaled by how bluffy THIS player is
            pool = population.interp(curve, size_frac) if size_frac is not None else population.curve_mean(curve)
            beta = pool * population.bluff_mult(street) * (p.bluff_share / max(1e-3, PRIORS["river_bluff"][0]))
        else:
            beta = p.bluff_share * STREET_BETA.get(street, 1.0)
        if size_frac is not None:
            if size_frac >= 0.75:
                beta *= p.bigbet_bluff / PRIORS["bigbet_bluff"][0]
            else:
                beta *= p.smallbet_bluff / PRIORS["smallbet_bluff"][0]
        beta = float(np.clip(beta, 0.02, 0.85))
        cap = float(np.clip(0.72 + 0.35 * (p.afq - 0.35), 0.55, 0.92))
        if situation == "donk":
            cap *= 0.6
        # Value capacity: you can't value-bet air.  If the observed frequency exceeds what the
        # reference range can bet for value, the excess must be bluffs (maniacs, tilters).
        capacity = cap * float((ref * (s >= 0.5)).sum()) / Wref
        if f > 1e-6 and capacity < f * (1 - beta):
            beta = float(np.clip(1.0 - capacity / f, beta, 0.9))
        value = solve_top(s, ref, f * (1 - beta) * Wref, 0.05, cap)
        bluff_score = np.clip(0.45 - s, 0, None)
        if street != "river" and board is not None:
            bluff_score = bluff_score + 0.8 * np.clip(s - raw_strength_vec(board), 0, None) * (s < 0.7)
        vmass = float((w * value).sum())
        # bluffs proportional to value actually held (plus a small floor for pure stabs)
        bmass = vmass * beta / (1 - beta) + 0.15 * f * beta * W
        room = w * np.clip(1 - value, 0, 1)
        bs_mass = float((room * bluff_score).sum())
        if bs_mass <= 0 or bmass <= 0:
            return _temper(np.clip(value, 0, 1), w, TEMPER)
        kk = bmass / bs_mass
        return _temper(np.clip(value + (1 - value) * np.clip(kk * bluff_score, 0, 1), 0, 1), w, TEMPER)

    def response_probs(self, w: np.ndarray, s: np.ndarray, street: str, size_frac: float,
                       vs_cbet: bool = False, base_fold: Optional[float] = None,
                       w_ref: Optional[np.ndarray] = None, facing_raise: bool = False,
                       commit: float = 0.0, multiway: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(P fold, P call, P raise) per combo when villain faces a bet of `size_frac` x pot.

        facing_raise: villain already bet/raised this street and is now being raised.
        commit: share of villain's remaining stack that calling costs (1 = calling puts him all-in)."""
        p = self.p
        if base_fold is not None:
            base = base_fold
        elif facing_raise:
            base = p.fold_vs_raise
        else:
            base = p.fold_to_cbet if vs_cbet else p.fold_vs_bet.get(street, 0.45)
        fc = rc = None
        if USE_POPULATION and base_fold is None and street != "preflop":
            fc = population.curve("fold", street, facing_raise, multiway, vs_cbet)
            rc = population.curve("raise", street, facing_raise, multiway, vs_cbet)
        if fc:
            # real pool's fold rate at this size, shifted by how much more/less this player folds than the pool
            stat = "fold_vs_raise" if facing_raise else ("fold_to_cbet" if vs_cbet else f"fold_vs_bet_{street}")
            fold = population.sigmoid(population.logit(population.interp(fc, size_frac))
                                      + population.logit(base) - population.logit(PRIORS[stat][0])
                                      + population.commit_shift(commit))
            fold = float(np.clip(fold, 0.02, 0.95))
        else:
            mdf = lambda x: 1.0 / (1.0 + max(0.05, x))
            # size effect saturates: whoever calls a 2.5x-pot bet also calls a bigger shove
            ratio = mdf(min(size_frac, 2.5)) / mdf(0.6)
            fold = 1.0 - (1.0 - base) * ratio ** p.size_sensitivity
            fold = float(np.clip(fold, 0.02, 0.92))
        W = float(w.sum())
        if W <= 0:
            z = np.zeros(N)
            return z, z, z
        ref = w if (w_ref is None or (fc and POP_THRESHOLDS == "current")) else w_ref
        Wref = float(ref.sum()) or W
        # absolute thresholds from the reference range: a strong current range folds less
        pf = solve_top(-s, ref, fold * Wref, 0.05)
        # nobody folds a genuinely strong hand because of bet size alone
        strong_floor = 0.90 if street != "preflop" else 0.97
        pf = np.where(s >= strong_floor, np.minimum(pf, 0.05), pf)
        if COMMIT_STRENGTH is not None and not fc and commit > 0.35 and street != "preflop":   # bot mode only
            t = float(np.clip(COMMIT_STRENGTH + 0.4 * (base - 0.45), 0.45, 0.85))
            pf = np.maximum(pf, (s < t) * 0.9 * min(1.0, (commit - 0.35) / 0.4))
        if rc:
            rr = population.interp(rc, size_frac) * float(np.clip(p.raise_vs_bet / max(1e-3, PRIORS["raise_vs_bet"][0]),
                                                                  0.3, 3.0))
            rr = float(np.clip(rr, 0.0, 0.6))
        else:
            rr = float(np.clip(p.raise_vs_bet * (1.0 if size_frac <= 1.0 else 0.6), 0.0, 0.5))
        cap = float(np.clip(0.6 + 0.4 * (p.afq - 0.35), 0.35, 0.85))   # strong hands often just call
        pr = solve_top(s, ref * (1 - pf), rr * Wref, 0.04, cap) * (1 - pf)
        if facing_raise and not fc:
            # bluffs give up against a raise (river: nothing left to draw to)
            air = (s < (0.35 if street == "river" else 0.2)).astype(float)
            pf = np.maximum(pf, air * (0.92 if street == "river" else 0.7))
            pr = np.minimum(pr, 1 - pf)
        pc = np.clip(1.0 - pf - pr, 0.0, 1.0)
        return _temper(pf, w, TEMPER), _temper(pc, w, TEMPER), _temper(pr, w, TEMPER)


# ---------------------------------------------------------------------------
# Range estimation from the action history
# ---------------------------------------------------------------------------
def estimate_range(view: GameView, seat: int, model: VillainModel,
                   dead_cards: Optional[list] = None, with_ref: bool = False):
    """Posterior weights over villain combos given everything villain did this hand.

    with_ref=True also returns the preflop (reference) range used to calibrate thresholds."""
    dead = list(view.hole) + list(view.board) + list(dead_cards or [])
    w = dead_mask(dead)
    pos = view.players[seat].position
    # ---- preflop
    raises, limpers = 0, 0
    for a in view.actions:
        if a.street != "preflop":
            break
        if a.kind.startswith("post"):
            continue
        if a.seat == seat:
            if raises == 0:
                situation = "unopened" if limpers == 0 else "limped"
            elif raises == 1:
                situation = "vs_raise"
            elif raises == 2:
                situation = "vs_3bet"
            else:
                situation = "vs_4bet"
            kind = a.kind if a.kind in ("raise", "call", "check", "fold") else "call"
            w = w * _temper(model.preflop_likelihood(situation, kind, pos), w, PF_TEMPER)
        if a.kind == "raise":
            raises += 1
        elif a.kind == "call" and raises == 0:
            limpers += 1
    pre_raiser = None
    for a in view.actions:
        if a.street == "preflop" and a.kind == "raise":
            pre_raiser = a.seat
    prev_aggressor = pre_raiser
    w_pre = w.copy()
    # ---- postflop
    for street, nb in (("flop", 3), ("turn", 4), ("river", 5)):
        acts = [a for a in view.actions if a.street == street]
        if not acts or len(view.board) < nb:
            break
        board = tuple(view.board[:nb])
        s = strength_vec(board)
        ref = w_pre * dead_mask(board)
        street_aggr = None
        street_aggressors: set = set()
        level, street_in = 0, {}
        first_bettor = None
        someone_checked = False
        for a in acts:
            facing = level > street_in.get(a.seat, 0)
            if a.seat == seat and a.kind != "fold":
                if not facing:
                    sit = bet_situation(street, seat, prev_aggressor, someone_checked)
                    if a.kind in ("bet", "raise"):
                        size = a.to / max(1, a.pot_before)
                        pb = model.bet_probs(w, s, street, sit, size, board, ref)
                        w = w * pb
                    else:
                        pb = model.bet_probs(w, s, street, sit, None, board, ref)
                        w = w * (1 - pb)
                else:
                    call_amt = level - street_in.get(seat, 0)
                    size = call_amt / max(1, a.pot_before - call_amt)
                    vs_cbet = street == "flop" and first_bettor == pre_raiser and first_bettor is not None
                    pf, pc, pr = model.response_probs(w, s, street, size, vs_cbet, None, ref,
                                                      facing_raise=seat in street_aggressors)
                    w = w * (pr if a.kind == "raise" else pc)
            if a.kind in ("bet", "raise"):
                level = a.to
                if first_bettor is None:
                    first_bettor = a.seat
                street_aggr = a.seat
                street_aggressors.add(a.seat)
            elif a.kind == "check":
                someone_checked = True
            street_in[a.seat] = a.to
        prev_aggressor = street_aggr
        w = w * dead_mask(view.board[:nb])
    total = w.sum()
    if total <= 1e-9:
        # contradictory evidence: fall back to strong-ish hands
        w = dead_mask(dead) * sig((0.25 - preflop_pct()) / 0.05)
    if with_ref:
        return w, w_pre * dead_mask(dead)
    return w


def bet_situation(street: str, seat: int, prev_aggressor: Optional[int], someone_checked: bool) -> str:
    """Classify a not-facing-a-bet spot: cbet / barrel / donk / stab."""
    if prev_aggressor == seat:
        return "cbet" if street == "flop" else "barrel"
    if prev_aggressor is not None and not someone_checked:
        return "donk"
    return "stab"


def range_from_vec(w: np.ndarray, threshold: float = 1e-4):
    from .ranges import Range
    m = w.max() if w.size else 0
    if m <= 0:
        return Range()
    return Range({ALL_COMBOS[i]: float(w[i] / m) for i in np.nonzero(w > threshold * m)[0]})


def range_summary(w: np.ndarray, board) -> dict:
    """Composition of a weighted range by current hand class (for prompts)."""
    from .texture import hand_features
    tot = float(w.sum())
    if tot <= 0:
        return {}
    buckets: dict[str, float] = {}
    idx = np.nonzero(w > 1e-6 * w.max())[0]
    if len(board) < 3:
        from .ranges import Range
        r = range_from_vec(w)
        return {"top_classes": r.describe(max_classes=25), "combos": round(tot, 1)}
    for i in idx:
        f = hand_features(ALL_COMBOS[i], board)
        key = f.strength_class
        buckets[key] = buckets.get(key, 0.0) + float(w[i])
    return {k: round(v / tot, 3) for k, v in sorted(buckets.items(), key=lambda kv: -kv[1])}
