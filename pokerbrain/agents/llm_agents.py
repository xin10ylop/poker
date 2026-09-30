"""LLM-driven agents: Opus as the decision-maker, Jev as the intuition module.

All of them keep the QuantAgent as backbone and fallback: preflop charts,
the EV engine, opponent tracking.  What differs is who gets the final say.

  JevReadsAgent   quant EV, but villain assumptions are blended with Jev's
                  calibrated reads (P(bluff), P(fold), tilt)
  JevDeciderAgent Jev picks from the engine's action menu (Choice question)
  OpusAgent       Opus reads the full dashboard (quant + dossier + Jev reads +
                  bankroll) and chooses a (mixed) strategy over the menu, in
                  the spots an escalation policy deems worth a model call
"""
from __future__ import annotations

import json
import math
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Callable, Optional

from ..bankroll import BankrollManager
from ..llm.jev import JevClient, JevError, decision_question, reads_questions, verify_question
from ..llm.prompts import VARIANTS
from ..llm.render import jev_state, render_dashboard
from ..opponents import OpponentDB
from ..quant import QuantReport
from ..view import Decision, GameView
from .quant_agent import QuantAgent


# ---------------------------------------------------------------------------
# Jev reads -> villain parameter adjustments
# ---------------------------------------------------------------------------
def jev_reads(jev: JevClient, view: GameView, rep: QuantReport, db: OpponentDB,
              bankroll_ctx: Optional[dict] = None) -> dict:
    """One Jev call; returns {villain_name: {...probabilities...}}."""
    if not rep.villains:
        return {}
    state = jev_state(view, rep, db, bankroll_ctx, include_quant=False)
    facing = view.legal.call_amount > 0
    can_bet = view.legal.can_raise
    questions = {}
    for k, vi in enumerate(rep.villains):
        for qid, q in reads_questions(vi.name, facing and k == 0, can_bet).items():
            questions[f"{qid}__{k}"] = q
    ans = jev.ask(state, questions, tag="reads")
    out: dict = {}
    for k, vi in enumerate(rep.villains):
        r: dict = {}
        a = ans.get(f"villain_bluffing__{k}")
        if a:
            r["villain_bluffing"] = float(a["noul"])
        a = ans.get(f"villain_folds_to_bet__{k}")
        if a:
            r["villain_folds_to_bet"] = float(a["noul"])
        a = ans.get(f"tilt__{k}")
        if a:
            r["tilt_score"] = float(a["score"])
            r["tilt_label"] = a.get("legend", {}).get(str(int(round(a["score"]))), "")
        a = ans.get(f"archetype__{k}")
        if a:
            r["archetype"] = {kk: float(v) for kk, v in a["probabilities"].items()}
        out[vi.name] = r
    return out


def reads_to_params(reads: dict, weight: float = 0.5) -> dict:
    """Map Jev probabilities onto VillainParams.apply_reads() inputs."""
    out = {}
    for name, r in reads.items():
        adj: dict = {"weight": weight}
        if "villain_bluffing" in r:
            adj["bluff_share"] = r["villain_bluffing"]
        if "villain_folds_to_bet" in r:
            adj["fold_prob"] = r["villain_folds_to_bet"]
        if "tilt_score" in r:
            adj["tilt"] = max(0.0, (r["tilt_score"] - 0.5) / 2.5)
        out[name] = adj
    return out


class JevReadsAgent(QuantAgent):
    def __init__(self, jev: JevClient, weight: float = 0.5, min_pot_bb: float = 4.0, **kw):
        kw.setdefault("name", "JevReads")
        super().__init__(**kw)
        self.jev = jev
        self.weight = weight
        self.min_pot_bb = min_pot_bb
        self.reads_provider = self._reads

    def _reads(self, view: GameView) -> Optional[dict]:
        if view.street == "preflop" or view.pot < self.min_pot_bb * view.bb:
            return None
        rep = self.engine.analyze(view, with_ev=False)
        try:
            r = jev_reads(self.jev, view, rep, self.db)
        except JevError:
            return None
        return reads_to_params(r, self.weight)


class JevDeciderAgent(QuantAgent):
    def __init__(self, jev: JevClient, min_pot_bb: float = 4.0, **kw):
        kw.setdefault("name", "JevDecider")
        super().__init__(**kw)
        self.jev = jev
        self.min_pot_bb = min_pot_bb

    def act(self, view: GameView) -> Decision:
        if view.street == "preflop" or view.pot < self.min_pot_bb * view.bb:
            return super().act(view)
        rep = self.engine.analyze(view)
        self.last_report = rep
        state = jev_state(view, rep, self.db)
        try:
            ans = self.jev.ask(state, decision_question([o.brief() for o in rep.options]), tag="decide")
            probs = {k: float(v) for k, v in ans["best_action"]["probabilities"].items()
                     if rep.option(k) and math.isfinite(float(v)) and float(v) > 0}
            if not probs:
                raise ValueError("no usable probabilities")
            # sample from Jev's calibrated distribution (natural mixed strategy)
            ids = list(probs)
            x = self.rng.random() * sum(probs.values())
            pick = ids[-1]
            for i in ids:
                x -= probs[i]
                if x <= 0:
                    pick = i
                    break
        except Exception:  # noqa: BLE001 - budget, network, shape: the engine decides
            return rep.best.decision
        o = rep.option(pick) or rep.best
        return Decision(o.decision.kind, o.decision.amount, source="jev", reason=o.label)


# ---------------------------------------------------------------------------
# Opus
# ---------------------------------------------------------------------------
TRICKY_QUESTION = {"tricky": {"type": "score",
                              "instructions": "How difficult is this decision for a strong professional?",
                              "criteria": ["Routine: the answer is obvious", "Some thought needed",
                                           "Close decision", "Very hard: depends heavily on reads"]}}


def jev_tricky(jev: JevClient, view: GameView, rep: QuantReport, db: OpponentDB) -> float:
    """Jev's 0-3 difficulty score (benchmark: best router of engine mistakes, AUC 0.66)."""
    ans = jev.ask(jev_state(view, rep, db), TRICKY_QUESTION, tag="router")
    return float(ans["tricky"]["score"])


@dataclass
class EscalationPolicy:
    """When is a model call worth it?  (latency + cost vs expected improvement)

    mode "postflop": every postflop decision (the benchmark's best: Opus + override gate);
    mode "jev": a Jev difficulty score gates Opus (System 1 decides when System 2 thinks) - about
    half the Opus calls; huge pots (>= always_pot_bb) always escalate;
    mode "key": big pots and close engine decisions.
    Every mode except "all" skips decisions whose pot is worth less than min_cost_ratio model calls."""
    mode: str = "key"                # "all" | "postflop" | "key" | "jev" | "never"
    tricky_threshold: float = 0.84   # Jev score (0-3); 0.84 = top ~40% of benchmark spots (captures 57% of EV loss)
    always_pot_bb: float = 40.0
    router_min_pot_bb: float = 6.0   # below this the engine decides alone (no Jev / Opus calls)
    min_pot_bb: float = 12.0         # always escalate pots this big (postflop)
    close_ev_bb: float = 1.0         # ...or when the top two engine options are this close
    close_frac_pot: float = 0.06
    preflop: bool = False            # escalate big preflop decisions (facing 3-bet+ / all-in)
    edge_fraction: float = 0.02      # expected gain from a model decision as a share of the pot (benchmark ~2%)
    min_cost_ratio: float = 3.0      # (deprecated: superseded by edge_fraction)

    def _worth(self, view: GameView, call_cost_usd: float, chip_value: Optional[float]) -> bool:
        """A model call must be expected to pay for itself: pot value x edge >= cost of the call."""
        return chip_value is None or view.pot * chip_value * self.edge_fraction >= call_cost_usd

    def should(self, view: GameView, rep: Optional[QuantReport], call_cost_usd: float = 0.05,
               chip_value: Optional[float] = None, tricky: Optional[float] = None) -> bool:
        if self.mode == "never":
            return False
        if self.mode == "all":
            return True
        if self.mode == "postflop":
            if view.street == "preflop" and not self.preflop:
                return False
            return self._worth(view, call_cost_usd, chip_value)
        if self.mode == "jev":
            if view.street == "preflop" and not self.preflop:
                return False
            if chip_value is not None and not self._worth(view, call_cost_usd, chip_value):
                return False
            if view.pot / view.bb >= self.always_pot_bb:
                return True
            if view.pot / view.bb < self.router_min_pot_bb:
                return False
            return tricky is not None and tricky >= self.tricky_threshold
        pot_bb = view.pot / view.bb
        if chip_value is not None and not self._worth(view, call_cost_usd, chip_value):
            return False                      # model call costs more than the decision can gain
        if view.street == "preflop":
            if not self.preflop:
                return False
            raises = sum(1 for a in view.actions if a.street == "preflop" and a.kind == "raise")
            return raises >= 2 and view.legal.call_amount > 0
        if pot_bb >= self.min_pot_bb:
            return True
        if rep is not None and len(rep.options) >= 2:
            evs = sorted((o.risk_adj_bb for o in rep.options), reverse=True)
            if evs[0] - evs[1] <= max(self.close_ev_bb, self.close_frac_pot * pot_bb):
                return True
        return False


Decider = Callable[[str, str, dict], dict]


class OpusAgent(QuantAgent):
    def __init__(self, decider: Decider, variant: str = "v3_elite", jev: Optional[JevClient] = None,
                 escalation: Optional[EscalationPolicy] = None, verifier: bool = False,
                 verifier_threshold: float = 0.85, jev_weight: float = 0.0, reads_in_dashboard: bool = False,
                 mix: bool = False, override_gate: Optional[float] = None, session_hand_counter: bool = True,
                 log: Optional[list] = None, stakes=None, require_stakes: Optional[bool] = None,
                 deadline_s: Optional[float] = None, call_cost_usd: Optional[float] = None, **kw):
        kw.setdefault("name", f"Opus[{variant}]")
        super().__init__(**kw)
        if variant not in VARIANTS:
            raise ValueError(f"unknown prompt variant {variant!r}; valid: {', '.join(VARIANTS)}")
        from ..config import ULTIMATE
        self.stakes = stakes if stakes is not None else (self.bankroll.stakes if self.bankroll is not None else None)
        self.require_stakes = ULTIMATE.get("require_stakes", True) if require_stakes is None else require_stakes
        self.deadline_s = float(deadline_s if deadline_s is not None else ULTIMATE.get("deadline_s", 25.0))
        self.call_cost_usd = float(call_cost_usd if call_cost_usd is not None else ULTIMATE.get("call_cost_usd", 0.05))
        self._pool: Optional[ThreadPoolExecutor] = None
        self.deadline_misses = 0
        self.decider = decider
        self.variant = variant
        self.jev = jev
        self.escalation = escalation or EscalationPolicy()
        self.verifier = verifier
        self.verifier_threshold = verifier_threshold
        self.jev_weight = jev_weight
        self.reads_in_dashboard = reads_in_dashboard
        self.mix = mix                  # benchmark: committing to the top action beats sampling Opus's mix
        # Overrides are trusted only when decisive: if Opus still puts more than this share of its own mix on
        # the engine's pick, the engine's pick is played (benchmark: hedged overrides lose, decisive ones win).
        self.override_gate = override_gate
        self.gated = 0
        self.hand_number = 0
        self.log = log if log is not None else []
        self.model_calls = 0
        self.fallbacks = 0
        if self.stakes is None and self.require_stakes and (escalation is None or escalation.mode != "all"):
            print("[pokerbrain] no stakes or bankroll given: Opus will not be consulted (the engine plays alone); "
                  "pass --stakes so Opus is used only where its fee is small next to the pot", file=sys.stderr)

    def new_hand(self, hand_index: int) -> None:
        super().new_hand(hand_index)
        self.hand_number += 1

    # ------------------------------------------------------------- helpers
    def _log(self, entry: dict) -> None:
        self.log.append(entry)
        if len(self.log) > 2000:
            del self.log[:-2000]

    def _chip_value(self, view: GameView) -> Optional[float]:
        if self.stakes is not None:
            return self.stakes.bb / view.bb
        if self.bankroll is not None:
            return self.bankroll.chip_value_for(view.bb)
        return None

    def _call_cost(self) -> float:
        client = getattr(self.decider, "client", None)
        fn = getattr(client, "mean_cost_usd", None)
        return float(fn(self.call_cost_usd)) if callable(fn) else self.call_cost_usd

    @staticmethod
    def _safe_decision(view: GameView) -> Decision:
        """Last resort when even the engine failed: check when free, call when cheap, else fold."""
        la = view.legal
        if la.can_check:
            return Decision("check", source="safe")
        if la.call_amount > 0 and la.call_amount <= 0.05 * max(1, view.hero.stack):
            return Decision("call", source="safe")
        return Decision("fold", source="safe").normalized(la)

    def _ask(self, view: GameView, system: str, user: str, meta: dict):
        """Run the decider with a hard wall-clock deadline (the table clock does not wait)."""
        if self.deadline_s <= 0:
            return self.decider(system, user, meta)
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=2)
        fut = self._pool.submit(self.decider, system, user, meta)
        try:
            return fut.result(timeout=self.deadline_s + 2.0)
        except FutureTimeout:
            self.deadline_misses += 1
            self._log({"hand": view.hand_id, "street": view.street, "error": f"deadline {self.deadline_s:.0f}s"})
            return None

    def act(self, view: GameView) -> Decision:
        """Always returns a legal action.  Failure order: Opus -> engine -> chart -> safe action."""
        eff_bb = view.effective_stack() / view.bb
        chart = None
        if view.street == "preflop" and eff_bb >= 25:
            chart = self.preflop(view)
            if chart is not None and not self.escalation.preflop:
                self.last_report = None
                return chart.normalized(view.legal)
        try:
            rep, reads = self._analyze(view)
            self.last_report = rep
            o = self.engine.choose(rep.options)
            fallback = Decision(o.decision.kind, o.decision.amount, source="quant", reason=o.label)
        except Exception as exc:  # noqa: BLE001
            self.fallbacks += 1
            self._log({"hand": view.hand_id, "street": view.street, "error": f"engine: {exc!r}"})
            return (chart if chart is not None else self._safe_decision(view)).normalized(view.legal)
        try:
            d = self._consult(view, rep, reads, chart, fallback)
        except Exception as exc:  # noqa: BLE001 - any model / formatting failure falls back to the engine
            self.fallbacks += 1
            self._log({"hand": view.hand_id, "street": view.street, "error": f"consult: {exc!r}"})
            d = fallback
        return d.normalized(view.legal)

    def _analyze(self, view: GameView):
        reads = None
        rep0 = self.engine.analyze(view, with_ev=False)
        want_reads = self.jev_weight > 0 or self.reads_in_dashboard
        if want_reads and self.jev is not None and rep0.villains and view.street != "preflop":
            try:
                reads = jev_reads(self.jev, view, rep0, self.db,
                                  self.bankroll.context() if self.bankroll else None)
            except Exception:  # noqa: BLE001
                reads = None
        rep = self.engine.analyze(view, reads=reads_to_params(reads, self.jev_weight)
                                  if (reads and self.jev_weight > 0) else None)
        return rep, reads

    def _consult(self, view: GameView, rep: QuantReport, reads, chart, fallback: Decision) -> Decision:
        chip_value = self._chip_value(view)
        if chip_value is None and self.require_stakes and self.escalation.mode != "all":
            return chart if chart is not None else fallback
        tricky = None
        if (self.escalation.mode == "jev" and self.jev is not None and view.street != "preflop"
                and view.pot >= self.escalation.router_min_pot_bb * view.bb):
            try:
                tricky = jev_tricky(self.jev, view, rep, self.db)
            except Exception:  # noqa: BLE001
                tricky = 3.0          # router down: fail open (let Opus decide)
        if not self.escalation.should(view, rep, call_cost_usd=self._call_cost(), chip_value=chip_value,
                                      tricky=tricky):
            return chart if chart is not None else fallback
        v = VARIANTS[self.variant]
        user = render_dashboard(view, rep, self.db, self.bankroll.context() if self.bankroll else None,
                                reads if self.reads_in_dashboard else None, {"hand_number": self.hand_number},
                                v["sections"])
        meta = {"hand_id": view.hand_id, "street": view.street, "options": [o.brief() for o in rep.options],
                "deadline_s": self.deadline_s}
        t0 = time.time()
        ans = self._ask(view, v["system"], user, meta)
        if ans is None:
            self.fallbacks += 1
            return fallback
        self.model_calls += 1
        usd = float(getattr(self.decider, "last_usd", 0.0) or 0.0)
        if usd and self.bankroll is not None:
            self.bankroll.record_llm_cost(usd)
        if not isinstance(ans, dict):
            raise ValueError("decider answer is not a JSON object")
        pick = self._sample(ans, rep)
        if pick is None:
            self.fallbacks += 1
            self._log({"hand": view.hand_id, "street": view.street, "error": "no usable mix in the answer"})
            return fallback
        opus_pick = pick
        if (self.override_gate is not None and pick.id != rep.best.id
                and self._engine_share(ans, rep) > self.override_gate):
            self.gated += 1
            pick = rep.best
        if self.verifier and self.jev is not None and pick.id != rep.best.id:
            try:
                st = jev_state(view, rep, self.db)
                vq = self.jev.ask(st, verify_question(pick.label), tag="verify")
                if float(vq["is_blunder"]["noul"]) >= self.verifier_threshold:
                    self._log({"hand": view.hand_id, "vetoed": pick.label})
                    pick = rep.best
            except Exception:  # noqa: BLE001
                pass
        note = ans.get("note")
        if isinstance(note, str) and note.strip() and rep.villains:
            self.db.add_note(rep.villains[0].name, note.strip()[:200], view.hand_id, source="opus")
        read = ans.get("read") if isinstance(ans.get("read"), str) else ""
        self._log({"hand": view.hand_id, "street": view.street, "choice": pick.label,
                   "engine": rep.best.label, "opus": opus_pick.label, "gated": pick is not opus_pick,
                   "read": read, "secs": round(time.time() - t0, 2)})
        return Decision(pick.decision.kind, pick.decision.amount, source="opus", reason=pick.label,
                        meta={"read": read})

    @staticmethod
    def _mix(ans: dict, rep: QuantReport) -> list:
        """Opus's mixed strategy as [(option id, p)], keeping only menu ids with a finite positive p."""
        out = []
        mix = ans.get("mix") if isinstance(ans, dict) else None
        for m in (mix if isinstance(mix, list) else []):
            try:
                oid, p = m.get("id"), float(m.get("p", 0))
            except (AttributeError, TypeError, ValueError):
                continue
            if math.isfinite(p) and p > 0 and rep.option(oid):
                out.append((oid, p))
        return out

    @classmethod
    def _engine_share(cls, ans: dict, rep: QuantReport) -> float:
        """Share of Opus's mix left on the engine's pick (1.0 when there is no usable mix: never a decisive override)."""
        valid = cls._mix(ans, rep)
        tot = sum(p for _, p in valid)
        return sum(p for oid, p in valid if oid == rep.best.id) / tot if tot > 0 else 1.0

    def _sample(self, ans: dict, rep: QuantReport):
        """The action to play from the answer; None when the answer carries no usable mix (the engine plays)."""
        valid = self._mix(ans, rep)
        if not valid:
            return None
        if not self.mix:
            return rep.option(max(valid, key=lambda t: t[1])[0])
        tot = sum(p for _, p in valid)
        x = self.rng.random() * tot
        for oid, p in valid:
            x -= p
            if x <= 0:
                return rep.option(oid)
        return rep.option(valid[-1][0])

def api_decider(client) -> Decider:
    """Adapter from OpusClient to the Decider signature (exposes .client and the last call's cost)."""
    def _d(system: str, user: str, meta: dict) -> dict:
        res = client.decide(system, user, tag=meta.get("street", ""), deadline_s=meta.get("deadline_s"))
        _d.last_usd = res.usd
        return res.data
    _d.client = client
    _d.last_usd = 0.0
    return _d
