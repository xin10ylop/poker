"""Jev (TypeSafe System One) client via OpenRouter's Decisions API.

Jev does not write text: it answers typed questions about a `state` with
calibrated probabilities (noul = P(yes), choice/score = distributions).
That makes it the perfect "gut feel" module for poker: fast (~100 ms),
very cheap (input tokens only), and its probabilities plug straight into EV
math instead of being parsed out of prose.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

import requests

from .budget import Budget, BudgetExceeded, default_budget

URL = "https://openrouter.ai/api/alpha/decisions"
DEFAULT_MODEL = "~typesafe/jev-latest"


class JevError(RuntimeError):
    pass


class JevClient:
    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL,
                 budget: Optional[Budget] = None, timeout: float = 12.0, retries: int = 2):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise JevError("OPENROUTER_API_KEY not set")
        self.model = model
        self.budget = budget or default_budget()
        self.timeout = timeout
        self.retries = retries
        self.calls = 0
        self.usd = 0.0
        self.latency: list[float] = []

    def ask(self, state: Any, questions: dict, tag: str = "") -> dict:
        """Returns the `answers` dict. Raises JevError / BudgetExceeded."""
        approx_tokens = len(json.dumps(state)) / 3.5 + len(json.dumps(questions)) / 3.5
        self.budget.check(approx_tokens * 0.042e-6 * 1.5)
        body = {"model": self.model, "state": state, "questions": questions}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json",
                   "X-Title": "pokerbrain"}
        last = None
        for attempt in range(self.retries + 1):
            t0 = time.time()
            try:
                r = requests.post(URL, headers=headers, json=body, timeout=self.timeout)
            except requests.RequestException as exc:
                last = exc
                time.sleep(0.5 * (attempt + 1))
                continue
            self.latency.append(time.time() - t0)
            if r.status_code == 200:
                data = r.json()
                cost = float((data.get("usage") or {}).get("cost") or approx_tokens * 0.042e-6)
                self.budget.record("jev", cost, tag)
                self.calls += 1
                self.usd += cost
                return data.get("answers", {})
            last = JevError(f"HTTP {r.status_code}: {r.text[:300]}")
            if r.status_code in (400, 401, 402, 403):
                break
            time.sleep(0.5 * (attempt + 1))
        raise JevError(str(last))


# ---------------------------------------------------------------------------
# Poker question sets
# ---------------------------------------------------------------------------
ARCHETYPE_CRITERIA = {
    "nit": "Very tight and cautious: few hands, rarely bluffs, folds to aggression, big bets = the nuts.",
    "tag": "Solid tight-aggressive regular with sensible ranges and balanced-ish aggression.",
    "lag": "Loose-aggressive: plays many hands, 3-bets and barrels often, capable of big bluffs.",
    "calling_station": "Loose-passive: calls far too much with weak pairs/draws, rarely folds, rarely bluffs.",
    "maniac": "Hyper-aggressive: bets and raises almost always, a large share of it pure bluffs.",
    "weak_passive": "Loose-passive fish who limps/calls preflop but gives up and folds when he misses.",
}

TILT_LEVELS = [
    "Calm: playing his normal game",
    "Slightly off: a bit looser or more aggressive than usual",
    "Tilted: clearly loose/aggressive after losses, chasing",
    "Full tilt: reckless, spewing chips, punting stacks",
]


def reads_questions(villain_name: str, facing_bet: bool, hero_can_bet: bool) -> dict:
    q: dict = {
        "archetype": {"type": "choice",
                      "instructions": f"Which player type best describes {villain_name}, given his stats, "
                                      f"showdowns and notes?",
                      "criteria": ARCHETYPE_CRITERIA},
        "tilt": {"type": "score",
                 "instructions": f"How tilted is {villain_name} right now, judging from his recent results "
                                 f"and any change in his betting behaviour?",
                 "criteria": TILT_LEVELS},
    }
    if facing_bet:
        q["villain_bluffing"] = {
            "type": "noul",
            "instructions": f"Is {villain_name}'s current bet a bluff (a hand that cannot beat most of "
                            f"hero's calling range if called)?",
            "criteria": {"true": "Weak hand or missed draw betting to make hero fold",
                         "false": "Value bet with a hand that expects to be ahead when called"}}
    if hero_can_bet:
        q["villain_folds_to_bet"] = {
            "type": "noul",
            "instructions": f"If hero makes a normal-sized bet or raise now (about 2/3 to full pot), "
                            f"will {villain_name} fold?",
            "criteria": {"true": "He folds: his range is weak or he gives up easily",
                         "false": "He calls or raises: sticky, strong, or never folds"}}
    return q


def decision_question(options: list[dict]) -> dict:
    """Jev-as-decider: a Choice over the engine's action menu."""
    crit = {}
    for o in options:
        desc = o["action"]
        if "ev_bb" in o:
            desc += f" | engine EV {o['ev_bb']:+.1f}bb"
        if "villain_fold_prob" in o:
            desc += f", villain folds ~{o['villain_fold_prob']:.0%}"
        crit[o["id"]] = desc
    return {"best_action": {"type": "choice",
                            "instructions": "Which action maximises hero's expected winnings against THIS "
                                            "opponent, combining the math with the psychological read?",
                            "criteria": crit}}


def verify_question(action_label: str) -> dict:
    return {"is_blunder": {"type": "noul",
                           "instructions": f"Hero is about to: {action_label}. Is this a clear mistake that a "
                                           f"strong professional would never make here?",
                           "criteria": {"true": "Clear blunder (e.g. folding the nuts, calling off with air "
                                                "vs a nit's shove, bluffing a calling station)",
                                        "false": "Reasonable or good play"}}}
