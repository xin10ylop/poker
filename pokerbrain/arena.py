"""Match runner with variance reduction.

* `Table.play_hand` drives one hand between agents on the local engine.
* `duplicate_hu` plays every deck twice with seats swapped (fresh agent
  instances per side), so card luck largely cancels.
* `ring_session` plays a 6-max table where the button rotates; comparing two
  hero variants on the *same seeds* (common random numbers) gives a paired
  comparison with far lower variance than independent runs.
* Win rates use the all-in-adjusted result (`ev_net`) by default.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from .agents.base import Agent
from .cards import ALL_CARDS
from .engine import HandResult, HandState, IllegalAction
from .view import Decision


def seeded_deck(seed: int) -> list[str]:
    d = list(ALL_CARDS)
    random.Random(seed).shuffle(d)
    return d


@dataclass
class MatchStats:
    name: str
    results_bb: list = field(default_factory=list)      # realized per hand (hero)
    ev_bb: list = field(default_factory=list)           # all-in adjusted per hand
    decisions: int = 0
    errors: int = 0
    seconds: float = 0.0
    extra: dict = field(default_factory=dict)

    def bb100(self, adjusted: bool = True) -> float:
        xs = self.ev_bb if adjusted else self.results_bb
        return 100.0 * sum(xs) / max(1, len(xs))

    def ci95(self, adjusted: bool = True) -> float:
        xs = self.ev_bb if adjusted else self.results_bb
        n = len(xs)
        if n < 2:
            return float("nan")
        m = sum(xs) / n
        var = sum((x - m) ** 2 for x in xs) / (n - 1)
        return 1.96 * 100.0 * math.sqrt(var / n)

    def summary(self) -> dict:
        return {"agent": self.name, "hands": len(self.ev_bb), "bb100_adj": round(self.bb100(), 2),
                "ci95": round(self.ci95(), 2), "bb100_raw": round(self.bb100(False), 2),
                "errors": self.errors, "sec_per_hand": round(self.seconds / max(1, len(self.ev_bb)), 4),
                **self.extra}


class Table:
    def __init__(self, agents: list[Agent], sb: int = 50, bb: int = 100, stack_bb: float = 100,
                 platform: str = "sim", table_id: str = "t1", on_decision: Optional[Callable] = None):
        self.agents = agents
        self.sb, self.bb = sb, bb
        self.stack = int(stack_bb * bb)
        self.platform = platform
        self.table_id = table_id
        self.on_decision = on_decision
        self.errors = 0

    def play_hand(self, deck: list[str], button: int, hand_id: str, hand_index: int = 0,
                  stacks: Optional[list[int]] = None) -> HandResult:
        n = len(self.agents)
        for ag in self.agents:
            ag.new_hand(hand_index)
        h = HandState(stacks or [self.stack] * n, button=button, sb=self.sb, bb=self.bb, deck=deck,
                      names=[a.name for a in self.agents], hand_id=hand_id)
        guard = 0
        while not h.finished:
            seat = h.to_act
            view = h.view_for(seat, platform=self.platform, table_id=self.table_id)
            try:
                d = self.agents[seat].act(view)
                d = d.normalized(view.legal)
                h.apply(d)
            except (IllegalAction, Exception) as exc:  # noqa: BLE001 - never let an agent crash the table
                self.errors += 1
                if self.on_decision:
                    self.on_decision("error", seat, view, exc)
                h.apply(Decision("check" if view.legal.can_check else "fold"))
            guard += 1
            if guard > 400:
                raise RuntimeError("hand did not terminate")
        res = h.result
        for i, ag in enumerate(self.agents):
            ag.observe(res.to_history(hero_seat=i, platform=self.platform, table_id=self.table_id), i)
        return res


def duplicate_hu(make_a: Callable[[], Agent], make_b: Callable[[], Agent], n_decks: int, seed: int = 0,
                 stack_bb: float = 100, sb: int = 50, bb: int = 100, progress: bool = False) -> MatchStats:
    """Heads-up duplicate: each deck is played twice with seats swapped.

    Two independent 'universes' (A1 vs B1 and B2 vs A2) so learning agents don't see
    the mirrored hands.  Stats are from agent A's perspective, per hand."""
    a1, b1, a2, b2 = make_a(), make_b(), make_a(), make_b()
    stats = MatchStats(name=a1.name)
    t1 = Table([a1, b1], sb=sb, bb=bb, stack_bb=stack_bb, table_id="dup1")
    t2 = Table([b2, a2], sb=sb, bb=bb, stack_bb=stack_bb, table_id="dup2")
    t0 = time.time()
    for i in range(n_decks):
        deck = seeded_deck(seed * 1_000_003 + i)
        button = i % 2
        r1 = t1.play_hand(deck, button, f"d{seed}-{i}a", i)
        r2 = t2.play_hand(deck, button, f"d{seed}-{i}b", i)
        stats.results_bb += [r1.net[0] / bb, r2.net[1] / bb]
        stats.ev_bb += [r1.ev_net[0] / bb, r2.ev_net[1] / bb]
        if a1.should_stop() or a2.should_stop():
            stats.extra["stopped_after_deck"] = i + 1
            break
        if progress and (i + 1) % 200 == 0:
            print(f"  {i + 1}/{n_decks} decks  {stats.bb100():+.1f} bb/100 ±{stats.ci95():.1f}", flush=True)
    stats.seconds = time.time() - t0
    stats.errors = t1.errors + t2.errors
    return stats


def ring_session(hero_factory: Callable[[], Agent], field_factory: Callable[[], list[Agent]], n_hands: int,
                 seed: int = 0, hero_seat: int = 0, stack_bb: float = 100, sb: int = 50, bb: int = 100,
                 progress: bool = False) -> MatchStats:
    """Hero vs a field at one table; button rotates.  Same seed => same decks & bot RNG."""
    hero = hero_factory()
    bots = field_factory()
    agents = bots[:hero_seat] + [hero] + bots[hero_seat:]
    table = Table(agents, sb=sb, bb=bb, stack_bb=stack_bb)
    stats = MatchStats(name=hero.name)
    per_opp: dict[str, float] = {}
    t0 = time.time()
    for i in range(n_hands):
        deck = seeded_deck(seed * 7_919 + i)
        res = table.play_hand(deck, i % len(agents), f"r{seed}-{i}", i)
        stats.results_bb.append(res.net[hero_seat] / bb)
        stats.ev_bb.append(res.ev_net[hero_seat] / bb)
        if hero.should_stop():                       # stop-loss / circuit breaker / broke: the session ends here
            stats.extra["stopped_after_hand"] = i + 1
            stats.extra["stop_reason"] = getattr(getattr(hero, "bankroll", None), "session_status", lambda: "stop")()
            break
        for j, ag in enumerate(agents):
            if j != hero_seat:
                per_opp[ag.name] = per_opp.get(ag.name, 0.0) + res.ev_net[j] / bb
        if progress and (i + 1) % 250 == 0:
            print(f"  {i + 1}/{n_hands} hands  {stats.bb100():+.1f} bb/100 ±{stats.ci95():.1f}", flush=True)
    stats.seconds = time.time() - t0
    stats.errors = table.errors
    stats.extra["field_bb100"] = {k: round(100 * v / n_hands, 1) for k, v in per_opp.items()}
    stats.extra.update(hero.stats())
    return stats


def paired_diff(a: MatchStats, b: MatchStats) -> tuple[float, float]:
    """Mean difference (bb/100) and 95% CI for two runs on the same seeds."""
    d = [x - y for x, y in zip(a.ev_bb, b.ev_bb)]
    n = len(d)
    m = sum(d) / n
    var = sum((x - m) ** 2 for x in d) / max(1, n - 1)
    return 100 * m, 1.96 * 100 * math.sqrt(var / n)
