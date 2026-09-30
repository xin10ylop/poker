"""Agent interface shared by bots, the quant agent and the LLM agents."""
from __future__ import annotations

from typing import Optional

from ..view import Decision, GameView, HandHistory


class Agent:
    name: str = "agent"

    def act(self, view: GameView) -> Decision:  # pragma: no cover - interface
        raise NotImplementedError

    def observe(self, history: HandHistory, my_seat: int) -> None:
        """Called after every hand with the public history (+ showdown cards)."""

    def new_hand(self, hand_index: int) -> None:
        """Called before each hand (bots reseed their RNG here)."""

    def should_stop(self) -> bool:
        """True when the session must end (stop-loss, circuit breaker, broke).  Runners check it after each hand."""
        return False

    def stats(self) -> dict:
        return {}


class CallingAgent(Agent):
    """Always checks/calls. Useful as a trivial baseline."""

    def __init__(self, name: str = "caller"):
        self.name = name

    def act(self, view: GameView) -> Decision:
        return Decision("call" if view.legal.call_amount > 0 else "check")


class RandomAgent(Agent):
    def __init__(self, name: str = "random", seed: int = 0):
        import random
        self.name = name
        self.rng = random.Random(seed)

    def act(self, view: GameView) -> Decision:
        la = view.legal
        r = self.rng.random()
        if la.can_raise and r < 0.2:
            return Decision("raise", self.rng.randint(la.min_raise_to, la.max_raise_to))
        if la.call_amount > 0:
            return Decision("call" if r < 0.6 else "fold")
        return Decision("check")
