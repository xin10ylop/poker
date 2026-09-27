"""Slumbot adapter (https://slumbot.com) - a public heads-up NLHE benchmark bot.

Protocol (verified against the official sample client):
  POST /slumbot/api/new_hand {"token"?}      POST /slumbot/api/act {"token","incr"}
  blinds 50/100, stacks 20000 (200bb) reset every hand
  client_pos 0 = big blind (acts 2nd preflop, 1st postflop), 1 = small blind/button
  action string: k=check c=call f=fold bN=bet/raise to N (chips on THIS street), '/' ends a street
  final response adds winnings, baseline_winnings (duplicate baseline), bot_hole_cards
We replay Slumbot's action string through our own engine, so every agent sees
exactly the same GameView it would see in the simulator.
"""
from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
from typing import Optional

import requests

from ..agents.base import Agent
from ..cards import ALL_CARDS
from ..engine import HandState
from ..view import Decision, GameView, HandHistory

HOST = "https://slumbot.com"
SB, BB, STACK = 50, 100, 20000


class SlumbotError(RuntimeError):
    pass


def _tokens(action: str) -> list[tuple[int, str, int]]:
    """Parse 'b200c/kb400' -> [(street, kind, amount)]."""
    out, street, i = [], 0, 0
    while i < len(action):
        ch = action[i]
        if ch == "/":
            street += 1
            i += 1
        elif ch in "kcf":
            out.append((street, {"k": "check", "c": "call", "f": "fold"}[ch], 0))
            i += 1
        elif ch == "b":
            j = i + 1
            while j < len(action) and action[j].isdigit():
                j += 1
            out.append((street, "raise", int(action[i + 1:j])))
            i = j
        else:
            raise SlumbotError(f"bad action string {action!r}")
    return out


def rebuild(hole: list, board: list, action: str, client_pos: int, hand_id: str,
            bot_hole: Optional[list] = None) -> HandState:
    """Our engine state for hero (seat 0) vs Slumbot (seat 1)."""
    button = 0 if client_pos == 1 else 1
    dead = set(hole) | set(board) | set(bot_hole or [])
    filler = [c for c in ALL_CARDS if c not in dead]
    random.Random(hand_id).shuffle(filler)
    bot = list(bot_hole) if bot_hole else filler[:2]
    rest = [c for c in filler if c not in bot]
    deck = list(hole) + bot + list(board) + rest[: 5 - len(board)]
    deck += [c for c in rest if c not in deck]
    h = HandState([STACK, STACK], button=button, sb=SB, bb=BB, deck=deck, names=["Hero", "Slumbot"],
                  hand_id=hand_id)
    for street, kind, amount in _tokens(action):
        if h.finished:
            break
        h.apply(Decision(kind, amount))
    return h


def to_incr(d: Decision) -> str:
    return {"fold": "f", "check": "k", "call": "c"}.get(d.kind) or f"b{int(d.amount)}"


@dataclass
class SlumbotSession:
    username: Optional[str] = None
    password: Optional[str] = None
    token: Optional[str] = None
    hands: int = 0
    winnings: float = 0.0
    baseline: float = 0.0
    results: list = field(default_factory=list)

    def _post(self, path: str, body: dict) -> dict:
        for attempt in range(4):
            try:
                r = requests.post(f"{HOST}/slumbot/api/{path}", json=body, timeout=20)
                data = r.json()
            except (requests.RequestException, ValueError) as exc:
                time.sleep(2 ** attempt)
                last = exc
                continue
            if data.get("error_msg"):
                raise SlumbotError(data["error_msg"])
            if data.get("token"):
                self.token = data["token"]
            return data
        raise SlumbotError(f"network error: {last}")

    def login(self) -> None:
        if self.username and self.password:
            self._post("login", {"username": self.username, "password": self.password})

    def play_hand(self, agent: Agent, hand_index: int, on_hand=None) -> dict:
        agent.new_hand(hand_index)
        body = {"token": self.token} if self.token else {}
        r = self._post("new_hand", body)
        hand_id = f"slumbot-{hand_index}"
        while "winnings" not in r:
            hole, board, action = r["hole_cards"], r.get("board", []), r.get("action", "")
            h = rebuild(hole, board, action, r["client_pos"], hand_id)
            if h.finished or h.to_act != 0:
                raise SlumbotError(f"desync: action={action!r}")
            view = h.view_for(0, platform="slumbot", table_id="slumbot")
            d = agent.act(view).normalized(view.legal)
            r = self._post("act", {"token": self.token, "incr": to_incr(d)})
        self.hands += 1
        self.winnings += r["winnings"]
        self.baseline += r.get("baseline_winnings", 0)
        res = {"hand": hand_index, "winnings": r["winnings"], "baseline": r.get("baseline_winnings", 0),
               "action": r.get("action"), "hole": r.get("hole_cards"), "bot_hole": r.get("bot_hole_cards"),
               "board": r.get("board")}
        self.results.append(res)
        # opponent tracking: only reveal the bot's cards if the hand reached showdown
        showdown = "f" not in (r.get("action") or "")
        h = rebuild(r["hole_cards"], r.get("board", []), r.get("action", ""), r["client_pos"], hand_id,
                    bot_hole=r.get("bot_hole_cards") if showdown else None)
        hist = HandHistory(hand_id=hand_id, sb=SB, bb=BB, button_seat=h.button, names=["Hero", "Slumbot"],
                           positions=list(h.positions), start_stacks=[STACK, STACK], actions=list(h.log),
                           board=list(r.get("board", [])),
                           shown={0: tuple(r["hole_cards"]), 1: tuple(r["bot_hole_cards"])} if showdown and
                           r.get("bot_hole_cards") else {},
                           net={0: r["winnings"], 1: -r["winnings"]}, hero_seat=0, platform="slumbot",
                           table_id="slumbot", hero_hole=tuple(r["hole_cards"]))
        agent.observe(hist, 0)
        if on_hand:
            on_hand(res)
        return res

    def bb100(self) -> float:
        return 100.0 * self.winnings / BB / max(1, self.hands)

    def bb100_baseline_adjusted(self) -> float:
        """winnings - baseline_winnings: Slumbot's duplicate-style variance reduction."""
        return 100.0 * (self.winnings - self.baseline) / BB / max(1, self.hands)


def run(agent: Agent, n_hands: int, username: Optional[str] = None, password: Optional[str] = None,
        progress: bool = True, pause: float = 0.05) -> SlumbotSession:
    s = SlumbotSession(username=username or os.environ.get("SLUMBOT_USER"),
                       password=password or os.environ.get("SLUMBOT_PASS"))
    s.login()
    for i in range(n_hands):
        s.play_hand(agent, i)
        if progress and (i + 1) % 25 == 0:
            print(f"  slumbot {i + 1}/{n_hands}: {s.bb100():+.1f} bb/100 raw, "
                  f"{s.bb100_baseline_adjusted():+.1f} baseline-adjusted", flush=True)
        time.sleep(pause)
    return s
