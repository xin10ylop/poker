"""Slumbot adapter (https://slumbot.com) - a public heads-up NLHE benchmark bot.

Protocol (verified against the official sample client):
  POST /slumbot/api/new_hand {"token"?}      POST /slumbot/api/act {"token","incr"}
  blinds 50/100, stacks 20000 (200bb) reset every hand
  client_pos 0 = big blind (acts 2nd preflop, 1st postflop), 1 = small blind/button
  action string: k=check c=call f=fold bN=bet/raise to N (chips on THIS street), '/' ends a street
  final response adds winnings, baseline_winnings (duplicate baseline), bot_hole_cards
We replay Slumbot's action string through our own engine, so every agent sees
exactly the same GameView it would see in the simulator.

Failure policy: POST /act (and /new_hand) is never retried - it is not idempotent, and a
lost response may already have been applied by the server.  Instead the hand is counted
in `broken_hands`, logged, and the run continues with a fresh /new_hand.  Any other
per-hand error (server error_msg, desync, bad response) is handled the same way; a token
error also triggers one re-login.  Agent exceptions fall back to a safe action
(`agent_errors`).  run() always returns the (partial) session.
"""
from __future__ import annotations

import logging
import os
import random
import time
from dataclasses import dataclass, field
from typing import Optional

import requests

from ..agents.base import Agent
from ..cards import ALL_CARDS
from ..engine import HandState, IllegalAction
from ..view import Decision, HandHistory
from .acpc import guarded_act

log = logging.getLogger("pokerbrain.slumbot")

HOST = "https://slumbot.com"
SB, BB, STACK = 50, 100, 20000
TIMEOUT = 20.0                 # seconds per HTTP request
LOGIN_RETRIES = 3              # login is idempotent, so it (and only it) is retried
MAX_CONSECUTIVE_ERRORS = 10    # run() gives up after this many failed hands in a row
_CARDS = frozenset(ALL_CARDS)
_BOARD_SIZES = (0, 3, 4, 5)


class SlumbotError(RuntimeError):
    pass


class SlumbotTransportError(SlumbotError):
    """The request's outcome is unknown (network error, HTTP 5xx, non-JSON body): never re-send it."""


class SlumbotTokenError(SlumbotError):
    """The server rejected our token / session (expired, invalid): log in again."""


def _is_token_error(msg: str) -> bool:
    m = msg.lower()
    return any(w in m for w in ("token", "session", "expired", "login", "log in", "not logged"))


def _tokens(action: str) -> list[tuple[int, str, int]]:
    """Parse 'b200c/kb400' -> [(street, kind, amount)].  Raises SlumbotError on malformed strings."""
    if not isinstance(action, str):
        raise SlumbotError(f"bad action string {action!r}")
    out, street, i = [], 0, 0
    while i < len(action):
        ch = action[i]
        if ch == "/":
            street += 1
            if street > 3:
                raise SlumbotError(f"too many streets in action string {action!r}")
            i += 1
        elif ch in "kcf":
            out.append((street, {"k": "check", "c": "call", "f": "fold"}[ch], 0))
            i += 1
        elif ch == "b":
            j = i + 1
            while j < len(action) and action[j].isdigit():
                j += 1
            if j == i + 1:
                raise SlumbotError(f"bet without a size at {i} in action string {action!r}")
            out.append((street, "raise", int(action[i + 1:j])))
            i = j
        else:
            raise SlumbotError(f"bad character {ch!r} at {i} in action string {action!r}")
    return out


def _check_cards(hole, board, bot_hole=None) -> None:
    if not isinstance(hole, (list, tuple)) or len(hole) != 2:
        raise SlumbotError(f"bad hole cards {hole!r}")
    if not isinstance(board, (list, tuple)) or len(board) not in _BOARD_SIZES:
        raise SlumbotError(f"bad board {board!r}")
    cards = list(hole) + list(board) + list(bot_hole or [])
    bad = [c for c in cards if c not in _CARDS]
    if bad or (bot_hole is not None and len(bot_hole) != 2):
        raise SlumbotError(f"bad cards in hole={hole!r} board={board!r} bot={bot_hole!r}")
    if len(set(cards)) != len(cards):
        raise SlumbotError(f"duplicate cards in hole={hole!r} board={board!r} bot={bot_hole!r}")


def rebuild(hole: list, board: list, action: str, client_pos: int, hand_id: str,
            bot_hole: Optional[list] = None) -> HandState:
    """Our engine state for hero (seat 0) vs Slumbot (seat 1).  Raises SlumbotError on malformed input."""
    if client_pos not in (0, 1):
        raise SlumbotError(f"bad client_pos {client_pos!r}")
    _check_cards(hole, board, bot_hole or None)
    toks = _tokens(action)
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
    for street, kind, amount in toks:
        if h.finished:
            raise SlumbotError(f"action after the end of the hand in {action!r}")
        if street != h.street_idx:
            raise SlumbotError(f"desync: {kind} on street {street} but the engine is on street "
                               f"{h.street_idx} in {action!r}")
        try:
            h.apply(Decision(kind, amount))
        except IllegalAction as exc:
            raise SlumbotError(f"illegal action in {action!r}: {exc}") from exc
    if not h.finished and action.count("/") > h.street_idx:     # a final '/' is optional, an extra one is not
        raise SlumbotError(f"desync: {action!r} has more streets than the betting reached")
    return h


def to_incr(d: Decision) -> str:
    return {"fold": "f", "check": "k", "call": "c"}.get(d.kind) or f"b{int(d.amount)}"


def _hand_fields(r: dict) -> tuple[list, list, str, int]:
    try:
        return r["hole_cards"], r.get("board") or [], r.get("action") or "", r["client_pos"]
    except (KeyError, TypeError, AttributeError) as exc:
        raise SlumbotError(f"unexpected response (missing {exc}): {str(r)[:200]}") from exc


@dataclass
class SlumbotSession:
    username: Optional[str] = None
    password: Optional[str] = field(default=None, repr=False)
    token: Optional[str] = field(default=None, repr=False)
    hands: int = 0
    winnings: float = 0.0
    baseline: float = 0.0
    results: list = field(default_factory=list, repr=False)
    broken_hands: int = 0          # hands abandoned (lost response, server error, desync): result unknown
    agent_errors: int = 0          # agent exceptions answered with the fallback action
    last_error: Optional[str] = None
    interrupted: bool = False
    aborted: bool = False          # stopped early (login failed, too many consecutive errors)

    def _post(self, path: str, body: dict, retries: int = 0) -> dict:
        """One POST (retried only when `retries` > 0, i.e. for idempotent calls such as login)."""
        url = f"{HOST}/slumbot/api/{path}"
        last: Optional[SlumbotError] = None
        for attempt in range(retries + 1):
            if attempt:
                time.sleep(min(2 ** attempt, 30))
            try:
                r = requests.post(url, json=body, timeout=TIMEOUT)
            except requests.RequestException as exc:
                last = SlumbotTransportError(f"{path}: network error: {exc}")
                continue
            status = int(getattr(r, "status_code", 200))
            if not 200 <= status < 300:
                msg = None
                try:
                    data = r.json()
                    msg = data.get("error_msg") if isinstance(data, dict) else None
                except ValueError:
                    pass
                if msg:
                    raise (SlumbotTokenError if _is_token_error(msg) else SlumbotError)(f"{path}: {msg}")
                if status >= 500:          # outcome unknown: the server may have applied it
                    last = SlumbotTransportError(f"{path}: HTTP {status}")
                    continue
                raise (SlumbotTokenError if status in (401, 403) else SlumbotError)(f"{path}: HTTP {status}")
            try:
                data = r.json()
            except ValueError as exc:
                last = SlumbotTransportError(f"{path}: non-JSON response (HTTP {status}): {exc}")
                continue
            if not isinstance(data, dict):
                last = SlumbotTransportError(f"{path}: unexpected response {str(data)[:200]}")
                continue
            if data.get("error_msg"):
                msg = str(data["error_msg"])
                raise (SlumbotTokenError if _is_token_error(msg) else SlumbotError)(f"{path}: {msg}")
            if data.get("token"):
                self.token = data["token"]
            return data
        assert last is not None
        raise last

    def login(self) -> None:
        if self.username and self.password:
            self._post("login", {"username": self.username, "password": self.password}, retries=LOGIN_RETRIES)

    def play_hand(self, agent: Agent, hand_index: int, on_hand=None) -> dict:
        agent.new_hand(hand_index)
        body = {"token": self.token} if self.token else {}
        r = self._post("new_hand", body)
        hand_id = f"slumbot-{hand_index}"
        while "winnings" not in r:
            hole, board, action, client_pos = _hand_fields(r)
            h = rebuild(hole, board, action, client_pos, hand_id)
            if h.finished or h.to_act != 0:
                raise SlumbotError(f"desync: action={action!r}")
            view = h.view_for(0, platform="slumbot", table_id="slumbot")
            d, failed = guarded_act(agent, view)
            self.agent_errors += failed
            r = self._post("act", {"token": self.token, "incr": to_incr(d)})
        try:
            winnings = float(r["winnings"])
            baseline = float(r.get("baseline_winnings") or 0)
        except (TypeError, ValueError) as exc:
            raise SlumbotError(f"bad winnings in final response: {str(r)[:200]}") from exc
        self.hands += 1
        self.winnings += winnings
        self.baseline += baseline
        res = {"hand": hand_index, "winnings": winnings, "baseline": baseline,
               "action": r.get("action"), "hole": r.get("hole_cards"), "bot_hole": r.get("bot_hole_cards"),
               "board": r.get("board")}
        self.results.append(res)
        try:
            self._observe(agent, r, hand_id, winnings)
        except Exception as exc:  # noqa: BLE001 - the result is already booked; only tracking is lost
            log.warning("slumbot hand %d: opponent tracking skipped (%s: %s)", hand_index, type(exc).__name__, exc)
        if on_hand:
            on_hand(res)
        return res

    def _observe(self, agent: Agent, r: dict, hand_id: str, winnings: float) -> None:
        # opponent tracking: only reveal the bot's cards if the hand reached showdown
        hole, board, action, client_pos = _hand_fields(r)
        showdown = "f" not in action
        bot_hole = r.get("bot_hole_cards") if showdown else None
        h = rebuild(hole, board, action, client_pos, hand_id, bot_hole=bot_hole)
        hist = HandHistory(hand_id=hand_id, sb=SB, bb=BB, button_seat=h.button, names=["Hero", "Slumbot"],
                           positions=list(h.positions), start_stacks=[STACK, STACK], actions=list(h.log),
                           board=list(board),
                           shown={0: tuple(hole), 1: tuple(bot_hole)} if bot_hole else {},
                           net={0: winnings, 1: -winnings}, hero_seat=0, platform="slumbot",
                           table_id="slumbot", hero_hole=tuple(hole))
        agent.observe(hist, 0)

    def bb100(self) -> float:
        return 100.0 * self.winnings / BB / max(1, self.hands)

    def bb100_baseline_adjusted(self) -> float:
        """winnings - baseline_winnings: Slumbot's duplicate-style variance reduction."""
        return 100.0 * (self.winnings - self.baseline) / BB / max(1, self.hands)

    def summary(self) -> dict:
        out = {"hands": self.hands, "bb100_raw": round(self.bb100(), 2),
               "bb100_baseline_adjusted": round(self.bb100_baseline_adjusted(), 2),
               "broken_hands": self.broken_hands, "agent_errors": self.agent_errors}
        if self.last_error:
            out["last_error"] = self.last_error
        if self.interrupted:
            out["interrupted"] = True
        if self.aborted:
            out["aborted"] = True
        return out


def run(agent: Agent, n_hands: int, username: Optional[str] = None, password: Optional[str] = None,
        progress: bool = True, pause: float = 0.05,
        max_consecutive_errors: int = MAX_CONSECUTIVE_ERRORS) -> SlumbotSession:
    """Play n_hands hand attempts.  Never raises for per-hand errors; always returns the (partial) session."""
    s = SlumbotSession(username=username or os.environ.get("SLUMBOT_USER"),
                       password=password or os.environ.get("SLUMBOT_PASS"))
    consecutive = 0
    try:
        try:
            s.login()
        except SlumbotError as exc:
            s.last_error, s.aborted = f"login failed: {exc}", True
            log.error("slumbot: %s", s.last_error)
            return s
        for i in range(n_hands):
            try:
                s.play_hand(agent, i)
            except Exception as exc:  # noqa: BLE001 - one bad hand must never end the run
                consecutive += 1
                s.broken_hands += 1
                s.last_error = f"hand {i}: {type(exc).__name__}: {exc}"
                log.warning("slumbot: hand %d abandoned, result unknown (%s: %s); starting a fresh hand",
                            i, type(exc).__name__, exc, exc_info=not isinstance(exc, SlumbotError))
                if isinstance(exc, SlumbotTokenError):
                    s.token = None
                    try:
                        s.login()
                    except SlumbotError as exc2:
                        s.last_error, s.aborted = f"re-login failed: {exc2}", True
                        log.error("slumbot: %s", s.last_error)
                        break
                if consecutive >= max_consecutive_errors:
                    s.aborted = True
                    log.error("slumbot: giving up after %d consecutive failed hands", consecutive)
                    break
                if consecutive > 1:
                    time.sleep(min(2 ** (consecutive - 2), 30))
                continue
            consecutive = 0
            if agent.should_stop():          # stop-loss / circuit breaker / broke: the session ends here
                s.last_error = "session stopped by the bankroll manager"
                log.warning("slumbot: %s after %d hands", s.last_error, s.hands)
                break
            if progress and (i + 1) % 25 == 0:
                print(f"  slumbot {i + 1}/{n_hands}: {s.bb100():+.1f} bb/100 raw, "
                      f"{s.bb100_baseline_adjusted():+.1f} baseline-adjusted", flush=True)
            time.sleep(pause)
    except KeyboardInterrupt:
        s.interrupted = True
        log.warning("slumbot: interrupted after %d hands", s.hands)
    finally:
        if progress:
            print(f"  slumbot done: {s.hands} hands, {s.broken_hands} broken, {s.agent_errors} agent errors: "
                  f"{s.bb100():+.1f} bb/100 raw, {s.bb100_baseline_adjusted():+.1f} baseline-adjusted", flush=True)
    return s
