"""Local HTTP decision service (stdlib only).

  POST /decide   body: GameView as JSON (GameView.to_dict() shape)
                 -> {"kind", "amount", "source", "reason", "session_status"}
  POST /observe  body: HandHistory as JSON, `hero_seat` required
                 -> {"ok": true}   ({"ok": true, "duplicate": true} for a (table_id, hand_id) already seen)
  GET  /health

Every POST must carry `Content-Type: application/json` and the header
`X-PokerBrain-Token: <token>`.  The token is $POKERBRAIN_API_TOKEN, or a random
one printed to stderr at startup.  Requests with an `Origin` header (browsers)
are refused, and so is any `Host` other than localhost / 127.0.0.1 / [::1]
while the server is bound to loopback (DNS rebinding).  Payloads are validated
before the agent sees them: 400 for malformed requests, 422 for an impossible
/decide state, 500 (details on stderr only) for agent failures.

Use it to connect the brain to environments that explicitly allow bots
(your own home-game server, bot competitions, research platforms).  It binds
to 127.0.0.1 and refuses any other address unless allow_remote=True.
"""
from __future__ import annotations

import dataclasses
import hmac
import ipaddress
import json
import math
import os
import re
import secrets
import socket
import sys
import threading
import traceback
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.parse import urlsplit

from ..agents.base import Agent
from ..cards import ALL_CARDS, normalize_card
from ..view import STREETS, ActionRecord, GameView, HandHistory, LegalActions, PlayerView

TOKEN_HEADER = "X-PokerBrain-Token"
MAX_BODY = 1 << 20                 # bytes
REQUEST_TIMEOUT_S = 10
DEDUP_KEEP = 2000                  # (table_id, hand_id) keys remembered by /observe
MAX_SEATS = 10
MAX_ACTIONS = 1000
MAX_CHIPS = 10 ** 12
BOARD_LEN = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}
ACTION_KINDS = frozenset({"post_sb", "post_bb", "post_ante", "fold", "check", "call", "bet", "raise"})
_KNOWN_CARDS = frozenset(ALL_CARDS)
_HOST_RE = re.compile(r"(localhost|127\.0\.0\.1|\[::1\])(:\d{1,5})?")


class BadRequest(ValueError):
    """A client error; `code` is the HTTP status to answer with."""

    def __init__(self, msg: str, code: int = 400):
        super().__init__(msg)
        self.code = code


# --------------------------------------------------------------------------- validation
class _Check:
    def __init__(self, code: int):
        self.code = code

    def fail(self, msg: str):
        raise BadRequest(msg, self.code)

    def obj(self, x, what: str) -> dict:
        if not isinstance(x, dict):
            self.fail(f"{what} must be an object")
        return x

    def lst(self, x, what: str, lo: int = 0, hi: int = MAX_ACTIONS) -> list:
        if not isinstance(x, list):
            self.fail(f"{what} must be a list")
        if not lo <= len(x) <= hi:
            self.fail(f"{what} must have {lo}..{hi} entries")
        return x

    def int_(self, x, what: str, lo: int = 0, hi: int = MAX_CHIPS) -> int:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            self.fail(f"{what} must be a number")
        if isinstance(x, float):
            if not math.isfinite(x) or not x.is_integer():
                self.fail(f"{what} must be a whole number")
            x = int(x)
        if not lo <= x <= hi:
            self.fail(f"{what} out of range ({lo}..{hi})")
        return x

    def num(self, x, what: str) -> float:
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or abs(x) > MAX_CHIPS:
            self.fail(f"{what} must be a finite number")
        return x

    def bool_(self, x, what: str) -> bool:
        if not isinstance(x, bool):
            self.fail(f"{what} must be true or false")
        return x

    def str_(self, x, what: str, maxlen: int = 100) -> str:
        if isinstance(x, int) and not isinstance(x, bool):
            x = str(x)
        if not isinstance(x, str) or len(x) > maxlen:
            self.fail(f"{what} must be a string of at most {maxlen} characters")
        return x

    def card(self, x, what: str) -> str:
        try:
            c = normalize_card(x) if isinstance(x, str) else None
        except ValueError:
            c = None
        if c not in _KNOWN_CARDS:
            self.fail(f"{what}: unknown card {x!r}")
        return c

    def cards(self, x, what: str, lo: int, hi: int) -> list:
        return [self.card(c, f"{what}[{i}]") for i, c in enumerate(self.lst(x, what, lo, hi))]

    def seat(self, x, what: str, n: int) -> int:
        return self.int_(x, what, 0, n - 1)

    def seat_key(self, k, what: str, n: int) -> int:
        if isinstance(k, str) and re.fullmatch(r"\d{1,2}", k):
            k = int(k)
        return self.seat(k, what, n)

    def actions(self, x, n: int) -> list:
        out = []
        for i, a in enumerate(self.lst(x, "actions", 0, MAX_ACTIONS)):
            w = f"actions[{i}]"
            a = self.obj(a, w)
            unknown = set(a) - {f.name for f in dataclasses.fields(ActionRecord)}
            if unknown:
                self.fail(f"{w}: unknown field {sorted(unknown)[0]!r}")
            if a.get("street") not in STREETS:
                self.fail(f"{w}.street must be one of {', '.join(STREETS)}")
            if a.get("kind") not in ACTION_KINDS:
                self.fail(f"{w}.kind must be one of {', '.join(sorted(ACTION_KINDS))}")
            out.append(ActionRecord(street=a["street"], seat=self.seat(a.get("seat"), f"{w}.seat", n),
                                    name=self.str_(a.get("name", ""), f"{w}.name"), kind=a["kind"],
                                    to=self.int_(a.get("to"), f"{w}.to"), added=self.int_(a.get("added"), f"{w}.added"),
                                    pot_before=self.int_(a.get("pot_before"), f"{w}.pot_before"),
                                    all_in=self.bool_(a.get("all_in", False), f"{w}.all_in")))
        return out


def _unique(c: _Check, groups: list[list]) -> None:
    seen: set = set()
    for g in groups:
        for card in g:
            if card in seen:
                c.fail(f"duplicate card {card}")
            seen.add(card)


def view_from_dict(d: dict) -> GameView:
    """Build a GameView from its JSON form, rejecting impossible states (BadRequest, code 422)."""
    c = _Check(422)
    d = c.obj(d, "body")
    for key in ("sb", "bb", "hero_seat", "hole", "street", "pot", "players", "legal", "button_seat"):
        if key not in d:
            c.fail(f"missing field {key!r}")
    raw_players = c.lst(d["players"], "players", 2, MAX_SEATS)
    n = len(raw_players)
    players = []
    for i, p in enumerate(raw_players):
        w = f"players[{i}]"
        p = c.obj(p, w)
        if c.seat(p.get("seat"), f"{w}.seat", n) != i:
            c.fail("players must be listed in seat order (players[i].seat == i)")
        hole = p.get("hole")
        players.append(PlayerView(
            seat=i, name=c.str_(p.get("name"), f"{w}.name"), stack=c.int_(p.get("stack"), f"{w}.stack"),
            bet=c.int_(p.get("bet"), f"{w}.bet"), total_in=c.int_(p.get("total_in"), f"{w}.total_in"),
            in_hand=c.bool_(p.get("in_hand"), f"{w}.in_hand"), all_in=c.bool_(p.get("all_in"), f"{w}.all_in"),
            position=c.str_(p.get("position"), f"{w}.position"),
            hole=tuple(c.cards(hole, f"{w}.hole", 2, 2)) if hole else None,
            start_stack=c.int_(p.get("start_stack", 0), f"{w}.start_stack")))
    sb, bb = c.int_(d["sb"], "sb"), c.int_(d["bb"], "bb", 1)
    if sb > bb:
        c.fail("sb must not exceed bb")
    hero_seat = c.seat(d["hero_seat"], "hero_seat", n)
    extra = d.get("extra") if isinstance(d.get("extra"), dict) else {}
    to_act = next((src[k] for src in (d, extra) for k in ("to_act", "acting_seat") if src.get(k) is not None), None)
    if to_act is not None and c.seat(to_act, "to_act", n) != hero_seat:
        c.fail("hero_seat is not the seat to act")
    street = d["street"]
    if street not in BOARD_LEN:
        c.fail(f"street must be one of {', '.join(STREETS)}")
    hole = c.cards(d["hole"], "hole", 2, 2)
    board = c.cards(d.get("board", []), "board", 0, 5)
    if len(board) != BOARD_LEN[street]:
        c.fail(f"board has {len(board)} cards but street is {street} ({BOARD_LEN[street]} expected)")
    hp = players[hero_seat]
    if hp.hole is not None and set(hp.hole) != set(hole):
        c.fail("players[hero_seat].hole does not match hole")
    _unique(c, [hole + board] + [list(p.hole) for p in players if p.hole and p.seat != hero_seat])
    if not hp.in_hand or hp.all_in:
        c.fail("hero is not in the hand or is already all-in")
    la = c.obj(d["legal"], "legal")
    legal = LegalActions(can_fold=c.bool_(la.get("can_fold"), "legal.can_fold"),
                         can_check=c.bool_(la.get("can_check"), "legal.can_check"),
                         call_amount=c.int_(la.get("call_amount"), "legal.call_amount"),
                         can_raise=c.bool_(la.get("can_raise"), "legal.can_raise"),
                         min_raise_to=c.int_(la.get("min_raise_to"), "legal.min_raise_to"),
                         max_raise_to=c.int_(la.get("max_raise_to"), "legal.max_raise_to"))
    if not (legal.can_fold or legal.can_check or legal.can_raise or legal.call_amount):
        c.fail("no legal action for hero_seat (not hero's turn?)")
    if legal.can_check and legal.call_amount:
        c.fail("legal.can_check with a call_amount > 0")
    if legal.call_amount > hp.stack:
        c.fail("legal.call_amount exceeds hero's stack")
    if not legal.min_raise_to <= legal.max_raise_to <= hp.stack + hp.bet:
        c.fail("legal raise bounds must satisfy min_raise_to <= max_raise_to <= stack + bet")
    if legal.can_raise and legal.min_raise_to <= hp.bet:
        c.fail("legal.min_raise_to must exceed hero's current bet")
    return GameView(hand_id=c.str_(d.get("hand_id", "h"), "hand_id", 200), sb=sb, bb=bb, hero_seat=hero_seat,
                    hole=tuple(hole), board=board, street=street, pot=c.int_(d["pot"], "pot"), players=players,
                    actions=c.actions(d.get("actions", []), n), legal=legal,
                    button_seat=c.seat(d["button_seat"], "button_seat", n),
                    platform=c.str_(d.get("platform", "api"), "platform"),
                    table_id=c.str_(d.get("table_id", "t1"), "table_id", 200))


def history_from_dict(d: dict, require_hero: bool = True) -> HandHistory:
    """Build and fully validate a HandHistory from its JSON form (BadRequest, code 400)."""
    c = _Check(400)
    d = c.obj(d, "body")
    fields = {f.name for f in dataclasses.fields(HandHistory)}
    unknown = set(d) - fields
    if unknown:
        c.fail(f"unknown field {sorted(unknown)[0]!r}")
    for key in ("hand_id", "sb", "bb", "button_seat", "names", "positions", "start_stacks", "actions", "board"):
        if key not in d:
            c.fail(f"missing field {key!r}")
    if require_hero and d.get("hero_seat") is None:
        c.fail("missing field 'hero_seat'")
    names = [c.str_(x, f"names[{i}]", 64) for i, x in enumerate(c.lst(d["names"], "names", 2, MAX_SEATS))]
    n = len(names)
    positions = [c.str_(x, f"positions[{i}]") for i, x in enumerate(c.lst(d["positions"], "positions", n, n))]
    stacks = [c.int_(x, f"start_stacks[{i}]") for i, x in enumerate(c.lst(d["start_stacks"], "start_stacks", n, n))]
    sb, bb = c.int_(d["sb"], "sb"), c.int_(d["bb"], "bb", 1)
    hero = c.seat(d["hero_seat"], "hero_seat", n) if d.get("hero_seat") is not None else None
    board = c.cards(d["board"], "board", 0, 5)
    shown = {c.seat_key(k, f"shown[{k!r}]", n): tuple(c.cards(v, f"shown[{k!r}]", 2, 2))
             for k, v in c.obj(d.get("shown", {}), "shown").items()}
    hero_hole = tuple(c.cards(d["hero_hole"], "hero_hole", 2, 2)) if d.get("hero_hole") else None
    holes = dict(shown)
    if hero_hole is not None and hero is not None:
        if hero in holes and set(holes[hero]) != set(hero_hole):
            c.fail("hero_hole does not match shown[hero_seat]")
        holes[hero] = hero_hole
    _unique(c, [board] + [list(h) for h in holes.values()])

    def seat_map(x, what):
        return {c.seat_key(k, f"{what}[{k!r}]", n): c.num(v, f"{what}[{k!r}]") for k, v in c.obj(x, what).items()}

    out = dict(d)
    out.update(hand_id=c.str_(d["hand_id"], "hand_id", 200), sb=sb, bb=bb,
               button_seat=c.seat(d["button_seat"], "button_seat", n), names=names, positions=positions,
               start_stacks=stacks, actions=c.actions(d["actions"], n), board=board, shown=shown,
               net=seat_map(d.get("net", {}), "net"), hero_seat=hero,
               platform=c.str_(d.get("platform", "api"), "platform"),
               table_id=c.str_(d.get("table_id", "t1"), "table_id", 200), hero_hole=hero_hole,
               ev_net=seat_map(d["ev_net"], "ev_net") if d.get("ev_net") else None)
    if d.get("showdown") is not None:
        out["showdown"] = [c.seat(s, f"showdown[{i}]", n) for i, s in enumerate(c.lst(d["showdown"], "showdown", 0, n))]
    return HandHistory(**out)


# --------------------------------------------------------------------------- server
def _is_loopback(host: str) -> bool:
    h = (host or "").strip().strip("[]").lower()
    if h == "localhost":
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


def _log(msg: str) -> None:
    print(f"pokerbrain server: {msg}", file=sys.stderr, flush=True)


class _Server6(ThreadingHTTPServer):
    address_family = socket.AF_INET6


def _session_status(agent: Agent) -> str:
    br = getattr(agent, "bankroll", None)
    if br is None:
        return "ok"
    try:
        return str(br.session_status())
    except Exception:  # noqa: BLE001 - never lose a decision over the status line
        _log("bankroll.session_status() failed:\n" + traceback.format_exc())
        return "unknown"


def make_server(agent: Agent, host: str = "127.0.0.1", port: int = 8765, allow_remote: bool = False,
                token: Optional[str] = None) -> ThreadingHTTPServer:
    """Create (but do not start) the decision server.  `srv.token` holds the API token."""
    loopback = _is_loopback(host)
    if not loopback and not allow_remote:
        raise SystemExit(f"refusing to bind the decision server to {host!r}: it is not a loopback address. "
                         "Bind to 127.0.0.1, or pass allow_remote=True on a trusted network only.")
    if token is None:
        token = os.environ.get("POKERBRAIN_API_TOKEN") or None
        if token is None:
            token = secrets.token_urlsafe(24)
            _log(f"API token (send it as the {TOKEN_HEADER} header): {token}")
    token_b = token.encode()
    lock = threading.Lock()
    seen: "OrderedDict[tuple, bool]" = OrderedDict()
    hands = {"key": None, "index": -1}

    class H(BaseHTTPRequestHandler):
        timeout = REQUEST_TIMEOUT_S
        server_version = "pokerbrain"
        sys_version = ""

        def _send(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode()
            try:
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                self.close_connection = True

        def _guard(self, need_token: bool) -> bool:
            if self.headers.get("Origin") is not None:
                self._send(403, {"error": "cross-origin requests are not allowed"})
                return False
            if loopback and not _HOST_RE.fullmatch((self.headers.get("Host") or "").strip().lower()):
                self._send(403, {"error": "bad Host header"})
                return False
            if need_token:
                got = (self.headers.get(TOKEN_HEADER) or "").encode("utf-8", "surrogateescape")
                if not hmac.compare_digest(got, token_b):
                    self._send(401, {"error": f"missing or wrong {TOKEN_HEADER} header"})
                    return False
            return True

        def do_GET(self):  # noqa: N802
            if not self._guard(need_token=False):
                return
            if urlsplit(self.path).path == "/health":
                self._send(200, {"ok": True, "agent": agent.name})
            else:
                self._send(404, {"error": "unknown path"})

        def do_POST(self):  # noqa: N802
            if not self._guard(need_token=True):
                return
            path = urlsplit(self.path).path
            if path not in ("/decide", "/observe"):
                self._send(404, {"error": "unknown path"})
                return
            ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if ctype != "application/json":
                self._send(415, {"error": "Content-Type must be application/json"})
                return
            raw_len = self.headers.get("Content-Length")
            if raw_len is None:
                self._send(411, {"error": "Content-Length required"})
                return
            try:
                n = int(raw_len.strip())
            except ValueError:
                self._send(400, {"error": "bad Content-Length"})
                return
            if n < 0 or n > MAX_BODY:
                self._send(413, {"error": f"body must be 0..{MAX_BODY} bytes"})
                return
            try:
                raw = self.rfile.read(n)
            except (TimeoutError, OSError):
                self.close_connection = True
                return
            if len(raw) != n:
                self._send(400, {"error": "incomplete body"})
                return
            try:
                payload = json.loads(raw)
            except (ValueError, RecursionError):
                self._send(400, {"error": "body is not valid JSON"})
                return
            if not isinstance(payload, dict):
                self._send(400, {"error": "body must be a JSON object"})
                return
            try:
                out = self._decide(payload) if path == "/decide" else self._observe(payload)
            except BadRequest as exc:
                self._send(exc.code, {"error": str(exc)})
            except Exception:  # noqa: BLE001 - agent/internal fault: details stay server-side
                _log(f"internal error on {path}:\n{traceback.format_exc()}")
                self._send(500, {"error": "internal error"})
            else:
                self._send(200, out)

        def _decide(self, payload: dict) -> dict:
            v = view_from_dict(payload)
            with lock:
                key = (v.table_id, v.hand_id)
                if key != hands["key"]:
                    hands["key"] = key
                    hands["index"] += 1
                    agent.new_hand(hands["index"])
                d = agent.act(v).normalized(v.legal)
                status = _session_status(agent)
            return {"kind": d.kind, "amount": d.amount, "source": d.source, "reason": d.reason,
                    "session_status": status}

        def _observe(self, payload: dict) -> dict:
            hh = history_from_dict(payload, require_hero=True)
            key = (hh.table_id, hh.hand_id)
            with lock:
                if key in seen:
                    seen.move_to_end(key)
                    return {"ok": True, "duplicate": True}
                agent.observe(hh, hh.hero_seat)
                seen[key] = True
                while len(seen) > DEDUP_KEEP:
                    seen.popitem(last=False)
            return {"ok": True}

        def log_message(self, *args):  # quiet
            pass

    cls = _Server6 if ":" in host else ThreadingHTTPServer
    srv = cls((host.strip("[]"), port), H)
    srv.daemon_threads = True
    srv.token = token
    return srv


def serve(agent: Agent, host: str = "127.0.0.1", port: int = 8765, allow_remote: bool = False) -> None:
    srv = make_server(agent, host, port, allow_remote=allow_remote)
    if not _is_loopback(host):
        _log(f"WARNING: listening on non-loopback address {host!r}; anyone who can reach it and knows the "
             "token can drive the agent (no TLS)")
    print(f"pokerbrain decision server on http://{host}:{srv.server_address[1]}  (agent: {agent.name})")
    try:
        srv.serve_forever()
    finally:
        srv.server_close()
