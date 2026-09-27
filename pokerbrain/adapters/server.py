"""Local HTTP decision service (stdlib only).

  POST /decide   body: GameView as JSON (GameView.to_dict() shape)  -> {"kind","amount","source","reason"}
  POST /observe  body: HandHistory as JSON                          -> {"ok": true}
  GET  /health

Use it to connect the brain to environments that explicitly allow bots
(your own home-game server, bot competitions, research platforms).  It binds
to 127.0.0.1 by default.
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ..agents.base import Agent
from ..view import ActionRecord, GameView, HandHistory, LegalActions, PlayerView


def view_from_dict(d: dict) -> GameView:
    players = [PlayerView(**{**p, "hole": tuple(p["hole"]) if p.get("hole") else None}) for p in d["players"]]
    actions = [ActionRecord(**a) for a in d.get("actions", [])]
    return GameView(hand_id=d.get("hand_id", "h"), sb=d["sb"], bb=d["bb"], hero_seat=d["hero_seat"],
                    hole=tuple(d["hole"]), board=list(d.get("board", [])), street=d["street"], pot=d["pot"],
                    players=players, actions=actions, legal=LegalActions(**d["legal"]),
                    button_seat=d["button_seat"], platform=d.get("platform", "api"),
                    table_id=d.get("table_id", "t1"))


def history_from_dict(d: dict) -> HandHistory:
    d = dict(d)
    d["actions"] = [ActionRecord(**a) for a in d.get("actions", [])]
    d["shown"] = {int(k): tuple(v) for k, v in d.get("shown", {}).items()}
    d["net"] = {int(k): v for k, v in d.get("net", {}).items()}
    if d.get("ev_net"):
        d["ev_net"] = {int(k): v for k, v in d["ev_net"].items()}
    return HandHistory(**d)


def serve(agent: Agent, host: str = "127.0.0.1", port: int = 8765) -> None:
    class H(BaseHTTPRequestHandler):
        def _send(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            self._send(200, {"ok": True, "agent": agent.name}) if self.path == "/health" else self._send(404, {})

        def do_POST(self):  # noqa: N802
            n = int(self.headers.get("Content-Length", 0))
            try:
                payload = json.loads(self.rfile.read(n) or b"{}")
                if self.path == "/decide":
                    v = view_from_dict(payload)
                    d = agent.act(v).normalized(v.legal)
                    self._send(200, {"kind": d.kind, "amount": d.amount, "source": d.source, "reason": d.reason})
                elif self.path == "/observe":
                    hh = history_from_dict(payload)
                    agent.observe(hh, hh.hero_seat if hh.hero_seat is not None else 0)
                    self._send(200, {"ok": True})
                else:
                    self._send(404, {"error": "unknown path"})
            except Exception as exc:  # noqa: BLE001
                self._send(400, {"error": str(exc)})

        def log_message(self, *args):  # quiet
            pass

    print(f"pokerbrain decision server on http://{host}:{port}  (agent: {agent.name})")
    ThreadingHTTPServer((host, port), H).serve_forever()
