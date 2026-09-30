"""I/O hardening: spend ledger (budget.py), HTTP decision server (server.py), file bridge (bridge.py)."""
import dataclasses
import http.client
import io
import json
import multiprocessing
import os
import socket
import sys
import threading
import time

import pytest

from pokerbrain import bridge as B
from pokerbrain.adapters import server as S
from pokerbrain.agents.base import Agent
from pokerbrain.agents.quant_agent import QuantAgent
from pokerbrain.bankroll import BankrollManager, Stakes
from pokerbrain.cards import ALL_CARDS
from pokerbrain.engine import HandState
from pokerbrain.llm.budget import Budget, BudgetExceeded, load_dotenv
from pokerbrain.opponents import OpponentDB
from pokerbrain.view import Decision


# =========================================================================== budget
def _record_worker(path, n, usd):
    b = Budget(cap_usd=1e6, path=path)
    errors = 0
    for _ in range(n):
        try:
            b.record("m", usd)
        except Exception:  # noqa: BLE001
            errors += 1
    return errors


def _reserve_worker(path, n, usd):
    b = Budget(cap_usd=10.0, path=path)
    ok = 0
    for _ in range(n):
        try:
            rid = b.reserve(usd)
        except BudgetExceeded:
            continue
        b.settle(rid, "m", usd)
        ok += 1
    return ok


def _pool():
    return multiprocessing.get_context("fork").Pool(4)


def test_budget_cross_process_record_total_exact(tmp_path):
    path = str(tmp_path / "spend.json")
    with _pool() as pool:
        errors = pool.starmap(_record_worker, [(path, 200, 0.25)] * 4)
    assert errors == [0, 0, 0, 0]
    data = json.load(open(path))
    assert data["total_usd"] == 200.0 and data["calls"] == 800 and data["pending"] == {}
    assert Budget(cap_usd=1e6, path=path).total() == 200.0


def test_budget_cross_process_cap_is_hard(tmp_path):
    path = str(tmp_path / "spend.json")
    with _pool() as pool:
        ok = pool.starmap(_reserve_worker, [(path, 30, 0.25)] * 4)
    assert sum(ok) == 40                       # exactly $10.00 / $0.25
    assert Budget(cap_usd=10.0, path=path).total() == 10.0


def test_budget_reservations_and_record_enforce_cap(tmp_path):
    b = Budget(cap_usd=1.0, path=str(tmp_path / "s.json"))
    r1 = b.reserve(0.6)
    with pytest.raises(BudgetExceeded):
        b.reserve(0.6)                         # 0.6 already pending
    b.release(r1)
    r2 = b.reserve(0.6)
    b.settle(r2, "m", 0.5)
    assert b.total() == 0.5 and b.reserved() == 0.0
    b.check(0.3)                               # legacy pair: check reserves, record settles
    assert abs(b.reserved() - 0.3) < 1e-12
    with pytest.raises(BudgetExceeded):
        b.record("m", 0.7)                     # written, then reported over the cap
    assert abs(b.total() - 1.2) < 1e-12 and b.reserved() == 0.0
    with pytest.raises(BudgetExceeded):
        b.check(0.0)


def test_budget_session_cap(tmp_path, monkeypatch):
    path = str(tmp_path / "s.json")
    b = Budget(cap_usd=100.0, path=path, session_cap_usd=0.5)
    b.check(0.1)
    b.record("m", 0.4)
    with pytest.raises(BudgetExceeded):
        b.check(0.2)
    monkeypatch.setenv("POKERBRAIN_SESSION_SPEND_USD", "0.25")
    other = Budget(cap_usd=100.0, path=path)   # same ledger, new session
    assert other.session_cap == 0.25
    other.check(0.2)


def test_budget_corrupt_ledger_fails_closed(tmp_path):
    path = tmp_path / "spend.json"
    path.write_text('{"total_usd": 4.99, "by_mo')             # truncated
    b = Budget(cap_usd=5.0, path=str(path))
    with pytest.raises(BudgetExceeded):
        b.check(0.01)
    with pytest.raises(BudgetExceeded):
        b.total()
    with pytest.raises(BudgetExceeded):
        b.record("m", 0.01)
    assert path.read_text() == '{"total_usd": 4.99, "by_mo'   # never reset
    assert (tmp_path / "spend.json.bak").read_text() == '{"total_usd": 4.99, "by_mo'
    for bad in ("[]", "", '{"total_usd": "x"}', '{"total_usd": NaN}', "{}"):
        path.write_text(bad)
        with pytest.raises(BudgetExceeded):
            Budget(cap_usd=5.0, path=str(path)).check()


def test_budget_relative_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    b = Budget(cap_usd=1.0, path="spend.json")
    b.check(0.1)
    b.record("claude-opus-5-5", 0.3)
    assert json.load(open(tmp_path / "spend.json"))["total_usd"] == 0.3
    monkeypatch.setenv("POKERBRAIN_SPEND_FILE", "sub/dir/spend.json")
    Budget(cap_usd=1.0).record("m", 0.1)
    assert (tmp_path / "sub" / "dir" / "spend.json").exists()


@pytest.mark.parametrize("raw", ["nan", "inf", "-1", "5 # dollars", "abc"])
def test_budget_invalid_cap_uses_default(monkeypatch, tmp_path, raw):
    monkeypatch.setenv("POKERBRAIN_MAX_SPEND_USD", raw)
    assert Budget(path=str(tmp_path / "s.json")).cap == 5.0
    assert Budget(cap_usd=float("nan"), path=str(tmp_path / "s.json")).cap == 5.0


def test_load_dotenv(tmp_path, monkeypatch, capsys):
    for k in ("PB_A", "PB_B", "PB_C", "PB_D", "PB_E", "PB_BAD"):
        monkeypatch.delenv(k, raising=False)
    env = tmp_path / ".env"
    env.write_text("export PB_A=sk-123\nPB_B=5 # dollars\nPB_C=\"x # y\"\nnot a line\nPB BAD=1\n"
                   "PB_D='unterminated\nPB_E=plain\n")
    load_dotenv(str(env))
    assert os.environ["PB_A"] == "sk-123" and os.environ["PB_B"] == "5"
    assert os.environ["PB_C"] == "x # y" and os.environ["PB_E"] == "plain"
    assert "PB_D" not in os.environ and "export PB_A" not in os.environ
    monkeypatch.delenv("PB_E")
    env.chmod(0o666)                            # world-writable: ignored
    load_dotenv(str(env))
    assert "PB_E" not in os.environ and "ignoring" in capsys.readouterr().err


# =========================================================================== server
TOKEN = "test-token"


class RecAgent(Agent):
    name = "rec"

    def __init__(self, bankroll=None):
        self.acts, self.observed, self.hands = [], [], []
        self.bankroll = bankroll
        self.fail = False

    def act(self, view):
        if self.fail:
            raise RuntimeError("secret /path/to/ledger")
        self.acts.append(view.hand_id)
        return Decision("call" if view.legal.call_amount else "check", source="rec")

    def observe(self, history, my_seat):
        self.observed.append((history.hand_id, my_seat))

    def new_hand(self, hand_index):
        self.hands.append(hand_index)


def _deck(prefix):
    return prefix + [c for c in ALL_CARDS if c not in prefix]


def _hand(hand_id="h1"):
    return HandState([10000, 10000], button=0, sb=50, bb=100, hand_id=hand_id,
                     deck=_deck(["Ah", "Kh", "2c", "3d", "Qh", "7h", "2d", "9s", "4c"]), names=["Hero", "V"])


def _view(hand_id="h1"):
    h = _hand(hand_id)
    return json.loads(json.dumps(h.view_for(h.to_act).to_dict()))


def _history(hand_id="h1", hero_seat=0):
    h = _hand(hand_id)
    h.apply(Decision("raise", 300))
    h.apply(Decision("fold"))
    hh = h.result.to_history(hero_seat=hero_seat)
    return json.loads(json.dumps(dataclasses.asdict(hh)))


@pytest.fixture
def srv():
    started = []

    def start(agent):
        s = S.make_server(agent, "127.0.0.1", 0, token=TOKEN)
        threading.Thread(target=s.serve_forever, daemon=True).start()
        started.append(s)
        return s.server_address[1]

    yield start
    for s in started:
        s.shutdown()
        s.server_close()


def _post(port, path, obj, ctype="application/json", token=TOKEN, headers=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    body = obj if isinstance(obj, (bytes, str)) else json.dumps(obj)
    h = {"Content-Type": ctype}
    if token is not None:
        h[S.TOKEN_HEADER] = token
    h.update(headers or {})
    c.request("POST", path, body=body, headers=h)
    r = c.getresponse()
    out = r.status, json.loads(r.read() or b"{}")
    c.close()
    return out


def _raw(port, data):
    s = socket.create_connection(("127.0.0.1", port), timeout=5)
    s.sendall(data)
    out = s.recv(65536)
    s.close()
    return out


def test_server_decide_and_auth(srv):
    br = BankrollManager(bankroll=1000, stakes=Stakes(sb=1, bb=2))
    port = srv(QuantAgent("PB", db=OpponentDB(), bankroll=br, seed=0))
    code, out = _post(port, "/decide", _view())
    assert code == 200 and set(out) == {"kind", "amount", "source", "reason", "session_status"}
    assert out["session_status"] == "ok" and out["kind"] in ("fold", "call", "raise")
    assert _post(port, "/decide", _view(), token=None)[0] == 401
    assert _post(port, "/decide", _view(), token="wrong")[0] == 401
    assert _post(port, "/decide", _view(), headers={"Origin": "https://evil.example"})[0] == 403
    assert _post(port, "/decide", _view(), headers={"Host": "evil.example:8765"})[0] == 403
    assert _post(port, "/decide", _view(), headers={"Host": f"localhost:{port}"})[0] == 200
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    c.request("GET", "/health")
    r = c.getresponse()
    assert r.status == 200 and json.loads(r.read())["agent"] == "PB"


def test_server_rejects_non_json_and_bad_length(srv):
    agent = RecAgent()
    port = srv(agent)
    assert _post(port, "/decide", json.dumps(_view()), ctype="text/plain")[0] == 415
    head = f"POST /decide HTTP/1.0\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n{S.TOKEN_HEADER}: {TOKEN}\r\n"
    for length, status in (("2000000", b" 413 "), ("-1", b" 413 "), ("abc", b" 400 "),
                           ("999999999999999999", b" 413 ")):
        t0 = time.monotonic()
        reply = _raw(port, (head + f"Content-Length: {length}\r\n\r\n{{}}").encode())
        assert status in reply.split(b"\r\n")[0], (length, reply[:80])
        assert time.monotonic() - t0 < 3
    assert agent.acts == []


def test_server_decide_validation_422(srv):
    agent = RecAgent()
    port = srv(agent)
    v = _view()
    bad = []
    x = json.loads(json.dumps(v))
    x["legal"].update(min_raise_to=50000, max_raise_to=200)
    bad.append(x)
    x = json.loads(json.dumps(v))
    x["legal"]["max_raise_to"] = 10 ** 6                      # above stack + bet
    bad.append(x)
    x = json.loads(json.dumps(v))
    x.update(street="flop", board=["Ah", "7h", "2c"])            # Ah duplicates the hole
    bad.append(x)
    x = json.loads(json.dumps(v))
    x.update(street="turn", board=["Qh", "7h", "2c"])            # 3 cards on the turn
    bad.append(x)
    x = json.loads(json.dumps(v))
    x["hole"] = "AhKh"
    bad.append(x)
    x = json.loads(json.dumps(v))
    x["hole"] = ["Ah", "Zz"]
    bad.append(x)
    x = json.loads(json.dumps(v))
    x["hero_seat"] = 5
    bad.append(x)
    x = json.loads(json.dumps(v))
    x["to_act"] = 1                                              # payload says seat 1 acts
    bad.append(x)
    x = json.loads(json.dumps(v))
    x["pot"] = "150"
    bad.append(x)
    h = _hand()
    bad.append(json.loads(json.dumps(h.view_for(1).to_dict())))  # not seat 1's turn: no legal action
    for x in bad:
        code, out = _post(port, "/decide", x)
        assert code == 422 and "error" in out, (x, code, out)
    assert _post(port, "/decide", '{"pot": NaN}')[0] == 422
    assert _post(port, "/decide", "not json")[0] == 400
    assert agent.acts == []
    assert _post(port, "/decide", v)[0] == 200 and agent.acts == ["h1"]


def test_server_new_hand_and_internal_error(srv):
    agent = RecAgent(bankroll=type("BR", (), {"session_status": lambda self: "stop_loss"})())
    port = srv(agent)
    for hid in ("h1", "h1", "h2", "h2", "h3"):
        code, out = _post(port, "/decide", _view(hid))
        assert code == 200 and out["session_status"] == "stop_loss"
    assert agent.hands == [0, 1, 2]
    agent.fail = True
    code, out = _post(port, "/decide", _view("h4"))
    assert code == 500 and out == {"error": "internal error"}


def test_server_observe_validation_and_dedup(srv):
    agent = RecAgent()
    port = srv(agent)
    hh = _history("h1", hero_seat=1)
    no_hero = dict(hh)
    no_hero.pop("hero_seat")
    code, out = _post(port, "/observe", no_hero)
    assert code == 400 and "hero_seat" in out["error"]
    bad_seat = json.loads(json.dumps(hh))
    bad_seat["actions"].append(dict(bad_seat["actions"][-1], seat=7))
    bad_net = json.loads(json.dumps(hh))
    bad_net["net"] = {"0": "abc", "1": 5}
    bad_kind = json.loads(json.dumps(hh))
    bad_kind["actions"][-1]["kind"] = "shove"
    bad_len = json.loads(json.dumps(hh))
    bad_len["positions"] = ["BTN"]
    bad_hero = dict(hh, hero_seat=2)
    for x in (bad_seat, bad_net, bad_kind, bad_len, bad_hero):
        assert _post(port, "/observe", x)[0] == 400
    assert agent.observed == []
    assert _post(port, "/observe", hh) == (200, {"ok": True})
    assert _post(port, "/observe", hh) == (200, {"ok": True, "duplicate": True})
    assert _post(port, "/observe", dict(hh, table_id="t2"))[1] == {"ok": True}
    assert agent.observed == [("h1", 1), ("h1", 1)]


def test_server_refuses_remote_bind():
    with pytest.raises(SystemExit):
        S.make_server(RecAgent(), "0.0.0.0", 0, token=TOKEN)


# =========================================================================== bridge
def _pending(q):
    return sorted(os.listdir(os.path.join(q, "pending")))


def _wait_for_pending(q, timeout=5.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        live = [f for f in _pending(q) if B.RID_RE.fullmatch(f[:-5]) and int(f[:20]) >= B._read_run_id(q)]
        if live:
            return live[0][:-5]
        time.sleep(0.01)
    raise AssertionError("no live request appeared")


def test_bridge_skips_stale_requests(tmp_path, capsys):
    q = str(tmp_path / "q")
    os.makedirs(os.path.join(q, "pending"))
    open(os.path.join(q, "pending", "235900-aaaaaa.json"), "w").write('{"id": "235900-aaaaaa", "user": "OLD"}')
    open(os.path.join(q, "MATCH_DONE"), "w").write("previous match")
    d = B.FileBridgeDecider(q, timeout=10, poll=0.02)
    assert _pending(q) == [] and not os.path.exists(os.path.join(q, "MATCH_DONE"))
    assert d.timeout == 10 and B.FileBridgeDecider.__init__.__defaults__[0] == 120.0
    # stale requests appearing after start (crashed run, old id format) are never served
    stale = "%020d-bbbbbb" % (d.run_id - 1)
    for rid in (stale, "000100-cccccc"):
        open(os.path.join(q, "pending", rid + ".json"), "w").write(json.dumps({"id": rid, "user": "STALE"}))
    result = {}
    t = threading.Thread(target=lambda: result.update(ans=d("sys", "LIVE REQUEST", {"k": 1})))
    t.start()
    live = _wait_for_pending(q)
    B._next(q, wait=2)
    out = capsys.readouterr().out
    assert f"REQUEST_ID: {live}" in out and "LIVE REQUEST" in out and "STALE" not in out
    B._answer(q, live, '{"action_id": "A1"}')
    t.join(5)
    assert result["ans"] == {"action_id": "A1"}
    assert live + ".json" not in _pending(q) and live + ".json" in os.listdir(os.path.join(q, "done"))


def test_bridge_timeout_and_malformed_answer_cleanup(tmp_path):
    q = str(tmp_path / "q")
    d = B.FileBridgeDecider(q, timeout=0.2, poll=0.02)
    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        d("sys", "user", {})
    assert time.monotonic() - t0 < 2
    assert _pending(q) == [] and len(os.listdir(os.path.join(q, "done"))) == 1

    d = B.FileBridgeDecider(q, timeout=5, poll=0.02)
    err = {}

    def call():
        try:
            d("sys", "user", {})
        except ValueError as exc:
            err["e"] = exc
    t = threading.Thread(target=call)
    t.start()
    rid = _wait_for_pending(q)
    open(os.path.join(q, "answers", rid + ".json"), "w").write("{not json")
    t.join(5)
    assert "malformed" in str(err["e"]) and _pending(q) == []


def test_bridge_answer_validation_and_cli(tmp_path, capsys, monkeypatch):
    q = str(tmp_path / "q")
    d = B.FileBridgeDecider(q, timeout=5, poll=0.02)
    with pytest.raises(B.BridgeError):
        B._answer(q, "../../escaped", '{"x": 1}')
    assert not (tmp_path / "escaped.json").exists()
    with pytest.raises(B.BridgeError):
        B._answer(q, "%020d-abcdef" % d.run_id, '{"x": 1}')        # well-formed but not pending
    assert B.main(["answer", q, "../../escaped", "{}"]) == 1
    assert B.main([]) == 2 and "usage" in capsys.readouterr().err
    assert B.main(["answer", q]) == 2
    assert B.main(["status", str(tmp_path / "missing")]) == 0
    assert "pending=0" in capsys.readouterr().out
    result = {}
    t = threading.Thread(target=lambda: result.update(ans=d("sys", "u", {})))
    t.start()
    rid = _wait_for_pending(q)
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"action_id": "A2"}'))
    assert B.main(["answer", q, rid, "-"]) == 0
    t.join(5)
    assert result["ans"] == {"action_id": "A2"}
    assert not [f for f in os.listdir(os.path.join(q, "answers")) if not f.endswith(".json")]
