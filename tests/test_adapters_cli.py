"""Robustness tests for the I/O adapters (slumbot, acpc, manual) and the command line."""
import json
import os
import signal
import socket
import threading
import time

import pytest
import requests

from pokerbrain import __main__ as cli
from pokerbrain.adapters import acpc as A
from pokerbrain.adapters import slumbot as S
from pokerbrain.adapters.manual import build_spot
from pokerbrain.agents.base import Agent, CallingAgent
from pokerbrain.bankroll import Stakes
from pokerbrain.engine import HandState
from pokerbrain.cards import ALL_CARDS
from pokerbrain.view import Decision


# ====================================================================== slumbot
@pytest.mark.parametrize("bad", ["b", "b300x", "x", "kb", "b200c/k/k/k/k"])
def test_slumbot_tokens_malformed_raise_slumbot_error(bad):
    with pytest.raises(S.SlumbotError):
        S._tokens(bad)


@pytest.mark.parametrize("hole,board,action,pos", [
    (["Ah", "Kd"], [], "kk", 1),                 # illegal: SB cannot check facing the big blind
    (["Ah", "Kd"], [], "b200ck", 1),             # missing '/': check lands on the wrong street
    (["Ah", "Kd"], [], "b200/c", 1),             # '/' before the street is complete
    (["Ah", "Kd"], [], "b200/", 1),              # trailing '/' before the street is complete
    (["Ah", "Kd"], [], "fk", 1),                 # action after the hand ended
    (["Ah", "Kd"], [], "b50000", 1),             # raise above the stack
    (["Ah", "Ah"], [], "", 1),                   # duplicate card
    (["Ah", "Kd"], ["Qh", "7h"], "", 1),         # impossible board size
    (["Ah", "Kd"], ["Ah", "7h", "2c"], "", 1),   # hole/board overlap
    (["Zz", "Kd"], [], "", 1),                   # bad card
    (["Ah", "Kd"], [], "", 2),                   # bad client_pos
])
def test_slumbot_rebuild_malformed_raise_slumbot_error(hole, board, action, pos):
    with pytest.raises(S.SlumbotError):
        S.rebuild(hole, board, action, pos, "x")


def test_slumbot_rebuild_accepts_legal_strings():
    for board, action, pos in [([], "", 1), ([], "b300", 0), (["Qh", "7h", "2c", "3d", "9s"], "b20000c///", 1),
                               (["Qh", "7h", "2c"], "b300c/kb19700", 1), (["Qh", "7h", "2c"], "cb300c/", 0)]:
        S.rebuild(["Ah", "Kd"], board, action, pos, "x")


def test_slumbot_session_repr_hides_secrets():
    r = repr(S.SlumbotSession(username="me", password="hunter2", token="tok-123"))
    assert "hunter2" not in r and "tok-123" not in r and "me" in r


class _Resp:
    def __init__(self, data, status=200):
        self._data, self.status_code = data, status

    def json(self):
        if isinstance(self._data, Exception):
            raise self._data
        return self._data


def _fake_post(script):
    """requests.post replacement: `script(path, body)` returns a _Resp or raises."""
    calls = []

    def post(url, json=None, timeout=None):
        path = url.rsplit("/", 1)[1]
        calls.append((path, dict(json or {})))
        return script(path, dict(json or {}), calls)
    return post, calls


FINAL = {"token": "T", "hole_cards": ["Ah", "Kd"], "board": ["Qh", "7h", "2c", "3d", "9s"],
         "action": "b300c/kk/kk/kk", "client_pos": 0, "winnings": -300, "baseline_winnings": 0,
         "bot_hole_cards": ["Qs", "Qc"]}


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(S.time, "sleep", lambda s: None)


def test_slumbot_lost_act_response_is_never_resent(monkeypatch, no_sleep):
    """First /act dies with a ConnectionError (maybe after the server applied it): the same incr must NOT
    be re-posted; the hand is counted as broken and a fresh /new_hand is started."""
    state = {"hands": 0, "dropped": False}

    def script(path, body, calls):
        if path == "new_hand":
            state["hands"] += 1
            if state["hands"] == 1:    # hero (BB) called a raise, flop, hero to act
                return _Resp({"token": "T", "hole_cards": ["Ah", "Kd"], "board": ["Qh", "7h", "2c"],
                              "action": "b300c/", "client_pos": 0})
            return _Resp({"token": "T", "hole_cards": ["Ah", "Kd"], "board": [], "action": "b300",
                          "client_pos": 0})
        if path == "act" and not state["dropped"]:
            state["dropped"] = True
            raise requests.ConnectionError("response lost after the server applied the action")
        return _Resp(FINAL)

    post, calls = _fake_post(script)
    monkeypatch.setattr(S.requests, "post", post)
    s = S.run(CallingAgent(), 2, progress=False, pause=0)
    assert [p for p, _ in calls] == ["new_hand", "act", "new_hand", "act"]
    assert [b.get("incr") for p, b in calls if p == "act"] == ["k", "c"]     # 'k' was not re-sent
    assert s.broken_hands == 1 and s.hands == 1 and s.winnings == -300


def test_slumbot_run_survives_errors_relogs_in_and_returns_partial(monkeypatch, no_sleep):
    n = {"new_hand": 0}

    def script(path, body, calls):
        if path == "login":
            return _Resp({"token": "T1"})
        n["new_hand"] += 1
        if n["new_hand"] == 3:
            return _Resp({"error_msg": "Invalid token"}, status=400)
        if n["new_hand"] == 4:
            return _Resp(ValueError("<html>502 Bad Gateway</html>"), status=502)
        if n["new_hand"] == 5:
            return _Resp({"unexpected": True})
        # Slumbot (small blind) folds straight away
        return _Resp({"token": "T1", "hole_cards": ["Ah", "Kd"], "board": [], "action": "f", "client_pos": 0,
                      "winnings": 50, "baseline_winnings": 25})

    post, calls = _fake_post(script)
    monkeypatch.setattr(S.requests, "post", post)
    s = S.run(CallingAgent(), 7, username="me", password="pw", progress=False, pause=0)
    assert s.hands == 4 and s.broken_hands == 3 and not s.aborted
    assert s.winnings == 200 and s.baseline == 100
    assert [p for p, _ in calls].count("login") == 2                          # re-login after the token error
    assert [p for p, _ in calls].count("new_hand") == 7                       # nothing was retried


def test_slumbot_run_gives_up_after_consecutive_errors(monkeypatch, no_sleep):
    def script(path, body, calls):
        raise requests.ConnectionError("down")

    post, calls = _fake_post(script)
    monkeypatch.setattr(S.requests, "post", post)
    s = S.run(CallingAgent(), 100, progress=False, pause=0, max_consecutive_errors=3)
    assert s.aborted and s.broken_hands == 3 and len(calls) == 3


class _Boom(Agent):
    name = "boom"

    def act(self, view):
        raise RuntimeError("budget exceeded")


def test_slumbot_agent_exception_falls_back(monkeypatch, no_sleep):
    def script(path, body, calls):
        if path == "new_hand":    # hero BB faces a min-raise: calling 100 of 19900 is cheap -> call
            return _Resp({"token": "T", "hole_cards": ["Ah", "Kd"], "board": [], "action": "b200",
                          "client_pos": 0})
        return _Resp({**FINAL, "action": "b200c/kk/kk/kk", "winnings": -200})

    post, calls = _fake_post(script)
    monkeypatch.setattr(S.requests, "post", post)
    s = S.run(_Boom(), 1, progress=False, pause=0)
    assert s.agent_errors == 1 and s.hands == 1 and s.broken_hands == 0
    assert calls[1] == ("act", {"token": "T", "incr": "c"})


# ====================================================================== fallback policy
def _view_facing(bet_to: int):
    """Hero = big blind (seat 0) facing a preflop raise to `bet_to` (0 = limped to hero, check is free)."""
    h = HandState([20000, 20000], button=1, sb=50, bb=100, deck=list(ALL_CARDS))
    h.apply(Decision("raise", bet_to) if bet_to else Decision("call"))
    return h.view_for(0)


def test_fallback_policy():
    assert A.fallback_decision(_view_facing(0)).kind == "check"
    assert A.fallback_decision(_view_facing(1000)).kind == "call"    # 900 <= 5% of 19900
    assert A.fallback_decision(_view_facing(1100)).kind == "fold"    # 1000 > 995
    d, failed = A.guarded_act(_Boom(), _view_facing(20000))
    assert failed and d.kind == "fold"
    d, failed = A.guarded_act(CallingAgent(), _view_facing(300))
    assert not failed and d.kind == "call"


# ====================================================================== acpc
@pytest.mark.parametrize("line", ["MATCHSTATE:0:0:r200", "garbage", "MATCHSTATE:2:0::AhKh|",
                                  "MATCHSTATE:0:x::AhKh|", "MATCHSTATE:0:0::AhKh|:extra", ""])
def test_acpc_parse_matchstate_rejects_malformed(line):
    with pytest.raises(A.ACPCError):
        A.parse_matchstate(line)


def test_acpc_parse_matchstate_ok():
    assert A.parse_matchstate("MATCHSTATE:1:17:r300c/:|AhKh/Qh7h2c\r\n") == (1, 17, "r300c/", "|AhKh/Qh7h2c")


@pytest.mark.parametrize("cards,pos", [("AhAh|", 0), ("Zz9h|", 0), ("AhKh|/Qh7h", 0), ("AhKh|/Ah7h2c", 0),
                                       ("AhKh|", 1), ("AhK|", 0), ("AhKh|/Qh7h2c/3d9s", 0), ("AhKh", 0)])
def test_acpc_parse_cards_rejects_malformed(cards, pos):
    with pytest.raises(A.ACPCError):
        A.parse_cards(cards, pos)


def test_acpc_parse_cards_ok():
    assert A.parse_cards("AhKh|QdQc/Qh7h2c/3d/9s", 0) == (["Ah", "Kh"], [["Qd", "Qc"]],
                                                          ["Qh", "7h", "2c", "3d", "9s"])


@pytest.mark.parametrize("betting,stack", [("r200cx/c", 20000), ("rc", 20000), ("r200ck", 20000),
                                           ("r15000", 10000), ("r200c/c/", 20000), ("fc", 20000)])
def test_acpc_rebuild_rejects_bad_betting(betting, stack):
    with pytest.raises(A.ACPCError):
        A.rebuild(1, betting, ["Ah", "Kh"], ["Qh", "7h", "2c"], stack, 50, 100, "q")


def _dealer(lines, capture):
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]

    def run():
        c, _ = srv.accept()
        c.settimeout(0.3)
        try:
            capture.append(c.recv(100).decode().strip())
            for ln in lines:
                c.sendall((ln + "\r\n").encode())
                try:
                    d = c.recv(1000)
                    if d:
                        capture.extend(x for x in d.decode().split("\r\n") if x)
                except socket.timeout:
                    pass
        finally:
            c.close()
            srv.close()
    threading.Thread(target=run, daemon=True).start()
    return port


def test_acpc_play_skips_bad_input_and_returns_partial_results():
    sent = []
    port = _dealer(["MATCHSTATE:0:0:r200",                       # 4 fields -> skipped line
                    "hello dealer",                              # junk -> skipped line
                    "MATCHSTATE:1:1:rc:|AhKh",                   # limit raise -> hand 1 skipped, we answer f
                    "MATCHSTATE:1:1:rcx:|AhKh",                  # rest of the broken hand
                    "MATCHSTATE:1:2::|AhKh",                     # hand 2: we (button) act first
                    "MATCHSTATE:1:2:cr300:|AhKh",                # BB raises, we call
                    "MATCHSTATE:1:2:cr300c/:|AhKh/Qh7h2c",       # BB acts first postflop
                    "MATCHSTATE:1:2:cr300c/r900f:|AhKh/Qh7h2c",  # we folded (dealer view): -300
                    ], sent)
    res = A.play(CallingAgent(), "127.0.0.1", port, timeout=5)
    assert sent[0] == "VERSION:2:0:0"
    assert res["hands"] == 1 and res["net_chips"] == -300
    assert res["bad_lines"] == 2 and res["skipped_hands"] == 1 and "error" not in res
    assert "MATCHSTATE:1:1:rc:|AhKh:f" in sent
    assert "MATCHSTATE:1:2::|AhKh:c" in sent and "MATCHSTATE:1:2:cr300:|AhKh:c" in sent


def test_acpc_play_timeout_and_refused_return_partial():
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)                  # accepts (backlog) but never writes
    try:
        t0 = time.time()
        res = A.play(CallingAgent(), "127.0.0.1", srv.getsockname()[1], timeout=0.5)
        assert time.time() - t0 < 5 and res["hands"] == 0 and "error" in res
    finally:
        srv.close()
    free = socket.socket()
    free.bind(("127.0.0.1", 0))
    port = free.getsockname()[1]
    free.close()
    res = A.play(CallingAgent(), "127.0.0.1", port, timeout=0.5)
    assert res["hands"] == 0 and "error" in res


# ====================================================================== manual
def test_manual_pipe_must_close_exactly_one_street():
    with pytest.raises(ValueError, match="not complete"):
        build_spot("AhKh", "Qh7h2c3d", "r2.5 c | x | b5", hero_is_button=False)
    with pytest.raises(ValueError, match="'\\|'"):     # streets merged without '|'
        build_spot("AhKh", "Qh7h2c", "r2.5 c x b3", hero_is_button=False)
    with pytest.raises(ValueError):
        build_spot("AhKh", "Qh7h2c3d9s", "r2.5 c | x x | x x | x x | x", hero_is_button=False)


def test_manual_board_must_match_street():
    with pytest.raises(ValueError, match="needs 4 cards"):   # a turn card would be invented
        build_spot("AhKh", "Qh7h2c", "r2.5 c | x x |", hero_is_button=True)
    with pytest.raises(ValueError, match="needs 0 cards"):   # board would be dropped
        build_spot("AhKh", "Qh7h2c3d9s", "r2.5", hero_is_button=False)
    with pytest.raises(ValueError, match="needs 3 cards"):
        build_spot("AhKh", "", "r2.5 c |", hero_is_button=False)


@pytest.mark.parametrize("hole,board", [("AhKh", "Ah7h2c"), ("AhAh", ""), ("AhKh", "Qh7h7h")])
def test_manual_duplicate_cards_rejected(hole, board):
    with pytest.raises(ValueError, match="twice"):
        build_spot(hole, board, "r2.5 c |" if board else "", hero_is_button=False)


def test_manual_allin_and_sizing_forms():
    h = build_spot("AhKh", "", "allin", hero_is_button=True)          # hero (button) shoves 100bb
    assert h.street_bets[0] == 10000 and h.to_act == 1
    h = build_spot("AhKh", "", "shove", hero_is_button=False)         # villain shoves, hero to act
    assert h.street_bets[1] == 10000 and h.to_act == 0
    h = build_spot("AhKh", "", "r2.5 allin", hero_is_button=False)
    assert h.street_bets[0] == 10000 and h.to_act == 1
    for acts in ("r3 b9", "raise 3 bet 9", "r 3 b9bb", "raise3 raise 9"):
        h = build_spot("AhKh", "", acts, hero_is_button=True)
        assert h.street_bets == [300, 900], acts
    h = build_spot("AhKh", "Qh7h2c", "r2.5 c | x bet 10", hero_is_button=False)
    assert h.street == "flop" and h.street_bets == [0, 1000] and h.to_act == 0
    h = build_spot("JcJd", "Kd8c4h2s7d", "r2.5 c | x x | x x | x b10", False)    # README example
    assert h.street == "river" and h.to_act == 0


@pytest.mark.parametrize("hole,board,actions", [
    ("AhKh", "", "r"), ("AhKh", "", "r2,5"), ("AhKh", "", "zz"), ("AhKh", "", "r1.5"),
    ("AhKh", "Qh7h2c", "c x | b150"), ("AhKh", "", "f c"), ("AKs", "", ""), ("AhKhQh", "", ""),
    ("AhKh", "", "c c"), ("AhKh", "", "x"),
])
def test_manual_errors_are_one_line_value_errors(hole, board, actions):
    with pytest.raises(ValueError) as ei:
        build_spot(hole, board, actions, hero_is_button=True)
    assert "\n" not in str(ei.value) and str(ei.value)


def test_cli_analyze_error_exits_2_without_traceback(monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_dotenv", lambda *a, **k: None)
    with pytest.raises(SystemExit) as ei:
        cli.main(["analyze", "--hole", "AhKh", "--board", "Qh7h2c", "--actions", "r2.5 c | x x |"])
    assert ei.value.code == 2
    err = capsys.readouterr().err
    assert "error:" in err and "Traceback" not in err and "needs 4 cards" in err


# ====================================================================== __main__
@pytest.mark.parametrize("bad", ["1-2", "1", "0/0", "2/1", "1/2/4", "a/b", "-1/2", "nan/1", "1/inf", ""])
def test_stakes_validator_rejects(bad):
    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        cli.stakes_arg(bad)


def test_stakes_validator_accepts():
    assert cli.stakes_arg("0.05/0.1") == (0.05, 0.1)
    assert cli.stakes_arg("1/1") == (1.0, 1.0)


def test_cli_rejects_bad_stakes_and_variant(capsys):
    ap = cli.build_parser()
    with pytest.raises(SystemExit) as ei:
        ap.parse_args(["sim", "--stakes", "1-2"])
    assert ei.value.code == 2
    with pytest.raises(SystemExit) as ei:
        ap.parse_args(["sim", "--agent", "opus", "--variant", "nope"])
    assert ei.value.code == 2
    err = capsys.readouterr().err
    assert "--variant" in err and "v13_final" in err            # lists the valid names
    a = ap.parse_args(["sim", "--variant", "v13_final", "--stakes", "0.05/0.1"])
    assert a.variant == "v13_final" and a.stakes == (0.05, 0.1)


def test_cli_acpc_and_serve_options():
    ap = cli.build_parser()
    a = ap.parse_args(["acpc", "--host", "h", "--port", "1"])
    assert a.hands is None and (a.stack, a.sb, a.bb) == (20000, 50, 100)
    a = ap.parse_args(["acpc", "--host", "h", "--port", "1", "--stack", "10000", "--sb", "25", "--bb", "50"])
    assert (a.stack, a.sb, a.bb) == (10000, 25, 50)
    assert ap.parse_args(["serve", "--allow-remote"]).allow_remote is True
    assert ap.parse_args(["serve"]).allow_remote is False


def _run_main(monkeypatch, argv, agent_factory=lambda: CallingAgent("PokerBrain")):
    seen = {}

    def fake_make_agent(kind, db, bankroll=None, seed=0, variant=None, escalation=None, stakes=None):
        seen.update(kind=kind, bankroll=bankroll, stakes=stakes, variant=variant, escalation=escalation, db=db)
        return agent_factory()
    monkeypatch.setattr(cli, "make_agent", fake_make_agent)
    monkeypatch.setattr(cli, "load_dotenv", lambda *a, **k: None)
    cli.main(argv)
    return seen


def test_cli_stakes_bankroll_router_wiring(monkeypatch, capsys):
    seen = _run_main(monkeypatch, ["sim", "--agent", "opus", "--hands", "1", "--field", "nit"])
    assert seen["stakes"] is None and seen["bankroll"] is None
    assert "no --stakes given" in capsys.readouterr().err
    seen = _run_main(monkeypatch, ["sim", "--agent", "ultimate", "--stakes", "0.5/1", "--hands", "1",
                                   "--field", "nit"])
    assert isinstance(seen["stakes"], Stakes) and (seen["stakes"].sb, seen["stakes"].bb) == (0.5, 1.0)
    assert seen["bankroll"] is None and "no --stakes" not in capsys.readouterr().err
    seen = _run_main(monkeypatch, ["sim", "--bankroll", "0", "--stakes", "1/2", "--hands", "1", "--field", "nit"])
    assert seen["bankroll"] is not None and seen["bankroll"].bankroll == 0 and seen["bankroll"].stakes.bb == 2
    _run_main(monkeypatch, ["sim", "--agent", "quant", "--router", "key", "--hands", "1", "--field", "nit"])
    assert "--router only applies" in capsys.readouterr().err


def test_make_agent_passes_stakes_to_opus(monkeypatch):
    import pokerbrain.agents.llm_agents as L
    import pokerbrain.llm.claude as C
    got = {}

    class FakeOpus:
        def __init__(self, decider, **kw):
            got.update(kw)
    monkeypatch.setattr(L, "OpusAgent", FakeOpus)
    monkeypatch.setattr(L, "api_decider", lambda client: None)
    monkeypatch.setattr(C, "OpusClient", lambda **kw: None)
    st = Stakes(sb=0.5, bb=1.0)
    cli.make_agent("opus", db=None, bankroll=None, stakes=st)
    assert got["stakes"] is st and "override_gate" in got and "bankroll" in got


def test_autosave_wrapper_delegates_and_saves_every_n():
    saves = []
    inner = CallingAgent("Inner")
    w = cli._Autosave(inner, lambda: saves.append(1), every=3)
    assert w.name == "Inner" and w.act(_view_facing(0)).kind == "check"
    for _ in range(7):
        w.observe(None, 0)
    assert len(saves) == 2


def test_sigterm_saves_db_and_restores_handlers(monkeypatch, tmp_path, capsys):
    if not hasattr(signal, "SIGTERM"):
        pytest.skip("no SIGTERM")
    before = signal.getsignal(signal.SIGTERM)
    db = tmp_path / "db.json"

    class KillAt(CallingAgent):
        n = 0

        def act(self, view):
            KillAt.n += 1
            if KillAt.n > 300:
                pytest.fail("SIGTERM did not interrupt the run")
            if KillAt.n == 30:
                os.kill(os.getpid(), signal.SIGTERM)
                time.sleep(0.5)          # the handler raises KeyboardInterrupt here
            return super().act(view)
    with pytest.raises(SystemExit) as ei:
        _run_main(monkeypatch, ["sim", "--hands", "100000", "--field", "nit", "--db", str(db)],
                  agent_factory=lambda: KillAt("PokerBrain"))
    assert ei.value.code == 130
    assert db.exists() and json.loads(db.read_text())["platform"] == "sim"
    assert signal.getsignal(signal.SIGTERM) == before
