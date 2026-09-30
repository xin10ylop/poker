import random

from pokerbrain.adapters.acpc import rebuild as acpc_rebuild
from pokerbrain.adapters.acpc import to_acpc
from pokerbrain.adapters.slumbot import _tokens, rebuild as slum_rebuild, to_incr
from pokerbrain.bankroll import BankrollManager, Stakes, bankroll_for_ror, risk_of_ruin
from pokerbrain.cards import ALL_CARDS
from pokerbrain.engine import HandState
from pokerbrain.opponents import OpponentDB
from pokerbrain.quant import QuantEngine
from pokerbrain.ranges import Range
from pokerbrain.texture import hand_features
from pokerbrain.view import Decision


def _deck(prefix, seed=0):
    rest = [c for c in ALL_CARDS if c not in prefix]
    random.Random(seed).shuffle(rest)
    return prefix + rest


def test_range_parsing():
    r = Range.parse("TT+, AKs, A5s:0.5, K9s-KJs")
    cw = r.class_weights()
    assert cw["TT"] == 6 and cw["AA"] == 6 and cw["AKs"] == 4
    assert abs(cw["A5s"] - 2.0) < 1e-9
    assert set(k for k in cw if k.startswith("K")) == {"KK", "K9s", "KTs", "KJs"}


def test_hand_features():
    f = hand_features(("5c", "5d"), ["5h", "Ks", "9c"])
    assert f.made == "set" and f.strength_class == "strong"
    f = hand_features(("Ah", "Kh"), ["Qh", "7h", "2c"])
    assert f.nut_flush_draw and f.outs >= 9


def test_tracker_counts_preflop_and_cbet():
    # 3-handed: BTN opens, SB folds, BB calls; flop BB checks, BTN c-bets, BB folds
    d = _deck([])
    h = HandState([10000] * 3, button=0, sb=50, bb=100, deck=d, names=["Btn", "Sb", "Bb"])
    h.apply(Decision("raise", 250))
    h.apply(Decision("fold"))
    h.apply(Decision("call"))
    h.apply(Decision("check"))
    h.apply(Decision("raise", 300))
    h.apply(Decision("fold"))
    db = OpponentDB()
    db.update(h.result.to_history(hero_seat=None))
    btn, bb = db.get("Btn"), db.get("Bb")
    assert btn.counts["pfr"] == [1, 1] and btn.counts["steal"] == [1, 1]
    assert btn.counts["cbet"] == [1, 1]
    assert bb.counts["fold_to_cbet"] == [1, 1]
    assert bb.counts["vpip"] == [1, 1] and bb.counts["pfr"] == [0, 1]
    assert bb.counts["fold_to_steal"] == [0, 1]


def test_quant_basic_invariants():
    # hero has the nuts on the river facing a bet: calling/raising must beat folding
    d = _deck(["As", "Ks", "2c", "3d", "Qs", "Js", "Ts", "4h", "5d"])
    h = HandState([10000, 10000], button=0, sb=50, bb=100, deck=d, names=["Hero", "V"])
    h.apply(Decision("raise", 250)); h.apply(Decision("call"))
    h.apply(Decision("check")); h.apply(Decision("check"))
    h.apply(Decision("check")); h.apply(Decision("check"))
    h.apply(Decision("raise", 300))                       # V bets river
    rep = QuantEngine(OpponentDB()).analyze(h.view_for(0))
    fold = next(o for o in rep.options if o.label == "fold")
    assert fold.ev == 0
    assert rep.best.label != "fold"
    assert rep.nut_info == "the NUTS"


def test_slumbot_parsing_and_rebuild():
    assert _tokens("b200c/kb400") == [(0, "raise", 200), (0, "call", 0), (1, "check", 0), (1, "raise", 400)]
    h = slum_rebuild(["Ah", "Kh"], ["Qh", "7h", "2c"], "b200c/kb400", 1, "t")
    assert h.street_bets == [400, 0] and h.total_in == [600, 200] and h.to_act == 1
    assert to_incr(Decision("raise", 1200)) == "b1200" and to_incr(Decision("call")) == "c"


def test_acpc_matches_slumbot():
    h = acpc_rebuild(1, "r200c/cr600", ["Ah", "Kh"], ["Qh", "7h", "2c"], 20000, 50, 100, "x")
    assert h.street_bets == [400, 0] and h.total_in == [600, 200]
    h2 = acpc_rebuild(0, "r200c/c", ["Ah", "Kh"], ["Qh", "7h", "2c"], 20000, 50, 100, "y")
    assert h2.to_act == 1
    h3 = acpc_rebuild(1, "r200c/c", ["Ah", "Kh"], ["Qh", "7h", "2c"], 20000, 50, 100, "z")
    assert h3.to_act == 0
    assert to_acpc(Decision("raise", 300), h3) == "r500"   # 200 already in + 300 on this street


def test_bankroll_math():
    assert abs(bankroll_for_ror(5, 90, 0.01) - 3730) < 5
    assert risk_of_ruin(5, 90, 3730) < 0.0101
    br = BankrollManager(bankroll=500, stakes=Stakes(sb=1, bb=2))       # 2.5 buy-ins: short
    deep = BankrollManager(bankroll=500000, stakes=Stakes(sb=1, bb=2))
    assert br.risk_aversion() > deep.risk_aversion()
    assert br.risk_penalty(1e8) > deep.risk_penalty(1e8)


def test_opus_override_gate():
    from pokerbrain.agents.llm_agents import EscalationPolicy, OpusAgent
    d = _deck(["As", "Ks", "2c", "3d", "Qs", "Js", "Ts", "4h", "5d"])
    h = HandState([10000, 10000], button=0, sb=50, bb=100, deck=d, names=["Hero", "V"])
    h.apply(Decision("raise", 250)); h.apply(Decision("call"))
    for _ in range(4):
        h.apply(Decision("check"))
    h.apply(Decision("raise", 300))                       # V bets river into hero's nuts
    view = h.view_for(0)

    def play(engine_p):
        box = {}

        def decider(system, user, meta):              # Opus stand-in: prefers a non-engine action
            rep = box["agent"].last_report
            other = next(o.id for o in rep.options if o.id != rep.best.id and o.label != "fold")
            box["other"] = rep.option(other).label
            mix = [{"id": other, "p": 1 - engine_p}] + ([{"id": rep.best.id, "p": engine_p}] if engine_p else [])
            return {"action_id": other, "mix": mix}
        box["agent"] = OpusAgent(decider, variant="v13_final", escalation=EscalationPolicy(mode="all"),
                                 override_gate=0.2, name="Hero")
        dec = box["agent"].act(view)
        return dec.reason, box["other"], box["agent"]

    chosen, other, agent = play(0.4)                       # hedged override -> engine's pick is played
    assert chosen == agent.last_report.best.label and chosen != other and agent.gated == 1
    chosen, other, agent = play(0.1)                       # decisive override -> Opus's pick is played
    assert chosen == other and agent.gated == 0
    chosen, other, agent = play(0.0)
    assert chosen == other and agent.log[-1]["gated"] is False
    postflop = EscalationPolicy(mode="postflop")
    assert postflop.should(view, None)
    assert not postflop.should(view, None, chip_value=0.0001)     # pot worth less than 3 model calls
    pre = HandState([10000, 10000], button=0, sb=50, bb=100, deck=_deck([]), names=["Hero", "V"])
    assert not postflop.should(pre.view_for(pre.to_act), None)


def test_allin_has_no_reraise_branch():
    # regression (found by the live Opus decider): an all-in can't be re-raised, so hands that would
    # raise must count as calls - not as "hero loses his bet"
    d = _deck(["As", "Ks", "2c", "3d", "Qs", "Js", "Ts", "4h", "5d"])
    h = HandState([10000, 10000], button=0, sb=50, bb=100, deck=d, names=["Hero", "V"])
    h.apply(Decision("raise", 250)); h.apply(Decision("call"))
    for _ in range(4):
        h.apply(Decision("check"))
    h.apply(Decision("raise", 300))                       # V bets river into hero's royal flush
    rep = QuantEngine(OpponentDB()).analyze(h.view_for(0))
    shove = next(o for o in rep.options if o.label.startswith("all-in"))
    call = next(o for o in rep.options if o.label.startswith("call"))
    assert not shove.raise_prob
    assert shove.ev >= call.ev - 1e-6


def test_phh_replay(tmp_path):
    from pokerbrain.adapters.phh import load_phhs, replay
    p = tmp_path / "t.phhs"
    p.write_text("""[1]
variant = 'NT'
antes = [0, 0, 0]
blinds_or_straddles = [0.10, 0.25, 0]
min_bet = 0.25
starting_stacks = [38.30, 32.15, 25.90]
actions = ['d dh p1 ????', 'd dh p2 ????', 'd dh p3 ????', 'p3 cbr 1.00', 'p1 f', 'p2 f']
hand = 3
players = ['a', 'b', 'c']

[2]
variant = 'NT'
antes = [0, 0]
blinds_or_straddles = [0.10, 0.25]
min_bet = 0.25
starting_stacks = [22.45, 12.25]
actions = ['d dh p1 ????', 'd dh p2 ????', 'p2 cc', 'p1 cc', 'd db 3s4c9c', 'p1 cbr 0.50', 'p2 f']
hand = 4
players = ['x', 'y']
""")
    hands = load_phhs(str(p))
    r = replay(hands[0])                                  # p1 = small blind, p3 = button
    assert r.history.positions == ["SB", "BB", "BTN"]
    assert r.history.net[2] == 140 and r.history.net[0] == -40 and r.history.net[1] == -100
    assert all(dp.view.hole == () for dp in r.points)    # decision views never carry hole cards
    r = replay(hands[1])                                  # heads-up blinds are reversed: p1 = big blind
    assert r.history.positions == ["BB", "BTN"]
    assert [a.kind for a in r.history.actions if a.street == "flop"] == ["bet", "fold"]
    assert r.history.net == {0: 100, 1: -100}


def test_population_interp():
    from pokerbrain import population
    pts = [[0.25, 0.3, 100], [1.0, 0.6, 100]]
    assert population.interp(pts, 0.1) == 0.3 and population.interp(pts, 2.0) == 0.6
    assert abs(population.interp(pts, 0.5) - 0.45) < 1e-9        # linear in log(size)


def test_exploits_need_a_sample():
    # an UNKNOWN opener must get the plain charts: with real-pool priors the old absolute cutoffs
    # ("nit if PFR < 11%") fired on every unknown player
    from pokerbrain.agents.quant_agent import QuantAgent
    for hole in (["Ah", "Jh"], ["Kd", "Qc"], ["7s", "7d"], ["Ts", "9s"], ["Ac", "5c"]):
        for seed in range(6):
            decisions = []
            for exploit in (True, False):
                d = _deck(hole + ["2c", "3d", "4h", "5s", "6c", "8d", "9h", "Tc", "Jd", "Qs"], seed)
                # 6-max: hero on the button (seat 0); UTG (seat 3) opens, HJ and CO fold
                h = HandState([10000] * 6, button=0, sb=50, bb=100, deck=d, names=[f"P{i}" for i in range(6)])
                h.apply(Decision("raise", 250)); h.apply(Decision("fold")); h.apply(Decision("fold"))
                ag = QuantAgent("Hero", db=OpponentDB(), seed=seed, exploit=exploit)
                ag.new_hand(seed)
                dec = ag.preflop(h.view_for(0))
                decisions.append((dec.kind, dec.amount))
            assert decisions[0] == decisions[1], (hole, seed, decisions)


def test_wtsd_counts_mucked_showdowns():
    d = _deck(["As", "Ks", "2c", "3d", "Qs", "Js", "Ts", "4h", "5d"])
    h = HandState([10000, 10000], button=0, sb=50, bb=100, deck=d, names=["A", "B"])
    h.apply(Decision("raise", 250)); h.apply(Decision("call"))
    for _ in range(6):
        h.apply(Decision("check"))
    hh = h.result.to_history()
    assert sorted(hh.showdown) == [0, 1]
    hh.shown = {}                                   # real site: both mucked / cards not recorded
    db = OpponentDB()
    db.update(hh)
    assert db.get("A").counts["wtsd"] == [1, 1] and db.get("B").counts["wtsd"] == [1, 1]
    assert db.get("A").counts["wsd"] == [1, 1] and db.get("B").counts["wsd"] == [0, 1]


def _river_nuts_view():
    d = _deck(["As", "Ks", "2c", "3d", "Qs", "Js", "Ts", "4h", "5d"])
    h = HandState([10000, 10000], button=0, sb=50, bb=100, deck=d, names=["Hero", "V"])
    h.apply(Decision("raise", 250)); h.apply(Decision("call"))
    for _ in range(4):
        h.apply(Decision("check"))
    h.apply(Decision("raise", 300))                       # V bets river into hero's royal flush
    return h.view_for(0)


def test_opus_agent_never_escapes_and_never_hangs():
    import time as _t
    from pokerbrain.agents.llm_agents import EscalationPolicy, OpusAgent
    view = _river_nuts_view()
    # malformed answers: the engine plays and nothing escapes act()
    for bad in ([1, 2], {"action_id": "A2", "mix": 5}, {"action_id": "A2", "mix": [], "note": 7},
                {"action_id": "A9"}, {"mix": [{"id": "A2", "p": float("nan")}, {"id": "A3", "p": -1}]}):
        ag = OpusAgent(lambda s, u, m: bad, variant="v13_final", escalation=EscalationPolicy(mode="all"),
                       override_gate=0.2, name="Hero")
        d = ag.act(view)
        assert d.kind == "raise" and d.source == "quant"           # the engine's own choice with the nuts
        assert ag.last_report is not None and d.reason in {o.label for o in ag.last_report.options}
    # an exception inside the decider
    def boom(s, u, m):
        raise RuntimeError("model down")
    ag = OpusAgent(boom, variant="v13_final", escalation=EscalationPolicy(mode="all"), name="Hero")
    d = ag.act(view)
    assert d.kind == "raise" and d.source == "quant" and ag.fallbacks == 1
    # a decider that ignores the clock: the deadline returns the engine's pick
    def slow(s, u, m):
        _t.sleep(3.0)
        return {"action_id": "A2", "mix": [{"id": "A2", "p": 1.0}]}
    ag = OpusAgent(slow, variant="v13_final", escalation=EscalationPolicy(mode="all"), name="Hero", deadline_s=0.5)
    t = _t.time()
    d = ag.act(view)
    # deadline 0.5 s + a 2 s grace for the client's own timeout to surface; the 3 s sleeper never gets to answer
    assert _t.time() - t < 2.9 and d.kind == "raise" and d.source == "quant" and ag.deadline_misses == 1
    # unknown variant is rejected at construction, not hours later
    import pytest
    with pytest.raises(ValueError):
        OpusAgent(slow, variant="v13-final", name="Hero")


def test_opus_only_where_it_pays():
    from pokerbrain.agents.llm_agents import EscalationPolicy, OpusAgent
    calls = []

    def decider(s, u, m):
        calls.append(m["street"])
        return {"action_id": "A1", "mix": [{"id": "A1", "p": 1.0}]}
    view = _river_nuts_view()                              # pot 800 chips = 8bb
    # no stakes at all: Opus is never consulted in real-play modes
    ag = OpusAgent(decider, variant="v13_final", escalation=EscalationPolicy(mode="postflop"), name="Hero")
    ag.act(view)
    assert calls == []
    # micro stakes ($0.05/$0.10): an 8bb pot is worth $0.80; 2% of it cannot pay for a $0.05 call
    ag = OpusAgent(decider, variant="v13_final", escalation=EscalationPolicy(mode="postflop"), name="Hero",
                   stakes=Stakes(0.05, 0.10))
    ag.act(view)
    assert calls == []
    # $1/$2: the same pot is worth $16, 2% = $0.32 > $0.05: consulted
    ag = OpusAgent(decider, variant="v13_final", escalation=EscalationPolicy(mode="postflop"), name="Hero",
                   stakes=Stakes(1.0, 2.0))
    ag.act(view)
    assert calls == ["river"]


def test_bankroll_money_rules():
    bm = BankrollManager(bankroll=50.0, stakes=Stakes(0.05, 0.10))       # 5 buy-ins
    # a 100bb coin flip: variance = (10000 chips)^2 / 4; the penalty must stay far below the pot
    pen_bb = bm.risk_penalty(0.25 * 10000 ** 2, 100) / 100
    assert 0.5 < pen_bb < 6.0
    bm2 = BankrollManager(bankroll=5000.0, stakes=Stakes(0.05, 0.10))
    assert bm2.risk_penalty(0.25 * 10000 ** 2, 100) < bm.risk_penalty(0.25 * 10000 ** 2, 100) / 50
    # chip units follow the view's big blind
    bm.record_hand(-1000, view_bb=10)              # 100bb lost at $0.10/bb = $10
    assert abs(bm.session_pnl + 10.0) < 1e-9
    bm.bankroll = 0.0
    assert bm.session_status() == "broke"
    assert risk_of_ruin(5, 0, 1000) == 0.0 and bankroll_for_ror(5, 95, 0) == float("inf")


def test_phh_result_comes_from_the_record(tmp_path):
    from pokerbrain.adapters.phh import load_phhs, replay
    base = """variant = 'NT'
antes = [0, 0]
blinds_or_straddles = [0.10, 0.25]
min_bet = 0.25
starting_stacks = [25, 25]
actions = ['d dh p1 ????', 'd dh p2 ????', 'p2 cbr 0.75', 'p1 cc', 'd db 2c7d9h', 'p1 cc', 'p2 cc', 'd db Ts', 'p1 cc', 'p2 cc', 'd db 3d', 'p1 cc', 'p2 cc', 'p1 sm ????', 'p2 sm AhAd']
players = ['x', 'y']
"""
    p = tmp_path / "a.phhs"
    p.write_text("[1]\n" + base + "hand = 1\n\n[2]\n" + base + "hand = 2\nwinnings = [1.45, 0]\n")
    a, b = load_phhs(str(p))
    ra = replay(a)                                 # x mucked, no recorded result: cards and actions kept, money unknown
    assert ra is not None and ra.result_known is False if hasattr(ra, "result_known") else not ra.history.result_known
    assert sorted(ra.history.showdown) == [0, 1] and list(ra.history.shown) == [1]
    r = replay(b)
    assert r.history.result_known
    assert r.history.net == {0: 580 - 300, 1: -300}   # 400 chips/$: x collected $1.45 (rake 0.05), both put $0.75 in
    assert sorted(r.history.showdown) == [0, 1] and list(r.history.shown) == [1]


def test_full_ring_positions_and_charts():
    from pokerbrain.preflop import rfi_range, vs_open_ranges
    from pokerbrain.view import position_names
    assert position_names(10) == ["BTN", "SB", "BB", "UTG", "UTG1", "UTG2", "MP", "LJ", "HJ", "CO"]
    co, utg = rfi_range("CO").fraction_of_all(), rfi_range("UTG").fraction_of_all()
    for pos in ("UTG1", "UTG2", "MP", "P4"):
        assert rfi_range(pos).fraction_of_all() <= utg < co
    tb_utg, _ = vs_open_ranges("BTN", "UTG1")
    tb_co, _ = vs_open_ranges("BTN", "CO")
    assert tb_utg.fraction_of_all() <= tb_co.fraction_of_all()


def test_texture_paired_boards_and_card_order():
    from pokerbrain.texture import effective_strength, strengths_for
    assert hand_features(("Qh", "Qd"), ["Kc", "Ks", "5d"]).strength_class == "weak"          # not "air"
    assert hand_features(("5c", "4d"), ["5h", "4s", "Kc", "Kd", "Ad"]).strength_class == "weak"  # counterfeited
    assert hand_features(("Th", "9d"), ["Ts", "9h", "Kc", "4d", "2d"]).strength_class == "strong"
    b = ["Qh", "7h", "2c"]
    assert effective_strength(("Ah", "Kh"), b) == effective_strength(("Kh", "Ah"), b) > 0.7
    assert strengths_for([("Kh", "Ah")], b)[("Kh", "Ah")] > 0.7
