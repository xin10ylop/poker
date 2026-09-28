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
