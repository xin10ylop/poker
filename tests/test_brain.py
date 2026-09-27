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
