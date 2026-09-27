import random

import pytest

from pokerbrain.cards import ALL_CARDS
from pokerbrain.engine import HandState, IllegalAction
from pokerbrain.view import Decision


def deck_with(prefix: list[str], seed: int = 0) -> list[str]:
    rest = [c for c in ALL_CARDS if c not in prefix]
    random.Random(seed).shuffle(rest)
    return prefix + rest


def test_hu_raise_fold():
    h = HandState([10000, 10000], button=0, sb=50, bb=100, deck=deck_with([]))
    assert h.to_act == 0  # HU: button/SB acts first preflop
    h.apply(Decision("raise", 250))
    assert h.to_act == 1
    h.apply(Decision("fold"))
    assert h.finished
    assert h.result.net == [100, -100]


def test_hu_postflop_order_and_showdown():
    # seat0: AhAd, seat1: KhKd, board 2c 7d 9s Jc 3h -> seat0 wins
    d = deck_with(["Ah", "Ad", "Kh", "Kd", "2c", "7d", "9s", "Jc", "3h"])
    h = HandState([10000, 10000], button=0, sb=50, bb=100, deck=d)
    h.apply(Decision("call"))       # SB completes
    h.apply(Decision("check"))      # BB checks option
    assert h.street == "flop" and h.to_act == 1  # BB first postflop
    h.apply(Decision("check"))
    h.apply(Decision("raise", 100))  # bet
    h.apply(Decision("call"))
    for _ in range(2):
        h.apply(Decision("check"))
        h.apply(Decision("check"))
    assert h.finished
    assert h.result.net == [200, -200]
    assert h.result.showdown_seats == [0, 1]


def test_side_pots_three_way():
    # seat0 short all-in with best hand, seat1 second best, seat2 worst
    d = deck_with(["As", "Ac", "Ks", "Kc", "Qs", "Qc", "2d", "7h", "9c", "4d", "3s"])
    h = HandState([1000, 5000, 5000], button=2, sb=50, bb=100, deck=d)
    # button=2 -> SB=0, BB=1, preflop first to act = seat2
    assert h.to_act == 2
    h.apply(Decision("raise", 5000))   # seat2 shoves
    h.apply(Decision("call"))          # seat0 calls all-in for 1000 total
    h.apply(Decision("call"))          # seat1 calls
    assert h.finished
    # main pot 3000 -> seat0 (AA); side pot 8000 -> seat1 (KK)
    assert h.result.net == [2000, 3000, -5000]
    # all-in EV is an expectation; must sum to zero and favour seat0
    assert abs(sum(h.result.ev_net)) < 1e-6
    assert h.result.allin_street == "preflop"


def test_incomplete_raise_does_not_reopen():
    d = deck_with([])
    # 3-handed postflop: seat A bets, B short-shoves (incomplete), back to A: call/fold only
    h = HandState([10000, 10000, 1250], button=0, sb=50, bb=100, deck=d)
    # button 0, SB 1, BB 2, first preflop = seat0
    h.apply(Decision("call"))
    h.apply(Decision("call"))
    h.apply(Decision("check"))
    assert h.street == "flop"
    assert h.to_act == 1  # SB first
    h.apply(Decision("raise", 1000))     # seat1 bets 1000
    h.apply(Decision("raise", 1150))     # seat2 all-in 1150 (incomplete raise of 150)
    la = h.legal_actions()
    assert h.to_act == 0 and la.can_raise  # seat0 has not acted this street -> may raise
    h.apply(Decision("call"))
    la = h.legal_actions()
    assert h.to_act == 1
    assert not la.can_raise and la.call_amount == 150
    h.apply(Decision("call"))


def test_min_raise_rules():
    h = HandState([10000] * 3, button=0, sb=50, bb=100, deck=deck_with([]))
    la = h.legal_actions()
    assert la.min_raise_to == 200
    h.apply(Decision("raise", 350))        # raise by 250
    la = h.legal_actions()
    assert la.min_raise_to == 600          # 350 + 250
    with pytest.raises(IllegalAction):
        h.apply(Decision("raise", 500))


def test_bb_option_when_limped():
    h = HandState([10000, 10000, 10000], button=0, sb=50, bb=100, deck=deck_with([]))
    h.apply(Decision("call"))   # btn
    h.apply(Decision("call"))   # sb
    assert h.to_act == 2 and h.street == "preflop"
    la = h.legal_actions()
    assert la.can_check and la.can_raise


def _random_decision(rng, la):
    opts = []
    if la.can_fold:
        opts.append("fold")
    if la.can_check:
        opts.append("check")
    if la.call_amount > 0:
        opts.append("call")
    if la.can_raise:
        opts.append("raise")
    k = rng.choice(opts)
    if k == "raise":
        choice = rng.random()
        if choice < 0.3:
            amt = la.min_raise_to
        elif choice < 0.5:
            amt = la.max_raise_to
        else:
            amt = rng.randint(la.min_raise_to, la.max_raise_to)
        return Decision("raise", amt)
    return Decision(k)


def test_fuzz_conservation():
    rng = random.Random(1)
    for t in range(1500):
        n = rng.randint(2, 6)
        stacks = [rng.randint(80, 30000) for _ in range(n)]
        deck = list(ALL_CARDS)
        rng.shuffle(deck)
        h = HandState(stacks, button=rng.randrange(n), sb=50, bb=100, deck=deck)
        steps = 0
        while not h.finished:
            h.apply(_random_decision(rng, h.legal_actions()))
            steps += 1
            assert steps < 500
        assert sum(h.result.net) == 0
        assert abs(sum(h.result.ev_net)) < 1e-6
        for i in range(n):
            assert h.result.net[i] >= -stacks[i]


def test_crosscheck_against_pokerkit():
    pk = pytest.importorskip("pokerkit")
    from pokerkit import Automation, NoLimitTexasHoldem
    autos = (Automation.ANTE_POSTING, Automation.BET_COLLECTION, Automation.BLIND_OR_STRADDLE_POSTING,
             Automation.HOLE_CARDS_SHOWING_OR_MUCKING, Automation.HAND_KILLING, Automation.CHIPS_PUSHING,
             Automation.CHIPS_PULLING)
    rng = random.Random(42)
    mismatches = 0
    for t in range(1200):
        n = rng.randint(2, 6)
        stacks = [rng.choice([rng.randint(60, 600), rng.randint(600, 20000)]) for _ in range(n)]
        button = rng.randrange(n)
        deck = list(ALL_CARDS)
        rng.shuffle(deck)
        h = HandState(stacks, button=button, sb=50, bb=100, deck=deck)
        # pokerkit index j <-> our seat (button + 1 + j) % n
        seat_of = [(button + 1 + j) % n for j in range(n)]
        s = NoLimitTexasHoldem.create_state(autos, True, 0, (50, 100), 100,
                                            tuple(stacks[seat_of[j]] for j in range(n)), n)
        got = [0] * n
        while s.can_deal_hole():
            j = s.hole_dealee_index
            s.deal_hole(h.holes[seat_of[j]][got[j]])
            got[j] += 1
        board_ptr = 0
        while True:
            if s.can_burn_card():
                s.burn_card("??")
                continue
            if s.can_deal_board():
                k = 3 if board_ptr == 0 else 1
                s.deal_board("".join(h.full_board[board_ptr: board_ptr + k]))
                board_ptr += k
                continue
            if h.finished:
                break
            assert s.actor_index is not None, f"hand {t}: pokerkit has no actor but we do"
            assert seat_of[s.actor_index] == h.to_act, f"hand {t}: actor mismatch"
            la = h.legal_actions()
            pk_can = s.can_complete_bet_or_raise_to()
            if la.can_raise != pk_can:
                # Known divergence: pokerkit resets its "acted" set on a *short* all-in, so it lets a
                # player who already acted re-raise facing an incomplete raise.  TDA rules (which we
                # follow) say that player may only call or fold.  Accept only that exact case.
                i = h.to_act
                reopen_blocked = (h.acted[i] and h.level_after[i] is not None
                                  and h.level - h.level_after[i] < h.last_full_raise)
                assert (not la.can_raise) and pk_can and reopen_blocked, f"hand {t}: can_raise mismatch"
            if la.can_raise:
                assert la.min_raise_to == s.min_completion_betting_or_raising_to_amount, f"hand {t} min"
                assert la.max_raise_to == s.max_completion_betting_or_raising_to_amount, f"hand {t} max"
            d = _random_decision(rng, la)
            h.apply(d)
            if d.kind == "fold":
                s.fold()
            elif d.kind in ("check", "call"):
                s.check_or_call()
            else:
                s.complete_bet_or_raise_to(d.amount)
        while s.status:
            if s.can_burn_card():
                s.burn_card("??")
            elif s.can_deal_board():
                k = 3 if board_ptr == 0 else 1
                s.deal_board("".join(h.full_board[board_ptr: board_ptr + k]))
                board_ptr += k
            elif s.actor_index is not None and s.can_check_or_call() and s.checking_or_calling_amount == 0:
                # pokerkit sometimes asks for a no-choice check (nobody left who could call a
                # raise); our engine skips it.  Same outcome.
                s.check_or_call()
            else:
                break
        pk_net = {seat_of[j]: s.stacks[j] - stacks[seat_of[j]] for j in range(n)}
        ours = {i: h.result.net[i] for i in range(n)}
        if pk_net != ours:
            diffs = {k: ours[k] - pk_net[k] for k in ours if ours[k] != pk_net[k]}
            print("MISMATCH", t, "n", n, "ours", ours, "pk", pk_net, "diff", diffs,
                  "showdown", h.result.showdown_seats, "winners", h.result.winners)
            mismatches += 1
    assert mismatches == 0
