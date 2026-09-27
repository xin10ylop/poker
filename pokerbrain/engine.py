"""No-Limit Texas Hold'em hand engine with exact chip accounting.

Rules implemented:
  * blinds (+ optional ante), heads-up special case (button = small blind,
    acts first preflop and last postflop)
  * min-bet = 1bb, min-raise = last *full* raise increment
  * an all-in for less than a full raise does not reopen betting to players
    who already acted (TDA rule), but cumulative short all-ins totalling a full
    raise do
  * side pots, split pots, odd chips to the first winner left of the button
  * uncalled bets returned (via side-pot construction)
  * all-in EV: when betting ends with players all-in before the river, the exact
    equity of every pot is computed by enumerating the remaining board.  Arenas
    use this "all-in adjusted" result to remove run-out luck (variance reduction).

Hole cards are dealt by seat from a fixed deck order and the board is always
deck[2n:2n+5]; replaying a deck with agents swapped between seats therefore
produces identical situations (duplicate poker).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Optional

from .cards import evaluate, stable_hash
from .view import (STREETS, ActionRecord, Decision, GameView, HandHistory,
                   LegalActions, PlayerView, position_names)


class IllegalAction(Exception):
    pass


@dataclass
class HandResult:
    hand_id: str
    names: list
    net: list                 # chips won/lost per seat (realized)
    ev_net: list              # all-in-adjusted expectation per seat
    board: list
    holes: list
    showdown_seats: list      # seats that reached showdown (cards revealed)
    log: list
    button: int
    sb: int
    bb: int
    start_stacks: list
    pot_total: int
    allin_street: Optional[str] = None
    winners: list = field(default_factory=list)   # [(pot_amount, [seats])]
    final_street: str = "preflop"
    positions: list = field(default_factory=list)

    def to_history(self, hero_seat=None, platform: str = "sim", table_id: str = "t1") -> HandHistory:
        shown = {i: tuple(self.holes[i]) for i in self.showdown_seats}
        return HandHistory(hand_id=self.hand_id, sb=self.sb, bb=self.bb, button_seat=self.button,
                           names=list(self.names), positions=list(self.positions),
                           start_stacks=list(self.start_stacks), actions=list(self.log), board=list(self.board),
                           shown=shown, net={i: self.net[i] for i in range(len(self.names))},
                           hero_seat=hero_seat, platform=platform, table_id=table_id,
                           hero_hole=tuple(self.holes[hero_seat]) if hero_seat is not None else None,
                           ev_net={i: self.ev_net[i] for i in range(len(self.names))}
                           if self.allin_street else None)


class HandState:
    def __init__(self, stacks: list[int], button: int, sb: int, bb: int, deck: list[str],
                 names: Optional[list[str]] = None, hand_id: str = "h0", ante: int = 0):
        n = len(stacks)
        if n < 2:
            raise ValueError("need at least two players")
        if any(s <= 0 for s in stacks):
            raise ValueError("all seated players need chips (handle rebuys at the table level)")
        self.n = n
        self.sb, self.bb, self.ante = sb, bb, ante
        self.button = button % n
        self.names = list(names) if names else [f"P{i}" for i in range(n)]
        self.hand_id = hand_id
        self.start_stacks = list(stacks)
        self.stacks = list(stacks)
        self.street_bets = [0] * n
        self.total_in = [0] * n
        self.in_hand = [True] * n
        self.all_in = [False] * n
        self.deck = list(deck)
        self.holes = [(self.deck[2 * i], self.deck[2 * i + 1]) for i in range(n)]
        self.full_board = self.deck[2 * n: 2 * n + 5]
        self.board: list[str] = []
        self.street_idx = 0
        self.log: list[ActionRecord] = []
        self.acted = [False] * n
        self.level_after: list[Optional[int]] = [None] * n
        self.last_full_raise = bb
        self.finished = False
        self.result: Optional[HandResult] = None
        self.allin_ev: Optional[list[float]] = None
        self.allin_street: Optional[str] = None
        pos = position_names(n)
        self.positions = [pos[(i - self.button) % n] for i in range(n)]
        if n == 2:
            self.sb_seat = self.button
            self.bb_seat = (self.button + 1) % n
        else:
            self.sb_seat = (self.button + 1) % n
            self.bb_seat = (self.button + 2) % n
            self.positions[self.sb_seat] = "SB"
            self.positions[self.bb_seat] = "BB"
        self._post_blinds()
        if n == 2:
            first = self.sb_seat
        else:
            first = (self.bb_seat + 1) % n
        self.to_act: Optional[int] = None
        self.to_act = self._first_needing_action(first)
        if self.to_act is None:
            self._end_street()

    # ---------------------------------------------------------------- setup
    def _commit(self, seat: int, amount: int, street_bet: bool = True) -> int:
        amount = min(amount, self.stacks[seat])
        self.stacks[seat] -= amount
        self.total_in[seat] += amount
        if street_bet:
            self.street_bets[seat] += amount
        if self.stacks[seat] == 0:
            self.all_in[seat] = True
        return amount

    def _post_blinds(self) -> None:
        if self.ante:
            for i in range(self.n):
                pot_before = sum(self.total_in)
                added = self._commit(i, self.ante, street_bet=False)
                self.log.append(ActionRecord("preflop", i, self.names[i], "post_ante", 0, added, pot_before,
                                             self.all_in[i]))
        for seat, amt, kind in ((self.sb_seat, self.sb, "post_sb"), (self.bb_seat, self.bb, "post_bb")):
            pot_before = sum(self.total_in)
            added = self._commit(seat, amt)
            self.log.append(ActionRecord("preflop", seat, self.names[seat], kind, self.street_bets[seat], added,
                                         pot_before, self.all_in[seat]))

    # --------------------------------------------------------------- queries
    @property
    def street(self) -> str:
        return STREETS[self.street_idx]

    @property
    def level(self) -> int:
        return max(self.street_bets)

    def pot(self) -> int:
        return sum(self.total_in)

    def _can_act(self, i: int) -> bool:
        return self.in_hand[i] and not self.all_in[i]

    def _needs_action(self, i: int) -> bool:
        if not self._can_act(i):
            return False
        if self.street_bets[i] < self.level:
            return True          # owes chips: must fold / call / raise
        if self.acted[i]:
            return False
        # nothing to call: only a real decision if someone could call a bet/raise
        return any(self._can_act(j) and self.stacks[j] + self.street_bets[j] > self.level
                   for j in range(self.n) if j != i)

    def _first_needing_action(self, start: int) -> Optional[int]:
        for k in range(self.n):
            i = (start + k) % self.n
            if self._needs_action(i):
                return i
        return None

    def legal_actions(self, seat: Optional[int] = None) -> LegalActions:
        i = self.to_act if seat is None else seat
        if i is None or self.finished:
            return LegalActions(False, False, 0, False, 0, 0)
        to_call = self.level - self.street_bets[i]
        stack = self.stacks[i]
        all_in_to = self.street_bets[i] + stack
        # a raise is only meaningful if someone else can put in more than the current level
        others_can_act = any(self._can_act(j) and self.stacks[j] + self.street_bets[j] > self.level
                             for j in range(self.n) if j != i)
        reopened = (not self.acted[i]) or (self.level_after[i] is not None and
                                            self.level - self.level_after[i] >= self.last_full_raise)
        can_raise = stack > to_call and reopened and others_can_act
        min_to = min(self.level + self.last_full_raise, all_in_to)
        return LegalActions(can_fold=to_call > 0, can_check=to_call == 0,
                            call_amount=min(to_call, stack), can_raise=can_raise,
                            min_raise_to=min_to if can_raise else 0,
                            max_raise_to=all_in_to if can_raise else 0)

    # ---------------------------------------------------------------- actions
    def apply(self, decision: Decision) -> None:
        if self.finished or self.to_act is None:
            raise IllegalAction("hand is over")
        i = self.to_act
        la = self.legal_actions(i)
        kind = decision.kind
        pot_before = self.pot()
        if kind == "fold":
            if not la.can_fold:
                raise IllegalAction("cannot fold when checking is free")
            self.in_hand[i] = False
            self.acted[i] = True
            self.log.append(ActionRecord(self.street, i, self.names[i], "fold", self.street_bets[i], 0, pot_before))
        elif kind == "check":
            if not la.can_check:
                raise IllegalAction("cannot check facing a bet")
            self.acted[i] = True
            self.level_after[i] = self.level
            self.log.append(ActionRecord(self.street, i, self.names[i], "check", self.street_bets[i], 0, pot_before))
        elif kind == "call":
            if la.call_amount <= 0:
                raise IllegalAction("nothing to call")
            added = self._commit(i, la.call_amount)
            self.acted[i] = True
            self.level_after[i] = self.level
            self.log.append(ActionRecord(self.street, i, self.names[i], "call", self.street_bets[i], added,
                                         pot_before, self.all_in[i]))
        elif kind == "raise":
            if not la.can_raise:
                raise IllegalAction("raising not allowed")
            to = int(decision.amount)
            if to > la.max_raise_to:
                raise IllegalAction(f"raise to {to} exceeds all-in {la.max_raise_to}")
            if to < la.min_raise_to and to != la.max_raise_to:
                raise IllegalAction(f"raise to {to} below minimum {la.min_raise_to}")
            prev_level = self.level
            if to - prev_level >= self.last_full_raise:
                self.last_full_raise = to - prev_level
            added = self._commit(i, to - self.street_bets[i])
            self.acted[i] = True
            self.level_after[i] = self.level
            label = "bet" if prev_level == 0 else "raise"
            self.log.append(ActionRecord(self.street, i, self.names[i], label, self.street_bets[i], added,
                                         pot_before, self.all_in[i]))
        else:
            raise IllegalAction(f"unknown action {kind!r}")
        self._advance(i)

    def _advance(self, last: int) -> None:
        if sum(self.in_hand) == 1:
            self._finish_uncontested()
            return
        nxt = self._first_needing_action(last + 1)
        if nxt is None:
            self._end_street()
        else:
            self.to_act = nxt

    def _end_street(self) -> None:
        # reset per-street state
        self.street_bets = [0] * self.n
        self.acted = [False] * self.n
        self.level_after = [None] * self.n
        self.last_full_raise = self.bb
        n_can_act = sum(1 for i in range(self.n) if self._can_act(i))
        if self.street_idx == 3:
            self._showdown()
            return
        if n_can_act <= 1:
            # no more betting possible: compute all-in EV, then run it out
            if self.allin_ev is None:
                self.allin_street = self.street
                self.allin_ev = self._allin_expectation()
            self.board = list(self.full_board)
            self.street_idx = 3
            self._showdown()
            return
        self.street_idx += 1
        self.board = self.full_board[: {1: 3, 2: 4, 3: 5}[self.street_idx]]
        # postflop: first player left of the button acts first
        self.to_act = self._first_needing_action(self.button + 1)
        if self.to_act is None:
            self._end_street()

    # ------------------------------------------------------------- settlement
    def _pots(self) -> list[tuple[int, list[int]]]:
        """Side pots as (amount, eligible seats), smallest level first."""
        levels = sorted({self.total_in[i] for i in range(self.n) if self.in_hand[i] and self.total_in[i] > 0})
        pots, prev = [], 0
        for lv in levels:
            amount = sum(min(c, lv) - min(c, prev) for c in self.total_in)
            elig = [i for i in range(self.n) if self.in_hand[i] and self.total_in[i] >= lv]
            if amount > 0:
                pots.append((amount, elig))
            prev = lv
        return pots

    def _split(self, amount: int, winners: list[int], payout: list[float]) -> None:
        k = len(winners)
        share, rem = divmod(amount, k)
        order = sorted(winners, key=lambda s: (s - self.button - 1) % self.n)
        for idx, s in enumerate(order):
            payout[s] += share + (1 if idx < rem else 0)

    def _showdown(self) -> None:
        self.board = list(self.full_board)
        values = {i: evaluate(list(self.holes[i]) + self.board) for i in range(self.n) if self.in_hand[i]}
        payout = [0.0] * self.n
        winners_log = []
        for amount, elig in self._pots():
            best = max(values[i] for i in elig)
            winners = [i for i in elig if values[i] == best]
            self._split(amount, winners, payout)
            winners_log.append((amount, winners))
        showdown = [i for i in range(self.n) if self.in_hand[i]]
        self._finish(payout, showdown, winners_log)

    def _finish_uncontested(self) -> None:
        winner = next(i for i in range(self.n) if self.in_hand[i])
        payout = [0.0] * self.n
        payout[winner] = float(self.pot())
        self._finish(payout, [], [(self.pot(), [winner])])

    def _finish(self, payout: list[float], showdown: list[int], winners_log: list) -> None:
        net = [int(round(payout[i])) - self.total_in[i] for i in range(self.n)]
        ev = list(net) if self.allin_ev is None else [self.allin_ev[i] for i in range(self.n)]
        self.finished = True
        self.to_act = None
        self.result = HandResult(
            hand_id=self.hand_id, names=list(self.names), net=net, ev_net=ev, board=list(self.board),
            holes=list(self.holes), showdown_seats=showdown, log=list(self.log), button=self.button,
            sb=self.sb, bb=self.bb, start_stacks=list(self.start_stacks), pot_total=self.pot(),
            allin_street=self.allin_street, winners=winners_log, final_street=self.street,
            positions=list(self.positions))

    def _allin_expectation(self) -> list[float]:
        """Expected net per seat over the rest of the board.

        Exact enumeration when <= 2 cards are missing; otherwise Monte Carlo
        (eval7's C routine for a single heads-up pot, seeded sampling for
        multiway / side pots)."""
        known = list(self.board)
        need = 5 - len(known)
        dead = set(known)
        for i in range(self.n):
            dead.update(self.holes[i])
        rest = [c for c in _ALL if c not in dead]
        pots = self._pots()
        live = [i for i in range(self.n) if self.in_hand[i]]
        exp = [0.0] * self.n
        if need >= 3 and len(live) == 2 and len(pots) == 1:
            import eval7
            from .cards import to_eval7
            a, b = live
            eq = eval7.py_hand_vs_range_monte_carlo(to_eval7(self.holes[a]), [(tuple(to_eval7(self.holes[b])), 1.0)],
                                                    to_eval7(known), 40000)
            amount = pots[0][0]
            exp[a] = amount * eq
            exp[b] = amount * (1 - eq)
            return [exp[i] - self.total_in[i] for i in range(self.n)]
        if need >= 3:
            import random as _r
            rng = _r.Random(stable_hash(self.hand_id, tuple(self.holes)))
            runouts = (rng.sample(rest, need) for _ in range(3000))
        else:
            runouts = itertools.combinations(rest, need)
        count = 0
        for runout in runouts:
            board = known + list(runout)
            vals = {i: evaluate(list(self.holes[i]) + board) for i in live}
            for amount, elig in pots:
                best = max(vals[i] for i in elig)
                winners = [i for i in elig if vals[i] == best]
                for w in winners:
                    exp[w] += amount / len(winners)
            count += 1
        return [exp[i] / count - self.total_in[i] for i in range(self.n)]

    # ------------------------------------------------------------ observation
    def view_for(self, seat: int, platform: str = "sim", table_id: str = "t1",
                 reveal: bool = False) -> GameView:
        players = []
        for i in range(self.n):
            show = (i == seat) or reveal
            players.append(PlayerView(
                seat=i, name=self.names[i], stack=self.stacks[i], bet=self.street_bets[i],
                total_in=self.total_in[i], in_hand=self.in_hand[i], all_in=self.all_in[i],
                position=self.positions[i], hole=self.holes[i] if show else None,
                start_stack=self.start_stacks[i]))
        return GameView(hand_id=self.hand_id, sb=self.sb, bb=self.bb, hero_seat=seat, hole=self.holes[seat],
                        board=list(self.board), street=self.street, pot=self.pot(), players=players,
                        actions=list(self.log), legal=self.legal_actions(seat) if seat == self.to_act else
                        LegalActions(False, False, 0, False, 0, 0),
                        button_seat=self.button, platform=platform, table_id=table_id)


from .cards import ALL_CARDS as _ALL  # noqa: E402  (after class to avoid cycles in type hints)
