"""Platform-agnostic observation / decision types.

Every adapter (local simulator, Slumbot, ACPC, manual entry, HTTP bridge)
converts its native game state into a `GameView`, and every agent returns a
`Decision`.  Amounts are integers in *chips*; `bb` tells you the chip size of
one big blind so everything can be rendered in big blinds.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

STREETS = ("preflop", "flop", "turn", "river")


@dataclass
class LegalActions:
    can_fold: bool
    can_check: bool
    call_amount: int          # chips to add to call (0 when checking is possible)
    can_raise: bool
    min_raise_to: int         # street-level "raise to" minimum (all-in if smaller)
    max_raise_to: int         # street-level all-in amount

    def clamp_raise(self, to: int) -> int:
        return max(self.min_raise_to, min(self.max_raise_to, int(round(to))))


@dataclass
class PlayerView:
    seat: int
    name: str
    stack: int                # chips behind (not yet committed)
    bet: int                  # committed on this street
    total_in: int             # committed this hand (incl. blinds)
    in_hand: bool
    all_in: bool
    position: str
    hole: Optional[tuple] = None   # only hero (or revealed at showdown)
    start_stack: int = 0


@dataclass
class ActionRecord:
    street: str
    seat: int
    name: str
    kind: str                 # post_sb, post_bb, post_ante, fold, check, call, bet, raise
    to: int                   # street commitment after the action
    added: int                # chips added by this action
    pot_before: int           # total pot (all streets) before the action
    all_in: bool = False


@dataclass
class GameView:
    hand_id: str
    sb: int
    bb: int
    hero_seat: int
    hole: tuple
    board: list
    street: str
    pot: int                  # everything committed so far, all streets
    players: list
    actions: list
    legal: LegalActions
    button_seat: int
    platform: str = "sim"
    table_id: str = "t1"
    extra: dict = field(default_factory=dict)

    # ------------------------------------------------------------ helpers
    @property
    def hero(self) -> PlayerView:
        return self.players[self.hero_seat]

    def player(self, seat: int) -> PlayerView:
        return self.players[seat]

    def opponents(self, active_only: bool = True) -> list[PlayerView]:
        return [p for p in self.players
                if p.seat != self.hero_seat and (p.in_hand or not active_only)]

    def street_actions(self, street: Optional[str] = None) -> list[ActionRecord]:
        st = street or self.street
        return [a for a in self.actions if a.street == st and not a.kind.startswith("post")]

    def to_call(self) -> int:
        return self.legal.call_amount

    def current_bet_level(self) -> int:
        return max((p.bet for p in self.players), default=0)

    def effective_stack(self, vs_seats: Optional[list[int]] = None) -> int:
        """Largest amount hero can still lose/win vs the active opponents."""
        hero_total = self.hero.stack + self.hero.bet
        opps = [p for p in self.opponents() if vs_seats is None or p.seat in vs_seats]
        if not opps:
            return hero_total
        return min(hero_total, max(p.stack + p.bet for p in opps))

    def bb_amount(self, chips: float) -> float:
        return chips / self.bb

    def num_active(self) -> int:
        return sum(1 for p in self.players if p.in_hand)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Decision:
    kind: str                 # fold | check | call | raise  (bet == raise from 0)
    amount: int = 0           # for raise: street-level "raise to" in chips
    source: str = ""          # which component made it (quant, jev, opus, chart...)
    reason: str = ""
    meta: dict = field(default_factory=dict)

    def normalized(self, legal: LegalActions) -> "Decision":
        """Coerce into a legal action (never throws: unknown kinds / bad amounts become check or fold)."""
        k = str(self.kind or "").strip().lower()
        amount = self.amount
        if k in ("bet", "allin", "all-in", "shove", "all_in", "jam"):
            if k != "bet":
                amount = legal.max_raise_to
            k = "raise"
        if k not in ("fold", "check", "call", "raise"):
            k = "check" if legal.can_check else "fold"
        if k == "raise":
            if not legal.can_raise:
                k = "call" if legal.call_amount > 0 else "check"
            else:
                try:
                    amt = int(round(float(amount)))
                    if amt != amt:          # NaN
                        raise ValueError
                except (TypeError, ValueError, OverflowError):
                    amt = legal.min_raise_to
                return Decision("raise", legal.clamp_raise(amt), self.source, self.reason, self.meta)
        if k == "check" and not legal.can_check:
            k = "fold" if legal.call_amount > 0 else "check"
        if k == "call" and legal.call_amount == 0:
            k = "check"
        if k == "fold" and legal.can_check:
            k = "check"   # never fold when checking is free
        return Decision(k, 0, self.source, self.reason, self.meta)


@dataclass
class HandHistory:
    """A finished hand in platform-agnostic form (input to opponent tracking)."""
    hand_id: str
    sb: int
    bb: int
    button_seat: int
    names: list               # by seat
    positions: list           # by seat
    start_stacks: list        # by seat
    actions: list             # ActionRecord list (whole hand)
    board: list               # final board as far as it was dealt
    shown: dict               # seat -> (card, card) revealed at showdown
    net: dict                 # seat -> chips won/lost (as far as known)
    hero_seat: Optional[int] = None
    platform: str = "sim"
    table_id: str = "t1"
    hero_hole: Optional[tuple] = None
    ev_net: Optional[dict] = None      # seat -> all-in expectation (chips) when known
    showdown: Optional[list] = None    # seats that reached showdown, cards shown or mucked
    result_known: bool = True          # False when the money result could not be settled (net is unusable)

    def pot_total(self) -> int:
        return sum(a.added for a in self.actions)


def position_names(n: int) -> list[str]:
    """Position labels by offset from the button (offset 0 = button)."""
    if n == 2:
        return ["BTN", "BB"]  # HU: button is also the small blind
    names = {3: ["BTN", "SB", "BB"],
             4: ["BTN", "SB", "BB", "CO"],
             5: ["BTN", "SB", "BB", "UTG", "CO"],
             6: ["BTN", "SB", "BB", "UTG", "HJ", "CO"],
             7: ["BTN", "SB", "BB", "UTG", "MP", "HJ", "CO"],
             8: ["BTN", "SB", "BB", "UTG", "UTG1", "MP", "HJ", "CO"],
             9: ["BTN", "SB", "BB", "UTG", "UTG1", "UTG2", "MP", "HJ", "CO"],
             10: ["BTN", "SB", "BB", "UTG", "UTG1", "UTG2", "MP", "LJ", "HJ", "CO"]}
    return names.get(n, ["BTN", "SB", "BB"] + [f"P{i}" for i in range(3, n)])
