"""Decision-level benchmark: record real spots, score actions with a posterior oracle.

Full-match win rates are hopelessly noisy for comparing prompts (6-max CI is
±150 bb/100 after 600 hands).  Instead we score *decisions*:

  1. Record hero decision points from long simulated sessions (heads-up pots,
     so the opponent model is exact), including the opponent-profile snapshot
     hero had at that moment (stats, notes, tilt...).
  2. Oracle EV of each candidate action:
       * posterior over the villain's hole cards given his observed actions,
         computed from his *actual* bot policy with Monte Carlo over the bot's
         randomization (never his real cards, never his real RNG)
       * rollouts: sample (villain combo ~ posterior, future board), apply the
         action, finish the hand with the villain's bot policy and a TAG
         continuation policy for hero.  Common random numbers across actions.
  3. A decision's score = EV(chosen) - EV(best), in big blinds.
"""
from __future__ import annotations

import copy
import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

import numpy as np

from .agents.base import Agent
from .arena import Table, seeded_deck
from .bots import StyleBot
from .cards import ALL_CARDS, ALL_COMBOS, stable_hash
from .engine import HandState
from .opponents import OpponentDB
from .view import ActionRecord, Decision, GameView


@dataclass
class Spot:
    spot_id: str
    hand_index: int
    stacks: list
    button: int
    sb: int
    bb: int
    names: list
    deck: list                   # true deck (for bookkeeping only; oracle never uses villain's true cards)
    hero_seat: int
    villain_seat: int
    decisions: list              # [(seat, kind, amount)] applied so far in this hand
    street: str
    pot_bb: float
    villain_style: str
    villain_state: dict          # bot state (tilt counters...) at hand start
    db_snapshot: dict            # hero's opponent DB (json) at decision time
    session_context: dict = field(default_factory=dict)
    oracle: dict = field(default_factory=dict)       # label -> EV (bb, relative to folding now)
    options: list = field(default_factory=list)      # [{"id","label","kind","amount"}]
    quant_choice: Optional[str] = None
    tags: list = field(default_factory=list)

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "Spot":
        return cls(**d)

    # rebuild the exact decision state
    def replay(self, deck: Optional[list] = None) -> HandState:
        h = HandState(list(self.stacks), button=self.button, sb=self.sb, bb=self.bb, deck=deck or self.deck,
                      names=list(self.names), hand_id=self.spot_id)
        for seat, kind, amount in self.decisions:
            assert h.to_act == seat, "replay diverged"
            h.apply(Decision(kind, amount))
        return h

    def view(self) -> GameView:
        return self.replay().view_for(self.hero_seat)

    def db(self) -> OpponentDB:
        db = OpponentDB()
        from .opponents import PlayerProfile
        db.profiles = {k: PlayerProfile.from_json(copy.deepcopy(v)) for k, v in self.db_snapshot["profiles"].items()}
        db.hero_image = copy.deepcopy(self.db_snapshot.get("hero_image", db.hero_image))
        return db


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------
class RecordingHero(Agent):
    """Wraps a hero agent; snapshots every heads-up postflop decision (and big preflop ones)."""

    def __init__(self, inner: Agent, db: OpponentDB, keep: Callable[[GameView], bool]):
        self.inner = inner
        self.name = inner.name
        self.db = db
        self.keep = keep
        self.pending: list[dict] = []
        self.current_hand: Optional[dict] = None

    def new_hand(self, hand_index: int) -> None:
        self.inner.new_hand(hand_index)

    def observe(self, history, my_seat):
        self.inner.observe(history, my_seat)

    def act(self, view: GameView) -> Decision:
        d = self.inner.act(view)
        if self.keep(view):
            self.pending.append({"view": view, "db": {"profiles": {k: copy.deepcopy(v.to_json())
                                                                      for k, v in self.db.profiles.items()},
                                                       "hero_image": copy.deepcopy(self.db.hero_image)},
                                 "choice": d})
        return d


def default_keep(view: GameView) -> bool:
    if view.num_active() != 2:
        return False
    if view.street == "preflop":
        return view.pot >= 12 * view.bb and view.legal.call_amount > 0
    return view.pot >= 4 * view.bb


def record_spots(hero_factory: Callable[[OpponentDB], Agent], field_factory: Callable[[], list[StyleBot]],
                 n_hands: int, seed: int, hero_seat: int = 0, keep: Callable = default_keep,
                 max_spots: int = 10_000, progress: bool = False) -> list[Spot]:
    db = OpponentDB()
    hero = hero_factory(db)
    rec = RecordingHero(hero, db, keep)
    bots = field_factory()
    agents = bots[:hero_seat] + [rec] + bots[hero_seat:]
    table = Table(agents)
    spots: list[Spot] = []
    for i in range(n_hands):
        deck = seeded_deck(seed * 7_919 + i)
        button = i % len(agents)
        bot_states = {b.name: {"tilt_hands_left": b.tilt_hands_left, "seed": b.seed,
                               "style": b.base_style.name} for b in bots}
        rec.pending = []
        stacks = [table.stack] * len(agents)
        # play while capturing the decision list
        log_decisions: list = []
        orig_apply = HandState.apply

        def spy(self, decision, _log=log_decisions):
            _log.append((self.to_act, decision.kind, int(decision.amount)))
            return orig_apply(self, decision)
        HandState.apply = spy
        try:
            res = table.play_hand(deck, button, f"s{seed}-{i}", i)
        finally:
            HandState.apply = orig_apply
        for p in rec.pending:
            v: GameView = p["view"]
            n_before = sum(1 for a in v.actions if not a.kind.startswith("post"))
            vill = next(pp for pp in v.players if pp.in_hand and pp.seat != v.hero_seat)
            bot = agents[vill.seat]
            if not isinstance(bot, StyleBot):
                continue
            sp = Spot(spot_id=f"s{seed}-{i}-{n_before}", hand_index=i, stacks=stacks, button=button, sb=table.sb,
                      bb=table.bb, names=[a.name for a in agents], deck=deck, hero_seat=hero_seat,
                      villain_seat=vill.seat, decisions=log_decisions[:n_before], street=v.street,
                      pot_bb=v.pot / v.bb, villain_style=bot.base_style.name,
                      villain_state=bot_states[bot.name], db_snapshot=p["db"])
            if bot.base_style.name == "tilter" and bot_states[bot.name]["tilt_hands_left"] > 0:
                sp.tags.append("villain_tilted")
            spots.append(sp)
        if progress and (i + 1) % 250 == 0:
            print(f"  hand {i + 1}/{n_hands}: {len(spots)} spots", flush=True)
        if len(spots) >= max_spots:
            break
    return spots


# ---------------------------------------------------------------------------
# Oracle
# ---------------------------------------------------------------------------
def _make_bot(spot: Spot) -> StyleBot:
    st = spot.villain_state
    b = StyleBot(spot.names[spot.villain_seat], st["style"], seed=st["seed"])
    b.tilt_hands_left = st["tilt_hands_left"]
    b.new_hand(spot.hand_index)   # sets style for the hand (tilted or not)
    return b


def _deck_for(spot: Spot, villain_combo, rng: random.Random, board_known: list) -> list:
    n = len(spot.names)
    hero_cards = spot.deck[2 * spot.hero_seat: 2 * spot.hero_seat + 2]
    fixed = set(hero_cards) | set(villain_combo) | set(board_known)
    rest = [c for c in ALL_CARDS if c not in fixed]
    rng.shuffle(rest)
    deck = [None] * 52
    deck[2 * spot.hero_seat: 2 * spot.hero_seat + 2] = hero_cards
    deck[2 * spot.villain_seat: 2 * spot.villain_seat + 2] = list(villain_combo)
    for k, c in enumerate(board_known):
        deck[2 * n + k] = c
    it = iter(rest)
    for k in range(52):
        if deck[k] is None:
            deck[k] = next(it)
    return deck


def _action_matches(d: Decision, kind: str, amount: int, view: GameView) -> bool:
    d = d.normalized(view.legal)
    if d.kind != kind:
        return False
    if kind != "raise":
        return True
    return abs(d.amount - amount) <= max(view.bb, 0.25 * amount)


def villain_posterior(spot: Spot, k_samples: int = 24, rng: Optional[random.Random] = None) -> tuple:
    """(combos, probs) for the villain's hole cards given his observed actions."""
    rng = rng or random.Random(0)
    h0 = spot.replay()
    board_known = list(h0.board)
    hero_cards = spot.deck[2 * spot.hero_seat: 2 * spot.hero_seat + 2]
    dead = set(hero_cards) | set(board_known)
    cands = [c for c in ALL_COMBOS if c[0] not in dead and c[1] not in dead]
    bot = _make_bot(spot)
    weights = []
    vill_points = [idx for idx, (seat, _, _) in enumerate(spot.decisions) if seat == spot.villain_seat]
    for combo in cands:
        deck = _deck_for(spot, combo, rng, board_known)
        h = HandState(list(spot.stacks), button=spot.button, sb=spot.sb, bb=spot.bb, deck=deck,
                      names=list(spot.names), hand_id="post")
        like = 1.0
        for idx, (seat, kind, amount) in enumerate(spot.decisions):
            if seat == spot.villain_seat:
                view = h.view_for(seat)
                hits = 0
                for kk in range(k_samples):
                    bot.rng = random.Random(stable_hash(spot.spot_id, kk, idx))
                    if _action_matches(bot.act(view), kind, amount, view):
                        hits += 1
                like *= (hits + 0.15) / (k_samples + 0.3)
                if like < 1e-7:
                    break
            h.apply(Decision(kind, amount))
        weights.append(like)
    w = np.array(weights)
    w = w / w.sum()
    return cands, w


def oracle_evaluate(spot: Spot, options: list[Decision], n_samples: int = 300, k_samples: int = 24,
                    seed: int = 0, continuation: str = "tag") -> dict:
    """EV (bb, relative to folding now) of each option; common random numbers across options."""
    rng = random.Random(seed)
    cands, post = villain_posterior(spot, k_samples, rng)
    h0 = spot.replay()
    board_known = list(h0.board)
    hero_invested = h0.total_in[spot.hero_seat]
    npr = np.random.default_rng(seed)
    idxs = npr.choice(len(cands), size=n_samples, p=post)
    totals = [0.0] * len(options)
    sq = [0.0] * len(options)
    for t, ci in enumerate(idxs):
        combo = cands[ci]
        deck = _deck_for(spot, combo, random.Random(seed * 31 + t), board_known)
        for j, opt in enumerate(options):
            h = HandState(list(spot.stacks), button=spot.button, sb=spot.sb, bb=spot.bb, deck=deck,
                          names=list(spot.names), hand_id="roll")
            for seat, kind, amount in spot.decisions:
                h.apply(Decision(kind, amount))
            vb = _make_bot(spot)
            hero_bot = StyleBot("hero_cont", continuation, seed=seed + 17)
            vb.rng = random.Random(seed * 1_000 + t)
            hero_bot.rng = random.Random(seed * 2_000 + t)
            view = h.view_for(spot.hero_seat)
            h.apply(opt.normalized(view.legal))
            guard = 0
            while not h.finished:
                seat = h.to_act
                v = h.view_for(seat)
                ag = vb if seat == spot.villain_seat else hero_bot
                h.apply(ag.act(v).normalized(v.legal))
                guard += 1
                if guard > 60:
                    break
            val = (h.result.ev_net[spot.hero_seat] + hero_invested) / spot.bb
            totals[j] += val
            sq[j] += val * val
    n = float(n_samples)
    out = {}
    for j in range(len(options)):
        m = totals[j] / n
        var = max(0.0, sq[j] / n - m * m)
        out[j] = {"ev_bb": m, "se_bb": (var / n) ** 0.5}
    return {"evs": out, "posterior_top": _top_combos(cands, post)}


def _top_combos(cands, post, k: int = 12) -> list:
    order = np.argsort(-post)[:k]
    return [("".join(cands[i]), round(float(post[i]), 4)) for i in order]


def save_spots(spots: list[Spot], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump([s.to_json() for s in spots], f)


def load_spots(path: str) -> list[Spot]:
    with open(path) as f:
        return [Spot.from_json(d) for d in json.load(f)]
