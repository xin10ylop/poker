"""Opponent modeling: HUD statistics, showdown memory, sizing tells, tilt, notes.

Every statistic is a Beta-Binomial estimate shrunk toward a population prior,
so a player seen for 8 hands isn't labeled a maniac because he raised twice.
Profiles persist to JSON (one file per platform/player) so notes accumulate
across sessions.
"""
from __future__ import annotations

import json
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from .texture import hand_features, hand_strength
from .view import HandHistory

# Population priors: (mean, pseudo-count).  Roughly online low-stakes 6-max.
PRIORS: dict[str, tuple[float, float]] = {
    "vpip": (0.27, 18), "pfr": (0.18, 18), "limp": (0.10, 15), "threebet": (0.06, 20),
    "fold_to_3bet": (0.55, 12), "steal": (0.36, 10), "fold_to_steal": (0.62, 10),
    "cbet": (0.60, 10), "fold_to_cbet": (0.45, 10), "bet_checked_to": (0.40, 12),
    "donk": (0.10, 10), "barrel": (0.50, 10),
    "fold_vs_bet_flop": (0.42, 10), "fold_vs_bet_turn": (0.45, 10), "fold_vs_bet_river": (0.48, 10),
    "raise_vs_bet": (0.10, 15), "check_raise": (0.08, 12), "fold_vs_raise": (0.50, 10), "wtsd": (0.28, 12), "wsd": (0.50, 10),
    # Research (LeakBuster / GTO Wizard pool data): rivers are under-bluffed (12-25% weak hands vs
    # 18-37% GTO), overbets ~25% bluffs, small/block bets rarely bluffs; river raises are very strong.
    "river_bluff": (0.22, 6), "bigbet_bluff": (0.25, 6), "smallbet_bluff": (0.20, 6),
    "light_call_river": (0.30, 6), "bad_beat": (0.0, 1),
}
# Priors for unknown players = the real player pool's averages when a fitted population file exists
# (pokerbrain/population.py); the research values above are the fallback.
RESEARCH_PRIORS = dict(PRIORS)


def apply_population_priors() -> None:
    from . import population
    PRIORS.clear()
    PRIORS.update(RESEARCH_PRIORS)
    for k, m in (population.data().get("prior_means") or {}).items():
        if k in PRIORS and k != "bad_beat":
            PRIORS[k] = (float(m), PRIORS[k][1])


apply_population_priors()

ARCHETYPES = {
    "nit": "Very tight, passive-to-solid; plays few hands, rarely bluffs, folds to aggression.",
    "tag": "Tight-aggressive regular; solid ranges, c-bets, balanced-ish.",
    "lag": "Loose-aggressive; wide ranges, frequent 3-bets and barrels, bluffs a lot.",
    "calling_station": "Loose-passive; calls far too much, rarely folds pairs/draws, rarely bluffs.",
    "maniac": "Hyper-aggressive; raises/bets constantly with a very wide, bluff-heavy range.",
    "weak_passive": "Loose-passive fish who limps/calls preflop but folds when he misses.",
    "unknown": "Not enough information yet.",
}


def beta_mean(k: float, n: float, prior: tuple[float, float]) -> float:
    m, s = prior
    return (k + m * s) / (n + s)


@dataclass
class ShowdownRecord:
    hand_id: str
    hole: tuple
    board: list
    lines: list            # [(street, kind, size_frac, strength)]
    final_made: str
    net_bb: float
    ts: float = 0.0


@dataclass
class PlayerProfile:
    name: str
    platform: str = "sim"
    hands: int = 0
    counts: dict = field(default_factory=dict)       # stat -> [successes, opportunities]
    agg: dict = field(default_factory=lambda: {"bets": 0, "raises": 0, "calls": 0, "folds": 0, "checks": 0})
    bet_sizes: list = field(default_factory=list)    # recent postflop bet sizes as pot fractions
    showdowns: list = field(default_factory=list)    # ShowdownRecord dicts (recent)
    notes: list = field(default_factory=list)        # {"hand","text","source","ts"}
    recent: list = field(default_factory=list)       # last hands: {"net_bb","vpip","pfr","agg"}
    big_losses: list = field(default_factory=list)   # hand indices of big losses
    net_vs_hero_bb: float = 0.0
    archetype_probs: dict = field(default_factory=dict)
    last_seen: float = 0.0

    # ------------------------------------------------------------ stats
    def _inc(self, stat: str, success: bool) -> None:
        c = self.counts.setdefault(stat, [0, 0])
        c[1] += 1
        if success:
            c[0] += 1

    def stat(self, name: str) -> float:
        k, n = self.counts.get(name, [0, 0])
        return beta_mean(k, n, PRIORS[name])

    def samples(self, name: str) -> int:
        return self.counts.get(name, [0, 0])[1]

    def raw(self, name: str) -> Optional[float]:
        k, n = self.counts.get(name, [0, 0])
        return k / n if n else None

    def af(self) -> float:
        """Aggression factor (bets+raises)/calls, shrunk toward 2.0."""
        a = self.agg
        return (a["bets"] + a["raises"] + 2.0 * 6) / (a["calls"] + 6)

    def afq(self) -> float:
        a = self.agg
        aggressive = a["bets"] + a["raises"]
        return (aggressive + 0.40 * 20) / (aggressive + a["calls"] + a["folds"] + 20)

    def avg_bet_size(self) -> float:
        xs = self.bet_sizes[-60:]
        return (sum(xs) + 0.6 * 5) / (len(xs) + 5)

    # ------------------------------------------------------------ tilt
    def tilt_signals(self) -> dict:
        """P(tilt) from betting data: a trigger (big loss / bad beat) plus a measurable change in
        behaviour (recent VPIP/PFR vs the player's own baseline, as binomial z-scores).
        Research: humans loosen up and get more aggressive after big losses; decays over ~20 hands."""
        rec = self.recent[-12:]
        if len(rec) < 5 or self.hands < 25:
            return {"score": 0.0, "recent_vpip": None, "recent_pfr": None, "hands_since_big_loss": None}
        n = len(rec)
        rv = sum(r["vpip"] for r in rec) / n
        rp = sum(r["pfr"] for r in rec) / n
        base_v, base_p = self.stat("vpip"), self.stat("pfr")
        zv = (rv - base_v) / math.sqrt(max(1e-4, base_v * (1 - base_v) / n))
        zp = (rp - base_p) / math.sqrt(max(1e-4, base_p * (1 - base_p) / n))
        z = max(0.0, 0.5 * zv + 0.5 * zp)
        since = (self.hands - self.big_losses[-1]) if self.big_losses else None
        trig = 0.0
        if since is not None and since <= 20:
            trig = 1.0 - since / 20.0
        recent_net = sum(r["net_bb"] for r in rec)
        x = 1.2 * z + 1.5 * trig + (0.4 if recent_net < -60 else 0.0) - 3.0
        score = 1.0 / (1.0 + math.exp(-x))
        return {"score": round(score, 2), "recent_vpip": round(rv, 2), "recent_pfr": round(rp, 2),
                "behaviour_z": round(z, 2), "hands_since_big_loss": since, "recent_net_bb": round(recent_net, 1)}

    # ------------------------------------------------------------ archetype
    def heuristic_archetype(self) -> dict:
        """Nearest-prototype classification in HUD-stat space (softmax over distances)."""
        x = {"vpip": self.stat("vpip"), "pfr": self.stat("pfr"), "afq": self.afq(),
             "fcb": self.stat("fold_to_cbet"), "wtsd": self.stat("wtsd"), "tb": self.stat("threebet")}
        scale = {"vpip": 0.06, "pfr": 0.05, "afq": 0.12, "fcb": 0.14, "wtsd": 0.09, "tb": 0.04}
        d2 = {}
        for arch, proto in PROTOTYPES.items():
            d2[arch] = sum(((x[k] - v) / scale[k]) ** 2 for k, v in proto.items())
        conf = 1.0 - math.exp(-self.hands / 40.0)
        m = min(d2.values())
        ws = {k: math.exp(-0.5 * (v - m)) for k, v in d2.items()}
        tot = sum(ws.values())
        probs = {k: conf * v / tot for k, v in ws.items()}
        probs["unknown"] = 1.0 - conf
        return probs

    def archetype(self) -> tuple[str, float]:
        probs = self.archetype_probs or self.heuristic_archetype()
        k = max(probs, key=probs.get)
        return k, probs[k]

    # ------------------------------------------------------------ tells
    def sizing_tells(self) -> dict:
        """Bluff frequency split by bet size, from showdowns (with priors)."""
        big = [ln for sd in self.showdowns for ln in sd["lines"] if ln[1] in ("bet", "raise") and ln[2] >= 0.75
               and ln[0] in ("turn", "river")]
        small = [ln for sd in self.showdowns for ln in sd["lines"] if ln[1] in ("bet", "raise") and ln[2] < 0.75
                 and ln[0] in ("turn", "river")]
        def bluff_rate(lines, prior):
            b = sum(1 for ln in lines if ln[3] < 0.45)
            return beta_mean(b, len(lines), prior), len(lines)
        bb, nb = bluff_rate(big, PRIORS["bigbet_bluff"])
        sb, ns = bluff_rate(small, PRIORS["smallbet_bluff"])
        return {"bigbet_bluff": round(bb, 2), "bigbet_n": nb, "smallbet_bluff": round(sb, 2), "smallbet_n": ns}

    # ------------------------------------------------------------ export
    def summary(self, bb: int = 1) -> dict:
        arch, p = self.archetype()
        return {
            "name": self.name, "hands": self.hands,
            "vpip": round(self.stat("vpip"), 2), "pfr": round(self.stat("pfr"), 2),
            "3bet": round(self.stat("threebet"), 3), "fold_to_3bet": round(self.stat("fold_to_3bet"), 2),
            "steal": round(self.stat("steal"), 2), "fold_to_steal": round(self.stat("fold_to_steal"), 2),
            "cbet": round(self.stat("cbet"), 2), "fold_to_cbet": round(self.stat("fold_to_cbet"), 2),
            "fold_vs_river_bet": round(self.stat("fold_vs_bet_river"), 2),
            "fold_vs_raise": round(self.stat("fold_vs_raise"), 2),
            "check_raise": round(self.stat("check_raise"), 2),
            "af": round(self.af(), 2), "afq": round(self.afq(), 2),
            "wtsd": round(self.stat("wtsd"), 2), "wsd": round(self.stat("wsd"), 2),
            "avg_bet_size_pot": round(self.avg_bet_size(), 2),
            "archetype": arch, "archetype_conf": round(p, 2),
            "tilt": self.tilt_signals(), "sizing_tells": self.sizing_tells(),
            "net_vs_hero_bb": round(self.net_vs_hero_bb, 1),
            "notes": [n["text"] for n in self.notes[-8:]],
        }

    def to_json(self) -> dict:
        d = self.__dict__.copy()
        return d

    @classmethod
    def from_json(cls, d: dict) -> "PlayerProfile":
        p = cls(name=d["name"])
        p.__dict__.update(d)
        return p


PROTOTYPES = {
    "nit": {"vpip": 0.12, "pfr": 0.09, "afq": 0.35, "fcb": 0.60, "wtsd": 0.26, "tb": 0.025},
    "tag": {"vpip": 0.21, "pfr": 0.17, "afq": 0.45, "fcb": 0.45, "wtsd": 0.28, "tb": 0.07},
    "lag": {"vpip": 0.31, "pfr": 0.25, "afq": 0.55, "fcb": 0.38, "wtsd": 0.30, "tb": 0.11},
    "calling_station": {"vpip": 0.45, "pfr": 0.07, "afq": 0.22, "fcb": 0.15, "wtsd": 0.45, "tb": 0.02},
    "maniac": {"vpip": 0.58, "pfr": 0.44, "afq": 0.65, "fcb": 0.25, "wtsd": 0.40, "tb": 0.25},
    "weak_passive": {"vpip": 0.40, "pfr": 0.07, "afq": 0.25, "fcb": 0.58, "wtsd": 0.30, "tb": 0.02},
}


def _bump(x: float, scale: float) -> float:
    return 1.0 / (1.0 + math.exp(-x / scale))


def _bell(x: float, mu: float, sd: float) -> float:
    return math.exp(-((x - mu) ** 2) / (2 * sd * sd))


class OpponentDB:
    """In-memory profiles with optional JSON persistence."""

    def __init__(self, path: Optional[str] = None, platform: str = "sim"):
        self.path = path
        self.platform = platform
        self.profiles: dict[str, PlayerProfile] = {}
        self.hero_image: dict = {"showed_bluffs": 0, "showed_value": 0, "hands": 0, "vpip": 0, "pfr": 0}
        if path and os.path.exists(path):
            self.load()

    def get(self, name: str) -> PlayerProfile:
        if name not in self.profiles:
            self.profiles[name] = PlayerProfile(name=name, platform=self.platform)
        return self.profiles[name]

    # ------------------------------------------------------------ persistence
    def save(self) -> None:
        if not self.path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        data = {"platform": self.platform, "hero_image": self.hero_image,
                "profiles": {k: v.to_json() for k, v in self.profiles.items()}}
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, self.path)

    def load(self) -> None:
        with open(self.path) as f:
            data = json.load(f)
        self.platform = data.get("platform", self.platform)
        self.hero_image = data.get("hero_image", self.hero_image)
        self.profiles = {k: PlayerProfile.from_json(v) for k, v in data.get("profiles", {}).items()}

    # ------------------------------------------------------------ notes
    def add_note(self, name: str, text: str, hand_id: str = "", source: str = "auto") -> None:
        p = self.get(name)
        p.notes.append({"hand": hand_id, "text": text, "source": source, "ts": time.time()})
        if len(p.notes) > 60:
            p.notes = p.notes[-60:]

    # ------------------------------------------------------------ tracking
    def update(self, hh: HandHistory) -> list[str]:
        """Update every player's profile from a finished hand. Returns notable events."""
        events: list[str] = []
        names, pos = hh.names, hh.positions
        seats = list(range(len(names)))
        pre = [a for a in hh.actions if a.street == "preflop" and not a.kind.startswith("post")]
        acted: set[int] = set()
        vp: dict[int, bool] = {}
        pf: dict[int, bool] = {}
        raises = 0
        first_raiser: Optional[int] = None
        last_raiser: Optional[int] = None
        limpers: set[int] = set()
        callers_after_raise = 0
        faced_3bet: set[int] = set()
        for a in pre:
            s, prof = a.seat, self.get(names[a.seat])
            first = s not in acted
            if first:
                vp[s], pf[s] = False, False
                if raises == 0 and pos[s] != "BB":
                    prof._inc("limp", a.kind == "call")
                if raises == 0 and not limpers and pos[s] in ("CO", "BTN", "SB"):
                    prof._inc("steal", a.kind == "raise")
                if raises == 1 and s != first_raiser:
                    prof._inc("threebet", a.kind == "raise")
                    if (pos[s] in ("SB", "BB") and first_raiser is not None
                            and pos[first_raiser] in ("CO", "BTN", "SB") and callers_after_raise == 0 and not limpers):
                        prof._inc("fold_to_steal", a.kind == "fold")
            if raises == 2 and s == first_raiser and s not in faced_3bet:
                faced_3bet.add(s)
                prof._inc("fold_to_3bet", a.kind == "fold")
            if a.kind in ("call", "raise"):
                vp[s] = True
            if a.kind == "raise":
                pf[s] = True
                raises += 1
                last_raiser = s
                callers_after_raise = 0
                if raises == 1:
                    first_raiser = s
            elif a.kind == "call":
                if raises == 0:
                    limpers.add(s)
                else:
                    callers_after_raise += 1
            acted.add(s)
        for s in acted:
            prof = self.get(names[s])
            prof._inc("vpip", vp.get(s, False))
            prof._inc("pfr", pf.get(s, False))

        # ---------------- postflop
        saw_flop = {a.seat for a in hh.actions if a.street == "flop"}
        folded_at: dict[int, str] = {}
        for a in hh.actions:
            if a.kind == "fold":
                folded_at[a.seat] = a.street
        board_by_street = {"flop": hh.board[:3], "turn": hh.board[:4], "river": hh.board[:5]}
        lines: dict[int, list] = {s: [] for s in seats}
        prev_aggr = last_raiser
        for street in ("flop", "turn", "river"):
            acts = [a for a in hh.actions if a.street == street]
            if not acts:
                continue
            level = 0
            street_in = {s: 0 for s in seats}
            checked: set[int] = set()
            faced_cbet: set[int] = set()
            first_bettor: Optional[int] = None
            acted_st: set[int] = set()
            street_aggr: Optional[int] = None
            aggressors: set[int] = set()
            for a in acts:
                s, prof = a.seat, self.get(names[a.seat])
                facing = level > street_in[s]
                first = s not in acted_st
                if street == "flop" and s == last_raiser and first_bettor is None and first:
                    prof._inc("cbet", a.kind in ("bet", "raise"))
                if (street == "flop" and facing and first_bettor == last_raiser and s != last_raiser
                        and s not in faced_cbet):
                    faced_cbet.add(s)
                    prof._inc("fold_to_cbet", a.kind == "fold")
                if facing:
                    if s in aggressors:
                        prof._inc("fold_vs_raise", a.kind == "fold")
                    else:
                        prof._inc(f"fold_vs_bet_{street}", a.kind == "fold")
                    prof._inc("raise_vs_bet", a.kind == "raise")
                    if s in checked:
                        prof._inc("check_raise", a.kind == "raise")
                elif first and first_bettor is None:
                    if s == prev_aggr:
                        if street != "flop":
                            prof._inc("barrel", a.kind == "bet")
                    elif prev_aggr is not None and not checked:
                        prof._inc("donk", a.kind == "bet")
                    else:
                        prof._inc("bet_checked_to", a.kind == "bet")
                pot_before = max(1, a.pot_before)
                if a.kind in ("bet", "raise"):
                    size = (a.to - level) / (pot_before + (level - street_in[s]))
                    prof.bet_sizes.append(round(size, 3))
                    if len(prof.bet_sizes) > 200:
                        prof.bet_sizes = prof.bet_sizes[-200:]
                    prof.agg["bets" if a.kind == "bet" else "raises"] += 1
                    if first_bettor is None:
                        first_bettor = s
                    street_aggr = s
                    aggressors.add(s)
                    lines[s].append((street, a.kind, round(size, 2)))
                    level = a.to
                elif a.kind == "call":
                    prof.agg["calls"] += 1
                    lines[s].append((street, "call", round((a.added) / pot_before, 2)))
                elif a.kind == "fold":
                    prof.agg["folds"] += 1
                elif a.kind == "check":
                    prof.agg["checks"] += 1
                    checked.add(s)
                    lines[s].append((street, "check", 0.0))
                street_in[s] = a.to
                acted_st.add(s)
            prev_aggr = street_aggr

        showdown_seats = set(hh.shown.keys())
        for s in saw_flop:
            prof = self.get(names[s])
            reached = s in showdown_seats
            prof._inc("wtsd", reached)
            if reached:
                prof._inc("wsd", hh.net.get(s, 0) > 0)

        # ---------------- showdown memory & auto-notes
        for s, hole in hh.shown.items():
            prof = self.get(names[s])
            annotated = []
            for (street, kind, size) in lines[s]:
                board = board_by_street[street]
                hs = hand_strength(hole, board) if len(board) >= 3 else 0.5
                annotated.append((street, kind, size, round(hs, 3)))
            feats = hand_features(hole, hh.board) if len(hh.board) >= 3 else None
            rec = {"hand_id": hh.hand_id, "hole": list(hole), "board": list(hh.board), "lines": annotated,
                   "final_made": feats.made if feats else "preflop", "net_bb": hh.net.get(s, 0) / hh.bb,
                   "ts": time.time()}
            prof.showdowns.append(rec)
            if len(prof.showdowns) > 80:
                prof.showdowns = prof.showdowns[-80:]
            for (street, kind, size, hs) in annotated:
                if street == "river" and kind in ("bet", "raise"):
                    prof._inc("river_bluff", hs < 0.45)
                if street == "river" and kind == "call":
                    prof._inc("light_call_river", hs < 0.5)
            note = _auto_note(names[s], hole, hh, annotated, rec["final_made"])
            if note:
                self.add_note(names[s], note, hh.hand_id)
                events.append(f"{names[s]}: {note}")

        # ---------------- per-hand bookkeeping, tilt, hero image
        for s in seats:
            if s not in acted and pos[s] != "BB":
                continue
            prof = self.get(names[s])
            prof.hands += 1
            prof.last_seen = time.time()
            net_bb = hh.net.get(s, 0) / hh.bb
            prof.recent.append({"net_bb": round(net_bb, 1), "vpip": int(vp.get(s, False)),
                                "pfr": int(pf.get(s, False))})
            if len(prof.recent) > 40:
                prof.recent = prof.recent[-40:]
            bad_beat = bool(hh.ev_net and hh.net.get(s, 0) < 0 and hh.ev_net.get(s, 0) > 0
                            and net_bb <= -20)
            if net_bb <= -30 or bad_beat:
                prof.big_losses.append(prof.hands)
                prof.big_losses = prof.big_losses[-10:]
                if bad_beat:
                    prof.big_losses.append(prof.hands)   # counts double: strongest tilt trigger
                    events.append(f"{names[s]} suffered a BAD BEAT ({net_bb:.0f}bb as the all-in favorite)"
                                  " - high tilt risk")
                    self.add_note(names[s], f"Lost {-net_bb:.0f}bb all-in as the favorite (bad beat)",
                                  hh.hand_id)
                else:
                    events.append(f"{names[s]} lost a big pot ({net_bb:.0f}bb) - watch for tilt")
            if hh.hero_seat is not None and s != hh.hero_seat:
                prof.net_vs_hero_bb += net_bb if hh.net.get(hh.hero_seat, 0) != 0 else 0.0
        if hh.hero_seat is not None:
            hs_ = hh.hero_seat
            self.hero_image["hands"] += 1
            self.hero_image["vpip"] += int(vp.get(hs_, False))
            self.hero_image["pfr"] += int(pf.get(hs_, False))
            if hs_ in hh.shown:
                river_lines = [ln for ln in lines[hs_] if ln[0] == "river" and ln[1] in ("bet", "raise")]
                if river_lines and len(hh.board) == 5:
                    hs_val = hand_strength(hh.shown[hs_], hh.board)
                    key = "showed_bluffs" if hs_val < 0.45 else "showed_value"
                    self.hero_image[key] += 1
        return events

    def hero_image_summary(self) -> dict:
        h = self.hero_image
        n = max(1, h["hands"])
        return {"hands": h["hands"], "vpip_seen": round(h["vpip"] / n, 2), "pfr_seen": round(h["pfr"] / n, 2),
                "caught_bluffing": h["showed_bluffs"], "showed_value": h["showed_value"]}


def _auto_note(name: str, hole, hh: HandHistory, lines: list, made: str) -> Optional[str]:
    """Rule-based note for a notable showdown (the LLM can add richer notes)."""
    from .texture import pretty
    hand = pretty(hole)
    big_bluffs = [ln for ln in lines if ln[1] in ("bet", "raise") and ln[3] < 0.35 and ln[0] in ("turn", "river")]
    if big_bluffs:
        st, kind, size, hs = big_bluffs[-1]
        return (f"BLUFFED: {kind} {size:.0%} pot on the {st} with {hand} ({made.replace('_', ' ')}) "
                f"on {pretty(hh.board)}")
    light_calls = [ln for ln in lines if ln[1] == "call" and ln[0] == "river" and ln[3] < 0.4]
    if light_calls:
        return f"Called the river light with {hand} ({made.replace('_', ' ')}) on {pretty(hh.board)}"
    value_big = [ln for ln in lines if ln[1] in ("bet", "raise") and ln[2] >= 0.9 and ln[3] >= 0.8]
    if value_big:
        st, kind, size, hs = value_big[-1]
        return f"Big {kind} ({size:.0%} pot) on the {st} was strong: {hand} ({made.replace('_', ' ')})"
    slowplay = [ln for ln in lines if ln[1] in ("check", "call") and ln[3] >= 0.93 and ln[0] == "flop"]
    if slowplay and not any(ln[1] in ("bet", "raise") for ln in lines if ln[0] == "flop"):
        return f"Slow-played a monster on the flop: {hand} ({made.replace('_', ' ')})"
    return None
