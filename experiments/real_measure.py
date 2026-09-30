"""Measure, on real hands, every behaviour the villain model currently assumes from bot tuning.

One pass over the hand files (tracker learning in time order).  Writes results/real/measure.json.
  (a) fold / call / raise facing a bet by (street, bet|raise, hu|mw, size bucket, commitment bucket)
  (b) what the street's last bettor showed down, by street x size bucket (bluff = hand strength < 0.45)
  (c) tilt: behaviour in the 12 hands after a big loss vs the same player's baseline
  (d) pool stats by table size (<= 6 seats vs >= 7)
  (e) preflop continue rate facing a single open, by position and opener position

  python experiments/real_measure.py --files 'real/ps25/*.phhs' --out results/real/measure.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.real_players import label_of, situation_of  # noqa: E402
from pokerbrain.adapters.phh import load_phhs, replay  # noqa: E402
from pokerbrain.opponents import PRIORS, OpponentDB  # noqa: E402
from pokerbrain.texture import hand_strength  # noqa: E402

SIZE = [0.0, 0.3, 0.6, 1.0, 2.0, 99.0]
COMMIT = [0.0, 0.35, 0.6, 0.9, 1.01]
TILT_WINDOW = 12
BIG_LOSS = -30.0


def bucket(x, edges):
    for lo, hi in zip(edges, edges[1:]):
        if lo <= x < hi:
            return f"{lo:g}-{hi:g}"
    return f"{edges[-2]:g}+"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", required=True)
    ap.add_argument("--max-files", type=int, default=10 ** 9)
    ap.add_argument("--out", default="results/real/measure.json")
    a = ap.parse_args()
    t0 = time.time()
    hands = []
    for fn in sorted(glob.glob(a.files))[: a.max_files]:
        hands += load_phhs(fn)
    hands.sort(key=lambda h: int(h.get("hand", 0)))
    print(f"{len(hands)} hands ({time.time() - t0:.0f}s)", flush=True)

    facing = defaultdict(lambda: defaultdict(int))            # (a)
    shown_bet = defaultdict(lambda: [0, 0, 0.0])              # (b) key -> [bluffs, n, sum strength]
    tilt = defaultdict(lambda: {"w": defaultdict(lambda: [0, 0]), "b": defaultdict(lambda: [0, 0])})  # (c)
    hands_played, last_loss = defaultdict(int), {}
    db6, db9 = OpponentDB(platform="real"), OpponentDB(platform="real")    # (d)
    cont = defaultdict(lambda: defaultdict(int))              # (e)
    used = 0
    for k, h in enumerate(hands):
        r = replay(h)
        if r is None:
            continue
        used += 1
        hist = r.history
        n = len(hist.names)
        # ---- (a) facing-bet responses with commitment; per-player postflop events for (c)
        events = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        for dp in r.points:
            v = dp.view
            if v.street == "preflop":
                continue
            f = situation_of(v, dp.seat)
            y = label_of(dp.kind, f["facing"])
            if f["facing"]:
                key = "|".join([v.street, "raise" if f["facing_raise"] else "bet", "mw" if f["multiway"] else "hu",
                                bucket(f["size"], SIZE), bucket(f["commit"], COMMIT)])
                facing[key][y] += 1
                if not f["facing_raise"]:
                    events[dp.name]["fold_vs_bet"][1] += 1
                    events[dp.name]["fold_vs_bet"][0] += y == "fold"
                    events[dp.name]["raise_vs_bet"][1] += 1
                    events[dp.name]["raise_vs_bet"][0] += y == "raise"
            elif f["situation"] == "stab":
                events[dp.name]["bet_checked_to"][1] += 1
                events[dp.name]["bet_checked_to"][0] += y == "raise"
        # ---- (b) last bettor of each street, if his cards were really shown
        for st, nb in (("flop", 3), ("turn", 4), ("river", 5)):
            acts = [x for x in hist.actions if x.street == st]
            bets = [x for x in acts if x.kind in ("bet", "raise")]
            if not bets or len(hist.board) < nb:
                continue
            last = bets[-1]
            if last.seat not in r.shown:
                continue
            size = last.added / max(1, last.pot_before)
            hs = hand_strength(tuple(r.shown[last.seat]), hist.board[:nb])
            key = f"{st}|{bucket(size, SIZE)}|{'raise' if last.kind == 'raise' else 'bet'}"
            c = shown_bet[key]
            c[0] += hs < 0.45; c[1] += 1; c[2] += hs
        # ---- (c) tilt window bookkeeping and (e) preflop continue
        pre = [x for x in hist.actions if x.street == "preflop" and not x.kind.startswith("post")]
        seen, raises, callers, first_raiser = set(), 0, 0, None
        vp, pf = {}, {}
        for x in pre:
            s = x.seat
            if s not in seen:
                seen.add(s)
                if raises == 1 and callers == 0 and s != first_raiser:
                    opos = hist.positions[first_raiser]
                    key = f"{hist.positions[s]}|{'steal' if opos in ('CO', 'BTN', 'SB') else 'early'}"
                    cont[key][label_of(x.kind, True)] += 1
            vp[s] = vp.get(s, False) or x.kind in ("call", "raise")
            pf[s] = pf.get(s, False) or x.kind == "raise"
            if x.kind == "raise":
                raises += 1; callers = 0
                if raises == 1:
                    first_raiser = s
            elif x.kind == "call" and raises:
                callers += 1
        for s in seen:
            name = hist.names[s]
            hands_played[name] += 1
            since = hands_played[name] - last_loss.get(name, -10 ** 9)
            side = "w" if since <= TILT_WINDOW else "b"
            t = tilt[name][side]
            t["vpip"][1] += 1; t["vpip"][0] += vp.get(s, False)
            t["pfr"][1] += 1; t["pfr"][0] += pf.get(s, False)
            for stat, (kk, nn) in events.get(name, {}).items():
                t[stat][0] += kk; t[stat][1] += nn
            if hist.net.get(s, 0) / hist.bb <= BIG_LOSS:
                last_loss[name] = hands_played[name]
        # ---- (d)
        (db6 if r.seat_count <= 6 else db9).update(hist)
        if (k + 1) % 50000 == 0:
            print(f"  {k + 1} hands ({time.time() - t0:.0f}s)", flush=True)

    # ---- aggregate
    out = {"hands": used}
    out["facing"] = {k: dict(v) for k, v in sorted(facing.items())}
    out["shown_bettor"] = {k: {"bluff_rate": round(v[0] / v[1], 3), "n": v[1], "mean_strength": round(v[2] / v[1], 3)}
                           for k, v in sorted(shown_bet.items()) if v[1] >= 20}
    stats = ["vpip", "pfr", "fold_vs_bet", "raise_vs_bet", "bet_checked_to"]
    diffs = {s: [] for s in stats}
    for name, t in tilt.items():
        if t["w"]["vpip"][1] < 20 or t["b"]["vpip"][1] < 100:
            continue
        for s in stats:
            kw, nw = t["w"][s]; kb, nb_ = t["b"][s]
            if nw >= 10 and nb_ >= 30:
                diffs[s].append((kw / nw - kb / nb_, nw, kb / nb_))
    out["tilt"] = {s: {"players": len(d), "mean_change": round(sum(x[0] * x[1] for x in d) / max(1, sum(x[1] for x in d)), 4),
                       "baseline": round(sum(x[2] * x[1] for x in d) / max(1, sum(x[1] for x in d)), 4)}
                   for s, d in diffs.items()}
    pool = {}
    for label, db in (("6max", db6), ("9max", db9)):
        pool[label] = {}
        for name in PRIORS:
            kk = sum(p.counts.get(name, [0, 0])[0] for p in db.profiles.values())
            nn = sum(p.counts.get(name, [0, 0])[1] for p in db.profiles.values())
            if nn >= 200:
                pool[label][name] = [round(kk / nn, 4), nn]
    out["by_table_size"] = pool
    out["preflop_continue"] = {k: dict(v) for k, v in sorted(cont.items())}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)

    print("\n(a) fold rate facing a bet by commitment (heads-up, bets only):")
    for st in ("flop", "turn", "river"):
        for sb in ["0-0.3", "0.3-0.6", "0.6-1", "1-2", "2-99"]:
            row = []
            for cb in ["0-0.35", "0.35-0.6", "0.6-0.9", "0.9-1.01"]:
                c = facing.get(f"{st}|bet|hu|{sb}|{cb}")
                nn = sum(c.values()) if c else 0
                row.append(f"{c['fold'] / nn:.2f}({nn})" if nn >= 30 else "   -    ")
            print(f"  {st:5s} size {sb:8s} commit: " + "  ".join(row))
    print("\n(b) last bettor shown down: bluff rate (strength<0.45) by street x size:")
    for k, v in out["shown_bettor"].items():
        print(f"  {k:22s} bluff {v['bluff_rate']:.2f}  mean strength {v['mean_strength']:.2f}  n={v['n']}")
    print("\n(c) tilt: change in the 12 hands after a >=30bb loss vs own baseline:")
    for s, v in out["tilt"].items():
        print(f"  {s:16s} baseline {v['baseline']:.3f}  change {v['mean_change']:+.3f}  (players {v['players']})")
    print("\n(d) pool by table size:")
    for name in ("vpip", "pfr", "limp", "threebet", "fold_to_steal", "cbet", "fold_vs_bet_flop", "wtsd"):
        a6, a9 = pool["6max"].get(name), pool["9max"].get(name)
        print(f"  {name:18s} 6max {a6[0] if a6 else '-'}  9max {a9[0] if a9 else '-'}")
    print("\n(e) preflop facing a single open, no callers: continue rate by position | opener:")
    for k, c in out["preflop_continue"].items():
        nn = sum(c.values())
        if nn >= 200:
            print(f"  {k:14s} n={nn:6d}  fold {c.get('fold', 0) / nn:.2f}  call {c.get('call', 0) / nn:.2f}  raise {c.get('raise', 0) / nn:.2f}")
    print(f"done ({time.time() - t0:.0f}s) -> {a.out}")


if __name__ == "__main__":
    main()
