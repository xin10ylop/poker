"""Psychology test on REAL players: can Opus predict what a real player does next better than the stats?

Replays the real hands in time order (tracker learning as live), then picks facing-a-bet postflop decisions
from the later half by players the tracker has studied for 150+ hands.  For each it writes a prompt
(public hand history + the tracker's dossier on that player, no model predictions) and stores the
model's predictions separately.  Blind Opus subagents answer; score with --score.

  python experiments/real_opus_export.py --files 'real/ps25/*.phhs' --n 90 --out bridge_real
  python experiments/real_opus_export.py --score bridge_real
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.real_players import hud_probs, label_of, model_probs, situation_of  # noqa: E402
from pokerbrain import population  # noqa: E402
from pokerbrain.adapters.phh import load_phhs, replay  # noqa: E402
from pokerbrain.cards import stable_hash  # noqa: E402
from pokerbrain.opponents import OpponentDB, PlayerProfile  # noqa: E402
from pokerbrain.texture import pretty  # noqa: E402
from pokerbrain.villain import VillainParams  # noqa: E402

SYSTEM = """You are an expert poker psychologist and player-profiler.  You see a real online no-limit hold'em
cash game hand (2009, $0.10/$0.25 blinds) up to the moment one player must act facing a bet, plus a dossier
of everything a tracker has learned about that player over the previous hands (stats shrunk toward
population averages, showdowns, sizing tells, tilt evidence).  You do NOT see anyone's cards.
Predict what THIS player actually does next.  Give calibrated probabilities (they are scored with
log-loss against what really happened), then commit.
Answer with ONE JSON object: {"fold": p, "call": p, "raise": p, "reasoning": "<max 2 sentences>"}"""

TRIM = ("His result in pots against you", "Engine estimate of his range", "  top hand classes", "  hero equity")


def dossier(db: OpponentDB, name: str, view, seat: int) -> str:
    from pokerbrain.llm.render import render_dossier

    class _VI:                                       # minimal stand-in for the engine's VillainInfo
        pass
    vi = _VI()
    p = view.players[seat]
    vi.name, vi.position, vi.stack_bb = name, p.position, (p.stack + p.bet) / view.bb
    vi.params = VillainParams.from_profile(db.get(name))
    vi.composition, vi.combos, vi.range_text, vi.equity_vs = {}, 0.0, "", 0.0
    lines = render_dossier(db, vi, view, None).splitlines()
    lines[0] = lines[0].replace("observed this session", "observed so far")
    return "\n".join(ln for ln in lines if not ln.startswith(TRIM))


def history(view, actor: int) -> str:
    bb = view.bb
    out = []
    who = {p.seat: f"{'PLAYER' if p.seat == actor else 'P' + str(p.seat + 1)} ({p.position})" for p in view.players}
    for street, nb in (("preflop", 0), ("flop", 3), ("turn", 4), ("river", 5)):
        acts = [a for a in view.actions if a.street == street and not a.kind.startswith("post")]
        if street != "preflop" and len(view.board) < nb:
            break
        txt = []
        for a in acts:
            if a.kind in ("bet", "raise"):
                txt.append(f"{who[a.seat]} {'bets' if a.kind == 'bet' else 'raises to'} {a.to / bb:.1f}bb")
            elif a.kind == "call":
                txt.append(f"{who[a.seat]} calls")
            else:
                txt.append(f"{who[a.seat]} {a.kind}s")
        board = f" [{pretty(view.board[:nb])}]" if nb else ""
        out.append(f"{street.upper()}{board}: " + ("; ".join(txt) if txt else "(no action yet)"))
    return "\n".join(out)


def export(a):
    hands = []
    for fn in sorted(glob.glob(a.files)):
        hands += load_phhs(fn)
    hands.sort(key=lambda h: int(h.get("hand", 0)))
    start = int(0.5 * len(hands))
    db = OpponentDB(platform="real")
    cands = []
    for k, h in enumerate(hands):
        r = replay(h)
        if r is None:
            continue
        if k >= start:
            for dp in r.points:
                v = dp.view
                if v.street == "preflop":
                    continue
                prof = db.get(dp.name)
                if prof.samples("vpip") < 150:
                    continue
                f = situation_of(v, dp.seat)
                if not f["facing"] or not f["can_raise"]:
                    continue
                key = stable_hash("opus-real", r.history.hand_id, dp.seat, len(v.actions))
                if key % 1000 >= 4:                 # ~0.4% of eligible points
                    continue
                players = ", ".join(f"{'PLAYER' if p.seat == dp.seat else 'P' + str(p.seat + 1)} {p.position} "
                                    f"{(p.stack + p.bet) / v.bb:.0f}bb{'' if p.in_hand else ' (folded)'}" for p in v.players)
                call = v.legal.call_amount / v.bb
                user = (f"Table: {r.seat_count}-max, {len(v.players)} players dealt. Stacks at the start of this street: {players}\n\n"
                        f"Hand so far:\n{history(v, dp.seat)}\n\n"
                        f"NOW: PLAYER ({v.players[dp.seat].position}) faces a {'raise' if f['facing_raise'] else 'bet'}: "
                        f"calling costs {call:.1f}bb, pot is {v.pot / v.bb:.1f}bb including the bet "
                        f"(bet = {f['size']:.0%} of the pot before it), {'multiway' if f['multiway'] else 'heads-up'} pot, "
                        f"his remaining stack {v.players[dp.seat].stack / v.bb:.0f}bb.  He can fold, call or raise.\n\n"
                        f"Dossier on PLAYER:\n{dossier(db, dp.name, v, dp.seat)}")
                cands.append({"key": key, "id": f"r{r.history.hand_id}-{dp.seat}-{len(v.actions)}", "user": user,
                              "y": label_of(dp.kind, True), "f": f, "n": prof.samples("vpip"),
                              "hud": hud_probs(prof, f), "studied": model_probs(v, dp.seat, VillainParams.from_profile(prof), f),
                              "unknown": model_probs(v, dp.seat, VillainParams.from_profile(PlayerProfile(name="?")), f),
                              "pop": {"fold": population.interp(population.curve("fold", v.street, f["facing_raise"], f["multiway"]), f["size"]),
                                      "raise": population.interp(population.curve("raise", v.street, f["facing_raise"], f["multiway"]), f["size"])}})
        db.update(r.history)
    cands.sort(key=lambda c: c["key"])
    pick = cands[: a.n]
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "SYSTEM_PROMPT.txt"), "w") as fh:
        fh.write(SYSTEM)
    for c in pick:
        with open(os.path.join(a.out, c["id"] + ".txt"), "w") as fh:
            fh.write(c["user"])
        c["pop"]["call"] = max(0.0, 1 - c["pop"]["fold"] - c["pop"]["raise"])
    truth = {c["id"]: {k: c[k] for k in ("y", "n", "hud", "studied", "unknown", "pop")} for c in pick}
    json.dump(truth, open(os.path.join(os.path.dirname(a.out.rstrip("/")), os.path.basename(a.out.rstrip("/")) + "_truth.json"), "w"), indent=1)
    json.dump([c["id"] for c in pick], open(os.path.join(a.out, "index.json"), "w"), indent=1)
    print(f"{len(cands)} candidates, exported {len(pick)} prompts to {a.out}")
    print(" ".join(c["id"] for c in pick))


def score(a):
    d = a.score.rstrip("/")
    truth = json.load(open(d + "_truth.json"))
    ans = {}
    for fn in sorted(os.listdir(d)):
        if fn.startswith("answers_") and fn.endswith(".json"):
            ans.update(json.load(open(os.path.join(d, fn))))
    ll = lambda P, y: -math.log(min(0.99, max(0.01, float(P.get(y, 0.0)))))
    def norm(P):
        t = sum(max(0.0, float(P.get(k, 0))) for k in ("fold", "call", "raise")) or 1.0
        return {k: max(0.0, float(P.get(k, 0))) / t for k in ("fold", "call", "raise")}
    rows = {m: [] for m in ("opus", "studied", "unknown", "hud", "pop", "opus+studied")}
    for i, t in truth.items():
        if i not in ans:
            continue
        o = norm(ans[i])
        s = norm(t["studied"])
        preds = {"opus": o, "studied": s, "unknown": norm(t["unknown"]), "hud": norm(t["hud"]), "pop": norm(t["pop"]),
                 "opus+studied": {k: 0.5 * o[k] + 0.5 * s[k] for k in o}}
        for m, P in preds.items():
            rows[m].append(ll(P, t["y"]))
    n = len(rows["opus"])
    print(f"scored {n} real decisions (log-loss, lower is better; actual mix: "
          f"{ {y: sum(1 for i, t in truth.items() if i in ans and t['y'] == y) for y in ('fold', 'call', 'raise')} })")
    res = {}
    for m, L in sorted(rows.items(), key=lambda kv: sum(kv[1]) / max(1, len(kv[1]))):
        mu = sum(L) / len(L)
        se = math.sqrt(sum((x - mu) ** 2 for x in L) / max(1, len(L) - 1) / len(L))
        res[m] = [round(mu, 4), round(se, 4)]
        print(f"  {m:13s} {mu:.3f} ± {se:.3f}")
    base = rows["studied"]
    for m in ("opus", "opus+studied"):
        dd = [x - y for x, y in zip(rows[m], base)]
        mu = sum(dd) / len(dd)
        se = math.sqrt(sum((x - mu) ** 2 for x in dd) / max(1, len(dd) - 1) / len(dd))
        print(f"  paired {m} - studied model: {mu:+.3f} ± {se:.3f}")
        res[f"paired_{m}_minus_studied"] = [round(mu, 4), round(se, 4)]
    json.dump(res, open(d + "_scores.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", default="")
    ap.add_argument("--n", type=int, default=90)
    ap.add_argument("--out", default="bridge_real")
    ap.add_argument("--score", default="")
    a = ap.parse_args()
    score(a) if a.score else export(a)
