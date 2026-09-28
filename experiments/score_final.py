"""Score the final fresh round: the complete build (v13 prompt + decisive-override gate + Jev router).

  python experiments/score_final.py <answers_dir> [--ids results/bench/final_fresh90.json]

Reports, paired against the pure quant engine on the same spots:
  * Opus on every spot, argmax (ungated) and with the override gate at several thresholds;
  * the full live pipeline: Jev router (cached scores) decides which postflop spots reach Opus,
    gated Opus decides those, the engine decides the rest (preflop = charts/engine).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.bench import ev_loss, load, merge_answers  # noqa: E402
from pokerbrain.config import ULTIMATE  # noqa: E402
from pokerbrain.spots import Spot  # noqa: E402


def mean_se(x):
    n = len(x)
    m = sum(x) / n
    return m, math.sqrt(sum((v - m) ** 2 for v in x) / max(1, n - 1)) / math.sqrt(n)


def decide(ans: dict, spot: dict, gate):
    """Replicates OpusAgent: argmax of the valid mix, then the decisive-override gate."""
    ids = spot["oracle"]
    mix = []
    for m in ans.get("mix") or []:
        try:
            if m.get("id") in ids and float(m.get("p", 0)) > 0:
                mix.append((m["id"], float(m["p"])))
        except (TypeError, ValueError):
            pass
    pick = max(mix, key=lambda t: t[1])[0] if mix else ans.get("action_id")
    if pick not in ids:
        return spot["quant_choice"]
    tot = sum(p for _, p in mix)
    share = sum(p for i, p in mix if i == spot["quant_choice"]) / tot if tot else 0.0
    if gate is not None and pick != spot["quant_choice"] and share > gate:
        return spot["quant_choice"]
    return pick


def escalated(spot: dict, tricky: float | None) -> bool:
    e = ULTIMATE["escalation"]
    view = Spot.from_json({k: v for k, v in spot.items() if k in Spot.__dataclass_fields__}).view()
    if view.street == "preflop" and not e["preflop"]:
        return False
    pot_bb = view.pot / view.bb
    if pot_bb >= e["always_pot_bb"]:
        return True
    return pot_bb >= 6.0 and tricky is not None and tricky >= e["tricky_threshold"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--ids", default="results/bench/final_fresh90.json")
    ap.add_argument("--router", default="results/bench/jev_router.json")
    ap.add_argument("--out", default="results/bench/final_fresh_scores.json")
    a = ap.parse_args()
    answers = merge_answers(a.dir)
    ids = json.load(open(a.ids))
    allsp = {s["spot_id"]: s for s in load()}
    spots = [allsp[i] for i in ids if i in answers]
    router = json.load(open(a.router)) if os.path.exists(a.router) else {}
    q = {s["spot_id"]: ev_loss(s, {s["quant_choice"]: 1}) for s in spots}
    out = {"n": len(spots), "quant_ev_loss": round(mean_se(list(q.values()))[0], 3)}
    print(f"spots answered: {len(spots)}/{len(ids)}   quant EV loss {out['quant_ev_loss']:.2f} bb/decision")

    def report(name, picks):
        losses = [ev_loss(s, {picks[s['spot_id']]: 1}) for s in spots]
        gains = [q[s["spot_id"]] - l for s, l in zip(spots, losses)]
        m, se = mean_se(gains)
        ov = [(s, g) for s, g in zip(spots, gains) if picks[s["spot_id"]] != s["quant_choice"]]
        row = {"ev_loss": round(sum(losses) / len(losses), 3), "gain_vs_quant": round(m, 3), "se": round(se, 3),
               "overrides": len(ov), "override_gain_bb": round(sum(g for _, g in ov), 2),
               "override_wins": sum(g > 0.01 for _, g in ov), "override_losses": sum(g < -0.01 for _, g in ov)}
        out[name] = row
        print(f"{name:30s} EV loss {row['ev_loss']:5.2f}  gain vs quant {m:+.3f} ± {se:.3f} bb/decision  "
              f"overrides {len(ov):2d} (+{row['override_wins']}/-{row['override_losses']}, net {row['override_gain_bb']:+.1f}bb)")
        return row

    report("opus_all_spots_ungated", {s["spot_id"]: decide(answers[s["spot_id"]], s, None) for s in spots})
    for g in (0.0, 0.1, 0.2, 0.3):
        report(f"opus_all_spots_gate{g}", {s["spot_id"]: decide(answers[s["spot_id"]], s, g) for s in spots})
    gate = ULTIMATE.get("override_gate")
    if gate is None:
        gate = 0.2
    esc = {s["spot_id"]: escalated(s, (router.get(s["spot_id"]) or {}).get("tricky")) for s in spots}
    out["escalated"] = sum(esc.values())
    print(f"Jev router escalates {sum(esc.values())}/{len(spots)} spots to Opus")
    from experiments.bench import _report
    from pokerbrain.agents.llm_agents import EscalationPolicy
    keyp = EscalationPolicy(mode="key", min_pot_bb=12.0, close_ev_bb=1.0)
    routes = {"jev_router": esc, "all_postflop": {s["spot_id"]: s["street"] != "preflop" for s in spots}, "key_spots": {}}
    for s in spots:
        sp = Spot.from_json({k: v for k, v in s.items() if k in Spot.__dataclass_fields__})
        routes["key_spots"][s["spot_id"]] = s["street"] != "preflop" and keyp.should(sp.view(), _report(sp)[1])
    for rname, r in routes.items():
        print(f"-- route {rname}: {sum(r.values())}/{len(spots)} spots reach Opus")
        report(f"pipeline_{rname}+gated", {s["spot_id"]: decide(answers[s["spot_id"]], s, gate) if r[s["spot_id"]]
                                           else s["quant_choice"] for s in spots})
        report(f"pipeline_{rname}+ungated", {s["spot_id"]: decide(answers[s["spot_id"]], s, None) if r[s["spot_id"]]
                                             else s["quant_choice"] for s in spots})
    by = {}
    for s in spots:
        g = q[s["spot_id"]] - ev_loss(s, {decide(answers[s["spot_id"]], s, gate): 1})
        by.setdefault(s["street"], []).append(g)
    out["gated_by_street"] = {k: [len(v), round(sum(v) / len(v), 3)] for k, v in sorted(by.items())}
    print("gated gain by street:", out["gated_by_street"])
    print("\noverrides (ungated): spot style street engine->opus  engine_share  gain_bb")
    detail = []
    for s in spots:
        a_ = answers[s["spot_id"]]
        pick = decide(a_, s, None)
        if pick == s["quant_choice"]:
            continue
        mix = {m.get("id"): float(m.get("p", 0)) for m in (a_.get("mix") or []) if m.get("id") in s["oracle"]}
        tot = sum(v for v in mix.values() if v > 0) or 1.0
        share = mix.get(s["quant_choice"], 0.0) / tot
        g = q[s["spot_id"]] - ev_loss(s, {pick: 1})
        detail.append({"spot": s["spot_id"], "style": s["villain_style"], "street": s["street"],
                       "engine": s["quant_choice"], "opus": pick, "engine_share": round(share, 2),
                       "gain_bb": round(g, 2), "escalated": esc[s["spot_id"]], "rationale": a_.get("rationale", "")})
        print(f"  {s['spot_id']:13s} {s['villain_style']:8s} {s['street']:7s} {s['quant_choice']}->{pick}  "
              f"{share:4.2f}  {g:+7.2f}  {'router' if esc[s['spot_id']] else ''}")
    out["overrides"] = detail
    json.dump(out, open(a.out, "w"), indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
