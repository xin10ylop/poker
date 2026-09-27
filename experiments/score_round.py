"""Score a round of subagent answers: python experiments/score_round.py <bridge_dir> v1 v2 ..."""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.bench import load, score, paired, quant_strategies, load_answers

def main(bdir, variants, extra=None):
    allspots = {s["spot_id"]: s for s in load()}
    answers = {}
    for v in variants:
        try:
            answers[v] = load_answers(os.path.join(bdir, v))
        except Exception as e:  # noqa: BLE001
            print(v, "ERR", e)
    common = set.intersection(*[set(a) for a in answers.values()]) if answers else set()
    spots = [allspots[s] for s in sorted(common) if s in allspots]
    q = quant_strategies(spots)
    sq = score(spots, q)
    print(f"common spots: {len(spots)}")
    print(f"{'variant':16s} {'EVloss':>7s} {'se':>5s} {'best%':>6s} {'blund':>5s}  paired-vs-quant (neg=better)")
    print(f"{'quant':16s} {sq['ev_loss_bb']:7.2f} {sq['se']:5.2f} {sq['best_pct']:6.1f} {sq['blunders_gt10bb']:5d}")
    rows = {}
    for v, st in answers.items():
        st = {k: m for k, m in st.items() if k in common}
        invalid = sum(1 for sid, m in st.items() if any(oid not in allspots[sid]["oracle"] for oid in m))
        sc = score(spots, st)
        m, se = paired(spots, st, q)
        rows[v] = {**sc, "paired_vs_quant": round(m, 3), "paired_se": round(se, 3), "invalid": invalid}
        print(f"{v:16s} {sc['ev_loss_bb']:7.2f} {sc['se']:5.2f} {sc['best_pct']:6.1f} {sc['blunders_gt10bb']:5d}  "
              f"{m:+.2f} ± {se:.2f}  invalid={invalid}")
    return rows, spots

if __name__ == "__main__":
    rows, _ = main(sys.argv[1], sys.argv[2:])
    json.dump(rows, open(os.path.join(sys.argv[1], "scores.json"), "w"), indent=1)
