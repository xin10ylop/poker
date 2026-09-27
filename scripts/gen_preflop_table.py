"""Generate pokerbrain/data/preflop_equity.json: equity of each of the 169 classes
vs 1 and vs 2 random hands (Monte Carlo). Run once; output is committed."""
import json, random, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import eval7
from pokerbrain.cards import ALL_CLASSES, combos_of_class, _E7, ALL_CARDS

rng = random.Random(7)
full = eval7.HandRange("22+,A2+,K2+,Q2+,J2+,T2+,92+,82+,72+,62+,52+,42+,32")
out = {}
for cls in ALL_CLASSES:
    a, b = combos_of_class(cls)[0]
    hero = [_E7[a], _E7[b]]
    eq1 = eval7.py_hand_vs_range_monte_carlo(hero, full, [], 60000)
    # vs 2 random opponents
    deck = [c for c in ALL_CARDS if c not in (a, b)]
    wins = 0.0; N = 12000
    for _ in range(N):
        s = rng.sample(deck, 9)
        board = s[4:9]
        hv = eval7.evaluate(hero + [_E7[c] for c in board])
        v1 = eval7.evaluate([_E7[s[0]], _E7[s[1]]] + [_E7[c] for c in board])
        v2 = eval7.evaluate([_E7[s[2]], _E7[s[3]]] + [_E7[c] for c in board])
        best = max(hv, v1, v2)
        if hv == best:
            wins += 1.0 / ((hv == best) + (v1 == best) + (v2 == best))
    out[cls] = {"eq1": round(eq1, 4), "eq2": round(wins / N, 4)}
    print(cls, out[cls], flush=True)
path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pokerbrain", "data", "preflop_equity.json")
json.dump(out, open(path, "w"), indent=0, sort_keys=True)
print("wrote", path)
