"""Play N hands vs Slumbot with a chosen agent and save the session log."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pokerbrain.adapters.slumbot import run
from pokerbrain.agents.quant_agent import QuantAgent
n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
t = time.time()
s = run(QuantAgent("Hero", seed=11), n, progress=True)
out = {"agent": "quant", "hands": s.hands, "bb100_raw": round(s.bb100(), 2),
       "bb100_baseline_adjusted": round(s.bb100_baseline_adjusted(), 2), "secs": round(time.time() - t),
       "results": s.results}
os.makedirs("results", exist_ok=True)
json.dump(out, open(f"results/slumbot_quant_{n}.json", "w"), indent=1)
print({k: v for k, v in out.items() if k != "results"})
