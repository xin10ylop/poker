"""The 'ultimate' configuration: which prompt, which components, when to call Opus.

Filled in from the benchmark results (see docs/EXPERIMENTS.md).  Edit here to
change the default build; every field is overridable from the CLI or code.
"""

ULTIMATE = {
    "variant": "v3_elite",          # system prompt / dashboard variant for Opus
    "effort": "medium",             # Opus 5.5 effort (thinking depth / latency)
    "use_jev": True,                # Jev reads feed both the engine and the dashboard
    "jev_weight": 0.35,             # how strongly Jev reads move the engine's villain model
    "verifier": False,              # Jev blunder-check veto on Opus decisions
    "escalation": {                 # when a decision is worth a model call
        "mode": "key",
        "min_pot_bb": 12.0,
        "close_ev_bb": 1.0,
        "preflop": False,
    },
}
