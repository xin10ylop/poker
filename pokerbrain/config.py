"""The 'ultimate' configuration: which prompt, which components, when to call Opus.

Filled in from the benchmark results (see docs/EXPERIMENTS.md).  Edit here to
change the default build; every field is overridable from the CLI or code.
"""

ULTIMATE = {
    "variant": "v3_elite",          # system prompt / dashboard variant for Opus
    "effort": "medium",             # Opus 5.5 effort (thinking depth / latency)
    "use_jev": True,                # Jev = System-1 router that decides when Opus is consulted
    "jev_weight": 0.0,              # do NOT blend Jev bluff/fold reads into the math (benchmark: hurts)
    "reads_in_dashboard": False,    # show Jev reads to Opus as a second opinion (decided by round 2)
    "verifier": False,              # Jev blunder-check veto on Opus decisions
    "escalation": {                 # when a decision is worth a model call
        "mode": "jev",
        "tricky_threshold": 1.6,
        "always_pot_bb": 40.0,
        "min_pot_bb": 12.0,
        "close_ev_bb": 1.0,
        "preflop": False,
    },
}
