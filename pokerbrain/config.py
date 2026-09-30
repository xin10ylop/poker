"""The 'ultimate' configuration: which prompt, which components, when to call Opus.

Filled in from the benchmark results (see docs/EXPERIMENTS.md).  Edit here to
change the default build; every field is overridable from the CLI or code.
"""

ULTIMATE = {
    "variant": "v13_final",         # winner of the prompt tournament (see docs/EXPERIMENTS.md)
    "effort": "medium",             # Opus 5.5 effort (thinking depth / latency)
    "use_jev": False,               # Jev router (escalation mode "jev"): ~half the Opus calls, no EV gain
    "jev_weight": 0.0,              # do NOT blend Jev bluff/fold reads into the math (benchmark: hurts)
    "reads_in_dashboard": False,    # show Jev reads to Opus as a second opinion (hurts: Jev over-states bluffs)
    "verifier": False,              # Jev blunder-check veto on Opus decisions
    "mix": False,                   # sample Opus's mixed strategy (True) or commit to its top action
    "override_gate": 0.2,           # overrule the engine only if Opus leaves <= 20% of its mix on the engine's pick
    "require_stakes": True,         # without --stakes/--bankroll Opus is never consulted (its fee must be justified)
    "call_cost_usd": 0.05,          # assumed cost of an Opus call until the running mean is known
    "deadline_s": 25.0,             # hard wall-clock limit per Opus decision; past it the engine's pick is played
    "escalation": {                 # when a decision is worth a model call
        "mode": "postflop",             # every postflop decision worth > 3 model calls; "jev" = budget router
        "tricky_threshold": 0.84,       # Jev difficulty score gate (top ~40% of spots)
        "always_pot_bb": 40.0,
        "min_pot_bb": 12.0,
        "close_ev_bb": 1.0,
        "preflop": False,
        "edge_fraction": 0.02,          # a call must be worth it: pot value x 2% >= call cost (10bb pots at 25NL)
    },
}
