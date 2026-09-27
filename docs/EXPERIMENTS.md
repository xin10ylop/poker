# Experiments: how the ultimate setup was chosen

## Method

### Why not just play matches?
Full-match win rates are too noisy to rank prompts. In 6-max against a mixed
field, the 95% confidence interval after 600 hands was about ±160 bb/100. A
paired comparison of two agents on the same seeds was still ±200 bb/100.
Opus decisions via API cost cents each, so tens of thousands of hands per
variant are out of reach.

### Decision benchmark with a posterior oracle (`pokerbrain/spots.py`)
1. **Record** hero decision points from six long simulated sessions. Two are
   6-max: field A (nit, station, maniac, fish, tilter) and field B (tag, lag,
   sizer, station, nit). Four are heads-up: vs the sizing-tell bot, the tilter,
   the LAG and the station.
   - Recording starts only after a warm-up (80–150 hands), so the opponent
     dossiers contain real data.
   - Only heads-up pots of at least 6bb are kept, stratified by villain type
     and street.
2. **Oracle EV of every candidate action.**
   - A posterior over the villain's hole cards is computed from his *actual* bot
     policy, averaged over its randomization (24 samples per decision point).
     It never uses his real cards or his real random seed.
   - 400 rollouts on common random numbers per action. The villain follows his
     real policy; hero continues with a TAG policy.
3. **Score** = EV loss per decision against the oracle's best action, in bb.
   Mixed strategies are scored at their expected EV. Every agent plays the same
   spots, which makes the comparison paired.

The deterministic dev/test split is by spot-id hash. Tuning happens on dev;
the headline numbers come from test.

**Caveats**
- The oracle's hero continuation is a TAG policy, so river EVs are the
  cleanest and flop EVs carry some continuation bias.
- Oracle noise (median SE about 2bb per option) sets a floor on the measured
  loss for every agent. Differences between agents remain unbiased.
- Opus was run through Claude Code subagents on this model (Opus) acting on
  the exact rendered prompts, not through the API. API calls run at effort
  "medium", while subagents think at their own default depth.

### Bug caught by the benchmark
The first build showed "all-in" as the oracle's best action in 35% of spots.
The cause was a flaw in the sparring bots: their calling threshold grew
linearly with bet size, so they folded almost everything to large shoves. The
bots were fixed (the size effect now saturates, with sane all-in calling
ranges) and the benchmark was rebuilt. The first version is kept as
`results/bench/spots_v1_bots_overfold.json`.

## Results
(filled in below as experiments complete)

### Benchmark v2 (corrected bots): 330 spots, 313 where the choice matters
Villains: station 79, tilter 69, LAG 67, sizer 67, maniac 17, fish 11, nit 11, TAG 9.
Streets: flop 91, river 87, turn 87, preflop 65.

| Build (all 313 spots) | EV loss / decision (bb) | vs pure quant (paired) | Notes |
|---|---|---|---|
| Oracle | 0 | – | upper bound |
| **Pure quant engine** | **4.41** | – | best action 40% of the time |
| Jev decides (Choice over the scored menu) | 4.90 | +0.49 ± 0.15 worse | probability mass spread over options |
| Jev reads blended into the engine (w=0.5) | 6.26 | +1.85 ± 0.54 worse | bluff reads badly over-estimated (below) |

The quant engine's largest leak is **calling when folding is best** (26 spots, −8.7 bb each). It over-estimates how often these opponents bluff.

### Are the "psychology" reads accurate? (calibration against the oracle posterior)
In 132 facing-a-bet spots, the true bluff probability comes from the villain's real strategy.

| Read | Mean | Brier | Corr. with truth |
|---|---|---|---|
| Truth | 8.2% | – | – |
| Stats model (engine range) | 21.6% | 0.049 | 0.49 |
| Jev P(bluff) | 41.7% | 0.138 | 0.24 |
| Constant 8% | – | 0.022 | – |

Jev reports about 40% bluff whatever the opponent: tilter 46%, station 37%, nit 39%. Any blend with Jev worsens the Brier score. **Conclusion:** Jev's generic intuition is not calibrated for poker ranges, so its probabilities must not be plugged directly into EV math.

| Read | Jev | Stats model |
|---|---|---|
| Player-type accuracy (247 spots with a known type) | 83% | **94%** |
| Tilt detection AUC (tilter tilted vs calm) | 0.951 | **0.987** |

Jev's tilt reading is good (tilt score 1.99 vs 0.63), but it mostly re-reads the tilt evidence the stats model already provides.

### Jev as a router (when should Opus be consulted?)
Task: predict the spots where the engine's pick loses more than 2bb to the best action.

| Signal | AUC | EV loss captured by escalating the top 40% |
|---|---|---|
| EV gap between the engine's top two options | 0.649 | 46% |
| **Jev "how tricky is this decision" (Score)** | **0.663** | **57%** |
| Jev "is the engine's pick a mistake" (Noul) | 0.624 | – |
| Pot size | 0.48 | top 25% of pots hold 39% of the loss |

**Jev's role in the final build:** a fast System-1 gate that decides when to wake Opus (System 2). It isn't used as a source of probabilities for the math.

### Full-match sanity check of the quant backbone (corrected field, no LLM)
| Match | Quant engine (bb/100) | TAG-bot baseline (bb/100) |
|---|---|---|
| 6-max vs nit/station/maniac/fish/tilter, 1,500 hands | +130 ± 87 | +112 ± 95 |
| HU duplicate vs nit (800 hands) | +32 ± 57 | +9 ± 66 |
| HU duplicate vs TAG | +88 ± 84 | +0 ± 87 |
| HU duplicate vs LAG | +72 ± 124 | +10 ± 104 |
| HU duplicate vs station | +357 ± 165 | +70 ± 75 |
| HU duplicate vs maniac | +86 ± 247 | +130 ± 140 |
| HU duplicate vs fish | +48 ± 69 | −5 ± 82 |
| HU duplicate vs tilter | +32 ± 161 | +52 ± 103 |
| HU duplicate vs sizer | +6 ± 131 | +20 ± 100 |

The engine wins against every type. The intervals are still wide, which is why prompts are ranked on the decision benchmark and not on match results.
