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

## Opus prompt tournament (Opus via blind subagents in this session)
30 dev spots, stratified by villain type (8 types) and street. Every variant sees exactly the same spots.

### Round 1: eight framings
| Variant | Information | EV loss, mixed strategy | EV loss, argmax | Argmax vs quant | Overrides of the engine (EV gained per override) |
|---|---|---|---|---|---|
| quant engine | – | 4.43 | 4.43 | – | – |
| v1 raw | table + history only | 5.11 | 5.03 | +0.60 | 14 (−1.49) |
| v2 numbers only | + engine numbers | 4.35 | 4.43 | 0.00 | 0 (always follows the engine) |
| v3 elite | full dossier | 4.17 | 3.74 | −0.69 | 5 (+2.46) |
| **v4 exploit** | full dossier | **3.81** | **3.56** | **−0.87 ± 0.68** | 5 (**+3.54**) |
| v5 GTO-guard | full dossier | 4.08 | 3.57 | −0.86 | 8 (+2.90) |
| v6 auditor | full dossier | 4.09 | 4.33 | −0.10 | 8 (+1.08) |
| v7 checklist | full dossier | 4.25 | 3.72 | −0.71 | 7 (+3.06) |
| v8 council | full dossier | 4.27 | 4.47 | +0.04 | 4 (−0.16) |

Findings:
1. **Opus alone is worse than the engine** (v1). **Opus with only the numbers defers to them entirely** (v2).
2. **Opus with the full dossier beats the engine** under every good framing. When it overrides the engine, it gains 2.5–3.5bb per override. Examples:
   - Against the fish, it made a pot-size value bet instead of the engine's 95bb flop shove (+18bb).
   - Against the maniac, it checked to let him bluff.
   - Against the station, it bet big without shoving.
3. **Committing to the top action beats sampling Opus's mix** by 0.25–0.5bb here. Against non-adaptive opponents, randomizing between unequal options is pure cost.
4. Structured-procedure framings (checklist, council) add nothing over a clear exploitative identity.
5. Remaining shared leak: **missed value raises**. No variant shoved the river against the LAG (+27bb available), which mirrors the human population leak documented in the research.

### Round 2: refinements of v4
| Variant | Change | EV loss, argmax | vs v4 | Overrides (gain each) |
|---|---|---|---|---|
| v9 exploit+value | added value-extraction guidance **and** "deviate from a higher engine EV only with strong evidence" | 4.46 | +0.91 worse | 3 (−0.36) |
| v10 = v9 without the engine's pick | anchoring test | 4.40 | +0.84 worse | 2 (+0.49) |
| v11 = v9 without Jev reads | reads test | 3.80 | +0.24 | 5 (+3.75) |

Lesson: **Opus's value is its willingness to override the engine when reads justify it.** The "only with strong evidence" clause suppressed overrides and erased the gain. Removing the (miscalibrated) Jev reads helped. The final candidate is therefore **v12 = v4 without Jev reads**.

### Held-out test: 30 spots never used during prompt design
| Build | EV loss (argmax) | vs pure quant | Overrides (gain each) |
|---|---|---|---|
| Pure quant | 3.96 | – | – |
| v4 exploit (round-1 winner) | 4.59 | +0.63 ± 2.01 | 7 (**−2.68**) |
| **v5 GTO-guard** | **3.55** | **−0.41 ± 1.60** | 6 (**+2.06**) |
| v12 = v4 without reads | 4.60 | +0.64 ± 2.03 | 7 (−2.73) |
| 3-way majority vote | 4.77 | +0.81 | – |

**Over-exploitation case study.** Spot s1086-231-3 is preflop against a calling station who 4-bets.
- The exploit prompts **called** (−36.7bb). Their reasoning: "he lost 100bb last hand, is −900bb against us, and our 3-bet-happy image widens his range."
- GTO-guard **folded**. Its reasoning: "a passive station 4-betting is almost always QQ+/AK, even allowing for mild tilt."

One over-read psychological signal cost more than all the good exploits gained.

**Dev and test combined (60 spots):** v5 is the only prompt that beats the engine on both sets (−0.86 on dev, −0.41 on test). It overrides the engine on 20–25% of spots, and its overrides are profitable on both sets.

### Confirmation round: 30 fresh spots, v5 vs v13
v13 = v5 (GTO-guard) without the Jev reads, plus four "hard-won lessons" taken from v5's test-set failures: passive players' raises are strong even when they're tilted; tilt means more calls and bluff bets, not bluff raises; take the river value raise; commit unless options are truly equal.

| Build | EV loss (argmax) | vs pure quant | Overrides (gain each) |
|---|---|---|---|
| Pure quant | 3.90 | – | – |
| v5 GTO-guard | 4.38 | +0.48 ± 0.86 | 5 (−2.86) |
| **v13** | **3.64** | **−0.26 ± 0.40** | 5 (**+1.58**) |

**Jev reads caused v5's blunder.** v5's big loss was spot s540-483-13, a maniac jamming the turn. It called and cited "the Jev puts [bluffs] near 31%". The oracle says folding was right by 22.6bb. v13 sees no Jev reads and folded. This matches the calibration study above, where Jev over-states bluffs about 5×.

## The decisive-override gate
Every answer carries a mixed strategy. Pooling all 91 engine overrides from rounds 1–3 and the held-out test showed a sharp pattern:

| Override type | n | EV gained per override |
|---|---|---|
| **Decisive**: Opus puts 0% of its mix on the engine's pick | 25 | **+5.89bb** |
| Hedged: the engine's pick keeps some weight | 66 | −1.26bb |

Self-reported `confidence` is a much weaker signal. To rule out a fitting artefact, the gate was checked across the dev/held-out split:

| | Dev rounds (fit) | Held-out rounds (test + confirm) |
|---|---|---|
| Decisive overrides | +9.5bb each | +11.4bb each |
| Hedged overrides | +1.0bb each | **−5.6bb each** (they include every catastrophic blunder: −36.7, −25.4, −22.6bb) |
| Ungated build, gain vs quant per decision | +0.42 | −0.21 |
| Gate 0.2, gain vs quant per decision | +0.29 | +0.53 |

**Rule:** Opus's pick replaces the engine's only if Opus leaves at most 20% of its own mix on the engine's pick. The threshold was fixed at 0.2 before the final round was scored. Every threshold from 0 to 0.3 beat no gate; value collapses above 0.35. Implemented as `OpusAgent(override_gate=0.2)`. The gate is applied after the fact to the mix Opus reports, so the prompt doesn't change. Round 2 showed that telling Opus to "deviate only with strong evidence" destroys its value.

## When should Opus be consulted? (routing, re-examined with Opus's own answers)
The Jev router was calibrated to predict **engine errors**. Once Opus answers existed, the right question was: where does **Opus** improve on the engine? Using 420 answers (prompts v3–v13, all rounds) with the 0.2 gate:

| Spot group | Answers | Gain vs quant per decision |
|---|---|---|
| Spots the Jev router escalates | 172 | −0.30 ± 0.23 |
| Spots it does not escalate | 248 | +0.84 ± 0.31 |
| Pots ≥ 40bb (always escalated) | 46 | −0.20 (−1.23 ungated) |

Clustered by spot, the uncertainty is larger, so this is directional, not proof. Opus's value is in medium pots. In big, low-SPR pots the engine's exact arithmetic beats narrative reads, and "tricky" spots are hard for Opus too.

Routing rules compared per round, gain vs quant per decision with gate 0.2. Each round pools the prompts it tested: dev v3–v11, test v4/v5/v12, confirm v5/v13, fresh v13.

| Rule | Dev | Test | Confirm | Fresh (below) | Share of postflop spots sent to Opus |
|---|---|---|---|---|---|
| **Every postflop decision** | **+0.29** | **+0.76** | −0.12 | **+0.05** | 100% of postflop |
| Key spots (pot ≥ 12bb or close EVs) | +0.01 | +0.76 | −0.12 | +0.05 | ~87% |
| Jev router | +0.01 | −0.51 | −0.12 | +0.05 | ~55% |

"Every postflop decision" is never worse than the Jev router. So it is the default (`escalation.mode = "postflop"`). It skips pots worth less than three model calls, which is the "LLM rake" check. The Jev router remains as a budget mode (`--router jev`) that roughly halves Opus calls.

## Final round: 90 fresh spots, the complete build
90 spots never used in any earlier round, stratified by street. They are mostly LAG, tilter, sizing-tell and station opponents, because the other types' spots were used up. Prompt v13, answered blind by 6 Opus subagents.

| Build (v13) | EV loss | Gain vs quant per decision | Overrides |
|---|---|---|---|
| Pure quant | 4.39 | – | – |
| Opus on every spot, ungated | 4.72 | −0.34 ± 0.35 | 13 (+5/−8, net −30.4bb) |
| Opus on every spot, gate 0.2 | 4.54 | −0.15 ± 0.26 | 6 (+3/−3, net −13.5bb) |
| **Live pipeline: postflop only, gate 0.2** | **4.34** | **+0.05 ± 0.10** | 4 (+2/−2, net +4.2bb) |
| Live pipeline, ungated | 4.53 | −0.14 ± 0.25 | 11 (+4/−7) |

The largest loss was s541-334-8, a station limp-reraising preflop. Opus folded 75o and cited the lesson "passive players' raises are strong", but the oracle says a 4-bet was worth +21.6bb. Each lesson that fixes one case can break another. Preflop spots don't reach Opus in the live pipeline.

The gate blocked a good override once: +7.4bb, folding J-high to a station's turn donk, on a hedged 70/30 mix. It prevented a bad one once: −17.4bb, checking instead of a 150%-pot value bet.

### All held-out rounds pooled (240 answers: test, confirm, fresh)
| Policy | Gain vs pure quant per decision |
|---|---|
| Opus on every spot, ungated | −0.26 ± 0.44 |
| Opus on every spot, gate 0.2 | +0.27 ± 0.34 |
| Postflop only, ungated | +0.05 ± 0.37 |
| **Postflop only, gate 0.2 (final build)** | **+0.27 ± 0.32** |

**Honest bottom line.**
- The prompt tournament shows a **winner's curse**: every round's best prompt gained less, or lost, on the next fresh set.
- Against a strong quant engine with a calibrated opponent model, Opus's judgment is roughly break-even on these simulated opponents.
- The **gate** is the part that generalises. It improved results in 5 of 6 held-out prompt/round combinations and never hurt, because it removes the hedged overrides where the catastrophic blunders live.
- The final build therefore keeps Opus in the loop where it's safe: gated, postflop. The backbone is the engine.
- The simulator is a hard test for an LLM. Its opponents have stable, statistically learnable styles and no chat, timing or history outside the hand log. So the engine's HUD model already captures most of what a "read" can add. Human opponents are where the extra context Opus can use (notes, stories, meta-game) is most likely to matter, and the gate limits the cost of being wrong.
