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

## Two more checks
### Preflop: static charts vs the engine's opponent-aware EV (full matches)
On the 64 benchmark preflop spots where the choice matters, the live agent's charts lost **5.57bb per decision** and the engine's EV pick lost **3.70**. They disagreed on 23 of 64. Before switching, both were played in full paired matches (`experiments/preflop_test.py`: same decks and bot RNG, two 6-max fields × 6 seeds × 500 hands, plus heads-up duplicate against 7 types × 600 hands):

| Engine decides preflop… | vs charts, all 10,200 hands | 6-max field A | 6-max field B |
|---|---|---|---|
| when facing a raise | −18 ± 28 bb/100 | −65 ± 52 | −11 ± 41 |
| always | −19 ± 36 bb/100 | −99 ± 67 | −40 ± 57 |

**The charts stay.** The benchmark oracle plays the rest of the hand with a TAG bot, not with our postflop engine, so its preflop verdicts don't transfer. Full matches decide structural changes; the benchmark decides single-decision questions.

### Jev as a blunder-check verifier
Jev was asked "is this a clear mistake?" once for every override Opus made in any round: 90 overrides, 41 unique, $0.003.

| Veto rule | Overrides vetoed | EV change |
|---|---|---|
| P(blunder) ≥ 0.85 (preset) | 3 (all winners) | −14.6bb |
| P ≥ 0.7 | 4 | +8.0bb |
| P ≥ 0.5 | 18 (14 losers) | +98.5bb |
| gate 0.2, then Jev veto at 0.5 or 0.7 | 0 extra | ±0 |

Discrimination is weak: AUC 0.58 for predicting losing overrides. At the thresholds where Jev catches losers, it flags the same hedged overrides (engine share 0.25–0.45) that the free decisive-override gate already removes. On the held-out rounds, the gate alone gains +0.27 bb/decision and Jev's veto alone +0.11. **No verifier in the final build.**

## Live end-to-end match (the whole stack, with a real Opus deciding)
`experiments/live_match.py --hands 40 --field station,maniac,tilter --jev`

- **Setup:** 4-handed. Jev routed live, and a blind Opus subagent answered through the file bridge with the gate on.
- **Model calls:** Jev escalated 16 decisions, and Opus took 16 s per decision on average.
- **Gate:** 4 hedged overrides were gated back to the engine, 3 decisive ones were played, and there were 0 errors.
- **Result:** paired against an engine-only replay of the same decks, +114 ± 162 bb/100. Forty hands is noise; this run is an integration test, not a result.

## An engine bug found by the live Opus
In the live match, Opus faced a flop 4-bet holding a set (hand #33, SPR 0.8, the villain with 32.7bb behind). It flagged the engine's numbers as inconsistent:
- the engine rated "call" at +84bb and "all-in" at +10bb;
- the all-in line claimed "villain raises 55%", which is impossible over a shove.

Opus overrode the engine decisively and shoved. The correct value of the shove is about +110bb.

**Cause (`quant._ev_heads_up`).** When a bet puts either player all-in, the re-raise branch was correctly skipped. The villain model's raise mass was still used, though, and scored as "hero loses his bet". Every hand that would raise was counted as a loss, when in fact it calls. **Fix:** an all-in can't be re-raised, so the would-be raises become calls. There is a regression test.

**The fix exposed a second miscalibration.** The size effect in the villain model saturates at 2.5× pot. That's right for bluffs, since nobody folds everything to a huge shove, but it makes a 9× pot shove look called by the same range as a 2.5× bet. Measured against the oracle, the fixed engine over-valued overbet shoves (≥ 5× pot) by +3.5 ± 2.0bb. The bug had been hiding this by accident.

**Second fix: a stack-commitment rule (`villain.COMMIT_STRENGTH`).** When calling costs most of the villain's remaining stack, he continues only with genuinely strong hands. The threshold is effective strength ≥ 0.60, shifted by his folding tendency, and it ramps in from 35% of his stack. The value was tuned on the dev split.

| Engine | Benchmark EV loss vs old engine (313 spots) | Full matches vs old engine (10,200 paired hands) |
|---|---|---|
| Fix only | +0.27 ± 0.21 bb/decision | −7.9 ± 21.1 bb/100 |
| **Fix + commitment 0.60** | +0.09 ± 0.19 | **+1.1 ± 22.0** |
| Fix + commitment 0.70 | +0.19 ± 0.23 | +1.1 ± 22.0 |
| Fix + commitment 0.60, 6-max only, 12 fresh seeds | – | +3.4 ± 13.0 (12,000 hands) |

**Shipped: fix + commitment 0.60.** It performs the same as the old engine, and on the fresh 6-max seeds the early 6-max dip turned out to be noise. It is also *correct*: it stops showing Opus impossible numbers, and it no longer undervalues value shoves at low SPR, which matters most against real opponents who stack off light.

This is the clearest example of Opus's value in the whole project. It isn't out-reading the simulated opponents; it **audits the engine**. It noticed numbers that couldn't be right, acted correctly anyway, and led to a fix.

## Real players (the part that matters)
Everything above was measured against simulated bots. From here, every number comes from **real hands**:
- **HandHQ, PokerStars 25NL, July 2009.** 299,140 real-money online hands, of which 286,306 were replayed. Player names are anonymized but consistent, so each player can be followed across hands.
- **Pluribus.** 10,000 hands of professionals playing 6-max, with every hole card known.

Both come from the public PHH dataset (`pokerbrain/adapters/phh.py`). Every hand is replayed through the rules engine, so the tracker learns exactly as it would live.

### 1. The real pool is not the pool the research priors describe
| Tendency | Real 25NL (1.8M hands of stats) | Research prior it replaced |
|---|---|---|
| VPIP / PFR / 3-bet | 24% / 11% / 3.7% | 27% / 18% / 6% |
| Limp | 14% | 10% |
| Fold to 3-bet / fold to steal | 39% / 75% | 55% / 62% |
| C-bet / turn barrel | 69% / 62% | 60% / 50% |
| Fold to a flop bet | 60% | 42% |
| Fold when their own bet is raised | 30% | 50% |
| River bets that were bluffs (at showdown) | 13% | 22% |

Fold rate climbs steadily with bet size. Heads-up on the flop it goes 17% at 1/7 pot, 50% at 1/2, 67% at pot, 79% at 4× pot. On the river an overbet gets 86–90% folds. Small bets get raised a lot: 26% of flop bets under 30% pot are raised. Players who get raised rarely fold (16–56%).

What river bettors show down when called:
- **Pot-size and bigger:** strong hands 56–70% of the time.
- **Under 30% of the pot:** strong only 27% of the time, weak or nothing 61%.

### 2. Does studying players predict real humans?
The tracker learns in time order. At sampled real postflop decisions, taken before it sees that hand, PokerBrain predicts the player's action from public information only (23,392 scored decisions, log-loss, lower is better):

| Predictor | All | Facing a bet | Player seen 500+ hands |
|---|---|---|---|
| Model with its study of the player | 0.700 | 0.914 | 0.651 |
| Same model, player unknown | 0.724 | 0.931 | 0.714 |
| Player's own raw HUD frequencies | 0.697 | 0.924 | 0.640 |
| Population frequency table | 0.688 | 0.881 | 0.652 |

Two conclusions:
- **Studying players works.** The gain grows with the sample: 0.714 → 0.651 at 500+ hands.
- **The old model's assumptions were wrong for real players.** A plain frequency table beat it.

### 3. Range reading on real hole cards: the model was over-confident
Pluribus hands show every player's cards. For 25NL, the players' own showdowns do. That lets us score how much probability PokerBrain's estimated range puts on the hand the player really held:
- **Pros:** −2.05 bits vs a uniform guess.
- **25NL:** −1.28 bits vs a uniform guess.
- **Postflop narrowing** lost information compared with the preflop range alone.

The per-hand action probabilities were near 0 or 1, so hands real players actually hold got ruled out: slow-plays, thin bets, odd bluffs, loose preflop calls.

**Fix: tempering.** Every per-hand action probability is blended with the range's average for that action, so no hand is ever ruled out entirely. The two weights were chosen on a grid (`villain.TEMPER`, `villain.PF_TEMPER`):

| Postflop / preflop tempering | Pros: range vs uniform | Pros: action log-loss (cards known) | 25NL: range vs uniform | 25NL: action log-loss |
|---|---|---|---|---|
| 0 / 0 (old) | −2.05 bits | 0.699 | −1.28 bits | 0.768 |
| 0.2 / 0.1 | +0.21 | 0.577 | +0.31 | 0.645 |
| **0.35 / 0.45 (shipped)** | **+0.56** | **0.559** | **+0.60** | **0.610** |

With tempering, knowing a player's cards through the model predicts his actions far better than the range average does (0.610 vs 0.709). The card-level logic works once it stops being certain.

### 4. The fix: a population model fitted to real hands
`experiments/fit_population.py` writes `pokerbrain/data/population.json`. It contains:
- **The tracker's priors** for unknown players: the pool's averages.
- **Fold and raise curves** by street, bet-or-raise, heads-up or multiway, and bet size. These replace the MDF-style formula. Each player is shifted from the curve in logit space by how much more or less he folds than the pool.

You can refit the curves on your own hand histories. `POKERBRAIN_POPULATION=<file>` selects a population file, and `none` restores the research priors.

**Held-out result** (`experiments/real_eval.py`):
- **Setup:** 5,728 real decisions from the later half of the hands, every variant scored on identical decision points.
- **What was fitted where:** the population curves come from the earlier half only. The tempering weights were chosen on card-level scores; no part of the eval metric was used for fitting.
- **Metric:** log-loss, lower is better.

| Model | All | Facing a bet | Betting decisions | Player seen 0–19 hands | 500+ hands |
|---|---|---|---|---|---|
| Old model (research priors, formula) | 0.705 | 0.929 | 0.585 | 0.754 | 0.645 |
| Old + tempering | 0.691 | 0.903 | 0.579 | 0.749 | 0.628 |
| Population frequency table | 0.694 | 0.894 | 0.587 | 0.716 | 0.647 |
| Real-data curves | 0.670 | 0.845 | 0.577 | 0.705 | 0.619 |
| **Real-data curves + tempering (shipped)** | **0.665** | **0.843** | 0.571 | **0.701** | **0.612** |
| Same, player unknown (no study) | 0.681 | 0.851 | 0.591 | 0.703 | 0.643 |

- **Paired against the old model: −0.039 ± 0.004**, about nine standard errors.
- The model now beats the plain frequency table, where the old one lost to it.
- Studying the individual player still adds on top: 0.665 studied vs 0.681 unknown, and the gap grows to 0.612 vs 0.643 at 500+ hands.
- Solving fold thresholds on the villain's *current* range beats solving them on his preflop range: 0.670 vs 0.679.

### Trade-off: real players vs the simulated bots
Full paired matches against the simulated bots (10,200 hands, same decks), compared with the old engine:

| Engine | bb/100 vs old engine |
|---|---|
| Real-data priors + curves, no tempering | +4.6 ± 23.4 |
| **Real-data priors + curves + tempering (real-play default)** | **−43.8 ± 27.7** |
| (tempering alone) | −48.3 ± 24.8 |

The real-data curves cost nothing even against bots. **Tempering is the whole gap.** The bots play deterministic threshold strategies, which is exactly what the untempered model assumes, so narrow ranges are right against them. Real people are noisy. On real hole cards, untempered ranges ruled out hands they really held and scored worse than a blind guess.

The default is therefore real-play mode. For bot venues (Slumbot, ACPC, the simulator) run with `POKERBRAIN_POPULATION=none`, which restores the original engine exactly: research priors, formula curves, no tempering. The tempering values live in the population file next to the curves they were fitted with.

### 5. Psychology on real players: can Opus read people better than the statistics?
90 real facing-bet decisions from the later half, all by players the tracker had studied for 150+ hands. Blind Opus subagents saw the public hand and the tracker's dossier on the player: stats, showdowns, sizing tells, tilt evidence, notes. They were asked for fold/call/raise probabilities and scored against what the player really did.

| Predictor | Log-loss (lower is better) |
|---|---|
| **PokerBrain's statistical model** (real-data version) | **0.735 ± 0.053** |
| Opus shown the model's prediction as a baseline | 0.746 (paired +0.011 ± 0.014) |
| 50/50 blend of Opus and the model | 0.738–0.762 |
| Population curve / raw HUD / unknown player | 0.783 / 0.785 / 0.786 |
| **Opus reading the dossier alone** | **0.824** (paired +0.089 ± 0.039 worse) |

Opus reasons from pot odds ("he needs only 14% equity, so folding is rare"). Real players fold far more than pot odds say they should: Opus predicted 45% folds, the players actually folded 63%, and the model predicted 57%. Shown the model's numbers, Opus stops making big mistakes but still can't improve on them.

**On real people, the statistical player study is the psychology that works. The LLM adds no predictive power on top of it.** Opus's proven value in this project is different: it audits the engine, as with the all-in bug above. The final build reflects this. Opus can override the engine only when decisive, and the opponent predictions it sees come from the real-data model.

## Audit for real play
Everything was re-examined with one question: is it correct, stable and fitted for real people rather than for the simulated bots? Three independent code reviews (engine and rules, adapters and CLI, decision and money layer), a stress test on real hands, and measurements of every bot-tuned assumption on the 286,000 real hands.

### Stress test on real decisions
`experiments/stress_real.py` runs the full decision path (charts, engine, opponent tracker) at real decision points whose actor's cards are known: 10,752 from the Pluribus pros' hands and 5,952 from 25NL showdowns.

| | Pluribus (6-max pros), before | 25NL, before | Pluribus, after the audit |
|---|---|---|---|
| Decisions | 10,752 | 5,952 | 10,752 |
| Exceptions | 0 | 0 | 0 |
| Latency p50 / p99 / max | 0.00 / 0.20 / 0.83 s | 0.09 / 0.29 / 0.48 s | 0.00 / 0.17 / 0.30 s |
| Illegal decisions | 1 | 12 | 0 |
| Non-finite or impossible EVs | 0 | 0 | 0 |

The illegal decisions were chart raises in spots where raising was not allowed; adapters already coerced them, and the agent now normalizes chart decisions itself.

### Rules and engine bugs fixed
- **Short big blind.** When the big blind was all-in for less than a blind, the amount to call was set by what he posted rather than the full blind, the minimum raise was too small, and the small blind got a free check. Standard (TDA and online) rules apply now. pokerkit shares the old behaviour, so the cross-check skips short-blind deals.
- **Paired boards.** A pocket pair below a board pair (QQ on K-K-5) was classified as air; counterfeited two pair (54 on 5-4-K-K-A) as strong.
- **Hole-card order** changed a hand's strength (AhKh vs KhAh) in the draw-aware strength and in `strengths_for`.
- **Full-ring seats.** UTG+1, UTG+2 and MP fell back to the cutoff's opening range; 10-handed tables had no early positions at all. They now use the tightest chart, and an early open is treated as early when facing it.
- **Heads-up all-in equity** was unseeded (results varied between runs); now seeded.
- `Decision.normalized` never throws (bad kinds and amounts become check or fold); range strings raise `ValueError` only, and are case-insensitive.

### A selection bias caught on the way
The first version of the fix for unsettleable showdowns dropped those hands. Only 37 of 695 showdown hands in a sample carry recorded winnings, and the player who mucks is the loser, so the dropped hands were mostly ones where the bettor had won and shown value. The remaining shown bets looked twice as bluff-heavy as they are (33% instead of 15%). Those hands are now kept for their cards and actions, with only the money result marked unknown.

### Bot assumptions measured on real players
| Assumption | Measured on real hands | Change |
|---|---|---|
| Players fold more when a call commits their stack (commitment rule, tuned on bots) | They fold slightly *less* (turn, 60–100% pot: 54% fold when cheap vs 38% when it commits them) | Rule disabled in real-play mode; a fitted commitment shift replaces it |
| Tilt: VPIP +80% after a big loss | 12 hands after a 30bb+ loss: VPIP +0.5 points, PFR +0.2, fold-to-bet unchanged (1,102 players) | Tilt effects cut to a third; the score needs a visible change in the player's own play |
| Bluff shares: flop 1.9× and turn 1.4× the river share; big bets 25% / small 20% bluffs | River bets 13–18% bluffs, overbets 12%, raises 4–7%; small vs big is flat | Bluff share by street and size fitted from real showdowns; multipliers for flop/turn selection bias fitted on hole-card data |
| One prior for all tables | 6-max VPIP 28% / PFR 13.6% vs 9-max 22% / 8.6% | Priors per table size |
| Prior strength: 10–20 hands for every stat | Real players differ hugely in VPIP and limping (trust the player after ~5 hands) and very little in 3-bet, raise and showdown rates (trust the pool for 60–120) | Pseudo-counts fitted by empirical Bayes |
| Player types from the bots (45%-VPIP station, 58/44 maniac) | Real regulars cluster into nit, tight-passive regular, TAG, LAG, loose-passive; no station cluster | Prototypes and distance scales from k-means on 1,867 real regulars |
| Exploits with absolute cutoffs ("nit if PFR < 11%") | With real priors (PFR 10.8%) every unknown player tripped them | Exploits are relative to the pool and need a sample |
| BB continues ~40% vs an open | 21–26% | Calibrated |
| WTSD counted only shown cards | Real sites let losers muck: only 5% of 25NL showdowns carry recorded results, most involve a muck | Mucked showdowns count: WTSD 21% → 29%, W$SD 60% → 52% (the old figure was winner-biased) |

### Held-out check of the audited model
Same protocol as before (`experiments/real_eval.py`): the population file fitted on the earlier half of the hands, 5,728 later-half decisions scored on identical points. Log-loss, lower is better.

| Model | All | Facing a bet | Betting decisions | Player seen 0–19 hands | 500+ hands |
|---|---|---|---|---|---|
| Original (research priors, formula) | 0.701 | 0.922 | 0.583 | 0.754 | 0.641 |
| Previous build (curves + tempering) | 0.665 | 0.842 | 0.571 | 0.700 | 0.610 |
| **Audited build** (+ commitment shift, bluff by size, table-size priors, fitted pseudo-counts, corrected showdown data) | **0.665** | **0.842** | 0.571 | 0.700 | 0.612 |
| Audited build, player unknown | 0.682 | 0.852 | 0.591 | 0.702 | 0.642 |

Paired against the original: −0.036 ± 0.004 for both builds. **The audit's model changes are neutral on this metric.** That is expected: the fold and raise curves already carry what predicts the *acting* player's next move. What the new pieces change is the arithmetic behind hero's EV, which this metric cannot see: how a shove that commits the villain is valued, how much of a betting range is bluffs at each size, and what an unknown player at a 9-max table is assumed to be. The correctness of those inputs was checked directly against real hands (tables above) and, for the bluff composition, on real hole cards (below).

### Bluff composition on real hole cards
The shown-bettor curves come from showdowns. On the flop and turn a bluff that gives up is never shown, so those curves could understate bluffing. A correction multiplier for flop and turn bluff shares was fitted on hands with known cards (Pluribus pros; 25NL players who showed down), scoring how much probability the model's range puts on the hand actually held and how well it predicts the actor's move given his real cards:

| Flop / turn multiplier | Pros: range vs uniform | Pros: action log-loss | 25NL: range vs uniform | 25NL: action log-loss |
|---|---|---|---|---|
| 1.0 / 1.0 (measured shares as they are) | +0.497 bits | 0.5495 | **+0.603** | **0.5979** |
| 1.5 / 1.3 | +0.499 | 0.5462 | +0.604 | 0.5994 |
| 2.0 / 1.3 | +0.499 | **0.5457** | +0.603 | 0.6004 |
| 2.5 / 1.3 | +0.499 | 0.5465 | – | – |

The differences are within noise, and the 25NL pool, which is the target, is marginally best with no correction. **The measured shares ship uncorrected.** Compared with the previous build on the same card-level test, the audited build predicts the actor's move given his real cards better (pros 0.559 → 0.546, 25NL 0.610 → 0.598) with the same range fit (+0.50 / +0.60 bits vs a uniform guess).



### Money and decision-time safety
- **Stop-loss, circuit breaker and "broke" now stop the session.** Runners check the agent after every hand; the HTTP API reports `session_status` with every decision. Model fees are charged to the bankroll.
- **Opus is consulted only where its fee is justified:** expected gain (2% of the pot, the benchmark's figure) must cover the call. At $0.05/$0.10 that means pots over about 25bb; without `--stakes` the engine plays alone.
- **Every decision is bounded in time** (25 s by default, the table clock is shorter than any API timeout): no SDK retries, no waiting out a `retry-after`, and the engine's pick is played when the deadline passes.
- **Any model failure falls back to the engine,** including malformed answers (a list, a numeric `mix`, NaN probabilities). An answer with no usable mix plays the engine's pick instead of counting as a decisive override.
- **The risk penalty** used two scalings that both grew as the bankroll shrank; at 5 buy-ins it folded top pair top kicker to a shove. One CRRA scaling (γ = 1.5) now; a 100bb coin flip costs about 4bb of certainty equivalent at 5 buy-ins and nothing at 500.
- **Chip units** follow each view's big blind (clients that send cents no longer under-count losses 10×).
- **Spend cap:** file-locked across processes, fails closed on a corrupt ledger, reserves the worst case before each call, and a per-session cap.
- **HTTP server:** shared-secret token, JSON-only, Origin/Host checks, 10 s socket timeout, 1 MB bodies, full validation of views and histories, duplicate-hand rejection, loopback only unless `--allow-remote`.
- **Slumbot and ACPC:** a lost response never re-sends an action (the hand is abandoned instead), errors are per hand, partial results are kept, timeouts everywhere. SIGTERM saves the opponent notes; the DB is saved every 25 hands.

## The final build (`pokerbrain/config.py`)
| Component | Setting | Evidence |
|---|---|---|
| Backbone | Quant engine + Bayesian opponent model + preflop charts; all-in fix + stack-commitment rule | Wins against every simulated type; charts beat engine-preflop in full matches; all-in fix found by the live Opus |
| Opponent model | **Fitted to real players and audited**: real-pool priors per table size with fitted pseudo-counts, fold/raise curves by size with a commitment shift, bluff shares by street and size, tempering, real-player prototypes (`pokerbrain/data/population.json`; `POKERBRAIN_POPULATION=none` for bot venues) | Held-out real decisions: log-loss 0.665 vs 0.701 for the original model (paired −0.036 ± 0.004); 16,704 real decisions, 0 exceptions |
| Opus 5.5 | Prompt **v13**, full dossier, **no Jev reads**, effort medium | Prompt tournament (13 variants, 4 rounds) |
| When Opus is asked | **Every postflop decision** whose pot is worth > 3 model calls | Never worse than the Jev router in any round |
| How Opus's answer is used | Commit to its top action; **override the engine only when decisive** (≤20% of its mix on the engine's pick) | Gate validated out of sample; commitment beats sampling |
| Jev | Optional budget router (`--router jev`): about half the Opus calls | Five roles tested; none improved EV over the alternatives |
| Held-out result | +0.27 ± 0.32 bb/decision vs the engine alone (240 answers); +0.05 ± 0.10 on the final 90 fresh spots | See the tables above |
