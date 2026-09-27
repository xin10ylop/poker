# Strategy research: what the brain knows and why

This is the poker knowledge encoded in PokerBrain: the engine's formulas, the
opponent-model priors, and the playbook in the Opus system prompts. Sources are
listed at the end. Everything was checked on 2026-09-27.

## 1. What the LLM-poker literature says (and how the design responds)

| Finding | Design response |
|---|---|
| Frontier LLMs play poker unaided and lose to strong bots. Against GTO Wizard AI (heads-up, 200bb): best model −16 bb/100, Opus 4.6 −20.4. | The LLM never plays alone. A quant engine owns game state, arithmetic, legality and sizing. |
| **Scaffolding beats model size.** A deterministic state engine, a menu of viable actions and forced JSON output cut Opus 4.6's loss from −20.4 to −8.0 bb/100 (PokerSkill, 2026). Equity and solver tools brought a 7B model close to equilibrium (ToolPoker). | Opus chooses from an **action menu** already scored by the engine (EV, fold %, equity when called). Structured JSON output. |
| **LLMs misread their own hand** in about 2% of hands, even the best models (e.g. called trips "air", or a 4-flush "the nut flush"). | The engine's hand reading, nut status and outs are injected with the instruction "trust this, do not re-derive". |
| **Arithmetic errors** in pot odds, outs and pot fractions. | All numbers are precomputed: pot odds, MDF, break-even fold %, SPR, geometric sizing, equity vs range, EV. |
| LLMs don't think in ranges and **don't mix** actions, so their lines are transparent. | The dashboard gives the villain's estimated *range* composition. Opus outputs a **mixed strategy**, and code does the sampling. |
| **Knowing–doing gap**: the reasoning argues for one action and the output is another. | The output is the id of a menu item. Mixes are sampled in code. Illegal actions are impossible. |
| Weak reads from small samples, and stale notes that anchor decisions (Poker Arena 2026). | Every statistic shows its sample size *n* and is shrunk toward population priors. Notes carry recency. The prompt says "under ~30 hands lean on population tendencies". |
| Extreme styles (far too tight or too loose). | Preflop is solver-approximate charts with measured exploit adjustments, not free-form LLM play. |

## 2. Game theory the engine uses

For a bet B into pot P:

| Quantity | Formula | 33% | 50% | 75% | 100% | 150% |
|---|---|---|---|---|---|---|
| Break-even fold rate for a pure bluff (α) | B/(P+B) | 25% | 33% | 43% | 50% | 60% |
| Minimum defense frequency (MDF) | P/(P+B) | 75% | 67% | 57% | 50% | 40% |
| Caller's required equity (= optimal river bluff share) | B/(P+2B) | 20% | 25% | 30% | 33% | 37.5% |

- **Geometric sizing**: the pot fraction per street that gets all-in by the river is f = ((1+2S/P)^(1/n) − 1)/2. It's about 116% per street in a 100bb single-raised pot and about 53% in a 3-bet pot.
- **SPR commitment** (rough): at SPR ≤ 3–4, top pair or an overpair stacks off; at 4–10 you need two pair or a strong draw; above that, a set or better.
- **MDF caveat**: MDF assumes the bettor's bluffs have zero equity. Defend below MDF against players who under-bluff, and when out of position.

## 3. Population tendencies (priors when a player is unknown)

- **Rivers are under-bluffed** even at mid/high stakes. River bets contain 12–25% weak hands versus 18–37% in GTO play, and overbets about 25% versus 37%. The priors: `river_bluff` 0.22, `bigbet_bluff` 0.25, `smallbet_bluff` 0.20.
- **River raises are very strong.** Players just call rivers with the nuts instead of raising, so a raise means the nuts.
- **Low-stakes 3-bets are value-heavy**, and the big blind under-3-bets. Prior: `threebet` 0.06.
- **Players over-fold to delayed c-bets and turn probes.**
- **Some lines are over-bluffed**, e.g. the big blind's turn probe against the button at low stakes.

## 4. Exploits by player type

The opponent model infers the type from stats (nearest prototype); Jev also classifies it.

| Type | Signature | Counter |
|---|---|---|
| Nit | VPIP below 15 | Steal and isolate, c-bet more, fold to their big turn and river bets |
| TAG | VPIP about 18–25, PFR 15–22 | Play close to GTO; take the money from others |
| LAG | about 28/24 | Open tighter, 3-bet and 4-bet value wider, call down lighter |
| Calling station | VPIP above 35, low PFR, rarely folds | Never bluff; value-bet thinner and **bigger** |
| Maniac | Extreme VPIP, 3-bet and aggression | Bluff-catch wider, **don't raise their bluffs out**, trap |
| Weak-passive fish | PFR far below VPIP, folds when he misses | Isolate, bet when checked to |

HUD reliability: VPIP needs about 300 hands, 3-bet about 1,000, fold-to-3-bet about 1,500. That's why every statistic here is a Beta-Binomial estimate shrunk toward the priors.

## 5. Psychology that is measurable from betting data

- **Tilt triggers.** Losing a pot of 50–100bb or more, and above all losing an all-in as the favorite (a bad beat). Being down on the session also counts.
- **Tilt responses.** VPIP, PFR and 3-bet rise; fold-to-c-bet falls; more overbets and all-ins. Measure these as z-scores against the player's own baseline, and let the effect decay over about 20 hands.
- **How the model scores tilt.** P(tilt) = σ(1.2·z_behaviour + 1.5·trigger + … − 3). Bad beats count double.
- **Exploiting tilt.** Value-bet wider and bigger, call down lighter, and **don't bluff** a tilted player.
- **Sizing tells.** The bluff rate is estimated separately for big bets (≥75% pot) and small ones from showdowns. The villain model applies it to the size he actually chose.
- **Frequency-implied bluffing.** If a player bets far more often than his range has value hands to bet, the excess must be bluffs. The model infers bluff share from *frequency*, not only from rare showdowns.
- **Your own image.** Being caught bluffing gets you called more, so value-bet more after that.
- **Timing tells are weak evidence.** A widely quoted statistic was withdrawn for lack of a source, so timing is not used.

## 6. Preflop (6-max, 100bb, 2.5bb opens)

- **Opening ranges (RFI):** UTG about 17.5%, HJ 22%, CO 28–30%, BTN 44–47%, SB about 40% raise.
- **Big blind vs a button open:** defend about 52–60%, of which 10–14% 3-bets. About 30% against UTG.
- **3-bet ranges:** button vs cutoff about 12–16%, polarized. Small blind vs button 10–12%, mostly 3-bet or fold. Against UTG, linear and tight.
- **4-bets:** QQ+/AK for value (KK+ against tight 3-bettors); A5s–A3s as bluffs. At 100bb, a 5-bet is all-in.
- **Heads-up:** the button opens 80–90%; the big blind defends 62–80% and 3-bets 15–20%.
- **Exploit adjustments** come from HUD stats: steal wider against blinds that over-fold, 3-bet bluff against high fold-to-3-bet, go value-only against stations, tighten against nits, and widen value 3-bets against loose openers.

## 7. Bankroll and risk

- **Risk of ruin:** RoR = exp(−2μB/σ²). For a 1% risk at 5 bb/100 and σ = 90, you need about 3,730bb (37 buy-ins).
- **Kelly:** the Kelly bankroll is B* = σ²/μ. Use **⅓ Kelly** because the win rate is uncertain.
- **Per-decision risk penalty:** λ·Var/(2·bankroll), a log-utility approximation where λ grows as the bankroll shrinks. Short-rolled, it skips thin high-variance gambles; well-rolled, it is pure EV.
- **Stop-loss (3 buy-ins) and a drawdown circuit-breaker (3σ).** Stopping doesn't change EV for a bot, but it does protect against bugs and changed conditions.
- **The "LLM rake".** A model call costs money. Opus is only consulted when the pot is worth at least 3× the call cost, and only in big or close spots. At micro stakes, uncontrolled LLM calls can cost more than the win rate.

## Sources
- GTO Wizard LLM benchmark: https://arxiv.org/abs/2603.23660
- PokerSkill (scaffolded LLM poker): https://arxiv.org/abs/2605.30094
- ToolPoker: https://arxiv.org/abs/2602.00528
- PokerBench: https://arxiv.org/abs/2501.08328
- Poker Arena: https://arxiv.org/abs/2606.13815
- Suspicion-Agent: https://arxiv.org/abs/2309.17277
- Agent-Pro: https://arxiv.org/abs/2402.17574
- Kaggle Game Arena poker: https://www.kaggle.com/blog/game-arena-poker
- PokerBattle.ai: https://pokerbattle.ai/about
- Population data (LeakBuster/Upswing): https://upswingpoker.com/3-cash-game-exploits-regs-backed-by-data/
- GTO Wizard, over-bluffed lines: https://blog.gtowizard.com/calling-down-the-over-bluffed-lines-in-lower-limits/
- MDF and alpha: https://blog.gtowizard.com/mdf-alpha/
- Pot geometry: https://blog.gtowizard.com/pot-geometry/
- C-bet mechanics: https://blog.gtowizard.com/the-mechanics-of-c-bet-sizing/
- Risk of ruin: https://www.primedope.com/poker-risk-of-ruin-formula/
- Kelly for cash games: https://www.primedope.com/cash-kelly-calculator/
- Tilt after big losses: Smith et al. 2009 (Management Science); Eil & Lien 2014 (GEB); Palomäki et al. 2013
- Slumbot API: https://slumbot.com/sample_api.py
- ACPC protocol: http://www.computerpokercompetition.org/downloads/documents/protocols/protocol.pdf
