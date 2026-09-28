# PokerBrain: quant + psychology no-limit hold'em AI (Opus 5.5 + Jev)

PokerBrain splits each decision across three parts:

| Part | Role | What it does |
|---|---|---|
| **Quant engine** (code) | Calculator and backbone | Exact game state and hand reading. Estimates each villain's **range** by Bayesian narrowing through his actions. Computes equity against that range, pot odds, MDF, SPR and geometric sizing, then the **EV of every legal action**, and adjusts for risk using your bankroll. |
| **Claude Opus 5.5** | Judgment and psychology | Reads the whole dossier (numbers, per-player study notes, showdowns, tilt evidence, sizing tells, stakes and bankroll) and returns a mixed strategy over the engine's scored action menu. It is consulted on postflop decisions worth more than the model call. Its pick replaces the engine's **only when it is decisive** (≤20% of its own mix left on the engine's pick). Hedged overrides are where all the big blunders were. |
| **Jev** (TypeSafe System One, via OpenRouter) | Budget router (optional) | ~100 ms, about $0.00008 per call. `--router jev` lets Jev's "is this decision tricky?" score decide when Opus is woken, which roughly halves Opus calls. It was tested in five roles (below). |

Around them, an **opponent tracker** studies every player during the session. It keeps HUD stats shrunk toward population priors, a showdown memory, sizing tells, tilt signals (big losses, bad beats, looser play) and automatic notes. Everything persists across sessions.

> Design principle, backed by the research in `docs/STRATEGY.md`: an LLM playing poker unaided loses clearly to strong bots (Opus 4.6: −20 bb/100). With a deterministic engine and a scored action menu, the same model improved to −8 bb/100. So the engine owns the arithmetic and legality, and the LLM owns judgment and psychology.

## Where it plays

| Adapter | Command | Notes |
|---|---|---|
| Local simulator | `python -m pokerbrain sim` | 6-max or heads-up against a field of bot personalities (nit, TAG, LAG, station, maniac, fish, **tilter**, **sizing-tell**) |
| Slumbot | `python -m pokerbrain slumbot` | Public heads-up benchmark bot (200bb). Uses Slumbot's duplicate `baseline_winnings` for variance reduction |
| ACPC | `python -m pokerbrain acpc --host H --port P` | Standard bot-competition dealer protocol |
| HTTP API | `python -m pokerbrain serve` | `POST /decide` (GameView JSON) and `POST /observe` (HandHistory JSON), bound to localhost. Plug in any environment that permits bots: your own home-game server, research platforms |
| Hand review | `python -m pokerbrain analyze ...` | Study tool: the full dashboard and EV table for a spot you describe |

**Not included, on purpose.** There is no screen-reading or auto-clicking adapter for commercial real-money poker clients. PokerStars, GGPoker and essentially every other real-money site prohibit bots and real-time assistance in their terms of service. They actively detect both, ban accounts and confiscate balances. Running a bot there also takes money from players who believe they are playing people. Use PokerBrain where bots are welcome: the adapters above, bot competitions, private games whose players agree, and study.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # add OPENROUTER_API_KEY (Jev, and Opus via OpenRouter) or ANTHROPIC_API_KEY
python -m pytest -q         # engine is fuzz-tested against pokerkit
```

`POKERBRAIN_MAX_SPEND_USD` sets a hard spending cap across all model calls. The ledger lives in `~/.pokerbrain/spend.json`.

## Quick start

```bash
# free (no model calls): quant engine + opponent study vs the simulated field
python -m pokerbrain sim --agent quant --hands 1000

# the full build: quant + gated Opus on postflop decisions; persistent notes + bankroll awareness
python -m pokerbrain slumbot --agent ultimate --hands 200 --db data/slumbot_notes.json \
       --bankroll 2000 --stakes 1/2
# same, but Jev decides when Opus is worth waking (about half the Opus calls)
python -m pokerbrain sim --agent ultimate --router jev --hands 300

# a Claude Code session as the live Opus decider (no Anthropic key needed); a paired
# engine-only replay of the same decks is reported at the end
python experiments/live_match.py --qdir bridge/live --hands 100 [--jev]
#   answer side: python -m pokerbrain.bridge next bridge/live
#                python -m pokerbrain.bridge answer bridge/live <id> - < decision.json

# hand review
python -m pokerbrain analyze --hole JcJd --board Kd8c4h2s7d --actions "r2.5 c | x x | x x | x b10"
```

Agents: `quant` (no LLM), `jev-reads`, `jev-decide`, `opus`, and `ultimate` (configured in `pokerbrain/config.py`).

## Layout

```
pokerbrain/
  engine.py      NLHE rules: side pots, TDA short all-in rule, exact all-in EV
  quant.py       equity vs ranges, EV of each candidate action, risk adjustment
  villain.py     Bayesian range narrowing + calibrated villain response model
  opponents.py   HUD stats (Beta priors), showdowns, sizing tells, tilt, notes, archetypes
  preflop.py     solver-approximate charts (6-max 100bb, heads-up) + hand ranking
  bankroll.py    stakes, risk of ruin, fractional Kelly, stop-loss, "LLM rake" threshold
  bots.py        simulated opponents incl. psychological types
  spots.py       decision benchmark: recorder + posterior oracle
  arena.py       duplicate matches, paired comparisons, all-in-adjusted win rates
  llm/           Opus client (Anthropic SDK), Jev client, dashboard renderer, prompt variants
  agents/        QuantAgent, JevReadsAgent, JevDeciderAgent, OpusAgent (escalation + verifier)
  adapters/      slumbot, acpc, http server, manual analysis
  bridge.py      file bridge so an external session can make the live Opus decisions
experiments/     benchmark builder + harness (prompt/build comparisons)
docs/            STRATEGY.md (research), EXPERIMENTS.md (results)
```

## How the "ultimate setup" was chosen

See `docs/EXPERIMENTS.md` for everything, including the negative results. In short:
1. Record real decision points from long simulated sessions.
2. Score every candidate action with an **oracle**. It infers the villain's hand distribution from his actual strategy, never from his real cards, then runs rollouts on common random numbers. That gives luck-free EV per decision.
3. Compare builds on the same spots, confirm on held-out spots, then run a final round on 90 fresh spots.

| Question | Answer from the benchmark |
|---|---|
| Opus alone (table + history)? | Worse than the engine (+0.60 bb/decision EV loss) |
| Opus + engine numbers only? | Defers to the engine entirely |
| Best Opus prompt? | 13 framings tested. **v13**: a solver-trained baseline, deviate on strong evidence, plus lessons from reviewed blunders, **no Jev reads** |
| Jev as decider / reads in the math / reads shown to Opus / verifier / router? | Worse / worse (+1.85) / worse (caused blunders) / no gain / no better than consulting Opus on every postflop spot |
| Are the psychology reads accurate? | The statistical tracker beats Jev on bluff calibration (Brier 0.049 vs 0.138), player type (94% vs 83%) and tilt (AUC 0.987 vs 0.951) |
| Commit or randomize? | Commit to Opus's top action (random mixing costs 0.25–0.5bb vs non-adaptive players) |
| Biggest single win | **The decisive-override gate**: +0.5 bb/decision over ungated Opus on held-out spots |
| Final build vs pure engine, held-out spots | **+0.27 ± 0.32 bb/decision** pooled over 240 answers; +0.05 ± 0.10 on the final 90 fresh spots. Safe, but not a proven edge |
| Did Opus earn its place? | Yes, as an auditor. In the live test it noticed that the engine's all-in numbers were impossible ("villain raises 55%" over a shove), played the hand correctly anyway, and the bug is now fixed |
| Full matches, engine only | Wins against every simulated type. 6-max +130 ± 87 bb/100; Slumbot (strong HU bot) −31 bb/100 baseline-adjusted over 400 hands |

The honest summary: **the edge comes from the quant engine and the opponent study**. Opus, gated, adds judgment without adding blunders. On simulated opponents that is roughly break-even. It is most likely to matter against humans, whose stories, notes and meta-game the statistics can't capture.
