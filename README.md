# PokerBrain: quant + psychology no-limit hold'em AI (Opus 5.5 + Jev)

PokerBrain splits each decision across three parts:

| Part | Role | What it does |
|---|---|---|
| **Quant engine** (code) | Calculator | Exact game state and hand reading. Estimates each villain's **range** by Bayesian narrowing through his actions. Computes equity against that range, pot odds, MDF, SPR and geometric sizing, then the **EV of every legal action**, and adjusts for risk using your bankroll. |
| **Jev** (TypeSafe System One, via OpenRouter) | Intuition | ~100 ms, about $0.00004 per call. Returns calibrated gut reads: P(this bet is a bluff), P(he folds to a bet), tilt level, player type. |
| **Claude Opus 5.5** | Decision-maker | Reads the whole dossier (numbers, per-player study notes, showdowns, tilt evidence, sizing tells, Jev's reads, stakes and bankroll) and chooses a **mixed strategy** over the engine's scored action menu. It is called in the spots where that pays: big pots and close decisions. |

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

# Jev reads feeding the engine (fractions of a cent per hand)
python -m pokerbrain sim --agent jev-reads --hands 500

# the full build: quant + Jev + Opus in key spots; persistent notes + bankroll awareness
python -m pokerbrain slumbot --agent ultimate --hands 200 --db data/slumbot_notes.json \
       --bankroll 2000 --stakes 1/2

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

See `docs/EXPERIMENTS.md`. In short:
1. Record thousands of real decision points from long simulated sessions.
2. Score every candidate action with an **oracle**. It infers the villain's hand distribution from his actual strategy, never from his real cards, then runs rollouts on common random numbers. That gives luck-free EV per decision.
3. Compare builds on the same spots: quant only, Jev-as-decider, Jev reads → quant, Opus with 8 prompt variants, and combinations. Iterate on the prompts, then confirm on a held-out set.
