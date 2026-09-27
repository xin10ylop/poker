"""Stakes awareness, bankroll management and risk-adjusted decisions.

Formulas (see research notes in docs/STRATEGY.md):
  risk of ruin      RoR = exp(-2 * mu * B / sigma^2)          (mu, sigma in bb/100; B in bb)
  bankroll for RoR  B   = -sigma^2 * ln(RoR) / (2 * mu)
  Kelly bankroll    B*  = sigma^2 / mu ; fractional Kelly (1/4 - 1/3) recommended
  log-utility risk penalty for a single decision ~= Var / (2 * bankroll)

The penalty is what makes a short-rolled player pass on a +0.3bb coin-flip for
200bb while a well-rolled player takes it.  It is ~0 when the bankroll is deep.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

STAKES_LADDER = [  # (sb, bb) in currency units - typical online cash ladder
    (0.01, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.25), (0.25, 0.50), (0.50, 1.00),
    (1.00, 2.00), (2.50, 5.00), (5.00, 10.00), (10.0, 20.0), (25.0, 50.0), (50.0, 100.0),
]


@dataclass
class Stakes:
    sb: float = 0.5                 # currency
    bb: float = 1.0                 # currency
    chips_per_bb: int = 100         # engine chip units per big blind
    currency: str = "$"
    game: str = "NLHE cash"
    max_players: int = 6
    buy_in_bb: float = 100.0
    rake_pct: float = 0.05
    rake_cap_bb: float = 3.0
    tournament: bool = False

    @property
    def chip_value(self) -> float:
        return self.bb / self.chips_per_bb

    def label(self) -> str:
        return f"{self.currency}{self.sb:g}/{self.currency}{self.bb:g} {self.game} ({self.max_players}-max)"


def risk_of_ruin(winrate_bb100: float, sd_bb100: float, bankroll_bb: float) -> float:
    if winrate_bb100 <= 0:
        return 1.0
    return math.exp(-2.0 * winrate_bb100 * bankroll_bb / (sd_bb100 ** 2))


def bankroll_for_ror(winrate_bb100: float, sd_bb100: float, ror: float) -> float:
    if winrate_bb100 <= 0:
        return float("inf")
    return -sd_bb100 ** 2 * math.log(ror) / (2.0 * winrate_bb100)


@dataclass
class BankrollManager:
    bankroll: float = 5000.0            # currency
    stakes: Stakes = field(default_factory=Stakes)
    kelly_fraction: float = 0.33        # 1/3 Kelly (research: 1/4 - 1/3 given win-rate uncertainty)
    est_winrate_bb100: float = 5.0
    est_sd_bb100: float = 95.0
    stop_loss_buyins: float = 3.0
    stop_win_buyins: Optional[float] = None
    session_pnl: float = 0.0           # currency
    session_hands: int = 0
    peak: float = 0.0
    llm_cost: float = 0.0              # currency spent on model calls this session

    # ------------------------------------------------------------ numbers
    def bankroll_bb(self) -> float:
        return self.bankroll / self.stakes.bb

    def buyins(self) -> float:
        return self.bankroll / (self.stakes.buy_in_bb * self.stakes.bb)

    def bankroll_chips(self) -> float:
        return self.bankroll / self.stakes.chip_value

    def kelly_bankroll_bb(self) -> float:
        return self.est_sd_bb100 ** 2 / max(0.1, self.est_winrate_bb100)

    def risk_aversion(self) -> float:
        """1.0 at the Kelly-fraction target bankroll; grows when short-rolled."""
        target = self.kelly_bankroll_bb() / max(0.05, self.kelly_fraction)
        ratio = self.bankroll_bb() / max(1.0, target)
        return float(min(8.0, max(0.25, 1.0 / max(ratio, 1e-3))))

    def risk_penalty(self, variance_chips2: float) -> float:
        """Certainty-equivalent cost (chips) of taking on this variance."""
        br = self.bankroll_chips()
        if br <= 0:
            return 0.0
        return self.risk_aversion() * variance_chips2 / (2.0 * br)

    def ror(self) -> float:
        return risk_of_ruin(self.est_winrate_bb100, self.est_sd_bb100, self.bankroll_bb())

    # ------------------------------------------------------------ session
    def record_hand(self, net_chips: float) -> None:
        self.session_pnl += net_chips * self.stakes.chip_value
        self.bankroll += net_chips * self.stakes.chip_value
        self.session_hands += 1
        self.peak = max(self.peak, self.session_pnl)

    def record_llm_cost(self, usd: float) -> None:
        self.llm_cost += usd
        self.bankroll -= usd
        self.session_pnl -= usd

    def session_status(self) -> str:
        bi = self.stakes.buy_in_bb * self.stakes.bb
        if self.session_pnl <= -self.stop_loss_buyins * bi:
            return "stop_loss"
        # drawdown circuit breaker: > 3 sigma below expectation
        n = max(1, self.session_hands)
        sd_cur = self.est_sd_bb100 * math.sqrt(n / 100.0) * self.stakes.bb
        if (self.peak - self.session_pnl) > 3.0 * sd_cur and n > 200:
            return "circuit_breaker"
        if self.stop_win_buyins and self.session_pnl >= self.stop_win_buyins * bi:
            return "stop_win"
        return "ok"

    def recommend_stakes(self, ror_target: float = 0.02) -> tuple[float, float]:
        """Highest ladder stake where the bankroll keeps risk of ruin below the target."""
        need_bb = bankroll_for_ror(self.est_winrate_bb100, self.est_sd_bb100, ror_target)
        best = STAKES_LADDER[0]
        for sb, bb in STAKES_LADDER:
            if self.bankroll / bb >= need_bb:
                best = (sb, bb)
        return best

    def llm_value_threshold_chips(self, cost_per_call: float, edge_fraction: float = 0.03) -> float:
        """Smallest pot (chips) where a model call is worth it: expected EV gain >= call cost.

        edge_fraction = expected EV improvement from a better decision, as a fraction of the pot.
        """
        if cost_per_call <= 0:
            return 0.0
        return (cost_per_call / max(1e-9, edge_fraction)) / self.stakes.chip_value

    def context(self) -> dict:
        """Everything the decision-maker should know about stakes and money."""
        s = self.stakes
        mode = "normal"
        ra = self.risk_aversion()
        if ra >= 2.0:
            mode = "short-rolled: avoid high-variance marginal spots"
        elif ra <= 0.5:
            mode = "deep-rolled: pure EV maximization"
        return {
            "stakes": s.label(), "bb_value": f"{s.currency}{s.bb:g}", "buy_in": f"{s.currency}{s.buy_in_bb * s.bb:g}",
            "bankroll": f"{s.currency}{self.bankroll:,.2f}", "bankroll_buyins": round(self.buyins(), 1),
            "risk_of_ruin_est": round(self.ror(), 4), "risk_mode": mode, "risk_aversion": round(ra, 2),
            "session_pnl": f"{s.currency}{self.session_pnl:,.2f}", "session_hands": self.session_hands,
            "session_status": self.session_status(), "rake": f"{s.rake_pct:.0%} cap {s.rake_cap_bb:g}bb",
            "model_costs_session": f"{s.currency}{self.llm_cost:.2f}",
        }
