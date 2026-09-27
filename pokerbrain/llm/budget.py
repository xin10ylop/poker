"""Spend ledger with a hard cap, shared by every model client.

The ledger lives in a small JSON file (default ~/.pokerbrain/spend.json) so the
cap holds across processes and sessions.  Every call records its actual cost
(OpenRouter returns `usage.cost`; Anthropic usage is priced from tokens).
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Optional

# USD per million tokens (input, output, cache_read)
PRICES = {
    "claude-opus-5-5": (4.0, 20.0, 0.20),
    "anthropic/claude-opus-5.5": (4.0, 20.0, 0.20),
    "jev": (0.042, 0.0, 0.0),
}


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class SpendRecord:
    model: str
    usd: float
    ts: float
    tag: str = ""


class Budget:
    _lock = threading.Lock()

    def __init__(self, cap_usd: Optional[float] = None, path: Optional[str] = None):
        env_cap = os.environ.get("POKERBRAIN_MAX_SPEND_USD")
        self.cap = cap_usd if cap_usd is not None else (float(env_cap) if env_cap else 5.0)
        self.path = path or os.environ.get("POKERBRAIN_SPEND_FILE",
                                           os.path.join(os.path.expanduser("~"), ".pokerbrain", "spend.json"))
        self.session_usd = 0.0
        self.session_calls: dict[str, int] = {}

    def _read(self) -> dict:
        try:
            with open(self.path) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {"total_usd": 0.0, "by_model": {}, "calls": 0}

    def total(self) -> float:
        return float(self._read().get("total_usd", 0.0))

    def remaining(self) -> float:
        return self.cap - self.total()

    def check(self, estimate_usd: float = 0.0) -> None:
        if self.total() + estimate_usd > self.cap:
            raise BudgetExceeded(f"spend cap ${self.cap:.2f} reached (spent ${self.total():.4f})")

    def record(self, model: str, usd: float, tag: str = "") -> None:
        with self._lock:
            data = self._read()
            data["total_usd"] = float(data.get("total_usd", 0.0)) + usd
            bm = data.setdefault("by_model", {})
            bm[model] = float(bm.get(model, 0.0)) + usd
            data["calls"] = int(data.get("calls", 0)) + 1
            data["updated"] = time.time()
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f)
            os.replace(tmp, self.path)
            self.session_usd += usd
            self.session_calls[model] = self.session_calls.get(model, 0) + 1


_DEFAULT: Optional[Budget] = None


def default_budget() -> Budget:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Budget()
    return _DEFAULT


def load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader (no dependency)."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
