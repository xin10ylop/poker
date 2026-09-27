"""Claude Opus 5.5 client (official Anthropic SDK).

Works with either credential:
  * ANTHROPIC_API_KEY  -> Anthropic API directly, model "claude-opus-5-5"
  * OPENROUTER_API_KEY -> OpenRouter's Anthropic-compatible Messages endpoint,
                          model "anthropic/claude-opus-5.5"
The base URL is always passed explicitly so a stray ANTHROPIC_BASE_URL in the
environment can never redirect traffic.

Opus 5.5 notes: thinking is always on (effort is the latency/cost knob, default
"medium"), no temperature, no prefill; structured output via
output_config.format; the static system prompt is cached.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import anthropic

from .budget import PRICES, Budget, BudgetExceeded, default_budget

DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "action_id": {"type": "string", "description": "id of the chosen action from the menu, e.g. A3"},
        "mix": {"type": "array", "description": "optional mixed strategy over action ids (probabilities sum to 1)",
                "items": {"type": "object", "properties": {"id": {"type": "string"}, "p": {"type": "number"}},
                          "required": ["id", "p"], "additionalProperties": False}},
        "confidence": {"type": "number"},
        "read": {"type": "string", "description": "one-sentence read on the opponent's range/state of mind"},
        "rationale": {"type": "string", "description": "at most 3 short sentences"},
        "note": {"type": "string", "description": "optional new note about the opponent for future hands, or ''"},
    },
    "required": ["action_id", "mix", "confidence", "read", "rationale", "note"],
    "additionalProperties": False,
}


class ClaudeError(RuntimeError):
    pass


@dataclass
class ClaudeResult:
    data: dict
    usd: float
    latency: float
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    raw_text: str = ""
    extra: dict = field(default_factory=dict)


class OpusClient:
    def __init__(self, effort: str = "medium", max_tokens: int = 12000, timeout: float = 60.0,
                 budget: Optional[Budget] = None, provider: Optional[str] = None,
                 model: Optional[str] = None):
        self.budget = budget or default_budget()
        self.effort = effort
        self.max_tokens = max_tokens
        ak, ok = os.environ.get("ANTHROPIC_API_KEY"), os.environ.get("OPENROUTER_API_KEY")
        self.provider = provider or ("anthropic" if ak else "openrouter" if ok else None)
        if self.provider == "anthropic":
            if not ak:
                raise ClaudeError("ANTHROPIC_API_KEY not set")
            self.client = anthropic.Anthropic(api_key=ak, base_url="https://api.anthropic.com",
                                              timeout=timeout, max_retries=1)
            self.model = model or "claude-opus-5-5"
        elif self.provider == "openrouter":
            if not ok:
                raise ClaudeError("OPENROUTER_API_KEY not set")
            self.client = anthropic.Anthropic(auth_token=ok, api_key=None, base_url="https://openrouter.ai/api",
                                              timeout=timeout, max_retries=1)
            self.model = model or "anthropic/claude-opus-5.5"
        else:
            raise ClaudeError("set ANTHROPIC_API_KEY or OPENROUTER_API_KEY")
        self.structured_ok = True
        self.calls = 0
        self.usd = 0.0

    def _price(self, usage) -> float:
        pin, pout, pcache = PRICES.get(self.model, PRICES["claude-opus-5-5"])
        it = getattr(usage, "input_tokens", 0) or 0
        ot = getattr(usage, "output_tokens", 0) or 0
        cr = getattr(usage, "cache_read_input_tokens", 0) or 0
        cw = getattr(usage, "cache_creation_input_tokens", 0) or 0
        return (it * pin + ot * pout + cr * pcache + cw * pin * 1.25) / 1e6

    def decide(self, system: str, user: str, schema: dict = DECISION_SCHEMA, effort: Optional[str] = None,
               tag: str = "") -> ClaudeResult:
        self.budget.check(0.02)
        eff = effort or self.effort
        output_config: dict = {"effort": eff}
        if self.structured_ok:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        kwargs = dict(model=self.model, max_tokens=self.max_tokens,
                      system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                      messages=[{"role": "user", "content": user}], output_config=output_config)
        t0 = time.time()
        try:
            resp = self.client.messages.create(**kwargs)
        except anthropic.BadRequestError as exc:
            if self.structured_ok and "format" in str(exc).lower():
                self.structured_ok = False           # provider without structured outputs: parse text
                return self.decide(system, user, schema, effort, tag)
            raise ClaudeError(f"bad request: {exc}") from exc
        except anthropic.RateLimitError as exc:
            raise ClaudeError(f"rate limited: {exc}") from exc
        except anthropic.APIStatusError as exc:
            raise ClaudeError(f"API error {exc.status_code}: {exc}") from exc
        except anthropic.APIConnectionError as exc:
            raise ClaudeError(f"connection error: {exc}") from exc
        latency = time.time() - t0
        usd = self._price(resp.usage)
        self.budget.record(self.model, usd, tag)
        self.calls += 1
        self.usd += usd
        if resp.stop_reason == "refusal":
            raise ClaudeError(f"refusal: {getattr(resp, 'stop_details', None)}")
        text = "".join(b.text for b in resp.content if b.type == "text")
        data = parse_json_block(text)
        if data is None:
            raise ClaudeError(f"could not parse decision JSON: {text[:300]}")
        u = resp.usage
        return ClaudeResult(data=data, usd=usd, latency=latency, input_tokens=u.input_tokens,
                            output_tokens=u.output_tokens,
                            cache_read=getattr(u, "cache_read_input_tokens", 0) or 0, raw_text=text)


def parse_json_block(text: str) -> Optional[dict]:
    """Parse a JSON object from model text (plain JSON, ```json fences, or embedded)."""
    text = text.strip()
    for candidate in (text, *re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, TypeError):
            pass
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        break
                    break
        start = text.find("{", start + 1)
    return None
