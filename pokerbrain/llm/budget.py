"""Spend ledger with a hard cap, shared by every model client.

The ledger lives in a small JSON file (default ~/.pokerbrain/spend.json).  Every
read-modify-write holds an OS file lock (``fcntl.flock`` on ``<path>.lock``) and
publishes the new ledger atomically (unique temp file + ``os.replace``), so the
cap holds across threads, processes and sessions.  Every call records its actual
cost (OpenRouter returns ``usage.cost``; Anthropic usage is priced from tokens).

Reservations: ``reserve(estimate)`` books the estimate as *pending* in the ledger
and refuses (``BudgetExceeded``) when spent + pending + estimate would pass the
cap; ``settle(rid, model, usd)`` replaces it with the actual cost and
``release(rid)`` drops it when the call never happened.  The older pair keeps
working: ``check(estimate)`` reserves on behalf of the calling thread and that
thread's next ``record()`` settles it.  ``record()`` always writes the spend and
then raises ``BudgetExceeded`` if the ledger total (or the session total) is now
over its cap.  Reservations that are never settled expire after
``reservation_ttl`` seconds, or as soon as the process that made them is gone.

A corrupt or unreadable ledger fails closed: ``BudgetExceeded`` is raised, a
``.bak`` copy is kept and the file is never silently reset to $0.
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import math
import os
import re
import socket
import stat
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Iterator, Optional

# USD per million tokens (input, output, cache_read)
PRICES = {
    "claude-opus-5-5": (4.0, 20.0, 0.20),
    "anthropic/claude-opus-5.5": (4.0, 20.0, 0.20),
    "jev": (0.042, 0.0, 0.0),
}

DEFAULT_CAP_USD = 5.0
DEFAULT_RESERVATION_TTL_S = 600.0
_EPS = 1e-9            # float slack so a spend of exactly the cap is allowed


class BudgetExceeded(RuntimeError):
    pass


class LedgerError(BudgetExceeded):
    """The ledger cannot be read or parsed.  Spending is refused until it is fixed."""


@dataclass
class SpendRecord:
    model: str
    usd: float
    ts: float
    tag: str = ""


def _warn(msg: str) -> None:
    print(f"pokerbrain: {msg}", file=sys.stderr)


def _parse_usd(raw, name: str, default: Optional[float]) -> Optional[float]:
    """A finite amount >= 0, or `default` (with a warning) when `raw` is not one."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return default
    try:
        v = float("nan") if isinstance(raw, bool) else float(raw)
    except (TypeError, ValueError):
        v = float("nan")
    if not math.isfinite(v) or v < 0:
        _warn(f"invalid {name}={raw!r} (need a finite amount >= 0); using {default}")
        return default
    return v


def _amount(x, what: str) -> float:
    try:
        v = float("nan") if isinstance(x, bool) else float(x)
    except (TypeError, ValueError):
        v = float("nan")
    if not math.isfinite(v) or v < 0:
        raise ValueError(f"{what} must be a finite amount >= 0, got {x!r}")
    return v


def _pid_alive(pid) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return True           # unknown owner: let the TTL decide
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True           # e.g. EPERM: exists, owned by someone else
    return True


def _fresh() -> dict:
    return {"total_usd": 0.0, "by_model": {}, "calls": 0, "pending": {}}


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _validated(data) -> dict:
    """Return the ledger normalised, or raise ValueError if it is not a ledger."""
    if not isinstance(data, dict):
        raise ValueError("ledger is not a JSON object")
    total = data.get("total_usd")
    if not _finite(total) or total < 0:
        raise ValueError(f"bad total_usd {total!r}")
    data["total_usd"] = float(total)
    bm = data.setdefault("by_model", {})
    if not isinstance(bm, dict) or not all(_finite(v) for v in bm.values()):
        raise ValueError("bad by_model")
    calls = data.setdefault("calls", 0)
    if not isinstance(calls, int) or isinstance(calls, bool) or calls < 0:
        raise ValueError(f"bad calls {calls!r}")
    pend = data.setdefault("pending", {})
    if not isinstance(pend, dict):
        raise ValueError("bad pending")
    for r in pend.values():
        if not isinstance(r, dict) or not _finite(r.get("usd")) or r["usd"] < 0 or not _finite(r.get("ts")):
            raise ValueError("bad pending reservation")
    return data


class Budget:
    # Serialises threads of this process; fcntl.flock serialises processes.
    _lock = threading.RLock()

    def __init__(self, cap_usd: Optional[float] = None, path: Optional[str] = None, *,
                 session_cap_usd: Optional[float] = None, cap: Optional[float] = None,
                 reservation_ttl: float = DEFAULT_RESERVATION_TTL_S):
        if cap_usd is None:
            cap_usd = cap
        if cap_usd is not None:
            self.cap = _parse_usd(cap_usd, "cap_usd", DEFAULT_CAP_USD)
        else:
            self.cap = _parse_usd(os.environ.get("POKERBRAIN_MAX_SPEND_USD"), "POKERBRAIN_MAX_SPEND_USD",
                                  DEFAULT_CAP_USD)
        if session_cap_usd is not None:
            self.session_cap = _parse_usd(session_cap_usd, "session_cap_usd", None)
        else:
            self.session_cap = _parse_usd(os.environ.get("POKERBRAIN_SESSION_SPEND_USD"),
                                          "POKERBRAIN_SESSION_SPEND_USD", None)
        raw = path or os.environ.get("POKERBRAIN_SPEND_FILE") or os.path.join("~", ".pokerbrain", "spend.json")
        self.path = os.path.abspath(os.path.expanduser(raw))   # fixed now, whatever the cwd is later
        self.reservation_ttl = float(reservation_ttl)
        self.session_usd = 0.0
        self.session_calls: dict[str, int] = {}
        self._session_pending: dict[str, float] = {}
        self._tls = threading.local()       # the reservation made by this thread's last check()
        self._host = socket.gethostname()

    # ------------------------------------------------------------ storage
    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        with Budget._lock:
            try:
                os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
                fd = os.open(self.path + ".lock", os.O_RDWR | os.O_CREAT, 0o600)
            except OSError as exc:
                raise LedgerError(f"cannot open spend ledger lock {self.path}.lock: {exc}") from exc
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                yield
            finally:
                os.close(fd)          # also releases the flock

    def _load(self) -> dict:
        try:
            with open(self.path, "rb") as f:
                raw = f.read()
        except FileNotFoundError:
            return _fresh()
        except OSError as exc:
            raise LedgerError(f"spend ledger {self.path} is unreadable ({exc}); refusing to spend") from exc
        try:
            return _validated(json.loads(raw))
        except (ValueError, TypeError, RecursionError) as exc:
            bak = self._backup(raw)
            raise LedgerError(f"spend ledger {self.path} is corrupt ({exc}); refusing to spend. A copy is in "
                              f"{bak}. Repair or remove the ledger to continue.") from exc

    def _backup(self, raw: bytes) -> str:
        bak = self.path + ".bak"
        try:
            with open(bak, "rb") as f:
                if f.read() == raw:
                    return bak
        except OSError:
            pass
        try:
            self._write_bytes(bak, raw)
            _warn(f"spend ledger {self.path} is corrupt; copied it to {bak}")
        except OSError as exc:
            _warn(f"spend ledger {self.path} is corrupt and could not be copied to {bak}: {exc}")
        return bak

    def _write_bytes(self, dest: str, raw: bytes) -> None:
        d = os.path.dirname(dest) or "."
        fd, tmp = tempfile.mkstemp(prefix=os.path.basename(dest) + ".", suffix=".tmp", dir=d)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(raw)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, dest)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

    def _write(self, data: dict) -> None:
        data["updated"] = time.time()
        try:
            self._write_bytes(self.path, json.dumps(data).encode())
        except OSError as exc:
            raise LedgerError(f"cannot write spend ledger {self.path}: {exc}") from exc

    def _prune(self, data: dict) -> None:
        now = time.time()
        pend = data["pending"]
        for rid, r in list(pend.items()):
            if now - r["ts"] > self.reservation_ttl or (r.get("host") == self._host and not _pid_alive(r.get("pid"))):
                del pend[rid]

    # ------------------------------------------------------------ queries
    def total(self) -> float:
        """Settled spend in the ledger.  Raises LedgerError (a BudgetExceeded) if the ledger is corrupt."""
        return self._load()["total_usd"]

    def reserved(self) -> float:
        data = self._load()
        self._prune(data)
        return sum(r["usd"] for r in data["pending"].values())

    def remaining(self) -> float:
        data = self._load()
        self._prune(data)
        return self.cap - data["total_usd"] - sum(r["usd"] for r in data["pending"].values())

    # ------------------------------------------------------------ reservations
    def reserve(self, estimate_usd: float = 0.0, tag: str = "") -> str:
        """Book `estimate_usd` against the cap; returns a reservation id for settle()/release()."""
        est = _amount(estimate_usd, "estimate_usd")
        with self._locked():
            data = self._load()
            self._prune(data)
            spent = data["total_usd"]
            pending = sum(r["usd"] for r in data["pending"].values())
            if spent + pending + est > self.cap + _EPS:
                raise BudgetExceeded(f"spend cap ${self.cap:.2f} reached (spent ${spent:.4f}, "
                                     f"reserved ${pending:.4f}, this call ${est:.4f})")
            if self.session_cap is not None:
                s_pending = sum(self._session_pending.values())
                if self.session_usd + s_pending + est > self.session_cap + _EPS:
                    raise BudgetExceeded(f"session spend cap ${self.session_cap:.2f} reached "
                                         f"(session spent ${self.session_usd:.4f}, reserved ${s_pending:.4f})")
            rid = uuid.uuid4().hex
            data["pending"][rid] = {"usd": est, "ts": time.time(), "pid": os.getpid(), "host": self._host,
                                    "tag": str(tag)[:100]}
            self._write(data)
            self._session_pending[rid] = est
        return rid

    def release(self, reservation_id: Optional[str]) -> None:
        """Drop a reservation whose call never happened (no spend)."""
        if not reservation_id:
            return
        with self._locked():
            self._session_pending.pop(reservation_id, None)
            data = self._load()
            self._prune(data)
            if data["pending"].pop(reservation_id, None) is not None:
                self._write(data)

    def settle(self, reservation_id: Optional[str], model: str, usd: float, tag: str = "") -> None:
        """Record the actual cost of a call (replacing its reservation, if any).

        The spend is always written first; BudgetExceeded is raised afterwards if the ledger
        total or the session total is now over its cap.
        """
        usd = _amount(usd, "usd")
        with self._locked():
            if reservation_id:
                self._session_pending.pop(reservation_id, None)
            self.session_usd += usd
            self.session_calls[model] = self.session_calls.get(model, 0) + 1
            try:
                data = self._load()
            except LedgerError as exc:
                _warn(f"spend of ${usd:.4f} ({model}) could not be recorded: {exc}")
                raise
            self._prune(data)
            if reservation_id:
                data["pending"].pop(reservation_id, None)
            data["total_usd"] = data["total_usd"] + usd
            bm = data["by_model"]
            bm[model] = float(bm.get(model, 0.0)) + usd
            data["calls"] = data["calls"] + 1
            self._write(data)
            total = data["total_usd"]
        if total > self.cap + _EPS:
            raise BudgetExceeded(f"spend cap ${self.cap:.2f} exceeded (spent ${total:.4f})")
        if self.session_cap is not None and self.session_usd > self.session_cap + _EPS:
            raise BudgetExceeded(f"session spend cap ${self.session_cap:.2f} exceeded "
                                 f"(session spent ${self.session_usd:.4f})")

    # ------------------------------------------------------------ legacy pair
    def check(self, estimate_usd: float = 0.0) -> None:
        """Raise BudgetExceeded unless `estimate_usd` fits; reserves it for this thread's next record()."""
        prev = getattr(self._tls, "rid", None)
        self._tls.rid = None
        if prev:
            self.release(prev)        # the previous checked call never recorded: it did not happen
        self._tls.rid = self.reserve(estimate_usd)

    def record(self, model: str, usd: float, tag: str = "", reservation_id: Optional[str] = None) -> None:
        if reservation_id is None:
            reservation_id = getattr(self._tls, "rid", None)
            self._tls.rid = None
        self.settle(reservation_id, model, usd, tag)


def _reinit_lock_after_fork() -> None:
    Budget._lock = threading.RLock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reinit_lock_after_fork)


_DEFAULT: Optional[Budget] = None


def default_budget() -> Budget:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Budget()
    return _DEFAULT


_KEY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _dotenv_value(v: str) -> Optional[str]:
    """Value part of a .env line; None if malformed."""
    v = v.strip()
    if v[:1] in ("'", '"'):
        end = v.find(v[0], 1)
        if end == -1:
            return None                      # unterminated quote
        rest = v[end + 1:].strip()
        if rest and not rest.startswith("#"):
            return None                      # junk after the closing quote
        return v[1:end]
    if v.startswith("#"):
        return ""
    m = re.search(r"\s#", v)
    return v[:m.start()].rstrip() if m else v


def load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader (no dependency).  Existing environment variables win.

    Accepts ``KEY=value``, ``export KEY=value``, quoted values and inline ``  # comments``;
    malformed lines are ignored.  The file is skipped (with a warning) if it is not owned
    by the current user or is world-writable.
    """
    try:
        f = open(path, encoding="utf-8", errors="replace")
    except OSError:
        return
    with f:
        try:
            st = os.fstat(f.fileno())
        except OSError:
            return
        if not stat.S_ISREG(st.st_mode):
            return
        if st.st_uid != os.getuid() or st.st_mode & stat.S_IWOTH:
            _warn(f"ignoring {os.path.abspath(path)}: not owned by the current user or world-writable")
            return
        try:
            lines = f.read().splitlines()
        except OSError:
            return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export ") or line.startswith("export\t"):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        if not _KEY_RE.fullmatch(k):
            continue
        val = _dotenv_value(v)
        if val is None:
            continue
        os.environ.setdefault(k, val)
