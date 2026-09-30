"""File bridge: let an external process (e.g. a Claude Code session) be the Opus decider.

Match side (Python):   OpusAgent(FileBridgeDecider("bridge/live"), ...)
Answer side (shell):   python -m pokerbrain.bridge next   bridge/live      # prints the next prompt
                       python -m pokerbrain.bridge answer bridge/live <id> '<json decision>'
                       python -m pokerbrain.bridge answer bridge/live <id> - < decision.json   # via stdin
                       python -m pokerbrain.bridge status bridge/live
The match blocks on each escalated decision until an answer file appears (default
timeout 120 s; the agent then falls back to the engine).

Queue layout: <qdir>/pending/<id>.json (open requests), answers/<id>.json, done/<id>.json,
RUN_ID (start time of the current decider, ns) and MATCH_DONE (written when a match ends).
Request ids are "<time_ns, 20 digits>-<6 hex>", so they sort chronologically.  A new
FileBridgeDecider clears pending/ and MATCH_DONE, and `next` ignores requests older than RUN_ID.
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import sys
import tempfile
import time
import uuid
from typing import Optional

RID_RE = re.compile(r"(\d{20})-([0-9a-f]{6})")
DEFAULT_TIMEOUT_S = 120.0
USAGE = """usage:
  python -m pokerbrain.bridge next   <qdir>
  python -m pokerbrain.bridge answer <qdir> <id> '<json decision>'
  python -m pokerbrain.bridge answer <qdir> <id> -   < decision.json
  python -m pokerbrain.bridge status <qdir>"""


class BridgeError(ValueError):
    pass


def _rid_ns(rid: str) -> Optional[int]:
    m = RID_RE.fullmatch(rid)
    return int(m.group(1)) if m else None


def _atomic_write(path: str, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", prefix=".tmp-", suffix=".part")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _read_run_id(qdir: str) -> int:
    try:
        with open(os.path.join(qdir, "RUN_ID")) as f:
            return int(f.read().strip() or 0)
    except (OSError, ValueError):
        return 0


class FileBridgeDecider:
    def __init__(self, qdir: str, timeout: float = DEFAULT_TIMEOUT_S, poll: float = 0.25):
        self.qdir = qdir
        self.timeout = float(timeout)
        self.poll = poll
        for sub in ("pending", "answers", "done"):
            os.makedirs(os.path.join(qdir, sub), exist_ok=True)
        # A new match: requests and the end marker left by an earlier run must not be served.
        pend = os.path.join(qdir, "pending")
        for f in os.listdir(pend):
            src = os.path.join(pend, f)
            with contextlib.suppress(OSError):
                if f.endswith(".json"):
                    os.replace(src, os.path.join(qdir, "done", f))
                else:
                    os.remove(src)
        with contextlib.suppress(FileNotFoundError):
            os.remove(os.path.join(qdir, "MATCH_DONE"))
        self.run_id = time.time_ns()
        self._last_ns = self.run_id
        _atomic_write(os.path.join(qdir, "RUN_ID"), f"{self.run_id}\n")

    def _new_rid(self) -> str:
        ns = max(time.time_ns(), self._last_ns + 1)    # never older than RUN_ID, even if the clock steps back
        self._last_ns = ns
        return f"{ns:020d}-{uuid.uuid4().hex[:6]}"

    def __call__(self, system: str, user: str, meta: dict) -> dict:
        rid = self._new_rid()
        p = os.path.join(self.qdir, "pending", rid + ".json")
        a = os.path.join(self.qdir, "answers", rid + ".json")
        _atomic_write(p, json.dumps({"id": rid, "system": system, "user": user, "meta": meta}))
        try:
            deadline = time.monotonic() + self.timeout
            while not os.path.exists(a):
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"no answer for {rid} within {self.timeout:g}s")
                time.sleep(self.poll)
            return self._read_answer(a, rid)
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.replace(p, os.path.join(self.qdir, "done", rid + ".json"))

    def _read_answer(self, a: str, rid: str) -> dict:
        grace = time.monotonic() + 1.0          # tolerate a writer that is not atomic
        while True:
            try:
                with open(a) as f:
                    ans = json.load(f)
                if not isinstance(ans, dict):
                    raise ValueError("not a JSON object")
                return ans
            except ValueError as exc:
                if time.monotonic() >= grace:
                    raise ValueError(f"malformed answer for {rid}: {exc}") from exc
            time.sleep(min(self.poll, 0.1))


def _live_request(qdir: str) -> Optional[dict]:
    pend = os.path.join(qdir, "pending")
    try:
        names = os.listdir(pend)
    except OSError:
        return None
    run = _read_run_id(qdir)
    rids = []
    for f in names:
        if not f.endswith(".json"):
            continue
        rid = f[:-5]
        ns = _rid_ns(rid)
        if ns is None or ns < run or os.path.exists(os.path.join(qdir, "answers", f)):
            continue                      # stale (earlier run / old id format) or already answered
        rids.append(rid)
    for rid in sorted(rids):
        try:
            with open(os.path.join(pend, rid + ".json")) as fh:
                req = json.load(fh)
        except (OSError, ValueError):
            continue                      # timed out and moved meanwhile, or unreadable
        if isinstance(req, dict) and req.get("id") == rid and isinstance(req.get("user"), str):
            return req
    return None


def _next(qdir: str, wait: float = 600.0) -> None:
    deadline = time.monotonic() + wait
    done = os.path.join(qdir, "MATCH_DONE")
    while True:
        req = _live_request(qdir)
        if req is not None:
            print(f"REQUEST_ID: {req['id']}\n")
            print(req["user"])
            return
        if os.path.exists(done):
            print("MATCH_DONE")
            with contextlib.suppress(OSError), open(done) as f:
                print(f.read())
            return
        if time.monotonic() >= deadline:
            print("NO_REQUEST (timeout)")
            return
        time.sleep(0.3)


def _answer(qdir: str, rid: str, payload: str) -> None:
    if not RID_RE.fullmatch(rid):
        raise BridgeError(f"bad request id {rid!r} (expected <20 digits>-<6 hex digits>)")
    if not os.path.isfile(os.path.join(qdir, "pending", rid + ".json")):
        raise BridgeError(f"no pending request {rid} in {qdir} (already answered, timed out, or mistyped)")
    text = sys.stdin.read() if payload == "-" else payload
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise BridgeError(f"answer is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise BridgeError("answer must be a JSON object")
    _atomic_write(os.path.join(qdir, "answers", rid + ".json"), json.dumps(data))
    print("ok")


def _status(qdir: str) -> None:
    if not os.path.isdir(qdir):
        print(f"pending=0 done=0 (no bridge queue at {qdir})")
        return

    def count(sub: str) -> list:
        try:
            return [f for f in os.listdir(os.path.join(qdir, sub)) if f.endswith(".json")]
        except OSError:
            return []

    pend = count("pending")
    run = _read_run_id(qdir)
    live = sum(1 for f in pend if (_rid_ns(f[:-5]) or -1) >= run)
    extra = " MATCH_DONE" if os.path.exists(os.path.join(qdir, "MATCH_DONE")) else ""
    print(f"pending={len(pend)} done={len(count('done'))} live={live}{extra}")


def main(argv: Optional[list] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in ("-h", "--help", "help"):
        print(USAGE)
        return 0
    arity = {"next": 2, "status": 2, "answer": 4}
    if not args or arity.get(args[0]) != len(args):
        print(USAGE, file=sys.stderr)
        return 2
    cmd, qdir = args[0], args[1]
    try:
        if cmd == "next":
            _next(qdir)
        elif cmd == "answer":
            _answer(qdir, args[2], args[3])
        else:
            _status(qdir)
    except (BridgeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
