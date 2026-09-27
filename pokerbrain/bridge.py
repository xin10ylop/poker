"""File bridge: let an external process (e.g. a Claude Code session) be the Opus decider.

Match side (Python):   OpusAgent(FileBridgeDecider("bridge/live"), ...)
Answer side (shell):   python -m pokerbrain.bridge next   bridge/live      # prints the next prompt
                       python -m pokerbrain.bridge answer bridge/live <id> '<json decision>'
                       python -m pokerbrain.bridge status bridge/live
The match blocks on each escalated decision until an answer file appears.
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid


class FileBridgeDecider:
    def __init__(self, qdir: str, timeout: float = 900.0, poll: float = 0.25):
        self.qdir = qdir
        self.timeout = timeout
        self.poll = poll
        os.makedirs(os.path.join(qdir, "pending"), exist_ok=True)
        os.makedirs(os.path.join(qdir, "answers"), exist_ok=True)
        os.makedirs(os.path.join(qdir, "done"), exist_ok=True)

    def __call__(self, system: str, user: str, meta: dict) -> dict:
        rid = time.strftime("%H%M%S") + "-" + uuid.uuid4().hex[:6]
        p = os.path.join(self.qdir, "pending", rid + ".json")
        tmp = p + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"id": rid, "system": system, "user": user, "meta": meta}, f)
        os.replace(tmp, p)
        a = os.path.join(self.qdir, "answers", rid + ".json")
        t0 = time.time()
        while not os.path.exists(a):
            if time.time() - t0 > self.timeout:
                raise TimeoutError(f"no answer for {rid}")
            time.sleep(self.poll)
        time.sleep(0.05)
        with open(a) as f:
            ans = json.load(f)
        os.replace(p, os.path.join(self.qdir, "done", rid + ".json"))
        return ans


def _next(qdir: str, wait: float = 600.0) -> None:
    pend = os.path.join(qdir, "pending")
    t0 = time.time()
    while True:
        files = sorted(f for f in os.listdir(pend) if f.endswith(".json")) if os.path.isdir(pend) else []
        if files:
            with open(os.path.join(pend, files[0])) as f:
                req = json.load(f)
            print(f"REQUEST_ID: {req['id']}\n")
            print(req["user"])
            return
        if os.path.exists(os.path.join(qdir, "MATCH_DONE")):
            print("MATCH_DONE")
            with open(os.path.join(qdir, "MATCH_DONE")) as f:
                print(f.read())
            return
        if time.time() - t0 > wait:
            print("NO_REQUEST (timeout)")
            return
        time.sleep(0.3)


def _answer(qdir: str, rid: str, payload: str) -> None:
    data = json.loads(payload)
    a = os.path.join(qdir, "answers", rid + ".json")
    with open(a + ".tmp", "w") as f:
        json.dump(data, f)
    os.replace(a + ".tmp", a)
    print("ok")


if __name__ == "__main__":
    cmd, qdir = sys.argv[1], sys.argv[2]
    if cmd == "next":
        _next(qdir)
    elif cmd == "answer":
        _answer(qdir, sys.argv[3], sys.argv[4])
    elif cmd == "status":
        pend = os.listdir(os.path.join(qdir, "pending"))
        done = os.listdir(os.path.join(qdir, "done"))
        print(f"pending={len(pend)} done={len(done)}")
