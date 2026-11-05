"""Vectorized JumpEnv: each env in its own spawned process (CPU geometry); the policy stays in the caller.

Specs are dicts {"path": jsonl, "line": row index, "env_kwargs": {...}}. The env built from the last spec
can run inline in the caller's process (inline_last=True), so N envs can use N-1 worker processes.
Workers answer ("reset", start), ("step", action), ("stats", None) and ("close", None).
"""

from __future__ import annotations

import json
import multiprocessing as mp
import traceback


def load_row(path, line):
    with open(path, encoding="utf-8") as handle:
        for index, text in enumerate(handle):
            if index == line:
                return json.loads(text)
    raise IndexError(f"{path} has no line {line}.")


def make_env(spec):
    from rl_jump.env import JumpEnv

    return JumpEnv(load_row(spec["path"], spec["line"]), **spec.get("env_kwargs", {}))


def _worker(connection, spec):
    try:
        env = make_env(spec)
        connection.send(("ready", env.setup_seconds))
        while True:
            command, argument = connection.recv()
            if command == "close":
                break
            if command == "reset":
                connection.send(("ok", env.reset(argument)))
            elif command == "step":
                connection.send(("ok", env.step(argument)))
            elif command == "stats":
                connection.send(("ok", env.stats()))
            else:
                raise ValueError(f"unknown command {command!r}")
    except Exception:
        connection.send(("error", traceback.format_exc()))
    finally:
        connection.close()


class VecJumpEnv:
    def __init__(self, specs, *, inline_last=False):
        if not specs:
            raise ValueError("At least one env spec is required.")
        context = mp.get_context("spawn")
        self.size = len(specs)
        self.n_workers = self.size - (1 if inline_last else 0)
        self.workers = []
        for spec in specs[:self.n_workers]:
            parent, child = context.Pipe()
            process = context.Process(target=_worker, args=(child, spec), daemon=True)
            process.start()
            child.close()
            self.workers.append((parent, process))
        self.inline = make_env(specs[-1]) if inline_last else None
        self.setup_seconds = [self._receive(i, expect="ready") for i in range(self.n_workers)]
        if self.inline is not None:
            self.setup_seconds.append(self.inline.setup_seconds)

    def _receive(self, index, expect="ok"):
        kind, payload = self.workers[index][0].recv()
        if kind == "error":
            raise RuntimeError(f"env worker {index} failed:\n{payload}")
        if kind != expect and not (expect == "ok" and kind == "ok"):
            raise RuntimeError(f"env worker {index} sent {kind!r}, expected {expect!r}.")
        return payload

    def reset(self, starts):
        """starts: one start index per env. Returns one observation per env."""
        for index in range(self.n_workers):
            self.workers[index][0].send(("reset", starts[index]))
        out = [None] * self.size
        if self.inline is not None:
            out[-1] = self.inline.reset(starts[-1])
        for index in range(self.n_workers):
            out[index] = self._receive(index)
        return out

    def step(self, actions):
        """actions: {env_index: candidate index}. Returns {env_index: (obs, reward, done, info)}."""
        sent = [i for i in sorted(actions) if i < self.n_workers]
        for index in sent:
            self.workers[index][0].send(("step", actions[index]))
        out = {}
        if self.inline is not None and (self.size - 1) in actions:
            out[self.size - 1] = self.inline.step(actions[self.size - 1])
        for index in sent:
            out[index] = self._receive(index)
        return out

    def stats(self):
        for index in range(self.n_workers):
            self.workers[index][0].send(("stats", None))
        out = [None] * self.size
        if self.inline is not None:
            out[-1] = self.inline.stats()
        for index in range(self.n_workers):
            out[index] = self._receive(index)
        return out

    def close(self):
        for parent, process in self.workers:
            try:
                parent.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
        self.workers = []
