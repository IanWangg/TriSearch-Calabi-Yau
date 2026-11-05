"""Episode pool for training: slots that each run one episode (polytope line + start) at a time.

Each slot owns a JumpEnv, rebuilt when its polytope line changes (codebook setup is about 1.5 s on
h11=20). With inline=True the last slot runs in the caller's process, so n_slots slots use
n_slots processes in total (n_slots-1 spawned workers plus the caller).
"""

from __future__ import annotations

import multiprocessing as mp
import pickle
import traceback

from rl_jump.vec import load_row


class EnvSlot:
    def __init__(self, env_kwargs):
        self.env_kwargs = dict(env_kwargs)
        self.env = None
        self.key = None
        self.banked = {}  # counters of envs replaced by a rebuild (cumulative totals stay correct)

    def _bank(self):
        for key, value in self.env.stats().items():
            if isinstance(value, (int, float)):
                self.banked[key] = self.banked.get(key, 0) + value

    def begin(self, path, line, start_index):
        if self.env is None or self.key != (str(path), int(line)):
            from rl_jump.env import JumpEnv

            if self.env is not None:
                self._bank()
            self.env = JumpEnv(load_row(path, line), **self.env_kwargs)
            self.key = (str(path), int(line))
        return self.env.reset(start_index % len(self.env.starts))

    def step(self, action):
        return self.env.step(action)

    def stats(self):
        out = dict(self.banked)
        if self.env is not None:
            for key, value in self.env.stats().items():
                if isinstance(value, (int, float)):
                    out[key] = out.get(key, 0) + value
        return out


def _worker(connection, env_kwargs):
    try:
        slot = EnvSlot(env_kwargs)
        connection.send(("ready", None))
        while True:
            command, argument = connection.recv()
            if command == "close":
                break
            if command == "begin":
                connection.send(("ok", pickle.dumps(slot.begin(*argument))))
            elif command == "step":
                connection.send(("ok", pickle.dumps(slot.step(argument))))
            elif command == "stats":
                connection.send(("ok", pickle.dumps(slot.stats())))
            else:
                raise ValueError(f"unknown command {command!r}")
    except Exception:
        connection.send(("error", traceback.format_exc()))
    finally:
        connection.close()


class EpisodePool:
    """Slots 0..n-2 are workers; slot n-1 is inline when inline=True (otherwise all slots are workers)."""

    def __init__(self, env_kwargs, n_slots, *, inline=True):
        if n_slots < 1:
            raise ValueError("n_slots must be positive.")
        self.n_slots = int(n_slots)
        self.inline = bool(inline)
        n_workers = self.n_slots - (1 if self.inline else 0)
        context = mp.get_context("spawn")
        self.workers = []
        for slot in range(n_workers):
            parent, child = context.Pipe()
            process = context.Process(target=_worker, args=(child, dict(env_kwargs, seed=env_kwargs.get("seed", 0) + slot)),
                                      daemon=True)
            process.start()
            child.close()
            self.workers.append((parent, process))
        self.n_workers = n_workers
        self.local = EnvSlot(dict(env_kwargs, seed=env_kwargs.get("seed", 0) + self.n_slots - 1)) if self.inline else None
        for index in range(self.n_workers):
            self._receive(index, "ready")

    def _receive(self, index, expect="ok"):
        kind, payload = self.workers[index][0].recv()
        if kind == "ok":
            # Plain pickle bytes: tensors arrive as private copies, not shared-memory fds. Holding many
            # shared-fd tensors in the replay buffer exhausted the mmap limit (ENOMEM) in the first R2 Q run.
            payload = pickle.loads(payload)
        if kind == "error":
            raise RuntimeError(f"pool worker {index} failed:\n{payload}")
        if kind != expect:
            raise RuntimeError(f"pool worker {index} sent {kind!r}, expected {expect!r}.")
        return payload

    def begin(self, assignments):
        """assignments: {slot: (rows_path, line, start_index)}. Returns {slot: observation}."""
        for slot, request in assignments.items():
            if slot < self.n_workers:
                self.workers[slot][0].send(("begin", tuple(request)))
        out = {}
        if self.local is not None and (self.n_slots - 1) in assignments:
            out[self.n_slots - 1] = self.local.begin(*assignments[self.n_slots - 1])
        for slot in assignments:
            if slot < self.n_workers:
                out[slot] = self._receive(slot)
        return out

    def step(self, actions):
        """actions: {slot: candidate index}. Returns {slot: (observation, reward, done, info)}."""
        for slot, action in actions.items():
            if slot < self.n_workers:
                self.workers[slot][0].send(("step", int(action)))
        out = {}
        if self.local is not None and (self.n_slots - 1) in actions:
            out[self.n_slots - 1] = self.local.step(int(actions[self.n_slots - 1]))
        for slot in actions:
            if slot < self.n_workers:
                out[slot] = self._receive(slot)
        return out

    def stats(self):
        """Sum of all slots' counters (numeric values only)."""
        for slot in range(self.n_workers):
            self.workers[slot][0].send(("stats", None))
        collected = [self.local.stats()] if self.local is not None else []
        collected += [self._receive(slot) for slot in range(self.n_workers)]
        total = {}
        for item in collected:
            for key, value in item.items():
                if isinstance(value, (int, float)):
                    total[key] = total.get(key, 0) + value
        return total

    def close(self):
        for parent, process in self.workers:
            try:
                parent.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
            process.join(timeout=60)
            if process.is_alive():
                process.terminate()
        self.workers = []
