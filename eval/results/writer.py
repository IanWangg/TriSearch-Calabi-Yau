"""Keep partial results readable if geometry or an algorithm fails."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import asdict
import json
from pathlib import Path
import traceback


class ResultWriter:
    def __init__(self, path: Path):
        self.path = path
        self._resources = ExitStack()
        self._streams = {}

    def __enter__(self):
        try:
            for name in ("queries", "transitions", "expansions", "rollouts"):
                self._streams[name] = self._resources.enter_context(
                    (self.path / f"{name}.jsonl").open("w", encoding="utf-8")
                )
        except BaseException:
            self._resources.close()
            raise
        return self

    def write_json(self, name: str, payload: dict) -> None:
        from data.cy.pipeline import _write_json_atomic

        _write_json_atomic(self.path / f"{name}.json", payload)

    def append(self, name: str, payload: dict) -> None:
        stream = self._streams[name]
        stream.write(json.dumps(payload, allow_nan=False) + "\n")
        stream.flush()

    def query(self, event: dict) -> None:
        self.append("queries", event)

    def transition(self, event: dict) -> None:
        self.append("transitions", event)

    def expansion(self, event: dict) -> None:
        self.append("expansions", event)

    def rollout(self, result) -> None:
        self.append("rollouts", asdict(result))

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc is not None:
                self.write_json("failure", {"status": "failed", "error": str(exc),
                                           "traceback": "".join(traceback.format_exception(exc_type, exc, tb))})
        finally:
            self._resources.close()
