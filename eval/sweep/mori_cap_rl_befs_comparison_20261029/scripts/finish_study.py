"""Continue the running CPU pilot into the full benchmark after validation.

This stdlib-only supervisor uses no geometry or neural inference. Run with the
sage interpreter, CPU startup environment, and the original pilot's 20 CPUs.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu_ids", nargs=20, type=int, required=True)
    args = parser.parse_args()
    if os.environ.get("TRISEARCH_CPU_ONLY") != "1":
        parser.error("TRISEARCH_CPU_ONLY=1 is required")
    state_path = STUDY / "continuation.json"
    state = dict(status="waiting_for_pilot", pid=os.getpid(), cpu_only=True,
                 cpu_ids=args.cpu_ids,
                 started_at=datetime.now(timezone.utc).isoformat())

    def save():
        temporary = state_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(state_path)

    save()
    try:
        validation = json.loads((STUDY / "validation.json").read_text())
        if (validation["status"] != "complete" or validation["smoke_status"] != "complete"
                or any(suite["failures"] or suite["errors"] for suite in validation["tests"])):
            raise RuntimeError("Persisted CPU validation has not passed")
        pilot_launcher = json.loads((STUDY / "pilot_launcher.json").read_text())
        pilot_path = STUDY / "results/pilot/study_run.json"
        while True:
            pilot = json.loads(pilot_path.read_text())
            if pilot["status"] == "complete":
                break
            if pilot["status"] != "running":
                raise RuntimeError(f"Pilot did not complete: {pilot.get('error', pilot['status'])}")
            if not Path(f"/proc/{pilot_launcher['pid']}").exists():
                raise RuntimeError("Pilot launcher disappeared before completing its manifest")
            time.sleep(10)
        state.update(status="running_full", pilot_setup_id=pilot["setup_id"])
        save()
        command = [sys.executable, str(STUDY / "scripts/run_study.py"), "--phase", "full",
                   "--cpu_ids", *map(str, args.cpu_ids)]
        state["full_command"] = command
        with (STUDY / "logs/full_launcher.log").open("w") as log:
            process = subprocess.Popen(command, cwd=REPO, env=os.environ.copy(),
                                       stdout=log, stderr=subprocess.STDOUT)
            state["full_launcher_pid"] = process.pid
            save()
            returncode = process.wait()
        if returncode:
            raise RuntimeError(f"Full benchmark failed with exit code {returncode}")
        state.update(status="complete", finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        state.update(status="failed", error=str(exc))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
