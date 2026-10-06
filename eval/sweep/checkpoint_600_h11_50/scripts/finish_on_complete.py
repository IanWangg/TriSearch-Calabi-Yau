"""Produce the validated comparison when all four formal evaluations finish."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
STUDY = Path(__file__).resolve().parents[1]
notes = json.loads((STUDY / "experiment_notes.json").read_text())
runs = [ROOT / job["run"] for job in notes["jobs"].values()]
while not all((run / "summary.json").is_file() for run in runs):
    for run in runs:
        if (run / "failure.json").exists():
            raise RuntimeError(f"Evaluation failed: {run / 'failure.json'}")
    time.sleep(20)
environment = os.environ.copy()
environment["CUDA_VISIBLE_DEVICES"] = ""
environment["PYTHONDONTWRITEBYTECODE"] = "1"
with (STUDY / "logs" / f"analysis_{notes['stamp']}.log").open("w") as log:
    subprocess.run([sys.executable, "-B", str(STUDY / "scripts/analyze_results.py")],
                   cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
print(f"Validated results and plots: {STUDY / 'plots'}", flush=True)
