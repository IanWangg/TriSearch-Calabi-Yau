"""Read recent query progress and record resources for this study's jobs."""

from datetime import datetime, timezone
import json
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[4]
STUDY = Path(__file__).resolve().parents[1]
notes = json.loads((STUDY / "experiment_notes.json").read_text())
pids = dict(line.split() for line in (STUDY / "logs" / f"pids_{notes['stamp']}.txt").read_text().splitlines())
snapshot = dict(time=datetime.now(timezone.utc).isoformat(),
                available_memory_gb=psutil.virtual_memory().available / 2**30, jobs={})
all_cpus = set()
for tag, job in notes["jobs"].items():
    run = ROOT / job["run"]
    progress = {}
    path = run / "queries.jsonl"
    if path.is_file():
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 1500000))
            if stream.tell():
                stream.readline()
            for line in stream:
                try:
                    query = json.loads(line)
                except json.JSONDecodeError:
                    continue
                progress[str(query["polytope_index"])] = query["query_index"]
    summary_path = run / "summary.json"
    complete = summary_path.is_file()
    rollout_path = run / "rollouts.jsonl"
    if rollout_path.is_file():
        for line in rollout_path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            progress[str(row["polytope_index"])] = row["objective_queries"]
    rss = 0
    active = False
    try:
        process = psutil.Process(int(pids[tag]))
        active = process.is_running() and process.status() != psutil.STATUS_ZOMBIE
        for child in [process, *process.children(recursive=True)]:
            try:
                rss += child.memory_info().rss
            except psutil.NoSuchProcess:
                pass
        assert process.cpu_affinity() == job["cpu_ids"]
        if active and "CUDA_VISIBLE_DEVICES" in job:
            assert process.environ().get("CUDA_VISIBLE_DEVICES") == job["CUDA_VISIBLE_DEVICES"]
    except psutil.NoSuchProcess:
        pass
    assert not all_cpus.intersection(job["cpu_ids"])
    all_cpus.update(job["cpu_ids"])
    snapshot["jobs"][tag] = dict(queries_by_polytope=progress, rss_gb=rss / 2**30,
                                active=active, complete=complete, failure=(run / "failure.json").exists(),
                                cpu_ids=job["cpu_ids"], physical_gpu_index=job.get("physical_gpu_index"))
    print(f"{tag}: {progress}; RAM={rss / 2**30:.1f} GiB; active={active}; complete={complete}")
    if (run / "failure.json").exists():
        print((run / "failure.json").read_text()[:2000])
with (STUDY / "logs" / f"resources_{notes['stamp']}.jsonl").open("a") as stream:
    stream.write(json.dumps(snapshot) + "\n")
print(f"Available RAM: {snapshot['available_memory_gb']:.1f} GiB")
