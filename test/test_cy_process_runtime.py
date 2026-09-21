"""Real process failure tests; no geometry libraries are needed."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

import core.cy_process_runtime as process_runtime

from core.cy_process_runtime import (
    ManagedProcessError,
    ManagedProcessPool,
    MemoryBudgetExceeded,
    WorkerTaskError,
    WorkerTaskTimeout,
    _descendants,
    _process_table,
)


def _delayed_identity(value):
    time.sleep(0.1 if value == 0 else 0.002)
    return value


def _initialized_pid(value):
    return (os.getpid(), os.environ.get("CY_TEST_INITIALIZER"), value)


def _initializer(value):
    os.environ["CY_TEST_INITIALIZER"] = value


def _barrier_initializer(directory):
    directory = Path(directory)
    (directory / str(os.getpid())).touch()
    deadline = time.monotonic() + 4.0
    while time.monotonic() < deadline:
        if len(list(directory.iterdir())) == 2:
            return
        time.sleep(0.01)
    raise RuntimeError("Worker initializers did not start concurrently.")


def _raise_task(value):
    raise ValueError(f"invalid geometry {value}")


def _crash_task(_value):
    os._exit(17)


def _crash_once(path):
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return 42
    os.close(descriptor)
    os._exit(17)


def _sleep_task(seconds):
    time.sleep(seconds)


def _configuration_value(request):
    return len(request["configuration"])


def _configuration_crash_once(request):
    _crash_once(request["path"])
    return len(request["configuration"])


def _native_child_task(arguments):
    path, escape_group = arguments if isinstance(arguments, tuple) else (arguments, False)
    child = subprocess.Popen(
        [sys.executable, "-c", "import signal,time; "
         "signal.signal(signal.SIGTERM, signal.SIG_IGN); print('ready',flush=True); time.sleep(600)"],
        start_new_session=escape_group, stdout=subprocess.PIPE,
    )
    assert child.stdout.readline() == b"ready\n"
    Path(path).write_text(str(child.pid), encoding="ascii")
    time.sleep(600)


def _alive(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[0] not in {"Z", "X"}
    except (FileNotFoundError, ProcessLookupError):
        return False


def _wait_dead(pids, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if all(not _alive(pid) for pid in pids):
            return
        time.sleep(0.025)
    assert not [pid for pid in pids if _alive(pid)]


def _wait_pid_file(path, timeout=8.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            return int(path.read_text())
        except (FileNotFoundError, ValueError):
            time.sleep(0.02)
    pytest.fail(f"Native child did not announce its pid in {path}.")


def test_module_does_not_import_geometry_or_training():
    check = subprocess.run(
        [sys.executable, "-c", "import sys; import core.cy_process_runtime; "
         "assert not any(m.split('.')[0] in {'torch','sage','cytools','numpy'} "
         "for m in sys.modules)"],
        check=True, capture_output=True, text=True,
    )
    assert not check.stderr


def test_ordered_streaming_and_bounded_input_consumption():
    consumed = []

    def inputs():
        for value in range(20):
            consumed.append(value)
            yield value

    with ManagedProcessPool(num_workers=3) as pool:
        iterator = pool.imap(_delayed_identity, inputs())
        assert next(iterator) == 0
        assert len(consumed) == 3
        assert list(iterator) == list(range(1, 20))
        assert pool.stats["completed"] == 20
        pids = pool.worker_pids + (pool.guardian_pid,)
    _wait_dead(pids)
    pool.close()


def test_initializer_runs_in_each_managed_worker():
    with ManagedProcessPool(num_workers=2, initializer=_initializer, initargs=("registered",)) as pool:
        values = pool.map(_initialized_pid, [1, 2])
        assert {value[0] for value in values} == set(pool.worker_pids)
        assert [value[1] for value in values] == ["registered", "registered"]


def test_worker_initializers_are_dispatched_concurrently(tmp_path):
    with ManagedProcessPool(num_workers=2, initializer=_barrier_initializer,
                            initargs=(str(tmp_path),)) as pool:
        assert pool.map(abs, [-1, -2]) == [1, 2]


def test_dispatch_does_not_sleep_between_ready_windows(monkeypatch):
    with ManagedProcessPool(num_workers=1) as pool:
        with monkeypatch.context() as changes:
            def fail_on_sleep(_seconds):
                pytest.fail("Ready geometry dispatch was delayed by a sleep.")
            changes.setattr(process_runtime.time, "sleep", fail_on_sleep)
            assert pool.map(abs, range(10)) == list(range(10))


def test_single_result_next_keeps_an_idle_pool_reusable():
    with ManagedProcessPool(num_workers=2) as pool:
        assert next(pool.imap(abs, [-3])) == 3
        assert pool.map(abs, [-4, -5]) == [4, 5]


def test_repeated_configuration_uses_small_worker_local_references():
    configuration = tuple(range(20000))
    with ManagedProcessPool(num_workers=1) as pool:
        assert pool.map(_configuration_value, [{"configuration": configuration}]) == [20000]
        first_bytes = pool.stats["request_bytes_sent"]
        assert pool.map(_configuration_value, [{"configuration": configuration}]) == [20000]
        repeated_bytes = pool.stats["request_bytes_sent"] - first_bytes
        assert repeated_bytes < first_bytes / 20
        assert pool.stats["configuration_registrations"] == 1
        assert pool.stats["configuration_reuses"] == 1


def test_configuration_eviction_and_reclamation_keep_both_sides_in_sync(monkeypatch):
    monkeypatch.setattr(process_runtime, "_CONFIGURATION_CACHE_MAX_ENTRIES", 1)
    configurations = [(1, 2), (1, 2), (3, 4, 5), (1, 2)]
    with ManagedProcessPool(num_workers=1) as pool:
        assert pool.map(_configuration_value,
                        ({"configuration": value} for value in configurations)) == [2, 2, 3, 2]
        assert pool.stats["configuration_registrations"] == 3
        assert pool.stats["configuration_reuses"] == 1
        assert len(pool._transport_registries[0]) == 1
        pool._reclaim_worker(0)
        assert not pool._transport_registries[0]
        assert pool.map(_configuration_value, [{"configuration": (1, 2)}]) == [2]
        assert pool.stats["configuration_registrations"] == 4


def test_configuration_is_registered_again_after_worker_crash(tmp_path):
    request = {"configuration": (1, 2, 3), "path": str(tmp_path / "configuration_crash")}
    with ManagedProcessPool(num_workers=1) as pool:
        assert pool.map(_configuration_crash_once, [request]) == [3]
        assert pool.stats["configuration_registrations"] == 2
        assert pool.stats["restarts"] == 1


def test_zero_transport_budget_disables_configuration_registration():
    configuration = tuple(range(1000))
    with ManagedProcessPool(num_workers=1, configuration_cache_bytes=0) as pool:
        requests = [{"configuration": configuration} for _ in range(3)]
        assert pool.map(_configuration_value, requests) == [1000, 1000, 1000]
        assert pool.stats["configuration_registrations"] == 0
        assert pool.stats["configuration_reuses"] == 0
        assert pool.stats["configuration_identity_bytes"] == 0
        assert pool.stats["configuration_worker_cache_bytes"] == 0
        assert pool.memory_snapshot["trainer_rss_bytes"] > 0


def test_small_transport_budget_bounds_registration_without_changing_results():
    with ManagedProcessPool(num_workers=1, configuration_cache_bytes=128) as pool:
        assert pool.map(_configuration_value, [{"configuration": (1, 2)},
                                              {"configuration": tuple(range(1000))}]) == [2, 1000]
        assert pool.stats["configuration_identity_bytes"] <= 128
        assert pool.stats["configuration_worker_cache_bytes"] <= 128


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0, 0.0])
def test_invalid_deadlines_and_budgets_are_rejected(value):
    with pytest.raises(ValueError):
        ManagedProcessPool(task_timeout_sec=value)
    with pytest.raises(ValueError):
        ManagedProcessPool(memory_budget_gb=value)


def test_unsafe_start_method_is_rejected():
    with pytest.raises(ValueError, match="fork is unsafe"):
        ManagedProcessPool(start_method="fork")


def test_task_exception_has_remote_traceback_and_shuts_down():
    pool = ManagedProcessPool(num_workers=1)
    pids = pool.worker_pids + (pool.guardian_pid,)
    with pytest.raises(WorkerTaskError, match="invalid geometry 7"):
        pool.map(_raise_task, [7])
    _wait_dead(pids)


def test_worker_crash_is_retried_once(tmp_path):
    with ManagedProcessPool(num_workers=1) as pool:
        original = pool.worker_pids
        assert pool.map(_crash_once, [str(tmp_path / "crashed")]) == [42]
        assert pool.stats["restarts"] == 1
        assert pool.worker_pids != original
    _wait_dead(original)


def test_repeated_worker_crash_fails_without_inline_execution():
    pool = ManagedProcessPool(num_workers=1)
    with pytest.raises(ManagedProcessError, match="lost its worker twice"):
        pool.map(_crash_task, [None])
    _wait_dead(pool.worker_pids + (pool.guardian_pid,))


def test_hung_worker_has_a_bounded_deadline():
    pool = ManagedProcessPool(num_workers=1, task_timeout_sec=1.0)
    pids = pool.worker_pids + (pool.guardian_pid,)
    started = time.monotonic()
    with pytest.raises(WorkerTaskTimeout):
        pool.map(_sleep_task, [600])
    assert time.monotonic() - started < 5.0
    _wait_dead(pids)


def test_guardian_death_is_detected_and_workers_are_cleaned():
    pool = ManagedProcessPool(num_workers=2)
    pids = pool.worker_pids + (pool.guardian_pid,)
    os.kill(pool.guardian_pid, signal.SIGKILL)
    deadline = time.monotonic() + 5.0
    try:
        while time.monotonic() < deadline:
            try:
                pool.check_memory()
            except ManagedProcessError:
                break
            time.sleep(0.02)
        else:
            pytest.fail("Guardian death was not detected.")
        _wait_dead(pids)
    finally:
        pool.close()


@pytest.mark.parametrize("kill_target", ["trainer", "guardian"])
@pytest.mark.parametrize("escape_group", [False, True])
def test_sigkill_during_native_task_leaves_no_live_descendants(tmp_path, kill_target, escape_group):
    child_path = tmp_path / "native_pid"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).parent), os.getcwd()])
    script = (
        "import json,sys; from core.cy_process_runtime import ManagedProcessPool; "
        "from test_cy_process_runtime import _native_child_task; "
        "pool=ManagedProcessPool(num_workers=1); "
        "print(json.dumps([pool.guardian_pid, *pool.worker_pids]),flush=True); "
        "pool.map(_native_child_task,[(sys.argv[1],sys.argv[2]=='1')])"
    )
    trainer = subprocess.Popen(
        [sys.executable, "-c", script, str(child_path), str(int(escape_group))],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=environment,
    )
    owned = []
    try:
        owned = json.loads(trainer.stdout.readline())
        owned.append(_wait_pid_file(child_path))
        os.kill(trainer.pid if kill_target == "trainer" else owned[0], signal.SIGKILL)
        trainer.wait(timeout=6.0)
        _wait_dead(owned)
    finally:
        if trainer.poll() is None:
            trainer.kill()
            trainer.wait(timeout=6.0)
        for pid in owned:
            if _alive(pid):
                os.kill(pid, signal.SIGKILL)


def test_memory_budget_failure_is_explicit():
    with pytest.raises(MemoryBudgetExceeded):
        ManagedProcessPool(num_workers=1, memory_budget_gb=0.001)


def test_trainer_sigkill_during_worker_initialization(tmp_path):
    child_path = tmp_path / "initializing_native_pid"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).parent), os.getcwd()])
    trainer = subprocess.Popen(
        [sys.executable, "-c",
         "import sys; from core.cy_process_runtime import ManagedProcessPool; "
         "from test_cy_process_runtime import _native_child_task; "
         "ManagedProcessPool(num_workers=1,initializer=_native_child_task,initargs=(sys.argv[1],))",
         str(child_path)],
        env=environment, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    owned = set()
    try:
        native_pid = _wait_pid_file(child_path)
        owned = _descendants(trainer.pid, _process_table())
        assert native_pid in owned
        assert len(owned) == 3  # Guardian, initializing worker, native child.
        trainer.kill()
        trainer.wait(timeout=5.0)
        _wait_dead(owned)
    finally:
        if trainer.poll() is None:
            trainer.kill()
            trainer.wait(timeout=5.0)
        for pid in owned:
            if _alive(pid):
                os.kill(pid, signal.SIGKILL)
