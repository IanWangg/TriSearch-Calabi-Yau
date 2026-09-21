"""Bounded geometry workers with ownership independent of the training process.

This module deliberately imports only the standard library.  The guardian and
workers start via ``python -m`` instead of multiprocessing's main-module import,
so starting a worker does not import the trainer (or initialize CUDA).
"""

from __future__ import annotations

import atexit
import collections
import ctypes
import gc
import hashlib
import math
import multiprocessing.connection
import multiprocessing.reduction
import os
from pathlib import Path
import pickle
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import uuid


_subreaper_lock = threading.Lock()
_subreaper_users = 0
_previous_subreaper = 0
_CONFIGURATION_CACHE_MAX_ENTRIES = 4096


class ManagedProcessError(RuntimeError):
    """A managed worker or its owner failed."""


class WorkerTaskError(ManagedProcessError):
    """A task raised an exception in its geometry worker."""


class WorkerTaskTimeout(ManagedProcessError, TimeoutError):
    """A worker exceeded its task deadline."""


class MemoryBudgetExceeded(ManagedProcessError, MemoryError):
    """The owned process tree cannot operate within its memory budget."""


def _connection_pair():
    left, right = socket.socketpair()
    return (
        multiprocessing.connection.Connection(left.detach()),
        multiprocessing.connection.Connection(right.detach()),
    )


def _process_table():
    """Return pid -> (ppid, process group, resident bytes, start time, state)."""
    result = {}
    page_size = os.sysconf("SC_PAGE_SIZE")
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat", encoding="ascii") as stream:
                fields = stream.read().rsplit(")", 1)[1].split()
            result[int(name)] = (
                int(fields[1]), int(fields[2]), int(fields[21]) * page_size,
                int(fields[19]), fields[0],
            )
        except (OSError, ValueError, IndexError):
            continue
    return result


def _descendants(root_pid, table):
    children = collections.defaultdict(list)
    for pid, record in table.items():
        children[record[0]].append(pid)
    result = set()
    pending = [root_pid]
    while pending:
        for pid in children[pending.pop()]:
            if pid not in result:
                result.add(pid)
                pending.append(pid)
    return result


def _effective_memory_budget(requested_bytes):
    """Respect a smaller cgroup ceiling, including limits on parent cgroups."""
    if not requested_bytes:
        return 0
    ceilings = [requested_bytes]
    try:
        with open("/proc/self/cgroup", encoding="ascii") as stream:
            memberships = stream.read().splitlines()
        for membership in memberships:
            hierarchy, controllers, relative = membership.split(":", 2)
            if hierarchy == "0" and not controllers:
                root = Path("/sys/fs/cgroup")
                filename = "memory.max"
            elif "memory" in controllers.split(","):
                root = Path("/sys/fs/cgroup/memory")
                filename = "memory.limit_in_bytes"
            else:
                continue
            current = root / relative.lstrip("/")
            while current == root or root in current.parents:
                try:
                    value = (current / filename).read_text(encoding="ascii").strip()
                    if value != "max" and int(value) > 0:
                        ceilings.append(int(value))
                except (OSError, ValueError):
                    pass
                if current == root:
                    break
                current = current.parent
    except (OSError, ValueError):
        pass
    return min(ceilings)


def _memory_snapshot(parent_pid, budget_bytes, worker_pids, peak_bytes=0, table=None):
    table = _process_table() if table is None else table
    owned = _descendants(parent_pid, table) | {parent_pid}
    rss = sum(table[pid][2] for pid in owned if pid in table)
    return {
        "rss_bytes": rss,
        "trainer_rss_bytes": table[parent_pid][2] if parent_pid in table else 0,
        "peak_rss_bytes": max(rss, peak_bytes),
        "budget_bytes": budget_bytes,
        "fraction": rss / budget_bytes if budget_bytes else 0.0,
        "pressure": bool(budget_bytes and rss >= budget_bytes * 0.8),
        "process_count": len(owned),
        "worker_rss_bytes": {
            pid: sum(table[child][2] for child in
                     (_descendants(pid, table) | {pid}) if child in table)
            for pid in worker_pids
        },
        "sample_time": time.monotonic(),
    }


def _prctl(option, value):
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(option, value, 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def _register_subreaper():
    global _subreaper_users, _previous_subreaper
    with _subreaper_lock:
        if not _subreaper_users:
            previous = ctypes.c_int()
            _prctl(37, ctypes.byref(previous))  # PR_GET_CHILD_SUBREAPER
            _previous_subreaper = previous.value
            _prctl(36, 1)
        _subreaper_users += 1


def _unregister_subreaper():
    global _subreaper_users
    with _subreaper_lock:
        _subreaper_users -= 1
        if not _subreaper_users:
            _prctl(36, _previous_subreaper)


def _worker_parent_died(_signum, _frame):
    # Native descendants normally share this private process group.  Killing
    # the entire group also covers a guardian dying while a task is in C code.
    os.killpg(os.getpgrp(), signal.SIGKILL)


def _configuration_size(value, seen=None):
    """Approximate retained Python bytes for immutable configuration records."""
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    total = sys.getsizeof(value)
    if isinstance(value, dict):
        total += sum(_configuration_size(key, seen) + _configuration_size(item, seen)
                     for key, item in value.items())
    elif isinstance(value, (tuple, list, frozenset, set)):
        total += sum(_configuration_size(item, seen) for item in value)
    elif hasattr(value, "__dict__"):
        total += _configuration_size(vars(value), seen)
    return total


def _restore_configuration(item, transport, configurations, byte_limit):
    if transport is None:
        return item
    if transport[0] == "register":
        _, key, configuration, evicted, retained_bytes = transport
        for identity in evicted:
            configurations.pop(identity, None)
        if (retained_bytes > byte_limit or len(configurations) >= _CONFIGURATION_CACHE_MAX_ENTRIES or
                sum(record[1] for record in configurations.values()) + retained_bytes > byte_limit):
            raise ManagedProcessError("Configuration registration exceeded its transport cache budget.")
        configurations[key] = (configuration, retained_bytes)
    else:
        key = transport[1]
    item["configuration"] = configurations[key][0]
    return item


def _worker_main(fd, expected_parent):
    signal.signal(signal.SIGTERM, _worker_parent_died)
    _prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
    if os.getppid() != expected_parent:
        _worker_parent_died(None, None)
    connection = multiprocessing.connection.Connection(fd)
    reclaim = None
    configurations = {}
    configuration_limit = 0
    connection.send(("ready", os.getpid()))
    try:
        while True:
            message = connection.recv()
            if message[0] == "initialize":
                _, initializer, initargs, reclaim, configuration_limit = message
                try:
                    if initializer is not None:
                        initializer(*initargs)
                    connection.send(("initialized", os.getpid()))
                except BaseException:
                    connection.send(("initialization_error", traceback.format_exc()))
                    return
            elif message[0] == "run":
                _, task_id, function, item, transport = message
                try:
                    item = _restore_configuration(item, transport, configurations, configuration_limit)
                    result = function(item)
                    connection.send(("result", task_id, result))
                    del result, item, message, transport
                except BaseException:
                    connection.send(("task_error", task_id, traceback.format_exc()))
            elif message[0] == "reclaim":
                configurations.clear()
                if reclaim is not None:
                    reclaim()
                gc.collect()
                connection.send(("reclaimed",))
            elif message[0] == "stop":
                return
    except (EOFError, BrokenPipeError, ConnectionResetError):
        return
    finally:
        connection.close()


def _signal_processes(groups, pids, signum):
    own_group = os.getpgrp()
    for group in groups:
        if group <= 1 or group == own_group:
            continue
        try:
            os.killpg(group, signum)
        except ProcessLookupError:
            pass
    for pid in pids:
        if pid == os.getpid():
            continue
        try:
            os.kill(pid, signum)
        except ProcessLookupError:
            pass


def _reap_children():
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
            if not pid:
                return
        except ChildProcessError:
            return


def _cleanup_workers(processes, grace_seconds=2.0):
    groups = {process.pid for process in processes}
    table = _process_table()
    descendants = _descendants(os.getpid(), table)
    _signal_processes(groups, descendants, signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        for process in processes:
            process.poll()
        _reap_children()
        current = _descendants(os.getpid(), _process_table())
        if not current:
            return
        time.sleep(0.02)
    # Include adopted descendants: a native library may create a new session.
    descendants |= _descendants(os.getpid(), _process_table())
    _signal_processes(groups, descendants, signal.SIGKILL)
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        for process in processes:
            process.poll()
        _reap_children()
        current = _descendants(os.getpid(), _process_table())
        if not current:
            return
        # A native process may have forked between the previous snapshot and
        # SIGKILL. Its newly adopted children belong to this guardian too.
        _signal_processes(groups, current, signal.SIGKILL)
        time.sleep(0.02)


def _guardian_main(control_fd, parent_pid, parent_start, budget_bytes, worker_fds):
    _prctl(36, 1)  # PR_SET_CHILD_SUBREAPER
    control = multiprocessing.connection.Connection(control_fd)
    processes = {}
    reported_dead = set()
    stopping = False

    def request_stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def send(message):
        nonlocal stopping
        try:
            control.send(message)
        except (BrokenPipeError, EOFError, OSError):
            stopping = True

    def launch(index, fd):
        environment = os.environ.copy()
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
            environment[name] = "1"
        environment["CY_MANAGED_WORKER_INDEX"] = str(index)
        process = subprocess.Popen(
            [sys.executable, "-m", "core.cy_process_runtime", "--worker",
             str(fd), str(os.getpid())],
            pass_fds=(fd,), close_fds=True, start_new_session=True, env=environment,
        )
        os.close(fd)
        processes[index] = process
        reported_dead.discard(index)
        send(("worker_started", index, process.pid))

    peak = 0
    next_sample = 0.0
    memory_stop_deadline = None
    try:
        for index, fd in enumerate(worker_fds):
            launch(index, fd)
            if stopping:
                break
        while not stopping:
            now = time.monotonic()
            if now >= next_sample:
                table = _process_table()
                parent = table.get(parent_pid)
                if parent is None or parent[3] != parent_start or parent[4] in {"Z", "X"}:
                    break
                snapshot = _memory_snapshot(
                    parent_pid, budget_bytes,
                    [process.pid for process in processes.values()], peak, table=table,
                )
                peak = snapshot["peak_rss_bytes"]
                send(("memory", snapshot))
                if budget_bytes and snapshot["fraction"] >= 0.9:
                    if memory_stop_deadline is None:
                        send(("memory_stop", snapshot))
                        memory_stop_deadline = now + 2.0
                next_sample = now + 0.5
            if memory_stop_deadline is not None and now >= memory_stop_deadline:
                break
            for index, process in processes.items():
                if process.poll() is not None and index not in reported_dead:
                    reported_dead.add(index)
                    send(("worker_dead", index, process.returncode))
            if not control.poll(0.05):
                continue
            try:
                command = control.recv()
            except (EOFError, OSError):
                break
            if command[0] == "stop":
                break
            if command[0] == "restart":
                index = command[1]
                fd = multiprocessing.reduction.recv_handle(control)
                # Only this worker's group is stopped during replacement.
                process = processes[index]
                table = _process_table()
                owned = _descendants(process.pid, table)
                marker = f"CY_MANAGED_WORKER_INDEX={index}".encode()
                for pid in _descendants(os.getpid(), table):
                    try:
                        with open(f"/proc/{pid}/environ", "rb") as stream:
                            if marker in stream.read().split(b"\0"):
                                owned.add(pid)
                    except OSError:
                        pass
                _signal_processes({process.pid}, owned, signal.SIGKILL)
                process.wait(timeout=2.0)
                _reap_children()
                launch(index, fd)
    finally:
        _cleanup_workers(list(processes.values()))
        send(("stopped",))
        control.close()


class ManagedProcessPool:
    """A Linux pool with bounded ordered dispatch and supervised ownership.

    At most ``num_workers`` input/result objects are retained by ``imap``.  Tasks
    must be immutable and importable module-level callables: a request whose
    worker exits unexpectedly is retried once.  Python task exceptions, memory
    exhaustion and timeouts fail explicitly and are never run in the trainer.
    """

    def __init__(self, num_workers=0, start_method="spawn", memory_budget_gb=64,
                 task_timeout_sec=300, initializer=None, initargs=(),
                 reclaim_callback=None, worker_reclaim=None,
                 configuration_cache_bytes=64 * 1024 ** 2):
        if sys.platform != "linux":
            raise RuntimeError("Managed geometry workers require Linux process ownership.")
        if start_method != "spawn":
            raise ValueError("ManagedProcessPool requires start_method='spawn'; fork is unsafe.")
        if memory_budget_gb is not None and (
                not math.isfinite(float(memory_budget_gb)) or float(memory_budget_gb) <= 0):
            raise ValueError("memory_budget_gb must be positive or None.")
        if not math.isfinite(float(task_timeout_sec)) or float(task_timeout_sec) <= 0:
            raise ValueError("task_timeout_sec must be positive.")
        if (not math.isfinite(float(configuration_cache_bytes)) or
                int(configuration_cache_bytes) < 0):
            raise ValueError("configuration_cache_bytes must be finite and non-negative.")
        self.configuration_cache_bytes = int(configuration_cache_bytes)
        self.memory_budget_bytes = _effective_memory_budget(
            int(float(memory_budget_gb) * 1024 ** 3) if memory_budget_gb is not None else 0
        )
        requested = int(num_workers or 0)
        if requested < 0:
            raise ValueError("num_workers must be nonnegative.")
        table = _process_table()
        parent_record = table[os.getpid()]
        initial_memory = _memory_snapshot(os.getpid(), self.memory_budget_bytes, [], table=table)
        if self.memory_budget_bytes and initial_memory["fraction"] >= 0.9:
            raise MemoryBudgetExceeded(
                "Existing trainer processes already exceed 90% of the configured memory budget "
                f"({initial_memory['rss_bytes'] / 1024 ** 3:.2f} GiB RSS)."
            )
        if requested == 0:
            # Leave substantial headroom for geometry in automatic mode.
            headroom = max(1, int((self.memory_budget_bytes * 0.8 - parent_record[2]) /
                                 (2 * 1024 ** 3))) if self.memory_budget_bytes else 8
            requested = min(8, os.cpu_count() or 1, headroom)
        self.num_workers = requested
        self.task_timeout_sec = float(task_timeout_sec)
        self.initializer = initializer
        self.initargs = tuple(initargs)
        self.reclaim_callback = reclaim_callback
        self.worker_reclaim = worker_reclaim
        self._owner_pid = os.getpid()
        self._closed = False
        self._busy = False
        self._monitor_stop = threading.Event()
        self._control_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._failure = None
        self._memory = initial_memory
        self._worker_pids = {}
        self._dead = {}
        self._restarting = set()
        self._last_reclaim = 0.0
        self._stats = {"submitted": 0, "completed": 0, "restarts": 0,
                       "request_bytes_sent": 0, "configuration_registrations": 0,
                       "configuration_reuses": 0, "configuration_bytes_sent": 0}
        self._configuration_identities = collections.OrderedDict()
        self._configuration_identity_bytes = 0
        self._transport_registries = [collections.OrderedDict() for _ in range(self.num_workers)]
        self._transport_bytes = [0] * self.num_workers
        self._ownership_token = uuid.uuid4().hex
        self._subreaper_registered = False
        self._connections = []
        self._guardian = None
        self._control, guardian_control = _connection_pair()
        remote_connections = []
        for _ in range(self.num_workers):
            local, remote = _connection_pair()
            self._connections.append(local)
            remote_connections.append(remote)
        repository_root = str(Path(__file__).resolve().parents[1])
        environment = os.environ.copy()
        environment["CY_MANAGED_PROCESS_OWNER"] = self._ownership_token
        # Preserve pytest/script import paths for module-level task callables.
        paths = [repository_root] + [path for path in sys.path if path]
        environment["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(paths))
        descriptors = [guardian_control.fileno()] + [item.fileno() for item in remote_connections]
        try:
            _register_subreaper()
            self._subreaper_registered = True
            self._guardian = subprocess.Popen(
                [sys.executable, "-m", "core.cy_process_runtime", "--guardian",
                 str(descriptors[0]), str(self._owner_pid), str(parent_record[3]),
                 str(self.memory_budget_bytes)] + [str(fd) for fd in descriptors[1:]],
                pass_fds=descriptors, close_fds=True, env=environment,
            )
        except BaseException:
            if self._subreaper_registered:
                _unregister_subreaper()
                self._subreaper_registered = False
            self._control.close()
            for connection in self._connections:
                connection.close()
            raise
        finally:
            guardian_control.close()
            for connection in remote_connections:
                connection.close()
        self._monitor = threading.Thread(
            target=self._monitor_guardian, name="cy_guardian_monitor", daemon=True,
        )
        self._monitor.start()
        atexit.register(self.shutdown)
        try:
            deadline = time.monotonic() + self.task_timeout_sec
            for index in range(self.num_workers):
                self._begin_initialize_worker(index, deadline)
            for index in range(self.num_workers):
                self._finish_initialize_worker(index, deadline)
        except BaseException:
            self.shutdown()
            raise

    @property
    def guardian_pid(self):
        return self._guardian.pid if self._guardian is not None else None

    @property
    def worker_pids(self):
        with self._state_lock:
            return tuple(self._worker_pids[index] for index in sorted(self._worker_pids))

    @property
    def memory_snapshot(self):
        with self._state_lock:
            return dict(self._memory)

    @property
    def stats(self):
        with self._state_lock:
            result = dict(self._stats)
        result.update({"num_workers": self.num_workers, "worker_pids": self.worker_pids,
                       "guardian_pid": self.guardian_pid, "memory": self.memory_snapshot,
                       "configuration_cache_limit_bytes": self.configuration_cache_bytes,
                       "configuration_identity_bytes": self._configuration_identity_bytes,
                       "configuration_worker_cache_bytes": sum(self._transport_bytes)})
        return result

    def _monitor_guardian(self):
        try:
            while not self._monitor_stop.is_set():
                if not self._control.poll(0.05):
                    if self._guardian.poll() is not None:
                        break
                    continue
                message = self._control.recv()
                with self._state_lock:
                    if message[0] == "worker_started":
                        self._worker_pids[message[1]] = message[2]
                        self._dead.pop(message[1], None)
                        self._restarting.discard(message[1])
                    elif message[0] == "worker_dead":
                        if message[1] not in self._restarting:
                            self._dead[message[1]] = message[2]
                    elif message[0] == "memory":
                        self._memory = message[1]
                    elif message[0] == "memory_stop":
                        self._memory = message[1]
                        self._failure = MemoryBudgetExceeded(
                            "Owned processes reached 90% of the configured memory budget "
                            f"({message[1]['rss_bytes'] / 1024 ** 3:.2f} GiB RSS)."
                        )
                    elif message[0] == "stopped":
                        break
        except (EOFError, OSError):
            pass
        finally:
            if not self._closed:
                with self._state_lock:
                    if self._failure is None:
                        self._failure = ManagedProcessError("Geometry guardian exited unexpectedly.")
                    groups = set(self._worker_pids.values())
                self._cleanup_adopted_children(groups)

    def _cleanup_adopted_children(self, groups):
        """Clean only this pool's children if its guardian cannot reap them."""
        table = _process_table()
        descendants = _descendants(self._owner_pid, table)
        token = f"CY_MANAGED_PROCESS_OWNER={self._ownership_token}".encode()
        owned = set()
        for pid in descendants:
            if pid == self.guardian_pid:
                continue  # Popen owns the guardian's wait status.
            try:
                with open(f"/proc/{pid}/environ", "rb") as stream:
                    if token in stream.read().split(b"\0"):
                        owned.add(pid)
            except OSError:
                pass
            if table[pid][1] in groups:
                owned.add(pid)
        owned |= groups
        _signal_processes(groups, owned, signal.SIGKILL)
        deadline = time.monotonic() + 1.0
        while owned and time.monotonic() < deadline:
            for pid in list(owned):
                try:
                    reaped, _ = os.waitpid(pid, os.WNOHANG)
                    if reaped:
                        owned.discard(pid)
                except (ChildProcessError, ProcessLookupError):
                    owned.discard(pid)
            if owned:
                time.sleep(0.01)

    def check_memory(self):
        """Raise on supervision/budget failure and return the latest sample."""
        if os.getpid() != self._owner_pid:
            raise ManagedProcessError("ManagedProcessPool cannot be inherited by another process.")
        if self._closed:
            raise ManagedProcessError("ManagedProcessPool is closed.")
        with self._state_lock:
            failure = self._failure
            snapshot = dict(self._memory)
        if failure is not None:
            raise failure
        if snapshot.get("pressure") and time.monotonic() - self._last_reclaim >= 0.5:
            self._last_reclaim = time.monotonic()
            self._configuration_identities.clear()
            self._configuration_identity_bytes = 0
            if self.reclaim_callback is not None:
                self.reclaim_callback()
            gc.collect()
        return snapshot

    def _receive(self, index, deadline):
        connection = self._connections[index]
        while time.monotonic() < deadline:
            self.check_memory()
            if connection.poll(0.05):
                return connection.recv()
            with self._state_lock:
                if index in self._dead:
                    raise EOFError(f"Worker {index} exited with status {self._dead[index]}.")
        raise WorkerTaskTimeout(f"Geometry worker {index} exceeded {self.task_timeout_sec:g}s.")

    def _initialize_worker(self, index):
        deadline = time.monotonic() + self.task_timeout_sec
        self._begin_initialize_worker(index, deadline)
        self._finish_initialize_worker(index, deadline)

    def _begin_initialize_worker(self, index, deadline):
        message = self._receive(index, deadline)
        if message[0] != "ready":
            raise ManagedProcessError(f"Unexpected geometry worker startup: {message[0]}.")
        with self._state_lock:
            self._worker_pids[index] = message[1]
        self._connections[index].send((
            "initialize", self.initializer, self.initargs, self.worker_reclaim,
            self.configuration_cache_bytes,
        ))

    def _finish_initialize_worker(self, index, deadline):
        message = self._receive(index, deadline)
        if message[0] != "initialized":
            raise WorkerTaskError(f"Geometry worker initialization failed:\n{message[-1]}")

    def _restart_worker(self, index):
        self._connections[index].close()
        self._transport_registries[index].clear()
        self._transport_bytes[index] = 0
        local, remote = _connection_pair()
        self._connections[index] = local
        with self._state_lock:
            self._dead.pop(index, None)
            self._restarting.add(index)
            self._stats["restarts"] += 1
        try:
            with self._control_lock:
                self._control.send(("restart", index))
                multiprocessing.reduction.send_handle(
                    self._control, remote.fileno(), self.guardian_pid,
                )
        finally:
            remote.close()
        self._initialize_worker(index)

    def _reclaim_worker(self, index):
        self._connections[index].send(("reclaim",))
        response = self._receive(index, time.monotonic() + self.task_timeout_sec)
        if response[0] != "reclaimed":
            raise ManagedProcessError("Invalid geometry cache-reclamation response.")
        self._transport_registries[index].clear()
        self._transport_bytes[index] = 0

    def _prepare_configuration(self, index, item):
        """Intern immutable configuration fields without changing task APIs."""
        if (not self.configuration_cache_bytes or
                not isinstance(item, dict) or "configuration" not in item):
            return item, None, None
        configuration = item["configuration"]
        identity = id(configuration)
        metadata = self._configuration_identities.get(identity)
        if metadata is None:
            serialized = pickle.dumps(configuration, protocol=pickle.HIGHEST_PROTOCOL)
            key = hashlib.blake2b(serialized, digest_size=20).digest()
            size = _configuration_size(configuration)
            metadata = (configuration, key, size, len(serialized))
            if size <= self.configuration_cache_bytes:
                while self._configuration_identities and (
                        len(self._configuration_identities) >= _CONFIGURATION_CACHE_MAX_ENTRIES or
                        self._configuration_identity_bytes + size > self.configuration_cache_bytes):
                    _, removed = self._configuration_identities.popitem(last=False)
                    self._configuration_identity_bytes -= removed[2]
                self._configuration_identities[identity] = metadata
                self._configuration_identity_bytes += size
        else:
            self._configuration_identities.move_to_end(identity)
        _, key, size, serialized_size = metadata
        if size > self.configuration_cache_bytes:
            return item, None, None
        stripped = {name: value for name, value in item.items() if name != "configuration"}
        registry = self._transport_registries[index]
        if key in registry:
            return stripped, ("reference", key), (key, size, 0, ())
        evicted = []
        occupied = self._transport_bytes[index]
        for old_key, old_size in registry.items():
            if (occupied + size <= self.configuration_cache_bytes and
                    len(registry) - len(evicted) < _CONFIGURATION_CACHE_MAX_ENTRIES):
                break
            evicted.append(old_key)
            occupied -= old_size
        return (stripped, ("register", key, configuration, tuple(evicted), size),
                (key, size, serialized_size, tuple(evicted)))

    def _send_task(self, index, identity, function, item):
        wire_item, transport, registration = self._prepare_configuration(index, item)
        payload = multiprocessing.reduction.ForkingPickler.dumps(
            ("run", identity, function, wire_item, transport),
        )
        self._connections[index].send_bytes(payload)
        # Update sender state only once the entire registration message was
        # sent. A restarted worker always starts with an empty registry.
        with self._state_lock:
            self._stats["request_bytes_sent"] += len(payload)
            if registration is not None:
                key, size, sent_bytes, evicted = registration
                registry = self._transport_registries[index]
                for old_key in evicted:
                    self._transport_bytes[index] -= registry.pop(old_key)
                if key not in registry:
                    registry[key] = size
                    self._transport_bytes[index] += size
                registry.move_to_end(key)
                self._stats["configuration_bytes_sent"] += sent_bytes
                self._stats["configuration_registrations" if sent_bytes else "configuration_reuses"] += 1

    def imap(self, function, iterable, chunksize=1):
        """Yield ordered results with one active request per worker."""
        if chunksize != 1:
            raise ValueError("Bounded geometry dispatch requires chunksize=1.")
        if self._busy:
            raise ManagedProcessError("Only one imap iterator may run at a time.")
        self.check_memory()
        self._busy = True
        source = iter(iterable)
        pending = {}
        results = {}
        task_id = 0
        next_result = 0
        exhausted = False
        pressure_since = None
        idle_reclaimed = 0.0

        def dispatch(index, identity, item, retries=0):
            try:
                self._send_task(index, identity, function, item)
            except (EOFError, ConnectionResetError, BrokenPipeError, OSError):
                if retries:
                    raise ManagedProcessError(f"Geometry task {identity} lost its worker twice.")
                self._restart_worker(index)
                retries += 1
                self._send_task(index, identity, function, item)
            pending[index] = (identity, item, time.monotonic(), retries)
            with self._state_lock:
                self._stats["submitted"] += 1

        try:
            while pending or results or not exhausted:
                snapshot = self.check_memory()
                pressure = snapshot.get("pressure", False)
                if pressure:
                    if pressure_since is None:
                        pressure_since = time.monotonic()
                    if time.monotonic() - idle_reclaimed >= 0.5:
                        idle_reclaimed = time.monotonic()
                        for index in range(self.num_workers):
                            if index not in pending:
                                self._reclaim_worker(index)
                else:
                    pressure_since = None
                for index in range(self.num_workers):
                    if exhausted or pressure or task_id >= next_result + self.num_workers:
                        break
                    if index in pending:
                        continue
                    try:
                        item = next(source)
                    except StopIteration:
                        exhausted = True
                        break
                    dispatch(index, task_id, item)
                    task_id += 1
                    del item
                while next_result in results:
                    value = results.pop(next_result)
                    next_result += 1
                    yield value
                    del value
                if not pending:
                    if exhausted:
                        break
                    if pressure_since is not None and time.monotonic() - pressure_since >= 3.0:
                        raise MemoryBudgetExceeded(
                            "Memory remained above 80% of budget after reclamation; "
                            "geometry dispatch stopped."
                        )
                    if pressure:
                        time.sleep(0.02)
                    continue
                ready = multiprocessing.connection.wait(
                    [self._connections[index] for index in pending], timeout=0.05,
                )
                for index in list(pending):
                    identity, item, started, retries = pending[index]
                    if time.monotonic() - started >= self.task_timeout_sec:
                        raise WorkerTaskTimeout(
                            f"Geometry task {identity} exceeded {self.task_timeout_sec:g}s."
                        )
                    with self._state_lock:
                        dead = index in self._dead
                    if self._connections[index] not in ready and not dead:
                        continue
                    try:
                        if dead and self._connections[index] not in ready:
                            raise EOFError("Geometry worker exited.")
                        message = self._connections[index].recv()
                    except (EOFError, ConnectionResetError, BrokenPipeError, OSError) as error:
                        if retries:
                            raise ManagedProcessError(
                                f"Geometry task {identity} lost its worker twice."
                            ) from error
                        self._restart_worker(index)
                        dispatch(index, identity, item, retries=1)
                        continue
                    if message[0] == "task_error":
                        raise WorkerTaskError(f"Geometry task {identity} failed:\n{message[2]}")
                    if message[0] != "result" or message[1] != identity:
                        raise ManagedProcessError("Geometry worker returned an invalid task identity.")
                    del pending[index]
                    results[identity] = message[2]
                    del message, item
                    with self._state_lock:
                        self._stats["completed"] += 1
                    if pressure:
                        self._reclaim_worker(index)
        except GeneratorExit:
            # A single-result caller may use next(imap(...)).  Keep the pool
            # reusable if that request completed, but never abandon live work.
            if pending:
                self.shutdown()
            raise
        except BaseException:
            self.shutdown()
            raise
        finally:
            self._busy = False

    def map(self, function, iterable, chunksize=1):
        return list(self.imap(function, iterable, chunksize=chunksize))

    def shutdown(self, wait=True, cancel_futures=False):
        """Stop every owned process, with a bounded wait even for hung tasks."""
        del wait, cancel_futures
        if self._closed or os.getpid() != self._owner_pid:
            return
        self._closed = True
        try:
            with self._control_lock:
                self._control.send(("stop",))
        except (BrokenPipeError, EOFError, OSError):
            pass
        if self._guardian is not None:
            try:
                self._guardian.wait(timeout=3.5)
            except subprocess.TimeoutExpired:
                _signal_processes(set(self.worker_pids), set(), signal.SIGKILL)
                self._guardian.kill()
                self._guardian.wait(timeout=1.0)
        self._cleanup_adopted_children(set(self.worker_pids))
        self._monitor_stop.set()
        self._control.close()
        for connection in self._connections:
            connection.close()
        self._configuration_identities.clear()
        self._configuration_identity_bytes = 0
        for registry in self._transport_registries:
            registry.clear()
        self._transport_bytes[:] = [0] * self.num_workers
        if hasattr(self, "_monitor") and threading.current_thread() is not self._monitor:
            self._monitor.join(timeout=1.2)
        if self._subreaper_registered:
            _unregister_subreaper()
            self._subreaper_registered = False
        atexit.unregister(self.shutdown)

    close = shutdown

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback_value):
        self.shutdown()


def _main():
    if sys.argv[1] == "--guardian":
        _guardian_main(*(int(value) for value in sys.argv[2:6]),
                       [int(value) for value in sys.argv[6:]])
    elif sys.argv[1] == "--worker":
        _worker_main(int(sys.argv[2]), int(sys.argv[3]))
    else:
        raise SystemExit("This module is an internal geometry process launcher.")


if __name__ == "__main__":
    _main()
