"""Exact, disk-backed state identities and counts independent of cache eviction."""

from collections.abc import MutableMapping, MutableSet
from pathlib import Path
import sqlite3
import tempfile
import zlib


def _encoded_key(key):
    return zlib.compress(str(key).encode("utf-8"), level=1)


class StateHistory:
    def __init__(self, path=None, cache_bytes=8 * 1024**2):
        self._temporary = None
        if path is None:
            runtime_dir = Path(__file__).resolve().parents[1] / "runs" / "runtime"
            runtime_dir.mkdir(parents=True, exist_ok=True)
            self._temporary = tempfile.TemporaryDirectory(prefix="cy_history_", dir=runtime_dir)
            path = Path(self._temporary.name) / "states.sqlite3"
        self.path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute(f"PRAGMA cache_size={-(max(0, int(cache_bytes)) // 1024)}")
        self.connection.execute("PRAGMA mmap_size=0")
        self.connection.execute("PRAGMA temp_store=FILE")
        self.connection.execute("CREATE TABLE IF NOT EXISTS state_keys (namespace TEXT, key TEXT, value INTEGER NOT NULL, PRIMARY KEY(namespace,key)) WITHOUT ROWID")
        self._pending = 0
        self._lengths = {}

    def _write(self, sql, args):
        result = self.connection.execute(sql, args)
        self._pending += 1
        if self._pending >= 512:
            self.flush()
        return result

    def flush(self):
        if self.connection is not None:
            self.connection.commit()
            self._pending = 0

    def keys(self, namespace):
        return HistorySet(self, namespace)

    def counts(self, namespace):
        return HistoryCounts(self, namespace)

    def close(self):
        if self.connection is not None:
            self.flush()
            self.connection.close()
            self.connection = None
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class HistoryCounts(MutableMapping):
    def __init__(self, history, namespace):
        self.history, self.namespace = history, namespace
        if namespace not in history._lengths:
            self._length = history.connection.execute("SELECT COUNT(*) FROM state_keys WHERE namespace=?", (namespace,)).fetchone()[0]

    @property
    def _length(self):
        return self.history._lengths[self.namespace]

    @_length.setter
    def _length(self, value):
        self.history._lengths[self.namespace] = value

    def __getitem__(self, key):
        row = self.history.connection.execute("SELECT value FROM state_keys WHERE namespace=? AND key=?", (self.namespace, _encoded_key(key))).fetchone()
        if row is None:
            raise KeyError(key)
        return row[0]

    def __setitem__(self, key, value):
        key = _encoded_key(key)
        inserted = self.history._write("INSERT OR IGNORE INTO state_keys VALUES(?,?,?)", (self.namespace, key, int(value))).rowcount
        self._length += inserted
        if not inserted:
            self.history._write("UPDATE state_keys SET value=? WHERE namespace=? AND key=?", (int(value), self.namespace, key))

    def __delitem__(self, key):
        if not self.history._write("DELETE FROM state_keys WHERE namespace=? AND key=?", (self.namespace, _encoded_key(key))).rowcount:
            raise KeyError(key)
        self._length -= 1

    def __iter__(self):
        return (zlib.decompress(row[0]).decode("utf-8") for row in self.history.connection.execute("SELECT key FROM state_keys WHERE namespace=?", (self.namespace,)))

    def __len__(self):
        return self._length

    def clear(self):
        self.history._write("DELETE FROM state_keys WHERE namespace=?", (self.namespace,))
        self._length = 0


class HistorySet(MutableSet):
    def __init__(self, history, namespace):
        self.counts = HistoryCounts(history, namespace)

    def __contains__(self, key):
        return self.counts.get(key) is not None

    def __iter__(self):
        return iter(self.counts)

    def __len__(self):
        return len(self.counts)

    def add(self, key):
        inserted = self.counts.history._write("INSERT OR IGNORE INTO state_keys VALUES(?,?,0)", (self.counts.namespace, _encoded_key(key))).rowcount
        self.counts._length += inserted
        return bool(inserted)

    def discard(self, key):
        self.counts._length -= self.counts.history._write("DELETE FROM state_keys WHERE namespace=? AND key=?", (self.counts.namespace, _encoded_key(key))).rowcount

    def clear(self):
        self.counts.clear()
