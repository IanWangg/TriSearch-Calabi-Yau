"""Small, dependency-free caches with admission-time memory limits."""

from collections import OrderedDict
from collections.abc import MutableMapping
from dataclasses import fields, is_dataclass
import sys
from types import ModuleType


def retained_size(value, seen=None):
    """Conservative Python/tensor storage estimate; native heaps need RSS monitoring."""
    seen = set() if seen is None else seen
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        return size + sum(retained_size(k, seen) + retained_size(v, seen) for k, v in value.items())
    if isinstance(value, (tuple, list, set, frozenset)):
        return size + sum(retained_size(item, seen) for item in value)
    # Do not import Torch/NumPy into a geometry worker just to measure storage.
    if type(value).__module__.startswith("torch") and hasattr(value, "untyped_storage"):
        return size + value.untyped_storage().nbytes()
    if type(value).__module__.startswith("numpy") and hasattr(value, "nbytes"):
        return size + int(value.nbytes)
    if is_dataclass(value) and not isinstance(value, type):
        return size + sum(retained_size(getattr(value, f.name), seen) for f in fields(value))
    if not isinstance(value, (type, ModuleType)) and not callable(value):
        attributes = getattr(value, "__dict__", None)
        if attributes is not None:
            return size + retained_size(attributes, seen)
    return size


class BoundedLRU(MutableMapping):
    """An LRU whose limits apply on every insertion, including replacements."""

    def __init__(self, max_bytes, max_entries=None):
        self.max_bytes = max(0, int(max_bytes))
        self.max_entries = None if max_entries is None else max(0, int(max_entries))
        self._data = OrderedDict()
        self._sizes = {}
        self.bytes = 0
        self.hits = self.misses = self.evictions = self.bypasses = 0

    def __getitem__(self, key):
        try:
            value = self._data[key]
        except KeyError:
            self.misses += 1
            raise
        self._data.move_to_end(key)
        self.hits += 1
        return value

    def __setitem__(self, key, value):
        if key in self._data:
            del self[key]
        size = retained_size((key, value))
        if size > self.max_bytes or self.max_entries == 0:
            self.bypasses += 1
            return
        self._data[key] = value
        self._sizes[key] = size
        self.bytes += size
        self._trim()

    def __delitem__(self, key):
        del self._data[key]
        self.bytes -= self._sizes.pop(key)

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __contains__(self, key):
        return key in self._data

    def move_to_end(self, key, last=True):
        self._data.move_to_end(key, last=last)

    def popitem(self, last=True):
        key, value = self._data.popitem(last=last)
        self.bytes -= self._sizes.pop(key)
        return key, value

    def _trim(self):
        while self._data and (self.bytes > self.max_bytes or (
            self.max_entries is not None and len(self) > self.max_entries
        )):
            self.popitem(last=False)
            self.evictions += 1

    def resize(self, max_bytes=None, max_entries=None):
        if max_bytes is not None:
            self.max_bytes = max(0, int(max_bytes))
        if max_entries is not None:
            self.max_entries = max(0, int(max_entries))
        self._trim()

    def clear(self):
        self._data.clear()
        self._sizes.clear()
        self.bytes = 0

    def stats(self):
        return {"entries": len(self), "bytes": self.bytes, "max_bytes": self.max_bytes,
                "hits": self.hits, "misses": self.misses, "evictions": self.evictions,
                "bypasses": self.bypasses}
