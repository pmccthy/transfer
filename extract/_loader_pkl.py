"""Load a session loader pickle without its heavy deps or huge arrays.

The loader pickles (``<session>.pkl`` under ``SAT_dff_from_flu_processed``) were
written from a ``LoaderSAT`` object whose module imports ``matlab.engine`` and
lab packages at import time, and which stores very large arrays we don't need
(``flu``, ``flu_processed``, ``paq_data['data']`` and a duplicate session dict).
The largest of these are multiple GB, so we must avoid even *reading* their raw
bytes into memory (some session pkls are ~5 GB on disk).

This pure-Python Unpickler:
  * substitutes a harmless stub for every non-numpy/pandas class,
  * seeks past any byte blob larger than ``SKIP_BYTES`` (the raw data of the big
    arrays) instead of loading it, pushing an empty ``b""`` in its place, and
  * collapses the corresponding ndarray to an empty array on ``__setstate__``.

The per-condition dF/F arrays (float32, ≤ ~46 MB), ``trial_idxs`` and
``beh_df`` are well under the threshold and come back as ordinary numpy/pandas
objects. Oversized arrays come back empty — they are never used.
"""

from __future__ import annotations

import pickle
import struct

import numpy as np

try:
    from numpy.core.multiarray import _reconstruct as _np_reconstruct
except Exception:  # pragma: no cover
    from numpy._core.multiarray import _reconstruct as _np_reconstruct

_REAL = ("numpy", "pandas", "builtins", "collections", "functools",
         "_codecs", "datetime", "_pickle", "copyreg")

# Keep/drop decision is by ELEMENT COUNT (dtype-independent): a dF/F condition is
# (n_cells x ~90 time x n_trials) and is at most ~25M elements; flu/flu_processed
# (n_cells x ~58k) and paq (2 x tens-of-millions) are far larger. Note some
# sessions store dF/F as float64, so a byte threshold alone is unreliable.
MAX_ELEMS = 50_000_000
# Only used to avoid *reading* multi-GB raw blobs (paq / large flu) into memory.
# Sits above the largest dF/F condition (~176 MB as float64) and below flu.
SKIP_BYTES = 250_000_000
# Downcast kept floating arrays to float32 (halves memory for float64 sessions).
DOWNCAST_FLOAT32 = True
_CHUNK = 1 << 24  # 16 MB


class _LazyArray(np.ndarray):
    """ndarray subclass that drops its data on unpickle if it is too large."""

    def __setstate__(self, state):
        shape = state[1]
        n = 1
        for s in shape:
            n *= int(s)
        if n > MAX_ELEMS:
            super().__setstate__((state[0], (0,), state[2], state[3], b""))
        else:
            super().__setstate__(state)


def _reconstruct_factory(subtype, shape, dtype):
    return _np_reconstruct(_LazyArray, shape, dtype)


def _frombuffer_factory(buffer, dtype, shape, order):
    """Stand-in for numpy's _frombuffer. Drops oversized arrays (flu / paq) and
    any whose buffer was skipped; otherwise reconstructs, downcasting floats to
    float32 to save memory."""
    dtype = np.dtype(dtype)
    n = 1
    for s in shape:
        n *= int(s)
    if n > MAX_ELEMS:                       # flu / paq — never used
        return np.empty((0,), dtype=np.float32)
    need = n * dtype.itemsize
    if buffer is None or len(buffer) < need:
        return np.empty((0,), dtype=np.float32)
    arr = np.frombuffer(bytes(buffer), dtype=dtype).reshape(shape, order=order)
    if DOWNCAST_FLOAT32 and dtype.kind == "f" and dtype.itemsize > 4:
        arr = arr.astype(np.float32)
    return arr


class _Stub:
    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {"_state": state})

    def __init__(self, *a, **k):
        pass


class _SafeUnpickler(pickle._Unpickler):  # pure-Python unpickler (overridable)
    dispatch = dict(pickle._Unpickler.dispatch)

    def find_class(self, module, name):
        if module.endswith("multiarray") and name == "_reconstruct":
            return _reconstruct_factory
        if name == "_frombuffer":
            return _frombuffer_factory
        if module.split(".")[0] in _REAL:
            return super().find_class(module, name)
        return type(name, (_Stub,), {})

    def _read_or_skip(self, length):
        if length > SKIP_BYTES:
            remaining = length
            while remaining > 0:
                chunk = self.read(min(remaining, _CHUNK))
                if not chunk:
                    break
                remaining -= len(chunk)
            self.append(b"")
        else:
            self.append(self.read(length))

    def load_binbytes(self):
        (length,) = struct.unpack("<I", self.read(4))
        self._read_or_skip(length)
    dispatch[pickle.BINBYTES[0]] = load_binbytes

    def load_binbytes8(self):
        (length,) = struct.unpack("<Q", self.read(8))
        self._read_or_skip(length)
    dispatch[pickle.BINBYTES8[0]] = load_binbytes8

    def load_bytearray8(self):
        (length,) = struct.unpack("<Q", self.read(8))
        if length > SKIP_BYTES:
            remaining = length
            while remaining > 0:
                chunk = self.read(min(remaining, _CHUNK))
                if not chunk:
                    break
                remaining -= len(chunk)
            self.append(bytearray())
        else:
            b = bytearray(length)
            self.readinto(b)
            self.append(b)
    if hasattr(pickle, "BYTEARRAY8"):
        dispatch[pickle.BYTEARRAY8[0]] = load_bytearray8


def load_loader(path):
    """Unpickle a loader pkl, stubbing unknown classes and skipping huge arrays."""
    with open(path, "rb") as f:
        return _SafeUnpickler(f).load()
