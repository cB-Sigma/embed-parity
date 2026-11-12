"""Small numeric helpers. numpy only."""

from __future__ import annotations

import base64
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.floating[Any]]

_EPS = 1e-12


def as_matrix(x: Any) -> npt.NDArray[np.float64]:
    """Return ``x`` as a 2-D float64 array (a single vector becomes one row)."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"expected a vector or a matrix, got shape {arr.shape}")
    return arr


def norms(x: FloatArray) -> npt.NDArray[np.float64]:
    """Row-wise L2 norms."""
    return np.asarray(np.linalg.norm(as_matrix(x), axis=1), dtype=np.float64)


def normalize(x: FloatArray) -> npt.NDArray[np.float64]:
    """Row-wise L2 normalisation. Zero rows stay zero."""
    m = as_matrix(x)
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return np.asarray(m / np.maximum(n, _EPS), dtype=np.float64)


def row_cosine(a: FloatArray, b: FloatArray) -> npt.NDArray[np.float64]:
    """Cosine similarity between matching rows of ``a`` and ``b``."""
    am, bm = as_matrix(a), as_matrix(b)
    if am.shape != bm.shape:
        raise ValueError(f"shape mismatch: {am.shape} vs {bm.shape}")
    num = np.sum(am * bm, axis=1)
    den = np.linalg.norm(am, axis=1) * np.linalg.norm(bm, axis=1)
    return np.asarray(num / np.maximum(den, _EPS), dtype=np.float64)


def cosine(a: FloatArray, b: FloatArray) -> float:
    """Cosine similarity between two vectors."""
    return float(row_cosine(np.asarray(a)[None, :], np.asarray(b)[None, :])[0])


def max_abs_diff(a: FloatArray, b: FloatArray) -> float:
    """Largest element-wise absolute difference."""
    am, bm = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if am.shape != bm.shape:
        raise ValueError(f"shape mismatch: {am.shape} vs {bm.shape}")
    if am.size == 0:
        return 0.0
    return float(np.max(np.abs(am - bm)))


def rankdata(x: FloatArray) -> npt.NDArray[np.float64]:
    """Average ranks (1-based), ties share the mean rank."""
    v = np.asarray(x, dtype=np.float64).ravel()
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(v.size, dtype=np.float64)
    sorted_v = v[order]
    i = 0
    while i < v.size:
        j = i
        while j + 1 < v.size and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(a: FloatArray, b: FloatArray) -> float:
    """Spearman rank correlation. Returns 1.0 for constant-vs-constant input."""
    ra, rb = rankdata(a), rankdata(b)
    if ra.size != rb.size:
        raise ValueError("length mismatch")
    if ra.size < 2:
        return 1.0
    ca, cb = ra - ra.mean(), rb - rb.mean()
    den = float(np.sqrt(np.sum(ca * ca) * np.sum(cb * cb)))
    if den < _EPS:
        return 1.0 if np.allclose(ca, cb) else 0.0
    return float(np.sum(ca * cb) / den)


def sigmoid(x: FloatArray) -> npt.NDArray[np.float64]:
    """Numerically stable logistic function."""
    v = np.asarray(x, dtype=np.float64)
    out = np.empty_like(v)
    pos = v >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-v[pos]))
    ev = np.exp(v[~pos])
    out[~pos] = ev / (1.0 + ev)
    return out


# --------------------------------------------------------------------------- #
# Array (de)serialisation
# --------------------------------------------------------------------------- #

_DTYPES = {"float16": "<f2", "float32": "<f4", "float64": "<f8"}


def encode_array(x: FloatArray, dtype: str = "float32") -> dict[str, Any]:
    """Serialise an array as little-endian base64 with dtype and shape."""
    if dtype not in _DTYPES:
        raise ValueError(f"unsupported dtype {dtype!r}")
    arr = np.ascontiguousarray(np.asarray(x), dtype=_DTYPES[dtype])
    return {
        "dtype": dtype,
        "shape": list(arr.shape),
        "data": base64.b64encode(arr.tobytes()).decode("ascii"),
    }


def decode_array(obj: dict[str, Any]) -> npt.NDArray[np.float64]:
    """Inverse of :func:`encode_array`; always returns float64."""
    dtype = obj.get("dtype")
    if dtype not in _DTYPES:
        raise ValueError(f"unsupported dtype {dtype!r}")
    raw = base64.b64decode(obj["data"])
    arr = np.frombuffer(raw, dtype=_DTYPES[dtype])
    shape = tuple(int(s) for s in obj["shape"])
    expected = int(np.prod(shape)) if shape else 1
    if arr.size != expected:
        raise ValueError(f"array data has {arr.size} values, shape {shape} needs {expected}")
    return np.asarray(arr.reshape(shape), dtype=np.float64)


def decode_base64_embedding(data: str, dtype: str = "<f4") -> npt.NDArray[np.float64]:
    """Decode an OpenAI-style base64 embedding (little-endian float32 by default)."""
    raw = base64.b64decode(data)
    itemsize = np.dtype(dtype).itemsize
    if len(raw) % itemsize:
        raise ValueError(f"base64 payload of {len(raw)} bytes is not a multiple of {itemsize}")
    return np.asarray(np.frombuffer(raw, dtype=dtype), dtype=np.float64)


# --------------------------------------------------------------------------- #
# Quantisation
# --------------------------------------------------------------------------- #


def detect_quantization(x: FloatArray, full_dim: int | None = None) -> str:
    """Guess how the values of an endpoint's vectors were quantised.

    Returns one of ``float32``, ``float16``, ``int8``, ``uint8``, ``binary``
    (one value per dimension in {0,1} or {-1,1}) or ``packed-binary`` (bits
    packed into bytes, so the vector has ``full_dim / 8`` entries).

    The decision is made from value properties alone, so it also works when
    the direction of the vectors is wrong for some other reason.
    """
    m = as_matrix(x)
    if m.size == 0 or not np.all(np.isfinite(m)):
        return "float32"
    integral = bool(np.all(m == np.round(m)))
    if integral:
        values = np.unique(m)
        if values.size <= 2 and set(values.tolist()) <= {0.0, 1.0}:
            return "binary"
        if values.size <= 2 and set(values.tolist()) <= {-1.0, 1.0}:
            return "binary"
        lo, hi = float(np.min(m)), float(np.max(m))
        packed = full_dim is not None and m.shape[1] * 8 == full_dim
        if packed and lo >= -128 and hi <= 255:
            return "packed-binary"
        if lo >= -128 and hi <= 127:
            return "int8"
        if lo >= 0 and hi <= 255:
            return "uint8"
        return "float32"
    as16 = m.astype(np.float16).astype(np.float64)
    # JSON carries float32 values printed in decimal; a float16 survives the
    # round trip exactly, a genuine float32 almost never does for every entry.
    if np.array_equal(as16, m.astype(np.float32).astype(np.float64)):
        return "float16"
    return "float32"


def to_signs(x: FloatArray, kind: str, full_dim: int | None = None) -> npt.NDArray[np.float64]:
    """Map vectors to ±1 so binary outputs can be compared by cosine.

    ``kind`` is ``binary`` or ``packed-binary`` for endpoint output, or
    ``float`` for a reference vector (sign of each component).
    """
    m = as_matrix(x)
    if kind == "packed-binary":
        as_bytes = np.asarray(np.mod(m, 256), dtype=np.uint8)
        bits = np.unpackbits(as_bytes, axis=1)
        if full_dim is not None:
            bits = bits[:, :full_dim]
        return np.asarray(bits, dtype=np.float64) * 2.0 - 1.0
    return np.where(m > 0, 1.0, -1.0)
