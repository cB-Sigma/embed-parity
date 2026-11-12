"""Reference files: what the library returns, plus the variants needed for diagnosis.

A reference stores, for every probe:

* ``library`` - exactly what the reference library returns with default
  arguments (float32). This is the vector an endpoint is expected to match.
* ``variants`` - sentence vectors recomputed from token states under other
  implementation choices (pooling mode, prompt prefix, special tokens,
  truncation side and length, skipped Dense projection). They are stored
  *before* normalisation; normalisation, dimension truncation and
  quantisation are applied on the fly when comparing.

Variants are stored as float16 by default: they are only ever compared by
cosine similarity, where the rounding error is around 1e-7.
"""

from __future__ import annotations

import datetime as _dt
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from embed_parity import __version__
from embed_parity.backends import (
    POOLINGS,
    EmbeddingBackend,
    InputConfig,
    ModelInfo,
    RerankBackend,
    pool,
)
from embed_parity.mathx import decode_array, encode_array, normalize, row_cosine
from embed_parity.probes import ProbeSet, builtin_probes

FORMAT = "embed-parity/reference"
FORMAT_VERSION = 1

DEFAULT = "default"
DENSE_OFF = "dense=off"


class ReferenceFileError(ValueError):
    """The reference file is malformed or cannot answer the question asked."""


@dataclass
class VariantTable:
    """Vectors for one (input config, pooling) pair, for some or all probes."""

    probe_ids: list[str]
    matrix: npt.NDArray[np.float64]


@dataclass
class RerankReference:
    """Cross-encoder scores for every (query, document) pair of the probe set."""

    query_ids: list[str]
    document_ids: list[str]
    logits: npt.NDArray[np.float64]
    library: npt.NDArray[np.float64] | None
    token_counts: list[list[int]] | None = None


@dataclass(frozen=True)
class Expectation:
    """Which behaviour the endpoint is expected to show."""

    config: str = DEFAULT
    pooling: str = "mean"
    normalize: bool = True
    dimensions: int | None = None
    is_declared: bool = True

    def label(self) -> str:
        return f"{self.config}/{self.pooling}"


@dataclass
class Reference:
    """In-memory form of a reference file."""

    info: ModelInfo
    probes: ProbeSet
    input_configs: dict[str, InputConfig] = field(default_factory=dict)
    variants: dict[str, VariantTable] = field(default_factory=dict)
    library: npt.NDArray[np.float64] | None = None
    tokens: dict[str, dict[str, int]] = field(default_factory=dict)
    full_tokens: dict[str, int] = field(default_factory=dict)
    selfcheck: dict[str, Any] = field(default_factory=dict)
    rerank: RerankReference | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Lookups
    # ------------------------------------------------------------------ #

    @property
    def has_variants(self) -> bool:
        return bool(self.variants)

    @property
    def declared_dimension(self) -> int:
        return int(self.info.truncate_dim or self.info.dimension)

    def variant_keys(self) -> list[tuple[str, str]]:
        return [(k.split("/", 1)[0], k.split("/", 1)[1]) for k in self.variants]

    def base(self, config: str, pooling: str) -> npt.NDArray[np.float64] | None:
        """Un-normalised vectors for all probes under ``config`` and ``pooling``.

        Configs that only change long inputs are stored for those probes only;
        the remaining rows come from the default config.
        """
        table = self.variants.get(f"{config}/{pooling}")
        if table is None:
            return None
        ids = self.probes.ids
        if table.probe_ids == ids:
            return table.matrix
        fallback = self.variants.get(f"{DEFAULT}/{pooling}")
        if fallback is None or fallback.probe_ids != ids:
            return None
        if fallback.matrix.shape[1] != table.matrix.shape[1]:
            return None
        out = fallback.matrix.copy()
        for row, pid in enumerate(table.probe_ids):
            out[self.probes.index(pid)] = table.matrix[row]
        return np.asarray(out, dtype=np.float64)

    def affected(self, config: str, pooling: str) -> list[str]:
        """Probe ids for which ``config`` was actually computed."""
        table = self.variants.get(f"{config}/{pooling}")
        return list(table.probe_ids) if table else []

    def declared_expectation(self) -> Expectation:
        return Expectation(
            config=DEFAULT,
            pooling=self.info.pooling,
            normalize=self.info.normalize,
            dimensions=self.info.truncate_dim,
            is_declared=True,
        )

    def expectation(self, overrides: dict[str, str] | None = None) -> Expectation:
        """Declared behaviour with user overrides such as ``pooling=cls`` applied."""
        exp = self.declared_expectation()
        if not overrides:
            return exp
        config, pooling, norm, dims = exp.config, exp.pooling, exp.normalize, exp.dimensions
        for key, raw in overrides.items():
            value = raw.strip()
            if key == "pooling":
                if value not in POOLINGS:
                    raise ReferenceFileError(
                        f"--expect pooling must be one of {', '.join(POOLINGS)}"
                    )
                pooling = value
            elif key == "normalize":
                norm = _parse_bool(key, value)
            elif key in ("dimensions", "dim"):
                dims = int(value)
                if dims <= 0 or dims > self.info.dimension:
                    raise ReferenceFileError(
                        f"--expect dimensions must be between 1 and {self.info.dimension}"
                    )
            elif key == "prompt":
                declared = self.info.default_prompt_name if self.info.default_prompt else None
                wanted = None if value.lower() in ("none", "") else value
                config = DEFAULT if wanted == declared else f"prompt={wanted or 'none'}"
            elif key == "special_tokens":
                config = DEFAULT if _parse_bool(key, value) else "special=off"
            else:
                raise ReferenceFileError(
                    f"unknown --expect key {key!r}; use pooling, normalize, prompt, "
                    "special_tokens or dimensions"
                )
        changed = (config, pooling, norm, dims) != (
            exp.config,
            exp.pooling,
            exp.normalize,
            exp.dimensions,
        )
        if (config, pooling) != (exp.config, exp.pooling) and self.base(config, pooling) is None:
            raise ReferenceFileError(
                f"the reference has no variant {config}/{pooling}; "
                "rebuild it with variants enabled or check the prompt name"
            )
        return Expectation(config, pooling, norm, dims, is_declared=not changed)

    def expected_vectors(self, exp: Expectation) -> npt.NDArray[np.float64]:
        """The vectors the endpoint should return under ``exp``."""
        if exp.is_declared and self.library is not None:
            return self.library
        base = self.base(exp.config, exp.pooling)
        if base is None:
            if self.library is None:
                raise ReferenceFileError("the reference contains no vectors")
            # Declared-only reference: normalisation and dimension overrides still work.
            if (exp.config, exp.pooling) != (DEFAULT, self.info.pooling):
                raise ReferenceFileError(f"the reference has no variant {exp.label()}")
            if self.info.normalize and not exp.normalize:
                raise ReferenceFileError(
                    "cannot derive un-normalised vectors from a reference without variants"
                )
            base = self.library
        if exp.dimensions:
            base = base[:, : exp.dimensions]
        return normalize(base) if exp.normalize else base

    def token_count(self, config: str, probe_id: str) -> int | None:
        counts = self.tokens.get(config, {})
        if probe_id in counts:
            return counts[probe_id]
        return self.tokens.get(DEFAULT, {}).get(probe_id)

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def to_json(self, variant_dtype: str = "float16") -> dict[str, Any]:
        doc: dict[str, Any] = {
            "format": FORMAT,
            "version": FORMAT_VERSION,
            "meta": self.meta,
            "model": self.info.to_json(),
            "probe_set": self.probes.to_json(),
            "input_configs": {k: v.to_json() for k, v in self.input_configs.items()},
            "selfcheck": self.selfcheck,
            "tokens": self.tokens,
            "full_tokens": self.full_tokens,
        }
        if self.library is not None:
            doc["library"] = encode_array(self.library, "float32")
        if self.variants:
            doc["variants"] = {
                key: {"probe_ids": t.probe_ids, **encode_array(t.matrix, variant_dtype)}
                for key, t in self.variants.items()
            }
        if self.rerank is not None:
            rr = self.rerank
            doc["rerank"] = {
                "query_ids": rr.query_ids,
                "document_ids": rr.document_ids,
                "logits": encode_array(rr.logits, "float32"),
                "library": None if rr.library is None else encode_array(rr.library, "float32"),
                "token_counts": rr.token_counts,
            }
        return doc

    def save(self, path: str | Path, variant_dtype: str = "float16") -> None:
        Path(path).write_text(
            json.dumps(self.to_json(variant_dtype), ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def from_json(doc: dict[str, Any]) -> Reference:
        if doc.get("format") != FORMAT:
            raise ReferenceFileError("not an embed-parity reference file")
        if int(doc.get("version", 0)) > FORMAT_VERSION:
            raise ReferenceFileError(
                f"reference format version {doc.get('version')} is newer than this "
                f"tool understands ({FORMAT_VERSION}); upgrade embed-parity"
            )
        try:
            ref = Reference(
                info=ModelInfo.from_json(doc["model"]),
                probes=ProbeSet.from_json(doc["probe_set"]),
                input_configs={
                    k: InputConfig.from_json(v) for k, v in doc.get("input_configs", {}).items()
                },
                selfcheck=dict(doc.get("selfcheck", {})),
                tokens={
                    k: {p: int(n) for p, n in v.items()} for k, v in doc.get("tokens", {}).items()
                },
                full_tokens={p: int(n) for p, n in doc.get("full_tokens", {}).items()},
                meta=dict(doc.get("meta", {})),
            )
            if doc.get("library") is not None:
                ref.library = decode_array(doc["library"])
            for key, obj in doc.get("variants", {}).items():
                ref.variants[key] = VariantTable(list(obj["probe_ids"]), decode_array(obj))
            if doc.get("rerank") is not None:
                rr = doc["rerank"]
                ref.rerank = RerankReference(
                    query_ids=list(rr["query_ids"]),
                    document_ids=list(rr["document_ids"]),
                    logits=decode_array(rr["logits"]),
                    library=None if rr.get("library") is None else decode_array(rr["library"]),
                    token_counts=rr.get("token_counts"),
                )
        except (KeyError, TypeError) as exc:
            raise ReferenceFileError(f"malformed reference file: {exc!r}") from exc
        n = len(ref.probes.probes)
        if ref.library is not None and ref.library.shape[0] != n:
            raise ReferenceFileError("reference vectors do not match the probe list")
        return ref

    @staticmethod
    def load(path: str | Path) -> Reference:
        try:
            doc = json.loads(Path(path).read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ReferenceFileError(f"{path}: not valid JSON ({exc.msg})") from exc
        if not isinstance(doc, dict):
            raise ReferenceFileError(f"{path}: not an embed-parity reference file")
        return Reference.from_json(doc)


def _parse_bool(key: str, value: str) -> bool:
    low = value.lower()
    if low in ("1", "true", "yes", "on"):
        return True
    if low in ("0", "false", "no", "off"):
        return False
    raise ReferenceFileError(f"--expect {key} needs true or false, got {value!r}")


def _meta() -> dict[str, Any]:
    return {
        "tool": "embed-parity",
        "tool_version": __version__,
        "created": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def plan_input_configs(info: ModelInfo) -> dict[str, InputConfig]:
    """The input configurations worth computing for diagnosis."""
    default = info.default_input()
    configs: dict[str, InputConfig] = {DEFAULT: default}

    def variant(**changes: Any) -> InputConfig:
        data = default.to_json()
        data.update(changes)
        return InputConfig.from_json(data)

    declared_prompt = info.default_prompt_name if info.default_prompt else None
    for name, text in info.prompts.items():
        if text and name != declared_prompt:
            configs[f"prompt={name}"] = variant(prompt=text, prompt_name=name)
    if declared_prompt is not None:
        configs["prompt=none"] = variant(prompt="", prompt_name=None)
    configs["special=off"] = variant(add_special_tokens=False)
    other_side = "left" if default.truncation_side == "right" else "right"
    configs[f"trunc={other_side}"] = variant(truncation_side=other_side)
    lengths = {max(default.max_length // 2, 1)}
    if info.position_limit and info.position_limit > default.max_length:
        lengths.add(int(info.position_limit))
    for length in sorted(lengths - {default.max_length}):
        configs[f"maxlen={length}"] = variant(max_length=length)
    return configs


def build_reference(
    backend: EmbeddingBackend,
    probes: ProbeSet | None = None,
    *,
    variants: bool = True,
    batch_size: int = 8,
) -> Reference:
    """Compute a reference for an embedding model."""
    probe_set = probes or builtin_probes()
    info = backend.describe()
    texts, ids = probe_set.texts, probe_set.ids
    ref = Reference(info=info, probes=probe_set, meta=_meta())

    library = backend.encode_library(texts)
    if library is not None:
        ref.library = np.asarray(library, dtype=np.float64)

    configs = plan_input_configs(info) if variants else {DEFAULT: info.default_input()}
    default = configs[DEFAULT]
    full_counts: list[int] = []

    def run(config: InputConfig, subset: Sequence[int]) -> tuple[list[Any], list[int], list[int]]:
        states: list[Any] = []
        counts: list[int] = []
        starts: list[int] = []
        fulls: list[int] = []
        for at in range(0, len(subset), batch_size):
            chunk = [texts[i] for i in subset[at : at + batch_size]]
            enc = backend.encode_tokens(chunk, config)
            states.extend(enc.states)
            counts.extend(enc.token_counts)
            starts.extend(enc.pool_start)
            fulls.extend(enc.full_token_counts)
        if config is default:
            full_counts.extend(fulls)
        return states, counts, starts

    for name, config in configs.items():
        if name == DEFAULT:
            subset = list(range(len(texts)))
        elif name.startswith(("trunc=", "maxlen=")):
            limit = min(config.max_length, default.max_length)
            subset = [i for i, full in enumerate(full_counts) if full > limit]
            if not subset:
                continue
        else:
            subset = list(range(len(texts)))
        states, counts, starts = run(config, subset)
        sub_ids = [ids[i] for i in subset]
        ref.input_configs[name] = config
        ref.tokens[name] = dict(zip(sub_ids, counts, strict=True))
        modes = POOLINGS if variants else (info.pooling,)
        for mode in modes:
            pooled = np.stack(
                [pool(s, mode, st) for s, st in zip(states, starts, strict=True)]
            ).astype(np.float32)
            post = np.asarray(backend.post_pool(pooled), dtype=np.float64)
            ref.variants[f"{name}/{mode}"] = VariantTable(sub_ids, post)
            if name == DEFAULT and mode == info.pooling and info.post_modules and variants:
                ref.variants[f"{DENSE_OFF}/{mode}"] = VariantTable(
                    sub_ids, np.asarray(pooled, dtype=np.float64)
                )
                ref.input_configs[DENSE_OFF] = config

    ref.full_tokens = dict(zip(ids, full_counts, strict=True))
    ref.selfcheck = _selfcheck(ref)
    if ref.library is None:
        # No library output available: the recomputed declared variant is the reference.
        ref.library = ref.expected_vectors(
            Expectation(DEFAULT, info.pooling, info.normalize, info.truncate_dim, False)
        )
    if not variants:
        ref.variants.clear()
    return ref


def _selfcheck(ref: Reference) -> dict[str, Any]:
    """Compare the recomputed declared variant with the library's own output."""
    if ref.library is None:
        return {"status": "skipped", "reason": "backend has no library encode"}
    info = ref.info
    base = ref.base(DEFAULT, info.pooling)
    if base is None:
        return {"status": "skipped", "reason": "no variants"}
    mine = base[:, : info.truncate_dim] if info.truncate_dim else base
    if info.normalize:
        mine = normalize(mine)
    if mine.shape != ref.library.shape:
        return {
            "status": "failed",
            "reason": f"shape {mine.shape} differs from library output {ref.library.shape}",
        }
    cos = row_cosine(mine, ref.library)
    diff = float(np.max(np.abs(mine - ref.library)))
    ok = bool(np.min(cos) > 0.99999 and diff < 1e-3)
    return {
        "status": "ok" if ok else "failed",
        "cosine_min": float(np.min(cos)),
        "max_abs_diff": diff,
        "description": "declared variant recomputed from token states vs library encode()",
    }


def build_rerank_reference(backend: RerankBackend, probes: ProbeSet | None = None) -> Reference:
    """Compute a reference for a cross-encoder."""
    probe_set = probes or builtin_probes()
    if not probe_set.queries or not probe_set.documents:
        raise ReferenceFileError("a rerank reference needs probes tagged 'query' and 'document'")
    info = backend.describe()
    pairs = [
        (probe_set.get(q).text, probe_set.get(d).text)
        for q in probe_set.queries
        for d in probe_set.documents
    ]
    shape = (len(probe_set.queries), len(probe_set.documents))
    logits = np.asarray(backend.logits(pairs), dtype=np.float64).reshape(shape)
    lib = backend.library_scores(pairs)
    library = None if lib is None else np.asarray(lib, dtype=np.float64).reshape(shape)
    ref = Reference(info=info, probes=probe_set, meta=_meta())
    ref.rerank = RerankReference(
        query_ids=list(probe_set.queries),
        document_ids=list(probe_set.documents),
        logits=logits,
        library=library,
    )
    if library is not None:
        from embed_parity.mathx import sigmoid

        as_declared = sigmoid(logits) if info.activation == "sigmoid" else logits
        diff = float(np.max(np.abs(as_declared - library)))
        ref.selfcheck = {
            "status": "ok" if diff < 1e-4 else "failed",
            "max_abs_diff": diff,
            "description": f"{info.activation or 'identity'}(logits) vs library predict()",
        }
    return ref
