"""A deterministic toy encoder written in numpy.

It is not a language model. It exists so that the simulator, the demo and the
test-suite can exercise every code path of the checker without downloading
weights: it has a tokenizer with special tokens and truncation, position
embeddings, one masked self-attention layer, the four usual pooling modes,
optional normalisation, prompt prefixes and Matryoshka-style truncation.

All parameters are derived from integer hashing, so results are identical on
every platform and numpy version.
"""

from __future__ import annotations

import re
import zlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from embed_parity.backends import Encoded, InputConfig, ModelInfo, pool

PAD, CLS, SEP = 0, 1, 2
_VOCAB = 4096
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_PIECE = 5


def _hash_uniform(shape: tuple[int, ...], seed: int) -> npt.NDArray[np.float64]:
    """Uniform values in [-1, 1) from splitmix64; exact integer arithmetic."""
    n = int(np.prod(shape))
    with np.errstate(over="ignore"):
        x = (np.arange(n, dtype=np.uint64) + np.uint64(seed)) * np.uint64(0x9E3779B97F4A7C15)
        x ^= x >> np.uint64(30)
        x *= np.uint64(0xBF58476D1CE4E5B9)
        x ^= x >> np.uint64(27)
        x *= np.uint64(0x94D049BB133111EB)
        x ^= x >> np.uint64(31)
    u = (x >> np.uint64(11)).astype(np.float64) / float(1 << 53)
    return np.asarray((u * 2.0 - 1.0).reshape(shape), dtype=np.float64)


def pieces(text: str) -> list[str]:
    """Lower-case, split into words and punctuation, cut long words into pieces."""
    out: list[str] = []
    for word in _TOKEN_RE.findall(text.lower()):
        parts = [word[i : i + _PIECE] for i in range(0, len(word), _PIECE)]
        out.extend(part if n == 0 else "##" + part for n, part in enumerate(parts))
    return out


def tokenize(text: str) -> list[int]:
    """Token ids for ``text`` (no special tokens)."""
    return [3 + zlib.crc32(p.encode("utf-8")) % (_VOCAB - 3) for p in pieces(text)]


def _content(text: str) -> set[str]:
    """Word-initial pieces of four or more characters: a crude stop-word filter."""
    return {p for p in pieces(text) if len(p) >= 4 and not p.startswith("##")}


@dataclass(frozen=True)
class ToySpec:
    """Declared configuration of a toy model."""

    name: str = "toy-embed"
    hidden: int = 64
    max_seq_length: int = 48
    position_limit: int = 96
    pooling: str = "mean"
    normalize: bool = True
    prompts: dict[str, str] = field(
        default_factory=lambda: {"query": "search_query: ", "document": "search_document: "}
    )
    default_prompt_name: str | None = "document"
    matryoshka_dims: tuple[int, ...] = (64, 32, 16)
    similarity_fn: str = "cosine"
    seed: int = 7


TOY_MODELS: dict[str, ToySpec] = {
    "toy-embed": ToySpec(),
    "toy-embed-cls": ToySpec(name="toy-embed-cls", pooling="cls", default_prompt_name=None),
    "toy-embed-last": ToySpec(
        name="toy-embed-last", pooling="last", normalize=True, default_prompt_name="query"
    ),
    "toy-embed-raw": ToySpec(
        name="toy-embed-raw", normalize=False, similarity_fn="dot", default_prompt_name=None
    ),
}

TOY_RERANKERS: tuple[str, ...] = ("toy-rerank",)


class ToyEncoder:
    """Tokenizer plus a one-layer masked self-attention encoder."""

    def __init__(self, spec: ToySpec | None = None) -> None:
        self.spec = spec or ToySpec()
        h = self.spec.hidden
        s = self.spec.seed * 1_000_003
        self._emb = _hash_uniform((_VOCAB, h), s + 1)
        self._pos = 0.35 * _hash_uniform((self.spec.position_limit, h), s + 2)
        self._w_self = _hash_uniform((h, h), s + 3) / np.sqrt(h) * 1.6
        self._w_ctx = _hash_uniform((h, h), s + 4) / np.sqrt(h) * 2.4
        self._bias = 0.2 * _hash_uniform((h,), s + 5)

    # ------------------------------------------------------------------ #
    # Tokenisation
    # ------------------------------------------------------------------ #

    def build_ids(self, text: str, config: InputConfig) -> tuple[list[int], int]:
        """Token ids after prompt, special tokens and truncation; and the untruncated count."""
        body = tokenize(config.prompt + text)
        specials = 2 if config.add_special_tokens else 0
        full = len(body) + specials
        room = max(config.max_length - specials, 0)
        if len(body) > room:
            body = body[len(body) - room :] if config.truncation_side == "left" else body[:room]
        ids = [CLS, *body, SEP] if config.add_special_tokens else body
        if not ids:  # an empty input still needs one position to pool
            ids = [PAD]
            full = max(full, 1)
        return ids, full

    def count_tokens(self, text: str, config: InputConfig) -> int:
        return len(self.build_ids(text, config)[0])

    # ------------------------------------------------------------------ #
    # Forward pass
    # ------------------------------------------------------------------ #

    def forward(
        self,
        batch: Sequence[Sequence[int]],
        *,
        respect_mask: bool = True,
        uniform_attention: bool = False,
    ) -> list[npt.NDArray[np.float32]]:
        """Encode a batch that is padded to its longest member.

        ``respect_mask=False`` lets padding positions take part in attention,
        which is the classic batch-dependent bug. ``uniform_attention=True``
        replaces attention weights with a plain average.

        Returns one array per sequence. With ``respect_mask=False`` the arrays
        keep their padding rows, because a server with that bug pools them too.
        """
        if not batch:
            return []
        lengths = [len(ids) for ids in batch]
        width = max(lengths)
        if width > self.spec.position_limit:
            raise ValueError(
                f"sequence of {width} tokens exceeds the position limit {self.spec.position_limit}"
            )
        ids = np.full((len(batch), width), PAD, dtype=np.int64)
        mask = np.zeros((len(batch), width), dtype=bool)
        for row, seq in enumerate(batch):
            ids[row, : len(seq)] = np.asarray(seq, dtype=np.int64)
            mask[row, : len(seq)] = True
        x = self._emb[ids] + self._pos[:width][None, :, :]
        effective = np.ones_like(mask) if not respect_mask else mask
        if uniform_attention:
            scores = np.zeros((len(batch), width, width), dtype=np.float64)
        else:
            scores = x @ x.transpose(0, 2, 1) / np.sqrt(self.spec.hidden)
        scores = np.where(effective[:, None, :], scores, -1e30)
        scores -= scores.max(axis=-1, keepdims=True)
        weights = np.exp(scores)
        weights /= weights.sum(axis=-1, keepdims=True)
        ctx = weights @ x
        out = np.tanh(x @ self._w_self + ctx @ self._w_ctx + self._bias)
        result: list[npt.NDArray[np.float32]] = []
        for row, length in enumerate(lengths):
            keep = width if not respect_mask else length
            result.append(np.asarray(out[row, :keep], dtype=np.float32))
        return result

    # ------------------------------------------------------------------ #
    # EmbeddingBackend protocol
    # ------------------------------------------------------------------ #

    def describe(self) -> ModelInfo:
        sp = self.spec
        return ModelInfo(
            model_id=sp.name,
            backend="toy",
            kind="embedding",
            pooling=sp.pooling,
            normalize=sp.normalize,
            prompts=dict(sp.prompts),
            default_prompt_name=sp.default_prompt_name,
            max_seq_length=sp.max_seq_length,
            position_limit=sp.position_limit,
            truncation_side="right",
            hidden_size=sp.hidden,
            dimension=sp.hidden,
            matryoshka_dims=list(sp.matryoshka_dims),
            similarity_fn=sp.similarity_fn,
            modules=["ToyTransformer", f"Pooling({sp.pooling})"]
            + (["Normalize"] if sp.normalize else []),
            library_versions={"numpy": np.__version__},
        )

    def encode_tokens(self, texts: Sequence[str], config: InputConfig) -> Encoded:
        built = [self.build_ids(t, config) for t in texts]
        states = [self.forward([ids])[0] for ids, _ in built]
        return Encoded(
            states=states,
            token_counts=[len(ids) for ids, _ in built],
            full_token_counts=[full for _, full in built],
            pool_start=[0] * len(built),
        )

    def post_pool(self, pooled: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return pooled

    def encode_library(self, texts: Sequence[str]) -> npt.NDArray[np.float32] | None:
        return self.encode(texts)

    # ------------------------------------------------------------------ #
    # "Library" behaviour
    # ------------------------------------------------------------------ #

    def encode(
        self,
        texts: Sequence[str],
        *,
        config: InputConfig | None = None,
        pooling: str | None = None,
        normalize: bool | None = None,
    ) -> npt.NDArray[np.float32]:
        """Sentence embeddings the way the model declares them."""
        cfg = config or self.describe().default_input()
        enc = self.encode_tokens(texts, cfg)
        mode = pooling or self.spec.pooling
        out = np.stack([pool(s, mode) for s in enc.states]).astype(np.float64)
        if self.spec.normalize if normalize is None else normalize:
            out = out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
        return np.asarray(out, dtype=np.float32)


class ToyReranker:
    """A toy cross-encoder: lexical overlap plus a learned-looking term."""

    def __init__(self, name: str = "toy-rerank", max_length: int = 64, seed: int = 11) -> None:
        self.name = name
        self.max_length = max_length
        self._enc = ToyEncoder(ToySpec(name=name, max_seq_length=max_length, seed=seed))
        self._w = _hash_uniform((self._enc.spec.hidden,), seed * 7919 + 13)

    def describe(self) -> ModelInfo:
        return ModelInfo(
            model_id=self.name,
            backend="toy",
            kind="rerank",
            pooling="cls",
            max_seq_length=self.max_length,
            hidden_size=self._enc.spec.hidden,
            dimension=1,
            activation="sigmoid",
            num_labels=1,
            modules=["ToyCrossEncoder"],
            library_versions={"numpy": np.__version__},
        )

    def pair_ids(self, query: str, document: str) -> list[int]:
        q, d = tokenize(query), tokenize(document)
        room = max(self.max_length - 3 - len(q), 1)
        return [CLS, *q, SEP, *d[:room], SEP]

    def count_tokens(self, query: str, document: str) -> int:
        return len(self.pair_ids(query, document))

    def logits(self, pairs: Sequence[tuple[str, str]]) -> npt.NDArray[np.float32]:
        out: list[float] = []
        for query, document in pairs:
            q, d = _content(query), _content(document)
            overlap = len(q & d) / max(len(q), 1)
            states = self._enc.forward([self.pair_ids(query, document)])[0]
            learned = float(states.mean(axis=0).astype(np.float64) @ self._w)
            out.append(9.0 * overlap - 3.0 + 0.8 * learned)
        return np.asarray(out, dtype=np.float32)

    def library_scores(self, pairs: Sequence[tuple[str, str]]) -> npt.NDArray[np.float32] | None:
        z = self.logits(pairs).astype(np.float64)
        return np.asarray(1.0 / (1.0 + np.exp(-z)), dtype=np.float32)


def load_toy(model_id: str) -> ToyEncoder | ToyReranker:
    """Resolve ``toy-embed`` style names (an optional ``toy:`` prefix is accepted)."""
    name = model_id.removeprefix("toy:")
    if name in TOY_MODELS:
        return ToyEncoder(TOY_MODELS[name])
    if name in TOY_RERANKERS:
        return ToyReranker(name)
    known = ", ".join([*TOY_MODELS, *TOY_RERANKERS])
    raise KeyError(f"unknown toy model {model_id!r}; known: {known}")


def is_toy(model_id: str) -> bool:
    name = model_id.removeprefix("toy:")
    return model_id.startswith("toy:") or name in TOY_MODELS or name in TOY_RERANKERS


def spec_overrides(spec: ToySpec, **changes: Any) -> ToySpec:
    """Copy a spec with some fields replaced (used by tests)."""
    from dataclasses import replace

    return replace(spec, **changes)
