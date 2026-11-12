"""The interface a model implementation must offer to build a reference.

Two backends ship with the package: :mod:`embed_parity.toy` (pure numpy, used
by the simulator, the demo and the test-suite) and
:mod:`embed_parity.st_backend` (sentence-transformers, optional extra).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

POOLINGS: tuple[str, ...] = ("cls", "mean", "last", "max")


@dataclass(frozen=True)
class InputConfig:
    """How text is turned into model input.

    ``prompt`` is the literal prefix prepended to the text (empty for none) and
    ``prompt_name`` its declared name, kept only for reporting.
    """

    prompt: str = ""
    prompt_name: str | None = None
    add_special_tokens: bool = True
    max_length: int = 512
    truncation_side: str = "right"

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_json(obj: dict[str, Any]) -> InputConfig:
        return InputConfig(
            prompt=str(obj.get("prompt", "")),
            prompt_name=obj.get("prompt_name"),
            add_special_tokens=bool(obj.get("add_special_tokens", True)),
            max_length=int(obj.get("max_length", 512)),
            truncation_side=str(obj.get("truncation_side", "right")),
        )


@dataclass
class ModelInfo:
    """What the model declares about itself."""

    model_id: str
    backend: str
    kind: str = "embedding"  # or "rerank"
    revision: str | None = None
    pooling: str = "mean"
    normalize: bool = False
    include_prompt: bool = True
    prompts: dict[str, str] = field(default_factory=dict)
    default_prompt_name: str | None = None
    max_seq_length: int = 512
    position_limit: int | None = None
    truncation_side: str = "right"
    hidden_size: int = 0
    dimension: int = 0
    truncate_dim: int | None = None
    matryoshka_dims: list[int] = field(default_factory=list)
    similarity_fn: str = "cosine"
    modules: list[str] = field(default_factory=list)
    post_modules: list[str] = field(default_factory=list)
    activation: str | None = None  # rerank models: "sigmoid" or "identity"
    num_labels: int | None = None
    library_versions: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_json(obj: dict[str, Any]) -> ModelInfo:
        known = set(ModelInfo.__dataclass_fields__)
        return ModelInfo(**{k: v for k, v in obj.items() if k in known})

    @property
    def default_prompt(self) -> str:
        if self.default_prompt_name is None:
            return ""
        return self.prompts.get(self.default_prompt_name, "")

    def default_input(self) -> InputConfig:
        return InputConfig(
            prompt=self.default_prompt,
            prompt_name=self.default_prompt_name if self.default_prompt else None,
            add_special_tokens=True,
            max_length=self.max_seq_length,
            truncation_side=self.truncation_side,
        )


@dataclass
class Encoded:
    """Token-level output for a list of texts.

    ``states[i]`` has shape ``(tokens_i, hidden)`` with padding removed.
    ``pool_start[i]`` is the first token that pooling may use; it is non-zero
    only when the model declares ``include_prompt = false``.
    """

    states: list[npt.NDArray[np.float32]]
    token_counts: list[int]
    full_token_counts: list[int]
    pool_start: list[int]


class EmbeddingBackend(Protocol):
    """A model that can expose token states for variant computation."""

    def describe(self) -> ModelInfo:
        """Declared configuration."""
        ...

    def encode_tokens(self, texts: Sequence[str], config: InputConfig) -> Encoded:
        """Run the transformer and return per-token hidden states."""
        ...

    def post_pool(self, pooled: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Apply the modules that follow pooling, except normalisation."""
        ...

    def encode_library(self, texts: Sequence[str]) -> npt.NDArray[np.float32] | None:
        """What the library itself returns with default arguments, for self-checking."""
        ...


class RerankBackend(Protocol):
    """A cross-encoder."""

    def describe(self) -> ModelInfo:
        """Declared configuration."""
        ...

    def logits(self, pairs: Sequence[tuple[str, str]]) -> npt.NDArray[np.float32]:
        """Raw scores before any activation."""
        ...

    def library_scores(self, pairs: Sequence[tuple[str, str]]) -> npt.NDArray[np.float32] | None:
        """What the library returns by default, for self-checking."""
        ...


def pool(states: npt.NDArray[np.floating[Any]], mode: str, start: int = 0) -> npt.NDArray[Any]:
    """Pool one sequence of token states of shape ``(tokens, hidden)``."""
    if states.ndim != 2 or states.shape[0] == 0:
        raise ValueError(f"cannot pool token states of shape {states.shape}")
    body = states[start:] if 0 < start < states.shape[0] else states
    if mode == "cls":
        return np.asarray(states[0])
    if mode == "mean":
        return np.asarray(body.mean(axis=0))
    if mode == "last":
        return np.asarray(states[-1])
    if mode == "max":
        return np.asarray(body.max(axis=0))
    raise ValueError(f"unknown pooling mode {mode!r}")
