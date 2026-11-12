"""Reference backend built on sentence-transformers.

Imported lazily: it needs the ``reference`` extra (``sentence-transformers``
and ``torch``). Everything runs on CPU in float32.

The backend talks to the pieces of a ``SentenceTransformer`` directly - the
tokenizer, the underlying ``AutoModel`` and the modules that follow pooling -
so that pooling, prompts, special tokens and truncation can be varied
independently. The result for the declared configuration is compared with
``model.encode()`` when a reference is built (the *self-check*), which guards
against this decomposition drifting from the library.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from embed_parity.backends import Encoded, InputConfig, ModelInfo


class UnsupportedModelError(RuntimeError):
    """The model's architecture cannot be decomposed for variant computation."""


def _require() -> tuple[Any, Any]:
    try:
        import sentence_transformers
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "computing a reference from a real model needs the 'reference' extra:\n"
            "  pip install 'embed-parity[reference] @ "
            "git+https://github.com/cb-Sigma/embed-parity'\n"
            "(install a CPU-only torch first if you do not want the CUDA wheels:\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cpu)"
        ) from exc
    return sentence_transformers, torch


def _versions() -> dict[str, str]:
    out: dict[str, str] = {"numpy": np.__version__}
    for name in ("sentence_transformers", "transformers", "torch", "tokenizers"):
        try:
            module = __import__(name)
            out[name] = str(getattr(module, "__version__", "unknown"))
        except ImportError:  # pragma: no cover
            continue
    return out


def _pooling_mode(module: Any) -> str:
    """Read the pooling mode from old and new sentence-transformers layouts."""
    mode = getattr(module, "pooling_mode", None)
    if not isinstance(mode, str):
        getter = getattr(module, "get_pooling_mode_str", None)
        mode = getter() if callable(getter) else None
    if not isinstance(mode, str):
        raise UnsupportedModelError("cannot read the pooling mode of the Pooling module")
    aliases = {"lasttoken": "last", "last_token": "last"}
    mode = aliases.get(mode, mode)
    if mode not in ("cls", "mean", "last", "max"):
        raise UnsupportedModelError(
            f"pooling mode {mode!r} is not supported for variant computation"
        )
    return mode


def _module_label(module: Any) -> str:
    name = type(module).__name__
    if name == "Dense":
        linear = getattr(module, "linear", None)
        if linear is not None:
            return f"Dense({linear.in_features}->{linear.out_features})"
    return str(name)


class SentenceTransformerBackend:
    """An :class:`~embed_parity.backends.EmbeddingBackend` for bi-encoders."""

    def __init__(
        self,
        model_id: str,
        *,
        revision: str | None = None,
        trust_remote_code: bool = False,
        matryoshka_dims: Sequence[int] = (),
    ) -> None:
        st, torch = _require()
        self._torch = torch
        self.model_id = model_id
        self.model = st.SentenceTransformer(
            model_id, device="cpu", revision=revision, trust_remote_code=trust_remote_code
        )
        self.model.eval()
        modules = list(self.model)
        names = [type(m).__name__ for m in modules]
        if "Pooling" not in names or not hasattr(modules[0], "auto_model"):
            raise UnsupportedModelError(
                f"{model_id} is not a Transformer -> Pooling pipeline (modules: {names})"
            )
        split = names.index("Pooling")
        if split != 1:
            raise UnsupportedModelError(
                f"{model_id} has modules between the transformer and pooling: {names}"
            )
        self._transformer = modules[0]
        self._pooling = modules[split]
        self._post = [m for m in modules[split + 1 :] if type(m).__name__ != "Normalize"]
        self._normalize = "Normalize" in names[split + 1 :]
        self._names = names
        self._matryoshka = [int(d) for d in matryoshka_dims]
        self._revision = revision

    # ------------------------------------------------------------------ #

    def describe(self) -> ModelInfo:
        model = self.model
        tok = model.tokenizer
        cfg = self._transformer.auto_model.config
        hidden = int(getattr(cfg, "hidden_size", 0) or 0)
        limit = getattr(cfg, "max_position_embeddings", None)
        dim_getter = getattr(model, "get_embedding_dimension", None) or getattr(
            model, "get_sentence_embedding_dimension", None
        )
        truncate_dim = getattr(model, "truncate_dim", None)
        untruncated = getattr(model, "_get_untruncated_embedding_dimension", None)
        if callable(untruncated):
            dimension = untruncated()
        elif callable(dim_getter):
            dimension = dim_getter()
        else:
            dimension = None
        dimension = int(dimension or hidden)
        declared = self._matryoshka or list(getattr(cfg, "matryoshka_dimensions", None) or [])
        revision = self._revision
        card = getattr(model, "model_card_data", None)
        if revision is None and card is not None:
            revision = getattr(card, "base_model_revision", None)
        prompts = {str(k): str(v) for k, v in (getattr(model, "prompts", None) or {}).items()}
        notes: list[str] = []
        if not bool(getattr(self._pooling, "include_prompt", True)):
            notes.append("pooling excludes prompt tokens (include_prompt = false)")
        return ModelInfo(
            model_id=self.model_id,
            backend="sentence-transformers",
            kind="embedding",
            revision=str(revision) if revision else None,
            pooling=_pooling_mode(self._pooling),
            normalize=self._normalize,
            include_prompt=bool(getattr(self._pooling, "include_prompt", True)),
            prompts=prompts,
            default_prompt_name=getattr(model, "default_prompt_name", None),
            max_seq_length=int(model.max_seq_length),
            position_limit=int(limit) if isinstance(limit, int) and limit > 0 else None,
            truncation_side=str(getattr(tok, "truncation_side", "right")),
            hidden_size=hidden,
            dimension=dimension,
            truncate_dim=int(truncate_dim) if truncate_dim else None,
            matryoshka_dims=[int(d) for d in declared],
            similarity_fn=str(getattr(model, "similarity_fn_name", None) or "cosine"),
            modules=self._names,
            post_modules=[_module_label(m) for m in self._post],
            library_versions=_versions(),
            notes=notes,
        )

    def _prompt_tokens(self, prompt: str) -> int:
        """Tokens that pooling must skip when ``include_prompt`` is false."""
        if not prompt or bool(getattr(self._pooling, "include_prompt", True)):
            return 0
        tok = self.model.tokenizer
        ids = tok(prompt, add_special_tokens=True)["input_ids"]
        # Mirror sentence-transformers: a trailing EOS/SEP is not part of the prompt.
        trailing = {tok.eos_token_id, tok.sep_token_id} - {None}
        return len(ids) - 1 if ids and ids[-1] in trailing else len(ids)

    def encode_tokens(self, texts: Sequence[str], config: InputConfig) -> Encoded:
        torch = self._torch
        tok = self.model.tokenizer
        full_texts = [config.prompt + t for t in texts]
        previous = tok.truncation_side
        tok.truncation_side = config.truncation_side
        try:
            batch = tok(
                full_texts,
                padding=True,
                truncation=True,
                max_length=config.max_length,
                add_special_tokens=config.add_special_tokens,
                return_tensors="pt",
            )
            full = tok(
                full_texts,
                padding=False,
                truncation=False,
                add_special_tokens=config.add_special_tokens,
                verbose=False,
            )
        finally:
            tok.truncation_side = previous
        accepted = {"input_ids", "attention_mask", "token_type_ids"}
        inputs = {k: v for k, v in batch.items() if k in accepted}
        with torch.no_grad():
            hidden = self._transformer.auto_model(**inputs).last_hidden_state
        mask = batch["attention_mask"].bool()
        skip = self._prompt_tokens(config.prompt)
        states: list[npt.NDArray[np.float32]] = []
        counts: list[int] = []
        for row in range(hidden.shape[0]):
            kept = hidden[row][mask[row]]
            if kept.shape[0] == 0:
                # Nothing to pool (empty input without special tokens): use one zero state.
                kept = torch.zeros((1, hidden.shape[-1]), dtype=hidden.dtype)
            states.append(kept.to(torch.float32).numpy().copy())
            counts.append(int(mask[row].sum()))
        return Encoded(
            states=states,
            token_counts=counts,
            full_token_counts=[len(ids) for ids in full["input_ids"]],
            pool_start=[skip] * len(states),
        )

    def post_pool(self, pooled: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        if not self._post:
            return pooled
        torch = self._torch
        features: dict[str, Any] = {"sentence_embedding": torch.from_numpy(np.asarray(pooled))}
        with torch.no_grad():
            for module in self._post:
                features = module(features)
        return np.asarray(features["sentence_embedding"].to(torch.float32).numpy())

    def encode_library(self, texts: Sequence[str]) -> npt.NDArray[np.float32] | None:
        out = self.model.encode(list(texts), batch_size=8, convert_to_numpy=True)
        return np.asarray(out, dtype=np.float32)


class CrossEncoderBackend:
    """A :class:`~embed_parity.backends.RerankBackend` for cross-encoders."""

    def __init__(
        self, model_id: str, *, revision: str | None = None, trust_remote_code: bool = False
    ) -> None:
        st, torch = _require()
        self._torch = torch
        self.model_id = model_id
        self._revision = revision
        self.model = st.CrossEncoder(
            model_id, device="cpu", revision=revision, trust_remote_code=trust_remote_code
        )
        num_labels = int(getattr(self.model, "num_labels", 1) or 1)
        if num_labels != 1:
            raise UnsupportedModelError(
                f"{model_id} has {num_labels} labels; only single-score rerankers are supported"
            )

    def _activation(self) -> str:
        fn = getattr(self.model, "activation_fn", None) or getattr(
            self.model, "default_activation_function", None
        )
        name = type(fn).__name__.lower() if fn is not None else "identity"
        return "sigmoid" if "sigmoid" in name else "identity"

    def describe(self) -> ModelInfo:
        max_length = getattr(self.model, "max_length", None)
        if not isinstance(max_length, int):
            max_length = int(getattr(self.model.tokenizer, "model_max_length", 512))
        return ModelInfo(
            model_id=self.model_id,
            backend="sentence-transformers",
            kind="rerank",
            revision=self._revision,
            max_seq_length=min(int(max_length), 10**6),
            dimension=1,
            activation=self._activation(),
            num_labels=1,
            modules=["CrossEncoder"],
            library_versions=_versions(),
        )

    def logits(self, pairs: Sequence[tuple[str, str]]) -> npt.NDArray[np.float32]:
        scores = self.model.predict(
            [list(p) for p in pairs],
            activation_fn=self._torch.nn.Identity(),
            convert_to_numpy=True,
            batch_size=8,
        )
        return np.asarray(scores, dtype=np.float32).reshape(-1)

    def library_scores(self, pairs: Sequence[tuple[str, str]]) -> npt.NDArray[np.float32] | None:
        scores = self.model.predict([list(p) for p in pairs], convert_to_numpy=True, batch_size=8)
        return np.asarray(scores, dtype=np.float32).reshape(-1)
