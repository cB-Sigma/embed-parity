"""Validation against a real model. Opt-in: it downloads weights.

    pip install -e '.[reference]'
    EMBED_PARITY_REAL_MODEL=1 pytest tests/test_real_model.py

Every variant the reference stores is recomputed here through
sentence-transformers itself (by rebuilding the pipeline with a different
Pooling module, changing ``max_seq_length`` or the tokenizer's truncation
side), so the numpy decomposition in ``st_backend`` is checked against an
independent path through the library.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from embed_parity.mathx import row_cosine
from embed_parity.probes import builtin_probes
from embed_parity.reference import DEFAULT, Reference, build_reference

MODEL = os.environ.get("EMBED_PARITY_REAL_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")

pytestmark = [
    pytest.mark.real_model,
    pytest.mark.skipif(
        os.environ.get("EMBED_PARITY_REAL_MODEL") != "1",
        reason="set EMBED_PARITY_REAL_MODEL=1 to run against a downloaded model",
    ),
]


@pytest.fixture(scope="module")
def backend():  # type: ignore[no-untyped-def]
    pytest.importorskip("sentence_transformers")
    from embed_parity.st_backend import SentenceTransformerBackend

    return SentenceTransformerBackend(MODEL)


@pytest.fixture(scope="module")
def ref(backend) -> Reference:  # type: ignore[no-untyped-def]
    return build_reference(backend)


def _cos_min(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.min(row_cosine(a, b)))


def test_declared_configuration_is_read(ref: Reference) -> None:
    info = ref.info
    assert info.pooling == "mean"
    assert info.normalize is True
    assert info.max_seq_length == 256
    assert info.position_limit == 512
    assert info.dimension == 384
    assert info.truncation_side == "right"
    assert info.similarity_fn == "cosine"
    assert info.revision


def test_selfcheck_reproduces_library_output(ref: Reference) -> None:
    assert ref.selfcheck["status"] == "ok"
    assert ref.selfcheck["cosine_min"] > 0.999999
    assert ref.selfcheck["max_abs_diff"] < 1e-5


@pytest.mark.parametrize("mode", ["cls", "mean", "max", "lasttoken"])
def test_pooling_variants_match_library_pooling(backend, ref: Reference, mode: str) -> None:  # type: ignore[no-untyped-def]
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.models import Pooling

    transformer = backend.model[0]
    pipeline = SentenceTransformer(
        modules=[transformer, Pooling(ref.info.hidden_size, pooling_mode=mode)], device="cpu"
    )
    expected = pipeline.encode(builtin_probes().texts, batch_size=8)
    stored = ref.base(DEFAULT, "last" if mode == "lasttoken" else mode)
    assert stored is not None
    assert _cos_min(stored, expected) > 0.999999
    assert float(np.max(np.abs(stored - expected))) < 1e-4


def test_truncation_variants_match_library(backend, ref: Reference) -> None:  # type: ignore[no-untyped-def]
    model = backend.model
    texts = builtin_probes().texts
    try:
        model.max_seq_length = 128
        shorter = model.encode(texts, batch_size=8, normalize_embeddings=False)
        model.max_seq_length = 512
        longer = model.encode(texts, batch_size=8, normalize_embeddings=False)
        model.max_seq_length = 256
        model.tokenizer.truncation_side = "left"
        left = model.encode(texts, batch_size=8, normalize_embeddings=False)
    finally:
        model.max_seq_length = 256
        model.tokenizer.truncation_side = "right"
    for name, expected in (("maxlen=128", shorter), ("maxlen=512", longer), ("trunc=left", left)):
        stored = ref.base(name, "mean")
        assert stored is not None, name
        assert _cos_min(stored, expected) > 0.999999, name
        # and the variant really differs from the default on the long probe
        default = ref.base(DEFAULT, "mean")
        assert default is not None
        row = ref.probes.index("l-ledger")
        assert float(row_cosine(stored[row : row + 1], default[row : row + 1])[0]) < 0.999


def test_token_counts_match_tokenizer(backend, ref: Reference) -> None:  # type: ignore[no-untyped-def]
    tok = backend.model.tokenizer
    for probe in builtin_probes().probes:
        ids = tok(probe.text, truncation=True, max_length=256)["input_ids"]
        assert ref.token_count(DEFAULT, probe.id) == len(ids)
        bare = tok(probe.text, truncation=True, max_length=256, add_special_tokens=False)
        assert ref.token_count("special=off", probe.id) == len(bare["input_ids"])
    assert ref.full_tokens["l-ledger"] > 512


def test_diagnosis_on_real_vectors(backend, ref: Reference) -> None:  # type: ignore[no-untyped-def]
    from embed_parity.diagnose import diagnose

    model = backend.model
    texts = builtin_probes().texts
    raw = model.encode(texts, batch_size=8, normalize_embeddings=False)
    token_level = model.encode(texts, batch_size=8, output_value="token_embeddings")
    cls = np.stack([np.asarray(t)[0] for t in token_level])
    ids = builtin_probes().ids

    d = diagnose(ref, ids, np.asarray(cls, dtype=np.float64))
    assert d.summary.startswith("endpoint matches CLS pooling without normalisation")
    d = diagnose(ref, ids, np.asarray(raw, dtype=np.float64)[:, :128])
    assert "truncated to 128 of 384 dimensions" in d.summary
    d = diagnose(ref, ids, np.asarray(raw, dtype=np.float64).astype(np.float16).astype(np.float64))
    assert d.observed["quantization"] == "float16"
    assert d.best is not None
    assert d.best.key == "default/mean"
