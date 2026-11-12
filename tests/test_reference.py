from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from embed_parity.backends import InputConfig, ModelInfo
from embed_parity.demo import toy_reference
from embed_parity.mathx import norms, row_cosine
from embed_parity.probes import Probe, ProbeSet, builtin_probes, load_probes
from embed_parity.reference import (
    DEFAULT,
    Reference,
    ReferenceFileError,
    build_reference,
    plan_input_configs,
)
from embed_parity.toy import TOY_MODELS, ToyEncoder


def test_builtin_probes_are_consistent() -> None:
    probes = builtin_probes()
    assert len(probes.probes) == 24
    assert len(probes.queries) == 4
    assert len(probes.documents) == 8
    assert probes.get("l-ledger").text.count(" ") > 600
    assert all(p.text == p.text.strip() for p in probes.probes)
    assert ProbeSet.from_json(probes.to_json()) == probes


def test_probe_set_validation() -> None:
    with pytest.raises(ValueError, match="unique"):
        ProbeSet(probes=(Probe("a", "x"), Probe("a", "y")))
    with pytest.raises(ValueError, match="unknown probe ids"):
        ProbeSet(probes=(Probe("a", "x"),), queries=("b",))


def test_load_probes(tmp_path: Path) -> None:
    path = tmp_path / "p.jsonl"
    path.write_text(
        '{"id": "q1", "text": "hi", "tags": ["query"]}\n\n'
        '{"text": "a doc", "tags": ["document"]}\n',
        encoding="utf-8",
    )
    probes = load_probes(path)
    assert probes.ids == ["q1", "p3"]
    assert probes.queries == ("q1",)
    assert probes.documents == ("p3",)
    path.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_probes(path)
    path.write_text('{"id": "x"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="missing 'text'"):
        load_probes(path)
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no probes"):
        load_probes(path)


def test_planned_input_configs(toy_ref: Reference) -> None:
    configs = plan_input_configs(toy_ref.info)
    assert set(configs) == {
        DEFAULT,
        "prompt=query",
        "prompt=none",
        "special=off",
        "trunc=left",
        "maxlen=24",
        "maxlen=96",
    }
    assert configs[DEFAULT].prompt == "search_document: "
    assert configs["prompt=none"].prompt == ""
    assert configs["trunc=left"].truncation_side == "left"


def test_reference_contents(toy_ref: Reference) -> None:
    assert toy_ref.library is not None
    assert toy_ref.library.shape == (24, 64)
    assert toy_ref.selfcheck["status"] == "ok"
    assert toy_ref.selfcheck["cosine_min"] > 0.999999
    # truncation variants are stored only for probes that are actually too long
    long_ids = toy_ref.affected("trunc=left", "mean")
    assert "l-ledger" in long_ids
    assert "s-word" not in long_ids
    full = toy_ref.base("trunc=left", "mean")
    default = toy_ref.base(DEFAULT, "mean")
    assert full is not None
    assert default is not None
    row = toy_ref.probes.index("s-word")
    assert np.array_equal(full[row], default[row])
    row = toy_ref.probes.index("l-ledger")
    assert not np.allclose(full[row], default[row])
    assert toy_ref.token_count(DEFAULT, "l-ledger") == 48
    assert toy_ref.token_count("maxlen=24", "l-ledger") == 24
    assert toy_ref.token_count("maxlen=24", "s-word") == toy_ref.token_count(DEFAULT, "s-word")
    assert toy_ref.full_tokens["l-ledger"] > 500


def test_variants_match_direct_encoding(toy_ref: Reference) -> None:
    enc = ToyEncoder(TOY_MODELS["toy-embed"])
    texts = toy_ref.probes.texts
    for pooling in ("cls", "mean", "last", "max"):
        direct = enc.encode(texts, pooling=pooling, normalize=False)
        stored = toy_ref.base(DEFAULT, pooling)
        assert stored is not None
        assert np.allclose(stored, direct, atol=1e-6)
    no_prompt = enc.encode(texts, config=InputConfig(max_length=48), normalize=False)
    stored = toy_ref.base("prompt=none", "mean")
    assert stored is not None
    assert np.allclose(stored, no_prompt, atol=1e-6)


def test_save_load_roundtrip(tmp_path: Path, toy_ref: Reference) -> None:
    path = tmp_path / "ref.json"
    toy_ref.save(path)
    assert path.stat().st_size < 400_000
    loaded = Reference.load(path)
    assert loaded.info == toy_ref.info
    assert loaded.probes == toy_ref.probes
    assert loaded.tokens == toy_ref.tokens
    assert loaded.library is not None
    assert toy_ref.library is not None
    assert np.allclose(loaded.library, toy_ref.library, atol=1e-7)
    for key in toy_ref.variants:
        a, b = loaded.variants[key].matrix, toy_ref.variants[key].matrix
        # float16 storage: fine for cosine, which is all variants are used for
        assert float(row_cosine(a, b).min()) > 0.999999
    assert loaded.input_configs == toy_ref.input_configs


def test_load_rejects_bad_files(tmp_path: Path, toy_ref: Reference) -> None:
    path = tmp_path / "x.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ReferenceFileError, match="not an embed-parity reference"):
        Reference.load(path)
    path.write_text("{oops", encoding="utf-8")
    with pytest.raises(ReferenceFileError, match="not valid JSON"):
        Reference.load(path)
    doc = toy_ref.to_json()
    path.write_text(json.dumps({**doc, "version": 99}), encoding="utf-8")
    with pytest.raises(ReferenceFileError, match="newer"):
        Reference.load(path)
    broken = {k: v for k, v in doc.items() if k != "model"}
    path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ReferenceFileError, match="malformed"):
        Reference.load(path)
    doc["probe_set"]["probes"] = doc["probe_set"]["probes"][:3]
    doc["probe_set"]["queries"] = []
    doc["probe_set"]["documents"] = []
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ReferenceFileError, match="do not match the probe list"):
        Reference.load(path)


def test_expectation_overrides(toy_ref: Reference) -> None:
    declared = toy_ref.expectation()
    assert declared.is_declared
    assert (declared.config, declared.pooling, declared.normalize) == (DEFAULT, "mean", True)
    assert toy_ref.expected_vectors(declared) is toy_ref.library

    exp = toy_ref.expectation({"pooling": "cls", "normalize": "false"})
    assert not exp.is_declared
    vectors = toy_ref.expected_vectors(exp)
    assert not np.allclose(norms(vectors), 1.0, atol=1e-2)

    exp = toy_ref.expectation({"prompt": "query"})
    assert exp.config == "prompt=query"
    assert toy_ref.expectation({"prompt": "document"}).config == DEFAULT
    assert toy_ref.expectation({"prompt": "none"}).config == "prompt=none"
    assert toy_ref.expectation({"special_tokens": "off"}).config == "special=off"

    exp = toy_ref.expectation({"dimensions": "16"})
    out = toy_ref.expected_vectors(exp)
    assert out.shape == (24, 16)
    assert np.allclose(norms(out), 1.0)

    # an override that restates the declaration is still "declared"
    assert toy_ref.expectation({"pooling": "mean", "normalize": "true"}).is_declared


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"pooling": "median"}, "pooling must be"),
        ({"normalize": "maybe"}, "true or false"),
        ({"dimensions": "9999"}, "between 1 and 64"),
        ({"prompt": "nonexistent"}, "no variant"),
        ({"colour": "blue"}, "unknown --expect key"),
    ],
)
def test_expectation_errors(toy_ref: Reference, overrides: dict[str, str], message: str) -> None:
    with pytest.raises(ReferenceFileError, match=message):
        toy_ref.expectation(overrides)


def test_reference_without_variants() -> None:
    ref = toy_reference("toy-embed", variants=False)
    assert not ref.has_variants
    assert ref.library is not None
    assert ref.selfcheck["status"] == "ok"
    assert list(ref.tokens) == [DEFAULT]
    exp = ref.expectation({"dimensions": "8"})
    assert ref.expected_vectors(exp).shape == (24, 8)
    with pytest.raises(ReferenceFileError, match="no variant"):
        ref.expectation({"pooling": "cls"})
    with pytest.raises(ReferenceFileError, match="un-normalised"):
        ref.expected_vectors(ref.expectation({"normalize": "false"}))


class DenseBackend:
    """A toy encoder followed by a fixed linear projection, like a Dense module."""

    def __init__(self) -> None:
        self.enc = ToyEncoder(TOY_MODELS["toy-embed-cls"])
        rng = np.random.default_rng(3)
        self.weight = rng.normal(size=(64, 32)).astype(np.float32)

    def describe(self) -> ModelInfo:
        info = self.enc.describe()
        info.dimension = 32
        info.post_modules = ["Dense(64->32)"]
        return info

    def encode_tokens(self, texts, config):  # type: ignore[no-untyped-def]
        return self.enc.encode_tokens(texts, config)

    def post_pool(self, pooled):  # type: ignore[no-untyped-def]
        return pooled @ self.weight

    def encode_library(self, texts):  # type: ignore[no-untyped-def]
        out = self.enc.encode(texts, normalize=False) @ self.weight
        return out / np.linalg.norm(out, axis=1, keepdims=True)


def test_dense_projection_is_applied_and_recorded() -> None:
    ref = build_reference(DenseBackend())
    assert ref.selfcheck["status"] == "ok"
    assert ref.library is not None
    assert ref.library.shape == (24, 32)
    pre = ref.base("dense=off", "cls")
    assert pre is not None
    assert pre.shape == (24, 64)


class DriftingBackend(DenseBackend):
    """Library output that the decomposition cannot reproduce."""

    def encode_library(self, texts):  # type: ignore[no-untyped-def]
        return np.roll(super().encode_library(texts), 1, axis=1)


def test_selfcheck_flags_decomposition_drift() -> None:
    ref = build_reference(DriftingBackend())
    assert ref.selfcheck["status"] == "failed"
