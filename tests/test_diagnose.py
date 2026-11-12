from __future__ import annotations

import numpy as np

from embed_parity.diagnose import describe_behaviour, describe_config, diagnose
from embed_parity.mathx import normalize
from embed_parity.reference import DEFAULT, Reference


def _ids(ref: Reference) -> list[str]:
    return ref.probes.ids


def test_declared_behaviour_is_recognised(toy_ref: Reference) -> None:
    assert toy_ref.library is not None
    d = diagnose(toy_ref, _ids(toy_ref), toy_ref.library)
    assert d.status == "as-expected"
    assert d.findings == []
    assert d.best is not None
    assert d.best.key == "default/mean"
    assert "which is what the model declares" in d.summary


def test_summary_sentence_shape(toy_ref: Reference) -> None:
    base = toy_ref.base(DEFAULT, "cls")
    assert base is not None
    d = diagnose(toy_ref, _ids(toy_ref), base)
    assert d.status == "variant"
    assert d.summary.startswith(
        "endpoint matches CLS pooling without normalisation (cosine min 1.0"
    )
    assert d.summary.endswith("the model declares mean pooling with normalisation")
    assert d.observed["normalized"] == "no"
    assert len(d.findings) == 2


def test_ties_resolve_to_the_expected_variant(toy_ref: Reference) -> None:
    # Without the long probes, truncation variants are identical to the default.
    assert toy_ref.library is not None
    long_ids = set(toy_ref.affected("maxlen=24", "mean"))
    keep = [i for i, pid in enumerate(_ids(toy_ref)) if pid not in long_ids]
    ids = [_ids(toy_ref)[i] for i in keep]
    d = diagnose(toy_ref, ids, toy_ref.library[keep])
    assert d.status == "as-expected"
    assert d.best is not None
    assert d.best.config == DEFAULT


def test_expectation_changes_the_verdict(toy_ref: Reference) -> None:
    base = toy_ref.base(DEFAULT, "cls")
    assert base is not None
    exp = toy_ref.expectation({"pooling": "cls", "normalize": "false"})
    d = diagnose(toy_ref, _ids(toy_ref), base, exp)
    assert d.status == "as-expected"
    assert "which is what was expected" in d.summary


def test_unrelated_vectors_do_not_match(toy_ref: Reference) -> None:
    rng = np.random.default_rng(0)
    d = diagnose(toy_ref, _ids(toy_ref), normalize(rng.normal(size=(24, 64))))
    assert d.status == "no-match"
    assert "no computed variant reproduces the endpoint" in d.summary
    assert any("likely causes" in f for f in d.findings)


def test_oversized_dimension(toy_ref: Reference) -> None:
    d = diagnose(toy_ref, _ids(toy_ref), np.ones((24, 100)))
    assert d.status == "no-match"
    assert "100-dimensional" in d.summary


def test_nan_vectors(toy_ref: Reference) -> None:
    d = diagnose(toy_ref, _ids(toy_ref), np.full((24, 64), np.nan))
    assert d.status == "no-match"
    assert "NaN" in d.summary


def test_packed_binary(toy_ref: Reference) -> None:
    base = toy_ref.base(DEFAULT, "mean")
    assert base is not None
    packed = np.packbits(base > 0, axis=1).astype(np.float64)
    d = diagnose(toy_ref, _ids(toy_ref), packed)
    assert d.observed["quantization"] == "packed-binary"
    assert d.best is not None
    assert d.best.key == "default/mean"
    assert any("packed binary" in f for f in d.findings)


def test_unknown_scaling_is_reported(toy_ref: Reference) -> None:
    assert toy_ref.library is not None
    d = diagnose(toy_ref, _ids(toy_ref), toy_ref.library * 3.0)
    assert d.status == "variant"
    assert d.observed["normalized"] == "unknown"
    assert any("match neither" in f for f in d.findings)


def test_unavailable_without_variants(toy_ref: Reference) -> None:
    bare = Reference(info=toy_ref.info, probes=toy_ref.probes, library=toy_ref.library)
    assert toy_ref.library is not None
    assert diagnose(bare, _ids(toy_ref), toy_ref.library).status == "unavailable"
    assert diagnose(toy_ref, [], np.zeros((0, 64))).status == "unavailable"


def test_descriptions(toy_ref: Reference) -> None:
    assert describe_config(DEFAULT, toy_ref) == ""
    assert "'document' prompt prefix" in describe_config("prompt=none", toy_ref)
    assert "search_query" in describe_config("prompt=query", toy_ref)
    assert "from the left" in describe_config("trunc=left", toy_ref)
    assert "Dense projection" in describe_config("dense=off", toy_ref)
    text = describe_behaviour(
        toy_ref, pooling="last", config="special=off", normalized="yes", dimension=32
    )
    assert text == (
        "last-token pooling with normalisation, without special tokens, "
        "truncated to 32 of 64 dimensions"
    )


def test_near_match_is_reported_as_approximate(toy_ref: Reference) -> None:
    # Same recipe, slightly different numbers: what quantised weights look like.
    base = toy_ref.base(DEFAULT, "mean")
    assert base is not None
    rng = np.random.default_rng(5)
    noisy = base + rng.normal(scale=0.03, size=base.shape)
    d = diagnose(toy_ref, _ids(toy_ref), noisy)
    assert d.status == "approximate"
    assert d.summary.startswith("endpoint approximately matches mean pooling without normalisation")
    assert any("near match only" in f for f in d.findings)
    assert any("un-normalised vectors" in f for f in d.findings)
