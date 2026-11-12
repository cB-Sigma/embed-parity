"""Every fault the simulator can inject must be caught and named.

Each case starts a real HTTP server with one fault, runs the full check and
asserts on (a) the overall verdict, (b) which check failed and (c) the words
of the diagnosis.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from conftest import diagnosis_text, status_of
from embed_parity.reference import Reference
from embed_parity.report import FAIL, INFO, PASS, SKIP, WARN, Report

Check = Callable[..., Report]


def test_correct_server_passes_everything(check: Check, toy_ref: Reference) -> None:
    report = check(None, toy_ref)
    counts = report.counts()
    assert report.verdict() == PASS
    assert counts[FAIL] == 0
    assert counts[WARN] == 0
    assert "which is what the model declares" in diagnosis_text(report)
    assert status_of(report, "diagnosis/variant") == PASS


# fault spec -> (diagnosis must contain, checks that must FAIL)
VARIANT_FAULTS: list[tuple[str, list[str], list[str]]] = [
    (
        "pooling=cls",
        ["endpoint matches CLS pooling with normalisation", "expected mean pooling"],
        ["parity/cosine", "ranking/agreement", "diagnosis/variant"],
    ),
    (
        "pooling=last",
        ["endpoint matches last-token pooling", "endpoint uses last-token pooling"],
        ["parity/cosine", "diagnosis/variant"],
    ),
    (
        "pooling=max",
        ["endpoint matches max pooling", "endpoint uses max pooling, expected mean pooling"],
        ["parity/cosine", "diagnosis/variant"],
    ),
    (
        "no-normalize",
        [
            "endpoint matches mean pooling without normalisation",
            "the model declares mean pooling with normalisation",
            "endpoint returns un-normalised vectors, expected unit norm",
        ],
        ["parity/norm", "invariance/unit-norm", "diagnosis/variant"],
    ),
    (
        "pooling=cls,no-normalize",
        ["endpoint matches CLS pooling without normalisation", "expected mean pooling"],
        ["parity/cosine", "parity/norm", "diagnosis/variant"],
    ),
    (
        "drop-prompt",
        ["without the 'document' prompt prefix"],
        ["parity/cosine", "diagnosis/variant"],
    ),
    (
        "add-prompt=query",
        ["with the 'query' prompt prefix ('search_query: ')"],
        ["parity/cosine", "diagnosis/variant"],
    ),
    (
        "truncate-left",
        ["truncating long inputs from the left"],
        ["parity/cosine", "diagnosis/variant"],
    ),
    (
        "max-length=24",
        ["truncating at 24 tokens (the model declares 48)"],
        ["parity/cosine", "diagnosis/variant"],
    ),
    (
        "max-length=96",
        ["truncating at 96 tokens (the model declares 48)"],
        ["parity/cosine", "diagnosis/variant"],
    ),
    (
        "no-special-tokens",
        ["without special tokens"],
        ["parity/cosine", "diagnosis/variant"],
    ),
    (
        "quantize=int8",
        ["int8-quantised", "endpoint values are int8, expected float32"],
        ["parity/norm", "diagnosis/variant"],
    ),
    (
        "quantize=binary",
        ["binary-quantised", "endpoint values are binary"],
        ["parity/cosine", "diagnosis/variant"],
    ),
    (
        "dim=32",
        [
            "truncated to 32 of 64 dimensions",
            "normalised at full size and then cut to 32 dimensions without renormalising",
            "endpoint returns 32 dimensions (the first 32 of 64), expected 64",
        ],
        ["parity/dimension", "invariance/dimension", "diagnosis/variant"],
    ),
    (
        "pooling=cls,drop-prompt,dim=16",
        [
            "endpoint uses CLS pooling",
            "without the 'document' prompt prefix",
            "the first 16 of 64",
        ],
        ["parity/dimension", "diagnosis/variant"],
    ),
]


@pytest.mark.parametrize(
    ("faults", "phrases", "failing"), VARIANT_FAULTS, ids=[f[0] for f in VARIANT_FAULTS]
)
def test_diagnosis_names_the_injected_fault(
    check: Check, toy_ref: Reference, faults: str, phrases: list[str], failing: list[str]
) -> None:
    report = check(faults, toy_ref)
    assert report.verdict() == FAIL
    assert report.exit_code() == 1
    text = diagnosis_text(report)
    for phrase in phrases:
        assert phrase in text, f"{phrase!r} not in diagnosis:\n{text}"
    for check_id in failing:
        assert status_of(report, check_id) == FAIL, check_id
    # the diagnosis found an exact match, not merely the nearest variant
    section = report.section("diagnosis")
    assert section is not None
    assert section.data["status"] == "variant"
    assert section.data["best"]["cosine_min"] > 0.9995 or "quantize" in faults


def test_float16_output_is_noted_but_passes(check: Check, toy_ref: Reference) -> None:
    report = check("quantize=float16", toy_ref)
    assert report.verdict() == PASS
    assert report.verdict(strict=True) == FAIL
    assert status_of(report, "diagnosis/variant") == WARN
    assert "float16-quantised" in diagnosis_text(report)


def test_padding_mask_bug_is_caught_by_batch_checks(check: Check, toy_ref: Reference) -> None:
    report = check("batch-mask", toy_ref)
    assert report.verdict() == FAIL
    # single-request parity is fine: only batching reveals the bug
    assert status_of(report, "parity/cosine") == PASS
    assert status_of(report, "diagnosis/variant") == PASS
    assert status_of(report, "invariance/batch/longer") == FAIL
    assert status_of(report, "invariance/batch/mixed-8") == FAIL
    assert status_of(report, "invariance/batch/shorter") == PASS
    assert status_of(report, "invariance/batch/equal-length") == PASS


def test_equal_length_batch_bug_is_caught(check: Check, toy_ref: Reference) -> None:
    report = check("equal-length-batch", toy_ref)
    assert report.verdict() == FAIL
    assert status_of(report, "parity/cosine") == PASS
    assert status_of(report, "invariance/batch/duplicate") == FAIL
    assert status_of(report, "invariance/batch/equal-length") == FAIL
    # padded batches are unaffected, as in the upstream report
    assert status_of(report, "invariance/batch/longer") == PASS


def test_nondeterminism_is_caught_and_not_misattributed(check: Check, toy_ref: Reference) -> None:
    report = check("jitter=0.01", toy_ref)
    assert status_of(report, "invariance/determinism") == FAIL
    # small noise: the right recipe is still recognisable, but only approximately
    assert "endpoint approximately matches mean pooling" in diagnosis_text(report)
    section = report.section("diagnosis")
    assert section is not None
    assert section.data["status"] == "approximate"
    # large noise: nothing matches, and the tool says so instead of guessing
    report = check("jitter=0.2", toy_ref, skip=("invariance",))
    assert "no computed variant reproduces the endpoint" in diagnosis_text(report)
    section = report.section("diagnosis")
    assert section is not None
    assert section.data["status"] == "no-match"


def test_base64_payload_bug_is_caught(check: Check, toy_ref: Reference) -> None:
    report = check("base64-float64", toy_ref)
    assert status_of(report, "invariance/base64") == FAIL
    assert status_of(report, "parity/cosine") == PASS
    failed = [c.id for c in report.checks() if c.status == FAIL]
    assert failed == ["invariance/base64"]


def test_token_count_mismatch_is_explained(check: Check, toy_ref: Reference) -> None:
    report = check("usage-no-special", toy_ref)
    assert report.verdict() == PASS
    assert status_of(report, "tokens/usage") == WARN
    tokens = report.section("tokens")
    assert tokens is not None
    assert "every count is off by -2" in tokens.checks[0].detail

    report = check("drop-prompt", toy_ref)
    tokens = report.section("tokens")
    assert tokens is not None
    assert "without the 'document' prompt prefix" in tokens.checks[0].detail


def test_token_mismatch_can_fail_the_run(check: Check, toy_ref: Reference) -> None:
    from embed_parity.checks import Thresholds

    report = check("usage-no-special", toy_ref, thresholds=Thresholds(token_mismatch="fail"))
    assert status_of(report, "tokens/usage") == FAIL
    report = check("usage-no-special", toy_ref, thresholds=Thresholds(token_mismatch="ignore"))
    assert status_of(report, "tokens/usage") == INFO


def test_rejecting_long_input_is_not_a_failure(check: Check, toy_ref: Reference) -> None:
    report = check("reject-long", toy_ref)
    assert status_of(report, "parity/over-length") == INFO
    assert status_of(report, "parity/cosine") == PASS
    assert status_of(report, "invariance/very-long") == INFO
    assert status_of(report, "invariance/batch/longer") == SKIP
    assert report.verdict() == PASS


def test_expect_override_accepts_a_deliberate_deviation(check: Check, toy_ref: Reference) -> None:
    report = check(
        "pooling=cls,no-normalize", toy_ref, expect={"pooling": "cls", "normalize": "false"}
    )
    assert report.verdict() == PASS
    assert "which is what was expected" in diagnosis_text(report)


def test_matryoshka_dimensions_request(check: Check, toy_ref: Reference) -> None:
    report = check(None, toy_ref, dimensions=32)
    assert report.verdict() == PASS
    assert status_of(report, "parity/dimension") == PASS
    assert "truncated to 32 of 64 dimensions" in diagnosis_text(report)
    # a server that truncates without renormalising fails the same request
    report = check("dim=16", toy_ref, dimensions=32)
    assert status_of(report, "parity/dimension") == FAIL


@pytest.mark.parametrize("model", ["toy-embed-cls", "toy-embed-last", "toy-embed-raw"])
def test_other_declared_configurations(check: Check, model: str) -> None:
    from embed_parity.demo import toy_reference

    ref = toy_reference(model)
    assert check(None, ref, model=model).verdict() == PASS
    wrong = "pooling=mean" if ref.info.pooling != "mean" else "pooling=cls"
    report = check(wrong, ref, model=model)
    assert report.verdict() == FAIL
    assert f"endpoint uses {'mean' if 'mean' in wrong else 'CLS'} pooling" in diagnosis_text(report)


def test_black_box_mode_needs_no_reference(check: Check) -> None:
    report = check("batch-mask", None)
    assert [s.id for s in report.sections] == ["invariance"]
    assert status_of(report, "invariance/batch/longer") == FAIL
    assert status_of(report, "invariance/dimension") == INFO
    assert status_of(report, "invariance/unit-norm") == INFO
    report = check("no-normalize", None, expect={"normalize": "true", "dimensions": "64"})
    assert status_of(report, "invariance/unit-norm") == FAIL
    assert status_of(report, "invariance/dimension") == PASS


def test_reference_without_variants_still_checks_parity(check: Check) -> None:
    from embed_parity.demo import toy_reference

    ref = toy_reference(variants=False)
    report = check("pooling=cls", ref)
    assert status_of(report, "parity/cosine") == FAIL
    assert status_of(report, "diagnosis/variant") == SKIP


def test_skip_sections(check: Check, toy_ref: Reference) -> None:
    report = check(None, toy_ref, skip=("invariance", "tokens"))
    assert [s.id for s in report.sections] == ["parity", "ranking", "diagnosis"]
