from __future__ import annotations

from collections.abc import Callable

import numpy as np

from conftest import status_of
from embed_parity.reference import Reference
from embed_parity.report import FAIL, PASS, Report
from embed_parity.runner import CheckConfig, run_check
from embed_parity.simserver import SimServer

Sim = Callable[..., SimServer]


def _run(sim: Sim, ref: Reference, faults: str | None, **kw: object) -> Report:
    server = sim(faults)
    return run_check(CheckConfig(base_url=server.url, model="toy-rerank", **kw), ref)  # type: ignore[arg-type]


def test_rerank_reference(rerank_ref: Reference) -> None:
    rr = rerank_ref.rerank
    assert rr is not None
    assert rr.logits.shape == (4, 8)
    assert rerank_ref.info.kind == "rerank"
    assert rerank_ref.selfcheck["status"] == "ok"
    # the toy reranker ranks the matching document first for every query
    assert np.argmax(rr.logits, axis=1).tolist() == [0, 2, 4, 6]


def test_rerank_roundtrip(tmp_path, rerank_ref: Reference) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "rr.json"
    rerank_ref.save(path)
    loaded = Reference.load(path)
    assert loaded.rerank is not None
    assert rerank_ref.rerank is not None
    assert np.allclose(loaded.rerank.logits, rerank_ref.rerank.logits, atol=1e-6)
    assert loaded.info.activation == "sigmoid"


def test_correct_reranker_passes(sim: Sim, rerank_ref: Reference) -> None:
    report = _run(sim, rerank_ref, None)
    assert report.verdict() == PASS
    assert [s.id for s in report.sections] == ["rerank"]
    assert "sigmoid probabilities" in report.sections[0].checks[0].detail


def test_tei_rerank_shape(sim: Sim, rerank_ref: Reference) -> None:
    assert _run(sim, rerank_ref, None, rerank_api="tei").verdict() == PASS
    # TEI's raw_scores switch turns the same server into a logits endpoint
    report = _run(sim, rerank_ref, None, rerank_api="tei", extra_body={"raw_scores": True})
    assert status_of(report, "rerank/scores") == FAIL


def test_raw_logits_are_detected(sim: Sim, rerank_ref: Reference) -> None:
    report = _run(sim, rerank_ref, "rerank-raw")
    assert report.verdict() == FAIL
    assert status_of(report, "rerank/scores") == FAIL
    assert status_of(report, "rerank/ordering") == PASS  # ordering survives, scores do not
    assert any(
        "endpoint returns raw logits" in line and "the model declares sigmoid" in line
        for line in report.sections[0].lines
    )


def test_expected_activation_override(sim: Sim, rerank_ref: Reference) -> None:
    report = _run(sim, rerank_ref, "rerank-raw", expect={"activation": "identity"})
    assert report.verdict() == PASS


def test_batch_dependent_scores_are_caught(sim: Sim, rerank_ref: Reference) -> None:
    report = _run(sim, rerank_ref, "rerank-batch")
    assert status_of(report, "rerank/scores") == FAIL
    assert status_of(report, "rerank/batch") == FAIL
    assert any("match neither" in line for line in report.sections[0].lines)
