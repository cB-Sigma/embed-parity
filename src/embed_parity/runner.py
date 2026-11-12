"""Orchestrates one ``check`` run."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from embed_parity.checks import (
    Thresholds,
    collect,
    diagnosis_section,
    invariance_section,
    parity_section,
    ranking_section,
    rerank_section,
    token_section,
)
from embed_parity.client import EmbeddingClient, RerankClient, Transport
from embed_parity.diagnose import POOLING_NAMES, diagnose
from embed_parity.reference import Reference
from embed_parity.report import FAIL, Report

SECTIONS = ("parity", "ranking", "tokens", "diagnosis", "invariance", "rerank")


@dataclass
class CheckConfig:
    """Everything ``check`` needs to know."""

    base_url: str
    model: str | None = None
    api: str = "openai"
    path: str | None = None
    rerank_api: str = "cohere"
    rerank_path: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    api_key: str | None = None
    timeout: float = 60.0
    extra_body: dict[str, Any] = field(default_factory=dict)
    dimensions: int | None = None
    expect: dict[str, str] = field(default_factory=dict)
    thresholds: Thresholds = field(default_factory=Thresholds)
    skip: tuple[str, ...] = ()
    concurrency: int = 6
    verbose: bool = False


def describe_reference(ref: Reference) -> str:
    """One line describing what the model declares."""
    info = ref.info
    head = f"{info.model_id} ({info.backend}"
    if info.revision:
        head += f", revision {info.revision[:12]}"
    head += ")"
    if info.kind == "rerank":
        return f"{head}: cross-encoder, declares {info.activation or 'identity'} activation"
    bits = [
        f"{POOLING_NAMES.get(info.pooling, info.pooling)} pooling",
        "normalised" if info.normalize else "not normalised",
    ]
    if info.default_prompt:
        bits.append(f"default prompt '{info.default_prompt_name}'")
    bits.append(f"max {info.max_seq_length} tokens")
    bits.append(f"{ref.declared_dimension} dimensions")
    if info.post_modules:
        bits.append("post-pooling " + "+".join(info.post_modules))
    return f"{head}: declares " + ", ".join(bits)


def run_check(config: CheckConfig, ref: Reference | None) -> Report:
    """Run every applicable section against the endpoint."""
    transport = Transport(config.base_url, config.headers, config.api_key, config.timeout)
    thr = config.thresholds
    report = Report(
        endpoint={
            "base_url": config.base_url,
            "api": config.api,
            "model": config.model,
        },
        thresholds=thr.to_json(),
    )
    skip = set(config.skip)

    if ref is not None and ref.info.kind == "rerank":
        report.endpoint["api"] = config.rerank_api
        report.reference = {"summary": describe_reference(ref), "model": ref.info.to_json()}
        rclient = RerankClient(
            transport, config.model, config.rerank_api, config.rerank_path, config.extra_body
        )
        if "rerank" not in skip:
            report.sections.append(
                rerank_section(rclient, ref, thr, expect_activation=config.expect.get("activation"))
            )
        report.requests = rclient.requests_made
        return report

    client = EmbeddingClient(
        transport,
        config.model,
        config.api,
        config.path,
        dict(config.extra_body),
        config.dimensions,
    )
    expected_dim: int | None = None
    expect_normalized: bool | None = None

    if ref is not None:
        overrides = dict(config.expect)
        if config.dimensions and "dimensions" not in overrides:
            overrides["dimensions"] = str(config.dimensions)
        exp = ref.expectation(overrides)
        expected_dim = exp.dimensions or ref.info.dimension
        expect_normalized = exp.normalize
        report.reference = {
            "summary": describe_reference(ref),
            "model": ref.info.to_json(),
            "selfcheck": ref.selfcheck,
            "expectation": {
                "config": exp.config,
                "pooling": exp.pooling,
                "normalize": exp.normalize,
                "dimensions": expected_dim,
                "declared": exp.is_declared,
            },
        }
        outcome = collect(client, ref)
        parity = parity_section(ref, outcome, exp, thr, verbose=config.verbose)
        if ref.selfcheck.get("status") == "failed":
            parity.lines.append(
                "note: when this reference was built, the declared variant recomputed from "
                "token states did not reproduce the library output; diagnosis may be unreliable"
            )
        parity_failed = any(c.status == FAIL for c in parity.checks)
        if "parity" not in skip:
            report.sections.append(parity)
        if "ranking" not in skip:
            report.sections.append(ranking_section(ref, outcome, exp, thr))
        if "tokens" not in skip:
            report.sections.append(token_section(ref, outcome, exp, thr))
        if "diagnosis" not in skip and outcome.vectors is not None:
            diag = diagnose(ref, outcome.probe_ids, outcome.vectors, exp)
            report.sections.append(diagnosis_section(diag, parity_failed))
    else:
        if config.expect.get("dimensions"):
            expected_dim = int(config.expect["dimensions"])
        if "normalize" in config.expect:
            expect_normalized = config.expect["normalize"].lower() in ("1", "true", "yes", "on")

    if "invariance" not in skip:
        report.sections.append(
            invariance_section(
                client,
                thr,
                expected_dim=expected_dim,
                expect_normalized=expect_normalized,
                concurrency=config.concurrency,
            )
        )
    report.requests = client.requests_made
    return report
