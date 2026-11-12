"""Shared fixtures."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest

from embed_parity.checks import Thresholds
from embed_parity.demo import toy_reference, toy_rerank_reference
from embed_parity.reference import Reference
from embed_parity.report import Report
from embed_parity.runner import CheckConfig, run_check
from embed_parity.simserver import SimServer


@pytest.fixture(scope="session")
def toy_ref() -> Reference:
    return toy_reference("toy-embed")


@pytest.fixture(scope="session")
def rerank_ref() -> Reference:
    return toy_rerank_reference()


@pytest.fixture
def sim() -> Iterator[Callable[..., SimServer]]:
    """Factory for simulated servers that are stopped after the test."""
    servers: list[SimServer] = []

    def start(faults: str | None = None, model: str = "toy-embed") -> SimServer:
        server = SimServer(model, faults).start()
        servers.append(server)
        return server

    yield start
    for server in servers:
        server.stop()


@pytest.fixture
def check(sim: Callable[..., SimServer]) -> Callable[..., Report]:
    """Run a full check against a simulator with the given faults."""

    def run(
        faults: str | None,
        ref: Reference | None,
        *,
        model: str = "toy-embed",
        **config: object,
    ) -> Report:
        server = sim(faults, model)
        cfg = CheckConfig(
            base_url=server.url,
            model=model,
            thresholds=config.pop("thresholds", Thresholds()),  # type: ignore[arg-type]
            **config,  # type: ignore[arg-type]
        )
        return run_check(cfg, ref)

    return run


def status_of(report: Report, check_id: str) -> str:
    for c in report.checks():
        if c.id == check_id:
            return c.status
    raise AssertionError(f"no check {check_id!r} in {[c.id for c in report.checks()]}")


def diagnosis_text(report: Report) -> str:
    section = report.section("diagnosis")
    assert section is not None
    return "\n".join(section.lines)
