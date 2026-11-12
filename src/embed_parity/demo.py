"""``embed-parity demo``: run a real check against the bundled simulator."""

from __future__ import annotations

from embed_parity.checks import Thresholds
from embed_parity.reference import Reference, build_reference, build_rerank_reference
from embed_parity.report import Report
from embed_parity.runner import CheckConfig, run_check
from embed_parity.simserver import Faults, SimServer
from embed_parity.toy import TOY_MODELS, ToyEncoder, ToyReranker

DEFAULT_FAULTS = "pooling=cls,no-normalize"


def toy_reference(model: str = "toy-embed", *, variants: bool = True) -> Reference:
    """Reference for a toy embedding model, computed in-process."""
    return build_reference(ToyEncoder(TOY_MODELS[model]), variants=variants)


def toy_rerank_reference() -> Reference:
    return build_rerank_reference(ToyReranker())


def run_demo(
    faults: str | None = DEFAULT_FAULTS,
    *,
    model: str = "toy-embed",
    rerank: bool = False,
    api: str = "openai",
    thresholds: Thresholds | None = None,
    verbose: bool = False,
    skip: tuple[str, ...] = (),
) -> tuple[Report, list[str]]:
    """Start the simulator, check it, and return the report and the active faults.

    Everything is real: an HTTP server is started on a free local port and the
    same code path as ``embed-parity check`` runs against it. Only the model
    is a toy.
    """
    parsed = Faults.parse(faults)
    ref = toy_rerank_reference() if rerank else toy_reference(model)
    with SimServer(model, parsed) as server:
        config = CheckConfig(
            base_url=server.url,
            model=ref.info.model_id,
            api=api,
            rerank_api="tei" if api == "tei" else "cohere",
            thresholds=thresholds or Thresholds(),
            verbose=verbose,
            skip=skip,
        )
        report = run_check(config, ref)
    # The port is random; keep the printed report stable.
    report.endpoint["base_url"] = "http://127.0.0.1:<port> (simulator)"
    return report, parsed.active()
