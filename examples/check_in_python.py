#!/usr/bin/env python3
"""Use embed-parity as a library, for example inside an existing test-suite.

    python examples/check_in_python.py http://localhost:8000 served-model-name minilm.ref.json

Exits non-zero when the endpoint does not match the reference.
"""

from __future__ import annotations

import sys

from embed_parity.checks import Thresholds
from embed_parity.reference import Reference
from embed_parity.report import FAIL, render_text
from embed_parity.runner import CheckConfig, run_check


def main(base_url: str, model: str, reference_path: str) -> int:
    reference = Reference.load(reference_path)
    config = CheckConfig(
        base_url=base_url,
        model=model,
        api="openai",
        thresholds=Thresholds(min_cosine=0.9999),
    )
    report = run_check(config, reference)
    print(render_text(report))

    # Every verdict is available as data.
    failed = [check for check in report.checks() if check.status == FAIL]
    for check in failed:
        print(f"failed: {check.id}: {check.detail}", file=sys.stderr)

    # So is the diagnosis: which variant matched, and how it differs from the declaration.
    diagnosis = report.section("diagnosis")
    if diagnosis is not None and diagnosis.data:
        print("observed:", diagnosis.data["observed"].get("description"))
        print("expected:", diagnosis.data["expected"].get("description"))
    return report.exit_code()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    sys.exit(main(*sys.argv[1:]))
