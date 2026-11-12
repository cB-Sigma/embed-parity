from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from embed_parity import __version__
from embed_parity.cli import main
from embed_parity.simserver import Faults, SimServer

Sim = Callable[..., SimServer]


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as err:
        main(["--version"])
    assert err.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_no_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 2
    assert "COMMAND" in capsys.readouterr().out


def test_reference_inspect_check_roundtrip(
    tmp_path: Path, sim: Sim, capsys: pytest.CaptureFixture[str]
) -> None:
    ref = tmp_path / "ref.json"
    assert main(["reference", "--model", "toy-embed", "--out", str(ref)]) == 0
    out = capsys.readouterr().out
    assert "self-check ok" in out
    assert "declares mean pooling, normalised" in out

    assert main(["inspect", str(ref)]) == 0
    out = capsys.readouterr().out
    assert "default/mean" in out
    assert "search_query" in out

    good = sim()
    report_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"
    code = main(
        [
            "check",
            "--base-url",
            good.url,
            "--model",
            "toy-embed",
            "--reference",
            str(ref),
            "--json",
            str(report_path),
            "--markdown",
            str(md_path),
        ]
    )
    assert code == 0
    assert "RESULT: PASS" in capsys.readouterr().out
    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data["verdict"] == "PASS"
    assert data["counts"]["FAIL"] == 0
    assert {s["id"] for s in data["sections"]} == {
        "parity",
        "ranking",
        "tokens",
        "diagnosis",
        "invariance",
    }
    assert "## embed-parity: ✅ PASS" in md_path.read_text(encoding="utf-8")

    bad = sim("pooling=cls")
    code = main(["check", "--base-url", bad.url, "--model", "toy-embed", "--reference", str(ref)])
    assert code == 1
    out = capsys.readouterr().out
    assert "endpoint matches CLS pooling with normalisation" in out
    assert "RESULT: FAIL" in out

    code = main(
        [
            "check",
            "--base-url",
            bad.url,
            "--reference",
            str(ref),
            "--expect",
            "pooling=cls",
            "--json",
            "-",
        ]
    )
    assert code == 0
    assert json.loads(capsys.readouterr().out)["verdict"] == "PASS"


def test_check_with_model_id_and_no_reference(sim: Sim, capsys: pytest.CaptureFixture[str]) -> None:
    server = sim("no-normalize")
    assert main(["check", "--base-url", server.url, "--model-id", "toy-embed"]) == 1
    assert "without normalisation" in capsys.readouterr().out
    assert main(["check", "--base-url", server.url, "--model", "toy-embed"]) == 0
    assert "no reference given" in capsys.readouterr().out


def test_strict_turns_warnings_into_failures(sim: Sim, capsys: pytest.CaptureFixture[str]) -> None:
    server = sim("usage-no-special")
    args = ["check", "--base-url", server.url, "--model-id", "toy-embed"]
    assert main(args) == 0
    assert main([*args, "--strict"]) == 1
    assert main([*args, "--token-mismatch", "fail"]) == 1
    capsys.readouterr()


def test_thresholds_are_configurable(sim: Sim, capsys: pytest.CaptureFixture[str]) -> None:
    server = sim("jitter=0.006")
    base = [
        "check",
        "--base-url",
        server.url,
        "--model-id",
        "toy-embed",
        "--skip",
        "invariance,diagnosis,ranking",
    ]
    assert main(base) == 1
    loose = [*base, "--min-cosine", "0.99", "--norm-tolerance", "0.05"]
    assert main(loose) == 0
    assert main([*loose, "--max-abs-diff", "1e-9"]) == 1
    capsys.readouterr()


def test_rerank_cli(tmp_path: Path, sim: Sim, capsys: pytest.CaptureFixture[str]) -> None:
    ref = tmp_path / "rr.json"
    assert main(["reference", "--model", "toy-rerank", "--out", str(ref)]) == 0
    server = sim("rerank-raw")
    args = ["check", "--base-url", server.url, "--reference", str(ref)]
    assert main(args) == 1
    assert "endpoint returns raw logits" in capsys.readouterr().out
    assert main([*args, "--expect", "activation=identity"]) == 0
    assert main([*args, "--rerank-api", "tei", "--rerank-path", "/rerank"]) == 1
    capsys.readouterr()


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["check", "--base-url", "http://127.0.0.1:9", "--model-id", "toy-embed"], "cannot reach"),
        (["check", "--base-url", "http://x", "--reference", "/nonexistent.json"], "error"),
        (["check", "--base-url", "http://x", "--extra-body", "{bad"], "not valid JSON"),
        (["check", "--base-url", "http://x", "--extra-body", "[1]"], "must be a JSON object"),
        (["check", "--base-url", "http://x", "--skip", "everything"], "unknown section"),
        (["check", "--base-url", "http://x", "--expect", "pooling"], "KEY=VALUE"),
        (["check", "--base-url", "http://x", "--header", "broken"], "NAME=VALUE"),
        (["reference", "--model", "toy:missing", "--out", "x.json"], "unknown toy model"),
    ],
)
def test_usage_and_endpoint_errors_exit_2(
    argv: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(argv) == 2
    assert message in capsys.readouterr().err


def test_bad_expectation_exits_2(sim: Sim, capsys: pytest.CaptureFixture[str]) -> None:
    server = sim()
    argv = ["check", "--base-url", server.url, "--model-id", "toy-embed", "--expect", "pooling=x"]
    assert main(argv) == 2
    assert "pooling must be" in capsys.readouterr().err


def test_demo_commands(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--demo"]) == 1
    out = capsys.readouterr().out
    assert "injected faults: pooling=cls, no-normalize" in out
    assert "endpoint matches CLS pooling without normalisation" in out
    assert main(["demo", "--fault", "none"]) == 0
    assert "injected faults: none" in capsys.readouterr().out
    assert main(["demo", "--rerank"]) == 1
    assert "endpoint returns raw logits" in capsys.readouterr().out
    assert main(["demo", "--fault", "bogus"]) == 2
    assert "unknown fault" in capsys.readouterr().err


def test_verbose_shows_every_probe(capsys: pytest.CaptureFixture[str]) -> None:
    main(["demo", "--fault", "none", "--skip", "invariance"])
    short = capsys.readouterr().out
    main(["demo", "--fault", "none", "--skip", "invariance", "--verbose"])
    full = capsys.readouterr().out
    assert "6 of 24 probes shown" in short
    assert "l-ledger" in full
    assert full.count("\n") > short.count("\n") + 15


def test_probes_and_fault_listing(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["probes"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 24
    assert json.loads(lines[0])["id"] == "q-tides"
    assert main(["serve-sim", "--list-faults"]) == 0
    out = capsys.readouterr().out
    assert "pooling=<str>" in out
    assert "batch-mask" in out


def test_custom_probes(tmp_path: Path, sim: Sim, capsys: pytest.CaptureFixture[str]) -> None:
    probes = tmp_path / "probes.jsonl"
    probes.write_text(
        "\n".join(
            json.dumps(p)
            for p in [
                {"id": "q", "text": "red apples", "tags": ["query"]},
                {"id": "d1", "text": "apples are red or green", "tags": ["document"]},
                {"id": "d2", "text": "trains run on rails", "tags": ["document"]},
            ]
        ),
        encoding="utf-8",
    )
    ref = tmp_path / "ref.json"
    argv = ["reference", "--model", "toy-embed", "--out", str(ref), "--probes", str(probes)]
    assert main(argv) == 0
    server = sim()
    assert main(["check", "--base-url", server.url, "--reference", str(ref)]) == 0
    assert "3 probes" in capsys.readouterr().out


def test_fault_parsing() -> None:
    f = Faults.parse("pooling=cls, no-normalize,dim=8,jitter=0.5")
    assert (f.pooling, f.no_normalize, f.dim, f.jitter) == ("cls", True, 8, 0.5)
    assert f.active() == ["pooling=cls", "no-normalize", "dim=8", "jitter=0.5"]
    assert Faults.parse(None).active() == []
    assert Faults.parse("no-normalize=false").active() == []
    for bad in ("pooling=avg", "quantize=int4", "pooling", "nonsense"):
        with pytest.raises(ValueError, match=r"fault|pooling|quantize"):
            Faults.parse(bad)
