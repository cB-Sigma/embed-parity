"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from embed_parity import __version__
from embed_parity.checks import Thresholds
from embed_parity.client import EMBED_APIS, RERANK_APIS, EndpointError
from embed_parity.probes import ProbeSet, builtin_probes, load_probes
from embed_parity.reference import (
    Reference,
    ReferenceFileError,
    build_reference,
    build_rerank_reference,
)
from embed_parity.report import EXIT_ERROR, EXIT_OK, Report, render_markdown, render_text
from embed_parity.runner import SECTIONS, CheckConfig, describe_reference, run_check

API_KEY_ENV = "EMBED_PARITY_API_KEY"


class UsageError(Exception):
    """Bad command-line input."""


def _key_values(items: Sequence[str] | None, what: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items or ():
        for part in item.split(","):
            if not part.strip():
                continue
            key, sep, value = part.partition("=")
            if not sep or not key.strip():
                raise UsageError(f"{what} expects KEY=VALUE, got {part!r}")
            out[key.strip()] = value.strip()
    return out


def _headers(items: Sequence[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items or ():
        key, sep, value = item.partition("=")
        if not sep:
            key, sep, value = item.partition(":")
        if not sep or not key.strip():
            raise UsageError(f"--header expects NAME=VALUE, got {item!r}")
        out[key.strip()] = value.strip()
    return out


def _add_threshold_args(parser: argparse.ArgumentParser) -> None:
    d = Thresholds()
    g = parser.add_argument_group("thresholds")
    g.add_argument(
        "--min-cosine",
        type=float,
        default=d.min_cosine,
        metavar="X",
        help=f"minimum per-probe cosine with the reference (default {d.min_cosine})",
    )
    g.add_argument(
        "--norm-tolerance",
        type=float,
        default=d.norm_tolerance,
        metavar="X",
        help=f"allowed deviation of the norm ratio from 1 (default {d.norm_tolerance})",
    )
    g.add_argument(
        "--max-abs-diff",
        type=float,
        default=None,
        metavar="X",
        help="fail if any component differs by more than X (default: report only)",
    )
    g.add_argument(
        "--min-rank-correlation",
        type=float,
        default=d.min_rank_correlation,
        metavar="X",
        help=f"minimum mean Spearman correlation of retrieval scores "
        f"(default {d.min_rank_correlation})",
    )
    g.add_argument(
        "--invariance-min-cosine",
        type=float,
        default=d.invariance_min_cosine,
        metavar="X",
        help=f"minimum cosine between two encodings of the same text "
        f"(default {d.invariance_min_cosine})",
    )
    g.add_argument(
        "--token-mismatch",
        choices=("warn", "fail", "ignore"),
        default=d.token_mismatch,
        help="how to treat usage token counts that differ (default warn)",
    )
    g.add_argument(
        "--rerank-max-diff",
        type=float,
        default=d.rerank_max_diff,
        metavar="X",
        help=f"maximum absolute rerank score difference (default {d.rerank_max_diff})",
    )
    g.add_argument("--strict", action="store_true", help="treat warnings as failures")


def _thresholds(args: argparse.Namespace) -> Thresholds:
    return Thresholds(
        min_cosine=args.min_cosine,
        norm_tolerance=args.norm_tolerance,
        max_abs_diff=args.max_abs_diff,
        min_rank_correlation=args.min_rank_correlation,
        invariance_min_cosine=args.invariance_min_cosine,
        token_mismatch=args.token_mismatch,
        rerank_max_diff=args.rerank_max_diff,
    )


def _add_output_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("output")
    g.add_argument(
        "--json", metavar="PATH", help="also write the full report as JSON ('-' for stdout)"
    )
    g.add_argument("--markdown", metavar="PATH", help="also write a Markdown summary")
    g.add_argument("-v", "--verbose", action="store_true", help="show every probe")
    g.add_argument("--no-color", action="store_true", help="disable ANSI colours")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="embed-parity",
        description="Check that an embedding or rerank endpoint returns what the reference "
        "library returns, and explain the difference when it does not.",
    )
    parser.add_argument("--version", action="version", version=f"embed-parity {__version__}")
    parser.add_argument(
        "--demo", action="store_true", help="shortcut for the 'demo' command with default faults"
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # reference ---------------------------------------------------------------
    p = sub.add_parser(
        "reference",
        help="compute a reference file with sentence-transformers on CPU",
        description="Compute reference embeddings (or rerank scores) and the variants needed "
        "for diagnosis. Needs the 'reference' extra unless the model is a toy model.",
    )
    p.add_argument("--model", required=True, help="Hugging Face model id or local path")
    p.add_argument("--out", required=True, metavar="PATH", help="where to write the reference")
    p.add_argument(
        "--kind",
        choices=("embedding", "rerank"),
        default="embedding",
        help="'rerank' loads the model as a cross-encoder (default embedding)",
    )
    p.add_argument("--revision", help="model revision (commit, tag or branch) to pin")
    p.add_argument(
        "--probes", metavar="JSONL", help="use your own probes instead of the built-in set"
    )
    p.add_argument(
        "--no-variants",
        action="store_true",
        help="store only the declared output: smaller file, no diagnosis",
    )
    p.add_argument(
        "--variant-dtype",
        choices=("float16", "float32"),
        default="float16",
        help="precision of stored variants (default float16)",
    )
    p.add_argument(
        "--matryoshka-dims", metavar="N,N", help="dimensions the model supports truncating to"
    )
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="allow model repositories that ship custom code",
    )

    # check -------------------------------------------------------------------
    p = sub.add_parser(
        "check",
        help="check an endpoint against a reference",
        description="Send the probe set to an endpoint and compare with a reference. Without "
        "--reference or --model-id only the black-box invariance checks run.",
    )
    p.add_argument("--base-url", required=True, help="server root, e.g. http://localhost:8000")
    p.add_argument("--model", help="model name to send in requests")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--reference", metavar="PATH", help="reference file made by 'reference'")
    src.add_argument(
        "--model-id",
        metavar="ID",
        help="compute the reference now from this model (needs the 'reference' extra)",
    )
    p.add_argument(
        "--kind",
        choices=("embedding", "rerank"),
        default="embedding",
        help="with --model-id: what kind of model it is",
    )
    p.add_argument(
        "--api",
        choices=EMBED_APIS,
        default="openai",
        help="embeddings API shape (default openai: POST /v1/embeddings)",
    )
    p.add_argument("--path", help="override the embeddings route, e.g. /embeddings")
    p.add_argument(
        "--rerank-api",
        choices=RERANK_APIS,
        default="cohere",
        help="rerank API shape (default cohere: POST /v1/rerank with 'documents')",
    )
    p.add_argument("--rerank-path", help="override the rerank route, e.g. /rerank")
    p.add_argument("--api-key", help=f"bearer token (or set {API_KEY_ENV})")
    p.add_argument("--header", action="append", metavar="NAME=VALUE", help="extra HTTP header")
    p.add_argument("--timeout", type=float, default=60.0, help="request timeout in seconds")
    p.add_argument(
        "--extra-body",
        metavar="JSON",
        help="JSON object merged into every request, e.g. '{\"truncate\": true}'",
    )
    p.add_argument(
        "--dimensions",
        type=int,
        metavar="N",
        help="send 'dimensions': N and expect the reference truncated to N",
    )
    p.add_argument(
        "--expect",
        action="append",
        metavar="KEY=VALUE",
        help="override what the model declares: pooling=cls|mean|last|max, "
        "normalize=true|false, prompt=NAME|none, special_tokens=on|off, "
        "dimensions=N, activation=sigmoid|identity",
    )
    p.add_argument(
        "--skip",
        metavar="SECTIONS",
        default="",
        help=f"comma-separated sections to skip: {', '.join(SECTIONS)}",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=6,
        help="simultaneous requests in the concurrency check (0 disables; default 6)",
    )
    _add_threshold_args(p)
    _add_output_args(p)

    # demo --------------------------------------------------------------------
    p = sub.add_parser(
        "demo",
        help="run a check against the bundled simulated server",
        description="Start the bundled simulator with injected faults and check it. "
        "Run 'embed-parity serve-sim --list-faults' for the fault names.",
    )
    p.add_argument(
        "--fault",
        default=None,
        metavar="SPEC",
        help="comma-separated faults (default: pooling=cls,no-normalize; "
        "use 'none' for a correct server)",
    )
    p.add_argument("--model", default="toy-embed", help="toy model to serve (default toy-embed)")
    p.add_argument("--rerank", action="store_true", help="demonstrate the rerank checks")
    p.add_argument("--api", choices=EMBED_APIS, default="openai", help="API shape to exercise")
    p.add_argument("--skip", metavar="SECTIONS", default="", help="sections to skip")
    _add_threshold_args(p)
    _add_output_args(p)

    # serve-sim ---------------------------------------------------------------
    p = sub.add_parser(
        "serve-sim",
        help="run the simulated server (for trying the tool without a GPU)",
    )
    p.add_argument("--model", default="toy-embed", help="toy model to serve")
    p.add_argument("--fault", default=None, metavar="SPEC", help="comma-separated faults")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--list-faults", action="store_true", help="print the fault names and exit")

    # inspect -----------------------------------------------------------------
    p = sub.add_parser("inspect", help="show what a reference file contains")
    p.add_argument("reference", metavar="PATH")

    # probes ------------------------------------------------------------------
    sub.add_parser("probes", help="print the built-in probe set as JSON Lines")
    return parser


def _load_probes(path: str | None) -> ProbeSet:
    return load_probes(path) if path else builtin_probes()


def make_reference(
    model: str,
    kind: str,
    *,
    probes: ProbeSet,
    variants: bool = True,
    revision: str | None = None,
    trust_remote_code: bool = False,
    matryoshka_dims: Sequence[int] = (),
) -> Reference:
    """Build a reference from a toy model or through sentence-transformers."""
    from embed_parity.toy import ToyReranker, is_toy, load_toy

    if is_toy(model):
        toy = load_toy(model)
        if isinstance(toy, ToyReranker):
            return build_rerank_reference(toy, probes)
        return build_reference(toy, probes, variants=variants)

    from embed_parity.st_backend import (
        CrossEncoderBackend,
        SentenceTransformerBackend,
        UnsupportedModelError,
    )

    if kind == "rerank":
        return build_rerank_reference(
            CrossEncoderBackend(model, revision=revision, trust_remote_code=trust_remote_code),
            probes,
        )
    try:
        backend = SentenceTransformerBackend(
            model,
            revision=revision,
            trust_remote_code=trust_remote_code,
            matryoshka_dims=matryoshka_dims,
        )
    except UnsupportedModelError as exc:
        raise UsageError(
            f"{exc}\nembed-parity can only decompose Transformer -> Pooling models for now."
        ) from exc
    return build_reference(backend, probes, variants=variants)


def _cmd_reference(args: argparse.Namespace) -> int:
    dims = [int(x) for x in (args.matryoshka_dims or "").split(",") if x.strip()]
    ref = make_reference(
        args.model,
        args.kind,
        probes=_load_probes(args.probes),
        variants=not args.no_variants,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
        matryoshka_dims=dims,
    )
    ref.save(args.out, args.variant_dtype)
    size = Path(args.out).stat().st_size
    print(f"wrote {args.out} ({size / 1024:.0f} KiB)")
    print(describe_reference(ref))
    _print_selfcheck(ref)
    if ref.has_variants:
        print(f"variants: {len(ref.variants)} tables over {len(ref.input_configs)} input configs")
    return EXIT_OK if ref.selfcheck.get("status") != "failed" else EXIT_ERROR


def _print_selfcheck(ref: Reference) -> None:
    sc = ref.selfcheck
    if not sc:
        return
    if sc.get("status") == "ok":
        extra = f"cosine min {sc['cosine_min']:.7f}, " if "cosine_min" in sc else ""
        print(
            f"self-check ok: {sc.get('description', '')}: {extra}max|diff| {sc['max_abs_diff']:.2e}"
        )
    elif sc.get("status") == "failed":
        print(f"self-check FAILED: {sc}", file=sys.stderr)
        print(
            "the recomputed declared variant does not reproduce the library output; "
            "parity results remain valid, diagnosis may not be",
            file=sys.stderr,
        )
    else:
        print(f"self-check skipped: {sc.get('reason', '')}")


def _emit(report: Report, args: argparse.Namespace, preface: Sequence[str] = ()) -> int:
    color = sys.stdout.isatty() and not args.no_color and "NO_COLOR" not in os.environ
    text = render_text(report, strict=args.strict, color=color)
    if args.json == "-":
        print(report.dumps(args.strict))
    else:
        for line in preface:
            print(line)
        sys.stdout.write(text)
        if args.json:
            Path(args.json).write_text(report.dumps(args.strict) + "\n", encoding="utf-8")
    if args.markdown:
        Path(args.markdown).write_text(
            render_markdown(report, strict=args.strict), encoding="utf-8"
        )
    return report.exit_code(args.strict)


def _skip(value: str) -> tuple[str, ...]:
    items = tuple(s.strip() for s in value.split(",") if s.strip())
    unknown = [s for s in items if s not in SECTIONS]
    if unknown:
        raise UsageError(f"unknown section(s) {unknown}; choose from {', '.join(SECTIONS)}")
    return items


def _cmd_check(args: argparse.Namespace) -> int:
    extra: dict[str, Any] = {}
    if args.extra_body:
        try:
            extra = json.loads(args.extra_body)
        except json.JSONDecodeError as exc:
            raise UsageError(f"--extra-body is not valid JSON: {exc.msg}") from exc
        if not isinstance(extra, dict):
            raise UsageError("--extra-body must be a JSON object")
    ref: Reference | None = None
    if args.reference:
        ref = Reference.load(args.reference)
    elif args.model_id:
        ref = make_reference(args.model_id, args.kind, probes=builtin_probes())
        _print_selfcheck(ref)
    config = CheckConfig(
        base_url=args.base_url,
        model=args.model,
        api=args.api,
        path=args.path,
        rerank_api=args.rerank_api,
        rerank_path=args.rerank_path,
        headers=_headers(args.header),
        api_key=args.api_key or os.environ.get(API_KEY_ENV),
        timeout=args.timeout,
        extra_body=extra,
        dimensions=args.dimensions,
        expect=_key_values(args.expect, "--expect"),
        thresholds=_thresholds(args),
        skip=_skip(args.skip),
        concurrency=args.concurrency,
        verbose=args.verbose,
    )
    report = run_check(config, ref)
    preface = []
    if ref is None:
        preface.append(
            "no reference given: running black-box invariance checks only "
            "(add --reference or --model-id to compare with the library)"
        )
    return _emit(report, args, preface)


def _cmd_demo(args: argparse.Namespace) -> int:
    from embed_parity.demo import DEFAULT_FAULTS, run_demo

    spec = args.fault
    if spec is None:
        spec = "rerank-raw" if args.rerank else DEFAULT_FAULTS
    if spec.strip().lower() == "none":
        spec = ""
    report, active = run_demo(
        spec,
        model=args.model,
        rerank=args.rerank,
        api=args.api,
        thresholds=_thresholds(args),
        verbose=args.verbose,
        skip=_skip(args.skip),
    )
    faults = ", ".join(active) if active else "none"
    return _emit(report, args, [f"demo: simulated server, injected faults: {faults}", ""])


def _cmd_serve_sim(args: argparse.Namespace) -> int:
    from dataclasses import fields

    from embed_parity.simserver import Faults, SimServer

    if args.list_faults:
        for f in fields(Faults):
            kind = str(f.type).split(" ")[0]
            name = f.name.replace("_", "-")
            print(name if kind == "bool" else f"{name}=<{kind}>")
        return EXIT_OK
    server = SimServer(args.model, Faults.parse(args.fault), host=args.host, port=args.port)
    active = ", ".join(server.sim.faults.active()) or "none"
    print(f"simulated server for {args.model} on {server.url} (faults: {active})")
    print("routes: /v1/embeddings /embeddings /embed /api/embed /rerank /v1/rerank /health")
    print(f"try:    embed-parity check --base-url {server.url} --model-id {args.model}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    return EXIT_OK


def _cmd_inspect(args: argparse.Namespace) -> int:
    ref = Reference.load(args.reference)
    info = ref.info
    print(describe_reference(ref))
    print(
        f"probe set: {ref.probes.name}, {len(ref.probes.probes)} probes, "
        f"{len(ref.probes.queries)} queries x {len(ref.probes.documents)} documents"
    )
    meta = ref.meta
    print(f"created:   {meta.get('created', '?')} by embed-parity {meta.get('tool_version', '?')}")
    if info.library_versions:
        versions = ", ".join(f"{k} {v}" for k, v in sorted(info.library_versions.items()))
        print(f"libraries: {versions}")
    if info.kind == "embedding":
        print(f"modules:   {' -> '.join(info.modules)}")
        prompts = {k: v for k, v in info.prompts.items() if v}
        print(f"prompts:   {prompts or 'none'} (default: {info.default_prompt_name})")
        print(f"similarity: {info.similarity_fn}; matryoshka dims: {info.matryoshka_dims or 'n/a'}")
        print(
            f"truncation: {info.truncation_side} at {info.max_seq_length} tokens "
            f"(position limit {info.position_limit})"
        )
        for note in info.notes:
            print(f"note:      {note}")
        if ref.has_variants:
            print("variants (input config / pooling -> probes stored):")
            for key, table in ref.variants.items():
                print(f"  {key:<24} {len(table.probe_ids):>3} x {table.matrix.shape[1]}")
        else:
            print("variants:  none (built with --no-variants; no diagnosis possible)")
    _print_selfcheck(ref)
    return EXIT_OK


def _cmd_probes(_: argparse.Namespace) -> int:
    for probe in builtin_probes().probes:
        print(json.dumps(probe.to_json(), ensure_ascii=False))
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw and raw[0] == "--demo":
        raw = ["demo", *raw[1:]]
    args = parser.parse_args(raw)
    commands = {
        "reference": _cmd_reference,
        "check": _cmd_check,
        "demo": _cmd_demo,
        "serve-sim": _cmd_serve_sim,
        "inspect": _cmd_inspect,
        "probes": _cmd_probes,
    }
    if args.command is None:
        parser.print_help()
        return EXIT_ERROR
    try:
        return commands[args.command](args)
    except (UsageError, ReferenceFileError) as exc:
        print(f"embed-parity: error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except EndpointError as exc:
        print(f"embed-parity: endpoint error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"embed-parity: error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
