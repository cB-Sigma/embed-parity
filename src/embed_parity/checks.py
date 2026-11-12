"""The checks: parity against a reference, ranking, tokens, invariance, rerank."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from embed_parity.client import EmbeddingClient, EndpointError, ProtocolError, RerankClient
from embed_parity.diagnose import Diagnosis, diagnose
from embed_parity.mathx import (
    max_abs_diff,
    normalize,
    norms,
    row_cosine,
    sigmoid,
    spearman,
)
from embed_parity.reference import DEFAULT, Expectation, Reference
from embed_parity.report import FAIL, INFO, PASS, SKIP, WARN, Section, Table

Matrix = npt.NDArray[np.float64]


@dataclass
class Thresholds:
    """Pass/fail limits. See the README for how to choose them."""

    min_cosine: float = 0.999
    norm_tolerance: float = 0.01
    max_abs_diff: float | None = None
    min_rank_correlation: float = 0.98
    invariance_min_cosine: float = 0.9999
    unit_norm_tolerance: float = 1e-3
    token_mismatch: str = "warn"  # "warn", "fail" or "ignore"
    rerank_max_diff: float = 0.02
    base64_max_diff: float = 1e-6

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Parity against the reference
# --------------------------------------------------------------------------- #


@dataclass
class ParityOutcome:
    """Raw material shared by the parity, ranking, token and diagnosis sections."""

    probe_ids: list[str]
    vectors: Matrix | None
    tokens: dict[str, int]
    rejected: dict[str, str]
    failed: dict[str, str]


def collect(client: EmbeddingClient, ref: Reference) -> ParityOutcome:
    """Send every probe on its own, so batching cannot disturb the comparison."""
    ids: list[str] = []
    rows: list[Matrix] = []
    tokens: dict[str, int] = {}
    rejected: dict[str, str] = {}
    failed: dict[str, str] = {}
    max_len = ref.info.max_seq_length
    for n, probe in enumerate(ref.probes.probes):
        try:
            result = client.embed([probe.text])
        except EndpointError as exc:
            if exc.status is None and not isinstance(exc, ProtocolError) and n == 0:
                raise
            over_length = (
                "long" in probe.tags or ref.full_tokens.get(probe.id, 0) > max_len
            ) and exc.client_error
            (rejected if over_length else failed)[probe.id] = str(exc)
            continue
        vec = result.vectors[0]
        if rows and vec.shape != rows[0].shape:
            failed[probe.id] = f"dimension {vec.shape[0]} differs from earlier responses"
            continue
        ids.append(probe.id)
        rows.append(vec)
        if result.prompt_tokens is not None:
            tokens[probe.id] = result.prompt_tokens
    return ParityOutcome(ids, np.stack(rows) if rows else None, tokens, rejected, failed)


def parity_section(
    ref: Reference,
    outcome: ParityOutcome,
    exp: Expectation,
    thr: Thresholds,
    *,
    verbose: bool = False,
) -> Section:
    n_total = len(ref.probes.probes)
    section = Section("parity", f"Parity with the reference ({n_total} probes, one per request)")
    for pid, message in outcome.failed.items():
        section.add(f"probe/{pid}", f"probe {pid}", FAIL, message)
    if outcome.rejected:
        section.add(
            "over-length",
            "over-length input",
            INFO,
            f"endpoint rejected {', '.join(outcome.rejected)} instead of truncating "
            f"(the library truncates at {ref.info.max_seq_length} tokens)",
        )
    if outcome.vectors is None:
        section.add("vectors", "embeddings received", FAIL, "no probe could be embedded")
        return section

    E = outcome.vectors
    expected = ref.expected_vectors(exp)
    rows = [ref.probes.index(pid) for pid in outcome.probe_ids]
    X = expected[rows]
    d_e, d_x = E.shape[1], X.shape[1]
    finite = bool(np.all(np.isfinite(E)))
    section.add(
        "dimension",
        "dimension",
        PASS if d_e == d_x else FAIL,
        f"{d_e}" if d_e == d_x else f"endpoint returns {d_e}, expected {d_x}",
        endpoint=d_e,
        expected=d_x,
    )
    if not finite:
        section.add("finite", "finite values", FAIL, "vectors contain NaN or infinity")
        return section
    if d_e != d_x:
        section.lines.append(
            "vectors cannot be compared element-wise because the dimension differs; "
            "see the diagnosis below"
        )
        return section

    cos = row_cosine(E, X)
    ratio = norms(E) / np.maximum(norms(X), 1e-12)
    diff = np.max(np.abs(E - X), axis=1)
    section.data.update(
        {
            "probe_ids": outcome.probe_ids,
            "cosine": cos.tolist(),
            "norm_ratio": ratio.tolist(),
            "max_abs_diff": diff.tolist(),
        }
    )

    order = np.argsort(cos, kind="stable")
    shown = order if verbose or len(order) <= 8 else order[:6]
    table_rows = []
    for i in shown:
        pid = outcome.probe_ids[i]
        want = ref.token_count(exp.config, pid)
        got = outcome.tokens.get(pid)
        tok = f"{got if got is not None else '-'}/{want if want is not None else '-'}"
        table_rows.append([pid, f"{cos[i]:.6f}", f"{ratio[i]:.4f}", f"{diff[i]:.2e}", tok])
    note = ""
    if len(shown) < len(order):
        note = (
            f"({len(shown)} of {len(order)} probes shown, lowest cosine first; --verbose for all)"
        )
    section.table = Table(
        ["probe", "cosine", "norm ratio", "max|diff|", "tokens (endpoint/ref)"], table_rows, note
    )

    worst = int(np.argmin(cos))
    below = int(np.sum(cos < thr.min_cosine))
    section.add(
        "cosine",
        "cosine similarity",
        PASS if below == 0 else FAIL,
        f"min {np.min(cos):.6f} ({outcome.probe_ids[worst]}), mean {cos.mean():.6f}; "
        + (
            f"all >= {thr.min_cosine}"
            if below == 0
            else f"{below} of {len(cos)} probes below {thr.min_cosine}"
        ),
        min=float(np.min(cos)),
        mean=float(cos.mean()),
        threshold=thr.min_cosine,
    )
    dev = np.abs(ratio - 1.0)
    bad_norm = int(np.sum(dev > thr.norm_tolerance))
    section.add(
        "norm",
        "norm ratio (endpoint / reference)",
        PASS if bad_norm == 0 else FAIL,
        f"min {np.min(ratio):.4f}, max {np.max(ratio):.4f}; "
        + (
            f"all within {thr.norm_tolerance} of 1"
            if bad_norm == 0
            else f"{bad_norm} of {len(ratio)} probes off by more than {thr.norm_tolerance}"
        ),
        min=float(np.min(ratio)),
        max=float(np.max(ratio)),
        tolerance=thr.norm_tolerance,
    )
    if thr.max_abs_diff is not None:
        over = int(np.sum(diff > thr.max_abs_diff))
        section.add(
            "max-abs-diff",
            "max abs difference",
            PASS if over == 0 else FAIL,
            f"largest {np.max(diff):.2e}; limit {thr.max_abs_diff:g}",
            max=float(np.max(diff)),
        )
    else:
        section.add(
            "max-abs-diff",
            "max abs difference",
            INFO,
            f"largest {np.max(diff):.2e} (no limit set)",
            max=float(np.max(diff)),
        )
    return section


def _similarity(queries: Matrix, docs: Matrix, fn: str) -> Matrix:
    if fn == "dot":
        return np.asarray(queries @ docs.T)
    if fn == "euclidean":
        return np.asarray(-np.linalg.norm(queries[:, None, :] - docs[None, :, :], axis=2))
    if fn == "manhattan":
        return np.asarray(-np.sum(np.abs(queries[:, None, :] - docs[None, :, :]), axis=2))
    return np.asarray(normalize(queries) @ normalize(docs).T)


def ranking_section(
    ref: Reference, outcome: ParityOutcome, exp: Expectation, thr: Thresholds
) -> Section:
    fn = ref.info.similarity_fn or "cosine"
    section = Section("ranking", f"Retrieval ranking ({fn} similarity)")
    q_ids, d_ids = list(ref.probes.queries), list(ref.probes.documents)
    if not q_ids or len(d_ids) < 2:
        section.add("task", "rank agreement", SKIP, "the probe set has no retrieval task")
        return section
    if outcome.vectors is None or any(i not in outcome.probe_ids for i in (*q_ids, *d_ids)):
        section.add("task", "rank agreement", SKIP, "not all retrieval probes were embedded")
        return section
    pos = {pid: n for n, pid in enumerate(outcome.probe_ids)}
    E = outcome.vectors
    X = ref.expected_vectors(exp)
    eq, ed = E[[pos[i] for i in q_ids]], E[[pos[i] for i in d_ids]]
    xq = X[[ref.probes.index(i) for i in q_ids]]
    xd = X[[ref.probes.index(i) for i in d_ids]]
    s_end, s_ref = _similarity(eq, ed, fn), _similarity(xq, xd, fn)
    rhos, top1 = [], 0
    rows = []
    for qi, qid in enumerate(q_ids):
        rho = spearman(s_end[qi], s_ref[qi])
        rhos.append(rho)
        best_ref = int(np.argmax(s_ref[qi]))
        best_end = int(np.argmax(s_end[qi]))
        # A different top document only counts if the reference separates the two.
        same = best_end == best_ref or (s_ref[qi, best_ref] - s_ref[qi, best_end]) < 1e-4
        top1 += int(same)
        rows.append([qid, d_ids[best_ref], d_ids[best_end], f"{rho:.4f}"])
    section.table = Table(["query", "reference top-1", "endpoint top-1", "Spearman"], rows)
    mean_rho = float(np.mean(rhos))
    ok = mean_rho >= thr.min_rank_correlation and top1 == len(q_ids)
    section.add(
        "agreement",
        "rank-order agreement",
        PASS if ok else FAIL,
        f"top-1 agrees on {top1} of {len(q_ids)} queries; Spearman mean {mean_rho:.4f}, "
        f"min {min(rhos):.4f} (threshold {thr.min_rank_correlation})",
        top1=top1,
        queries=len(q_ids),
        spearman_mean=mean_rho,
        spearman_min=float(min(rhos)),
    )
    return section


def token_section(
    ref: Reference, outcome: ParityOutcome, exp: Expectation, thr: Thresholds
) -> Section:
    section = Section("tokens", "Token counts from usage")
    if not outcome.tokens:
        section.add("usage", "token-count parity", SKIP, "the endpoint reports no token usage")
        return section
    if not ref.tokens:
        section.add("usage", "token-count parity", SKIP, "the reference has no token counts")
        return section
    pairs = {
        pid: (got, ref.token_count(exp.config, pid))
        for pid, got in outcome.tokens.items()
        if ref.token_count(exp.config, pid) is not None
    }
    wrong = {pid: gw for pid, gw in pairs.items() if gw[0] != gw[1]}
    if not wrong:
        section.add(
            "usage",
            "token-count parity",
            PASS,
            f"usage.prompt_tokens equals the reference tokenizer on all {len(pairs)} probes",
        )
        return section
    explanation = ""
    for config in ref.tokens:
        if config == exp.config:
            continue
        if all(ref.token_count(config, pid) == got for pid, got in outcome.tokens.items()):
            from embed_parity.diagnose import describe_config

            explanation = (
                f"; the counts equal the reference tokenizer "
                f"{describe_config(config, ref) or 'with default input handling'}"
            )
            break
    if not explanation:
        deltas = sorted({gw[0] - (gw[1] or 0) for gw in wrong.values()})
        if len(deltas) == 1:
            explanation = f"; every count is off by {deltas[0]:+d}"
    sample = ", ".join(f"{pid} {g}/{w}" for pid, (g, w) in list(wrong.items())[:3])
    status = {"fail": FAIL, "ignore": INFO}.get(thr.token_mismatch, WARN)
    section.add(
        "usage",
        "token-count parity",
        status,
        f"{len(wrong)} of {len(pairs)} probes differ (endpoint/reference: {sample}){explanation}",
        mismatches=len(wrong),
    )
    return section


def diagnosis_section(diag: Diagnosis, parity_failed: bool) -> Section:
    section = Section("diagnosis", "Diagnosis")
    if diag.status == "unavailable":
        section.add("variant", "variant match", SKIP, diag.summary)
        return section
    section.lines.append(diag.summary)
    for finding in diag.findings:
        section.lines.append(f"  - {finding}")
    if diag.status == "as-expected":
        status = PASS
        detail = "the endpoint behaves as declared"
    elif diag.status == "variant":
        status = FAIL if parity_failed else WARN
        detail = f"{len(diag.findings)} difference(s) from the expected behaviour identified"
    elif diag.status == "approximate":
        status = FAIL if parity_failed else WARN
        detail = "closest variant is a near match, not an exact one"
    else:
        status = FAIL if parity_failed else INFO
        detail = "no variant explains the endpoint"
    section.add("variant", "variant match", status, detail)
    section.data.update(diag.to_json())
    return section


# --------------------------------------------------------------------------- #
# Black-box invariance checks (no reference needed)
# --------------------------------------------------------------------------- #

_A = "The quick brown fox jumps over the lazy dog near the quiet river bank."
_A_PERMUTED = "bank river quiet the near dog lazy the over jumps fox brown quick The."
_SHORT = "ok"
_LONGER = (
    "Regional rail services will run to a reduced timetable over the holiday period, "
    "with replacement buses between the two junction stations while the viaduct is "
    "inspected, and passengers are advised to check departure boards before travelling."
)
_FILL = ("a red door", "seven", "Where is the station?", "rain later, clearing by evening", "x y z")
_VERY_LONG = "The archive keeps one copy of every ledger and a second copy offsite. " * 400


def _cos1(a: Matrix, b: Matrix) -> float:
    if a.shape != b.shape:
        return float("nan")
    return float(row_cosine(a[None, :], b[None, :])[0])


def _verdict(cos: float, thr: Thresholds) -> str:
    return PASS if cos == cos and cos >= thr.invariance_min_cosine else FAIL


def invariance_section(
    client: EmbeddingClient,
    thr: Thresholds,
    *,
    expected_dim: int | None = None,
    expect_normalized: bool | None = None,
    concurrency: int = 6,
) -> Section:
    """Checks that hold for any correct server, whatever the model."""
    section = Section("invariance", "Invariance (black-box, no reference needed)")
    seen: list[Matrix] = []

    def embed(texts: list[str]) -> Matrix:
        vectors = client.embed(texts).vectors
        seen.append(vectors)
        return vectors

    base = embed([_A])[0]
    base_perm = embed([_A_PERMUTED])[0]

    # -- determinism ------------------------------------------------------
    repeats = [embed([_A])[0] for _ in range(2)]
    cos = min(_cos1(base, r) for r in repeats)
    diff = max(max_abs_diff(base, r) if r.shape == base.shape else float("inf") for r in repeats)
    section.add(
        "determinism",
        "repeat determinism",
        _verdict(cos, thr),
        "bit-identical across 3 requests"
        if diff == 0
        else f"cosine min {cos:.6f}, max|diff| {diff:.2e} across 3 requests",
        cosine_min=cos,
        max_abs_diff=diff,
    )

    # -- batch composition --------------------------------------------------
    def batch_check(
        check_id: str, title: str, batch: list[str], targets: dict[int, Matrix]
    ) -> None:
        try:
            got = embed(batch)
        except EndpointError as exc:
            if exc.client_error:
                section.add(check_id, title, SKIP, f"batch rejected ({_status(exc)})")
            else:
                section.add(check_id, title, FAIL, f"request failed: {exc}")
            return
        worst = min(_cos1(got[i], want) for i, want in targets.items())
        section.add(
            check_id,
            title,
            _verdict(worst, thr),
            f"cosine vs single request {worst:.6f}"
            + ("" if worst >= thr.invariance_min_cosine else " - output depends on the batch"),
            cosine_min=worst,
            batch_size=len(batch),
        )

    batch_check("batch/duplicate", "batch: same text twice", [_A, _A], {0: base, 1: base})
    batch_check(
        "batch/equal-length",
        "batch: two texts of equal length",
        [_A, _A_PERMUTED],
        {0: base, 1: base_perm},
    )
    batch_check("batch/shorter", "batch: with a shorter text", [_A, _SHORT], {0: base})
    batch_check("batch/longer", "batch: with a longer text", [_LONGER, _A], {1: base})
    batch_check(
        "batch/mixed-8",
        "batch: 8 mixed texts",
        [_SHORT, _LONGER, _A, *_FILL],
        {2: base},
    )

    # -- order ---------------------------------------------------------------
    try:
        fwd = embed([_A, _SHORT, _LONGER])
        rev = embed([_LONGER, _SHORT, _A])
        worst = min(_cos1(fwd[i], rev[2 - i]) for i in range(3))
        section.add(
            "order",
            "order invariance",
            _verdict(worst, thr),
            f"cosine min {worst:.6f} between a batch and its reverse",
            cosine_min=worst,
        )
    except EndpointError as exc:
        if exc.client_error:
            section.add("order", "order invariance", SKIP, f"batch rejected ({_status(exc)})")
        else:
            section.add("order", "order invariance", FAIL, f"request failed: {exc}")

    # -- concurrent single requests (server-side batching) -----------------
    if concurrency >= 2:
        texts = [_A if i % 2 == 0 else _A_PERMUTED for i in range(concurrency)]
        wants = [base if i % 2 == 0 else base_perm for i in range(concurrency)]
        barrier = threading.Barrier(concurrency)

        def one(text: str) -> Matrix:
            barrier.wait(timeout=30)
            return np.asarray(client.embed([text]).vectors[0], dtype=np.float64)

        try:
            worst = 1.0
            for _ in range(2):
                barrier.reset()
                with ThreadPoolExecutor(max_workers=concurrency) as pool_:
                    got_c = list(pool_.map(one, texts))
                worst = min(worst, *(_cos1(g, w) for g, w in zip(got_c, wants, strict=True)))
            section.add(
                "concurrency",
                f"concurrent requests ({concurrency} at once)",
                _verdict(worst, thr),
                f"cosine vs sequential request {worst:.6f}",
                cosine_min=worst,
            )
        except (EndpointError, threading.BrokenBarrierError) as exc:
            section.add("concurrency", "concurrent requests", FAIL, f"request failed: {exc}")

    # -- whitespace ----------------------------------------------------------
    try:
        padded = embed([f"  {_A}  \n"])[0]
        cos = _cos1(base, padded)
        section.add(
            "whitespace",
            "leading/trailing whitespace",
            PASS if cos >= thr.invariance_min_cosine else WARN,
            f"cosine vs trimmed text {cos:.6f}"
            + ("" if cos >= thr.invariance_min_cosine else " (tokenizer keeps outer whitespace)"),
            cosine=cos,
        )
    except EndpointError as exc:
        section.add("whitespace", "leading/trailing whitespace", WARN, f"request failed: {exc}")

    # -- empty input -----------------------------------------------------------
    try:
        empty = client.embed([""]).vectors[0]
        ok = bool(np.all(np.isfinite(empty)))
        section.add(
            "empty",
            "empty string",
            INFO if ok else FAIL,
            f"accepted, norm {float(np.linalg.norm(empty)):.4f}"
            if ok
            else "accepted but the vector contains NaN or infinity",
        )
    except EndpointError as exc:
        status = INFO if exc.client_error else WARN
        section.add("empty", "empty string", status, f"rejected ({_status(exc)})")

    # -- very long input ---------------------------------------------------------
    try:
        long_result = client.embed([_VERY_LONG])
        vec = long_result.vectors[0]
        ok = bool(np.all(np.isfinite(vec))) and vec.shape == base.shape
        tokens = (
            f", usage reports {long_result.prompt_tokens} tokens"
            if long_result.prompt_tokens is not None
            else ""
        )
        section.add(
            "very-long",
            "very long input (~5,200 words)",
            INFO if ok else FAIL,
            f"accepted{tokens}" if ok else "accepted but the vector is malformed",
            prompt_tokens=long_result.prompt_tokens,
        )
        if ok:
            seen.append(vec[None, :])
    except EndpointError as exc:
        if exc.client_error:
            section.add("very-long", "very long input (~5,200 words)", INFO, "rejected with 4xx")
        else:
            section.add(
                "very-long",
                "very long input (~5,200 words)",
                FAIL,
                f"server error instead of truncation or a 4xx ({_status(exc)})",
            )

    # -- shape and values --------------------------------------------------------
    dims = sorted({int(m.shape[1]) for m in seen})
    if len(dims) != 1:
        section.add("dimension", "dimension", FAIL, f"responses mix dimensions {dims}")
    elif expected_dim is None:
        section.add("dimension", "dimension", INFO, f"{dims[0]} (no expectation given)")
    else:
        section.add(
            "dimension",
            "declared dimension",
            PASS if dims[0] == expected_dim else FAIL,
            f"{dims[0]}" if dims[0] == expected_dim else f"{dims[0]}, expected {expected_dim}",
        )
    stacked = np.concatenate([m.reshape(-1, m.shape[-1]) for m in seen if m.shape[-1] == dims[0]])
    finite = bool(np.all(np.isfinite(stacked)))
    nonzero = bool(np.all(norms(stacked) > 0)) if finite else False
    section.add(
        "finite",
        "finite, non-zero vectors",
        PASS if finite and nonzero else FAIL,
        "" if finite and nonzero else "NaN, infinity or all-zero vectors returned",
    )
    if finite:
        vn = norms(stacked)
        unit = bool(np.all(np.abs(vn - 1.0) <= thr.unit_norm_tolerance))
        spread = f"norms {np.min(vn):.4f} to {np.max(vn):.4f}"
        if expect_normalized is None:
            section.add("unit-norm", "vector norm", INFO, spread + (" (unit norm)" if unit else ""))
        elif expect_normalized:
            section.add(
                "unit-norm",
                "unit norm (model declares normalisation)",
                PASS if unit else FAIL,
                spread,
            )
        else:
            section.add("unit-norm", "vector norm", INFO, spread + " (no normalisation declared)")

    # -- base64 ------------------------------------------------------------------
    if client.supports_base64:
        pair = [_A, _SHORT]
        try:
            as_float = client.embed(pair, encoding_format="float").vectors
            as_b64 = client.embed(pair, encoding_format="base64").vectors
            if as_b64.shape != as_float.shape:
                section.add(
                    "base64",
                    "base64 vs float encoding",
                    FAIL,
                    f"base64 decodes to {as_b64.shape[1]} float32 values, float encoding has "
                    f"{as_float.shape[1]} (wrong dtype or byte order in the payload?)",
                )
            else:
                diff = max_abs_diff(as_float, as_b64)
                section.add(
                    "base64",
                    "base64 vs float encoding",
                    PASS if diff <= thr.base64_max_diff else FAIL,
                    f"max|diff| {diff:.2e}",
                    max_abs_diff=diff,
                )
        except ProtocolError as exc:
            section.add("base64", "base64 vs float encoding", WARN, str(exc))
        except EndpointError as exc:
            status = SKIP if exc.client_error else FAIL
            section.add("base64", "base64 vs float encoding", status, f"rejected ({_status(exc)})")
    else:
        section.add("base64", "base64 vs float encoding", SKIP, f"not part of the {client.api} API")
    return section


def _status(exc: EndpointError) -> str:
    return f"HTTP {exc.status}" if exc.status is not None else str(exc)


# --------------------------------------------------------------------------- #
# Rerank
# --------------------------------------------------------------------------- #


def rerank_section(
    client: RerankClient,
    ref: Reference,
    thr: Thresholds,
    *,
    expect_activation: str | None = None,
) -> Section:
    """Compare rerank scores with a cross-encoder reference."""
    section = Section("rerank", "Rerank parity with the cross-encoder reference")
    rr = ref.rerank
    if rr is None:
        section.add("reference", "rerank reference", SKIP, "the reference has no rerank scores")
        return section
    declared = expect_activation or ref.info.activation or "identity"
    queries = [ref.probes.get(q).text for q in rr.query_ids]
    docs = [ref.probes.get(d).text for d in rr.document_ids]
    got = np.stack([client.rerank(q, docs).scores for q in queries])
    logits = rr.logits
    candidates = {"identity": logits, "sigmoid": sigmoid(logits)}
    diffs = {name: float(np.max(np.abs(got - cand))) for name, cand in candidates.items()}
    observed = min(diffs, key=lambda k: diffs[k])
    names = {"identity": "raw logits", "sigmoid": "sigmoid probabilities"}
    rows = [
        [qid, f"{float(np.max(np.abs(got[i] - candidates[declared][i]))):.4f}"]
        for i, qid in enumerate(rr.query_ids)
    ]
    section.table = Table(["query", f"max|diff| vs {names[declared]}"], rows)

    if diffs[observed] > thr.rerank_max_diff:
        section.lines.append(
            f"scores match neither raw logits (max|diff| {diffs['identity']:.4f}) nor "
            f"sigmoid(logits) (max|diff| {diffs['sigmoid']:.4f})"
        )
        section.add(
            "scores",
            "score parity",
            FAIL,
            f"max|diff| {diffs[declared]:.4f} vs {names[declared]}; limit {thr.rerank_max_diff}",
            diffs=diffs,
        )
    elif observed != declared:
        section.lines.append(
            f"endpoint returns {names[observed]} (max|diff| {diffs[observed]:.4f}); "
            f"the model declares {names[declared]}"
        )
        section.add(
            "scores",
            "score parity",
            FAIL,
            f"scores are {names[observed]}, expected {names[declared]}",
            diffs=diffs,
            observed=observed,
        )
    else:
        section.add(
            "scores",
            "score parity",
            PASS,
            f"endpoint returns {names[observed]}, max|diff| {diffs[observed]:.4f} "
            f"(limit {thr.rerank_max_diff})",
            diffs=diffs,
            observed=observed,
        )

    rhos = [spearman(got[i], logits[i]) for i in range(len(queries))]
    top1 = sum(
        int(logits[i, int(np.argmax(got[i]))] >= np.max(logits[i]) - 1e-4)
        for i in range(len(queries))
    )
    ok = float(np.mean(rhos)) >= thr.min_rank_correlation and top1 == len(queries)
    section.add(
        "ordering",
        "ordering agreement",
        PASS if ok else FAIL,
        f"top-1 agrees on {top1} of {len(queries)} queries; Spearman mean {np.mean(rhos):.4f}",
        spearman_mean=float(np.mean(rhos)),
        top1=top1,
    )

    # Batch and order invariance on the first query.
    q0 = queries[0]
    full = got[0]
    try:
        singles = np.asarray([client.rerank(q0, [d]).scores[0] for d in docs])
        diff = float(np.max(np.abs(singles - full)))
        section.add(
            "batch",
            "batch invariance",
            PASS if diff <= thr.rerank_max_diff else FAIL,
            f"max|diff| {diff:.4f} between scoring a document alone and in a list of {len(docs)}",
            max_abs_diff=diff,
        )
        reverse = client.rerank(q0, docs[::-1]).scores[::-1]
        diff = float(np.max(np.abs(reverse - full)))
        section.add(
            "order",
            "order invariance",
            PASS if diff <= thr.rerank_max_diff else FAIL,
            f"max|diff| {diff:.4f} after reversing the document list",
            max_abs_diff=diff,
        )
        again = client.rerank(q0, docs)
        diff = float(np.max(np.abs(again.scores - full)))
        section.add(
            "determinism",
            "repeat determinism",
            PASS if diff <= 1e-5 else FAIL,
            "bit-identical" if diff == 0 else f"max|diff| {diff:.2e}",
        )
        order = again.returned_order
        sorted_ok = all(
            again.scores[order[i]] >= again.scores[order[i + 1]] for i in range(len(order) - 1)
        )
        section.add(
            "sorted",
            "results sorted by score",
            PASS if sorted_ok else WARN,
            "" if sorted_ok else "results are not in descending score order",
        )
    except EndpointError as exc:
        section.add("batch", "batch invariance", FAIL, f"request failed: {exc}")
    return section


__all__ = [
    "DEFAULT",
    "ParityOutcome",
    "Thresholds",
    "collect",
    "diagnose",
    "diagnosis_section",
    "invariance_section",
    "parity_section",
    "ranking_section",
    "rerank_section",
    "token_section",
]
