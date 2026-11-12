"""Work out which implementation choice an endpoint's vectors correspond to.

The reference holds sentence vectors for many implementation variants. Cosine
similarity is unaffected by scaling, so the *direction* of an endpoint vector
identifies the pooling mode and the input handling, while its *norm*
identifies normalisation, and its length and value set identify dimension
truncation and quantisation. Each property is therefore read independently.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from embed_parity.mathx import detect_quantization, norms, row_cosine, to_signs
from embed_parity.reference import DEFAULT, DENSE_OFF, Expectation, Reference

POOLING_NAMES = {"cls": "CLS", "mean": "mean", "last": "last-token", "max": "max"}

#: Minimum per-probe cosine for "the endpoint matches this variant".
MATCH_COSINE = {
    "float32": 0.9995,
    "float16": 0.9995,
    "int8": 0.99,
    "uint8": 0.99,
    "binary": 0.95,
    "packed-binary": 0.95,
}


#: Below the match threshold but above this, the best variant is a "near match".
NEAR_COSINE = 0.95


@dataclass
class VariantScore:
    """How well one variant explains the endpoint."""

    config: str
    pooling: str
    cosine_min: float
    cosine_mean: float
    deviations: int
    truncated: bool
    matched_probes: int

    @property
    def key(self) -> str:
        return f"{self.config}/{self.pooling}"


@dataclass
class Diagnosis:
    """Result of matching an endpoint against the reference variants."""

    status: str  # "as-expected", "variant", "approximate", "no-match", "unavailable"
    summary: str
    findings: list[str] = field(default_factory=list)
    observed: dict[str, Any] = field(default_factory=dict)
    expected: dict[str, Any] = field(default_factory=dict)
    best: VariantScore | None = None
    ranking: list[VariantScore] = field(default_factory=list)
    probes_used: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "summary": self.summary,
            "findings": self.findings,
            "observed": self.observed,
            "expected": self.expected,
            "best": None if self.best is None else asdict(self.best),
            "ranking": [asdict(s) for s in self.ranking],
            "probes_used": self.probes_used,
        }


def describe_config(config: str, ref: Reference) -> str:
    """A human phrase for an input configuration, relative to the default."""
    if config == DEFAULT:
        return ""
    declared_prompt = ref.info.default_prompt_name if ref.info.default_prompt else None
    if config == "prompt=none":
        return f"without the '{declared_prompt}' prompt prefix"
    if config.startswith("prompt="):
        name = config.split("=", 1)[1]
        return f"with the '{name}' prompt prefix ({ref.info.prompts.get(name, '')!r})"
    if config == "special=off":
        return "without special tokens"
    if config.startswith("trunc="):
        return f"truncating long inputs from the {config.split('=', 1)[1]}"
    if config.startswith("maxlen="):
        return (
            f"truncating at {config.split('=', 1)[1]} tokens "
            f"(the model declares {ref.info.max_seq_length})"
        )
    if config == DENSE_OFF:
        return "without the Dense projection that follows pooling"
    return config


def describe_behaviour(
    ref: Reference,
    *,
    pooling: str,
    config: str,
    normalized: str,
    dimension: int | None,
    quantization: str = "float32",
) -> str:
    """Compose a phrase such as 'mean pooling without normalisation'."""
    parts = [f"{POOLING_NAMES.get(pooling, pooling)} pooling"]
    if normalized == "yes":
        parts.append("with normalisation")
    elif normalized == "no":
        parts.append("without normalisation")
    elif normalized == "truncated-not-renormalised":
        parts.append("normalised before truncation but not after")
    phrase = " ".join(parts)
    extra = describe_config(config, ref)
    if extra:
        phrase += f", {extra}"
    if dimension is not None and dimension < ref.info.dimension and config != DENSE_OFF:
        phrase += f", truncated to {dimension} of {ref.info.dimension} dimensions"
    if quantization not in ("float32", ""):
        phrase += f", {quantization}-quantised"
    return phrase


def _prepare(
    E: npt.NDArray[np.float64], B: npt.NDArray[np.float64], quant: str, full_dim: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None:
    """Bring endpoint and variant vectors into a comparable form."""
    if quant == "packed-binary":
        signs = to_signs(E, "packed-binary", full_dim)
        if B.shape[1] < signs.shape[1]:
            return None
        return signs, to_signs(B[:, : signs.shape[1]], "float")
    if B.shape[1] < E.shape[1]:
        return None
    Bc = B[:, : E.shape[1]]
    if quant == "binary":
        return to_signs(E, "binary"), to_signs(Bc, "float")
    if quant == "uint8":
        # sentence-transformers style uint8 is int8 shifted by 128
        return E - 128.0, Bc
    return E, Bc


def _normalisation(
    E: npt.NDArray[np.float64], B: npt.NDArray[np.float64], quant: str, loose: bool = False
) -> tuple[str, float]:
    """Classify the endpoint's norms relative to variant ``B`` (un-normalised).

    ``loose`` is used for near matches, where norms are only approximately
    right: the closer of "unit norm" and "raw norm" wins if it is within 10%.
    """
    if quant not in ("float32", "float16"):
        return "n/a", float("nan")
    d = E.shape[1]
    en = norms(E)
    b_cut = norms(B[:, :d])
    b_full = norms(B)
    ratio = float(np.median(en / np.maximum(b_cut, 1e-12)))
    unit_dev = np.abs(en - 1.0)
    raw_dev = np.abs(en / np.maximum(b_cut, 1e-12) - 1.0)
    if loose:
        if min(float(np.max(unit_dev)), float(np.max(raw_dev))) < 0.1:
            return ("yes" if np.max(unit_dev) <= np.max(raw_dev) else "no"), ratio
        return "unknown", ratio
    unit = bool(np.all(unit_dev < 2e-3))
    raw = bool(np.all(raw_dev < 2e-2))
    if unit:
        return "yes", ratio
    if raw:
        return "no", ratio
    if d < B.shape[1]:
        partial = b_cut / np.maximum(b_full, 1e-12)
        if bool(np.all(np.abs(en - partial) < 2e-3)):
            return "truncated-not-renormalised", ratio
    return "unknown", ratio


def diagnose(
    ref: Reference,
    probe_ids: list[str],
    endpoint: npt.NDArray[np.float64],
    expectation: Expectation | None = None,
) -> Diagnosis:
    """Find the reference variant that best explains ``endpoint``.

    ``endpoint`` holds one row per entry of ``probe_ids``.
    """
    exp = expectation or ref.declared_expectation()
    info = ref.info
    exp_dim = exp.dimensions or info.dimension
    expected = {
        "pooling": exp.pooling,
        "config": exp.config,
        "normalized": "yes" if exp.normalize else "no",
        "dimension": exp_dim,
        "quantization": "float32",
        "description": describe_behaviour(
            ref,
            pooling=exp.pooling,
            config=exp.config,
            normalized="yes" if exp.normalize else "no",
            dimension=exp_dim,
        ),
    }
    if not ref.has_variants:
        return Diagnosis(
            status="unavailable",
            summary="the reference was built without variants, so no diagnosis is possible",
            expected=expected,
        )
    if endpoint.size == 0 or not probe_ids:
        return Diagnosis(
            status="unavailable", summary="no endpoint vectors to diagnose", expected=expected
        )
    if not np.all(np.isfinite(endpoint)):
        return Diagnosis(
            status="no-match",
            summary="the endpoint returned NaN or infinite values",
            findings=["vectors contain NaN or infinite values"],
            expected=expected,
        )

    rows = [ref.probes.index(pid) for pid in probe_ids]
    d_e = int(endpoint.shape[1])
    quant = detect_quantization(endpoint, info.dimension)
    threshold = MATCH_COSINE[quant]

    scores: list[VariantScore] = []
    prepared: dict[str, npt.NDArray[np.float64]] = {}
    for config, pooling in ref.variant_keys():
        full = ref.base(config, pooling)
        if full is None:
            continue
        B = full[rows]
        pair = _prepare(endpoint, B, quant, info.dimension)
        if pair is None:
            continue
        cos = row_cosine(pair[0], pair[1])
        prepared[f"{config}/{pooling}"] = B
        scores.append(
            VariantScore(
                config=config,
                pooling=pooling,
                cosine_min=float(np.min(cos)),
                cosine_mean=float(cos.mean()),
                deviations=int(config != exp.config) + int(pooling != exp.pooling),
                truncated=pair[1].shape[1] < B.shape[1],
                matched_probes=int(np.sum(cos >= threshold)),
            )
        )

    observed: dict[str, Any] = {"dimension": d_e, "quantization": quant}
    if not scores:
        return Diagnosis(
            status="no-match",
            summary=(
                f"the endpoint returns {d_e}-dimensional vectors, larger than anything the "
                f"model produces ({info.dimension} declared, hidden size {info.hidden_size})"
            ),
            findings=[f"dimension is {d_e}; the model declares {info.dimension}"],
            observed=observed,
            expected=expected,
            probes_used=len(rows),
        )

    scores.sort(key=lambda s: (-round(s.cosine_min, 6), -round(s.cosine_mean, 6), s.deviations))
    best = scores[0]
    B = prepared[best.key]
    normalized, ratio = _normalisation(endpoint, B, quant)
    observed.update(
        {
            "pooling": best.pooling,
            "config": best.config,
            "normalized": normalized,
            "norm_ratio_median": ratio,
        }
    )
    obs_text = describe_behaviour(
        ref,
        pooling=best.pooling,
        config=best.config,
        normalized=normalized,
        dimension=d_e,
        quantization=quant,
    )
    observed["description"] = obs_text
    n = len(rows)

    if NEAR_COSINE <= best.cosine_min < threshold:
        normalized, ratio = _normalisation(endpoint, B, quant, loose=True)
        observed["normalized"] = normalized
        obs_text = describe_behaviour(
            ref,
            pooling=best.pooling,
            config=best.config,
            normalized=normalized,
            dimension=d_e,
            quantization=quant,
        )
        observed["description"] = obs_text
        differences = _differences(ref, exp, best, normalized, d_e, exp_dim, quant, ratio)
        findings = [
            f"near match only: cosine min {best.cosine_min:.5f}, mean {best.cosine_mean:.5f} "
            f"(an exact match needs {threshold}). This points to the same recipe computed "
            "with different numerics - quantised or reduced-precision weights, another "
            "runtime, or a slightly different tokenizer - rather than a different recipe",
            *differences,
        ]
        return Diagnosis(
            status="approximate",
            summary=(
                f"endpoint approximately matches {obs_text} (cosine min "
                f"{best.cosine_min:.5f}); {'the model declares' if exp.is_declared else 'expected'}"
                f" {expected['description']}"
            ),
            findings=findings,
            observed=observed,
            expected=expected,
            best=best,
            ranking=scores[:5],
            probes_used=n,
        )

    if best.cosine_min < threshold:
        worst = _worst_probes(ref, probe_ids, endpoint, B, quant, threshold)
        findings = [
            f"closest variant is {obs_text} (cosine min {best.cosine_min:.5f}, "
            f"mean {best.cosine_mean:.5f}); it reproduces {best.matched_probes} of {n} probes",
        ]
        if worst:
            findings.append("probes that no variant explains well: " + ", ".join(worst))
        findings.append(
            "likely causes outside the variant space: different weights or revision, a "
            "different tokenizer, reduced-precision or quantised weights, or a "
            "batch-dependent bug (see the invariance checks)"
        )
        return Diagnosis(
            status="no-match",
            summary=(
                f"no computed variant reproduces the endpoint; closest is {obs_text} "
                f"(cosine min {best.cosine_min:.5f})"
            ),
            findings=findings,
            observed=observed,
            expected=expected,
            best=best,
            ranking=scores[:5],
            probes_used=n,
        )

    findings = _differences(ref, exp, best, normalized, d_e, exp_dim, quant, ratio)
    status = "as-expected" if not findings else "variant"
    declares = "the model declares" if exp.is_declared else "expected"
    if status == "as-expected":
        summary = (
            f"endpoint matches {obs_text} (cosine min {best.cosine_min:.5f}), "
            f"which is what {'the model declares' if exp.is_declared else 'was expected'}"
        )
    else:
        summary = (
            f"endpoint matches {obs_text} (cosine min {best.cosine_min:.5f}); "
            f"{declares} {expected['description']}"
        )
    return Diagnosis(
        status=status,
        summary=summary,
        findings=findings,
        observed=observed,
        expected=expected,
        best=best,
        ranking=scores[:5],
        probes_used=n,
    )


def _worst_probes(
    ref: Reference,
    probe_ids: list[str],
    E: npt.NDArray[np.float64],
    B: npt.NDArray[np.float64],
    quant: str,
    threshold: float,
) -> list[str]:
    pair = _prepare(E, B, quant, ref.info.dimension)
    if pair is None:
        return []
    cos = row_cosine(pair[0], pair[1])
    order = np.argsort(cos)
    return [f"{probe_ids[i]} ({cos[i]:.4f})" for i in order[:4] if cos[i] < threshold]


def _differences(
    ref: Reference,
    exp: Expectation,
    best: VariantScore,
    normalized: str,
    d_e: int,
    exp_dim: int,
    quant: str,
    ratio: float,
) -> list[str]:
    """Sentences for every property where observed differs from expected."""
    out: list[str] = []
    name = POOLING_NAMES
    if best.pooling != exp.pooling:
        out.append(
            f"pooling: endpoint uses {name.get(best.pooling, best.pooling)} pooling, "
            f"expected {name.get(exp.pooling, exp.pooling)} pooling"
        )
    if best.config != exp.config:
        seen = describe_config(best.config, ref) or "with the model's default input handling"
        want = describe_config(exp.config, ref) or "the model's default input handling"
        out.append(f"input handling: endpoint encodes text {seen}; expected {want}")
    want_norm = "yes" if exp.normalize else "no"
    if normalized == "truncated-not-renormalised":
        out.append(
            "normalisation: vectors were normalised at full size and then cut to "
            f"{d_e} dimensions without renormalising (norm is about {ratio:.3f} of the "
            "un-normalised variant)"
        )
    elif normalized in ("yes", "no") and normalized != want_norm:
        if normalized == "no":
            out.append("normalisation: endpoint returns un-normalised vectors, expected unit norm")
        else:
            out.append(
                "normalisation: endpoint returns unit-norm vectors, expected un-normalised output"
            )
    elif normalized == "unknown":
        out.append(
            "normalisation: vector norms match neither the normalised nor the un-normalised "
            f"reference (median ratio to un-normalised {ratio:.4f})"
        )
    if quant == "packed-binary":
        out.append(
            f"quantisation: endpoint returns packed binary vectors ({d_e} bytes for "
            f"{ref.info.dimension} dimensions), expected float32"
        )
    else:
        if best.config != DENSE_OFF and d_e != exp_dim:
            out.append(
                f"dimension: endpoint returns {d_e} dimensions (the first {d_e} of "
                f"{ref.info.dimension}), expected {exp_dim}"
            )
        if quant != "float32":
            out.append(f"quantisation: endpoint values are {quant}, expected float32")
    return out
