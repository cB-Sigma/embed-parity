# embed-parity

**Does your embedding server return the vectors the model is supposed to produce? If not, which step differs?**

[![CI](https://github.com/cb-Sigma/embed-parity/actions/workflows/ci.yml/badge.svg)](https://github.com/cb-Sigma/embed-parity/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fcb-Sigma%2Fembed-parity%2Fmain%2Fpyproject.toml)](pyproject.toml)

An embedding or rerank server can answer every request with HTTP 200 and still return
vectors that differ from what sentence-transformers produces for the same model: the wrong
pooling mode, a missing normalisation step, a prompt prefix that was never applied, an
attention mask that is skipped for some batches, a different truncation length. Nothing
errors. Retrieval quality drops, and the cause is hard to see from the outside.

`embed-parity` sends a small probe set to the endpoint, compares the result with the
reference library, and when they differ it works out **which implementation choice the
endpoint actually made**:

```text
endpoint matches CLS pooling without normalisation (cosine min 1.00000); the model declares mean pooling with normalisation
```

Checking an endpoint needs only `numpy`. The reference is a small JSON file (about 270 KiB
for a 384-dimensional model) that you compute once on a laptop CPU and reuse in CI.

## What it looks like

This is real output. `embed-parity demo` starts a bundled simulated server with two
injected faults and runs the normal `check` code path against it over HTTP. The simulator
serves a deterministic toy encoder, not a language model, so the similarity values and the
"top-1" documents below carry no meaning beyond showing the report; see
[Validation](#validation) for runs against a real model.

<!-- output: embed-parity demo -->
```text
demo: simulated server, injected faults: pooling=cls, no-normalize

embed-parity 0.1.0
endpoint   http://127.0.0.1:<port> (simulator)  api=openai  model=toy-embed
reference  toy-embed (toy): declares mean pooling, normalised, default prompt 'document', max 48 tokens, 64 dimensions

== Parity with the reference (24 probes, one per request) ==
  probe       cosine     norm ratio  max|diff|  tokens (endpoint/ref)
  u-japanese  -0.074367  5.0802      1.16e+00   10/10
  u-chinese   -0.040043  5.0045      1.13e+00   12/12
  s-question  -0.005500  4.7386      1.16e+00   20/20
  u-emoji     0.134052   4.8001      1.18e+00   20/20
  s-code      0.182682   4.6568      1.05e+00   23/23
  d-tides     0.189375   4.3599      1.05e+00   48/48
  (6 of 24 probes shown, lowest cosine first; --verbose for all)
  [PASS] dimension                          64
  [FAIL] cosine similarity                  min -0.074367 (u-japanese), mean 0.285619; 24 of 24 probes below 0.999
  [FAIL] norm ratio (endpoint / reference)  min 4.3107, max 5.1368; 24 of 24 probes off by more than 0.01
  [INFO] max abs difference                 largest 1.18e+00 (no limit set)

== Retrieval ranking (cosine similarity) ==
  query             reference top-1  endpoint top-1  Spearman
  q-tides           d-python-dict    d-yeast         0.6667
  q-sourdough       d-sourdough      d-autumn        -0.4286
  q-python-sort     d-python-dict    d-python-dict   0.4286
  q-photosynthesis  d-yeast          d-chlorophyll   0.7619
  [FAIL] rank-order agreement  top-1 agrees on 1 of 4 queries; Spearman mean 0.3571, min -0.4286 (threshold 0.98)

== Token counts from usage ==
  [PASS] token-count parity  usage.prompt_tokens equals the reference tokenizer on all 24 probes

== Diagnosis ==
  endpoint matches CLS pooling without normalisation (cosine min 1.00000); the model declares mean pooling with normalisation
    - pooling: endpoint uses CLS pooling, expected mean pooling
    - normalisation: endpoint returns un-normalised vectors, expected unit norm
  [FAIL] variant match  2 difference(s) from the expected behaviour identified

== Invariance (black-box, no reference needed) ==
  [PASS] repeat determinism                        bit-identical across 3 requests
  [PASS] batch: same text twice                    cosine vs single request 1.000000
  [PASS] batch: two texts of equal length          cosine vs single request 1.000000
  [PASS] batch: with a shorter text                cosine vs single request 1.000000
  [PASS] batch: with a longer text                 cosine vs single request 1.000000
  [PASS] batch: 8 mixed texts                      cosine vs single request 1.000000
  [PASS] order invariance                          cosine min 1.000000 between a batch and its reverse
  [PASS] concurrent requests (6 at once)           cosine vs sequential request 1.000000
  [PASS] leading/trailing whitespace               cosine vs trimmed text 1.000000
  [INFO] empty string                              accepted, norm 5.1980
  [INFO] very long input (~5,200 words)            accepted, usage reports 48 tokens
  [PASS] declared dimension                        64
  [PASS] finite, non-zero vectors
  [FAIL] unit norm (model declares normalisation)  norms 4.3463 to 5.1707
  [PASS] base64 vs float encoding                  max|diff| 0.00e+00

RESULT: FAIL  (5 failed, 0 warnings, 14 passed; 52 requests)
```
<!-- /output -->

Some bugs do not show when texts are sent one at a time. The invariance checks need no
reference at all; here the simulated server ignores the attention mask in padded batches:

<!-- output: embed-parity demo --fault batch-mask --skip parity,ranking,tokens,diagnosis -->
```text
demo: simulated server, injected faults: batch-mask

embed-parity 0.1.0
endpoint   http://127.0.0.1:<port> (simulator)  api=openai  model=toy-embed
reference  toy-embed (toy): declares mean pooling, normalised, default prompt 'document', max 48 tokens, 64 dimensions

== Invariance (black-box, no reference needed) ==
  [PASS] repeat determinism                        bit-identical across 3 requests
  [PASS] batch: same text twice                    cosine vs single request 1.000000
  [PASS] batch: two texts of equal length          cosine vs single request 1.000000
  [PASS] batch: with a shorter text                cosine vs single request 1.000000
  [FAIL] batch: with a longer text                 cosine vs single request 0.320440 - output depends on the batch
  [FAIL] batch: 8 mixed texts                      cosine vs single request 0.320440 - output depends on the batch
  [PASS] order invariance                          cosine min 1.000000 between a batch and its reverse
  [PASS] concurrent requests (6 at once)           cosine vs sequential request 1.000000
  [PASS] leading/trailing whitespace               cosine vs trimmed text 1.000000
  [INFO] empty string                              accepted, norm 1.0000
  [INFO] very long input (~5,200 words)            accepted, usage reports 48 tokens
  [PASS] declared dimension                        64
  [PASS] finite, non-zero vectors
  [PASS] unit norm (model declares normalisation)  norms 1.0000 to 1.0000
  [PASS] base64 vs float encoding                  max|diff| 0.00e+00

RESULT: FAIL  (2 failed, 0 warnings, 11 passed; 52 requests)
```
<!-- /output -->

## Install

```bash
pip install git+https://github.com/cb-Sigma/embed-parity
```

That is enough to check endpoints against an existing reference file. To *compute* a
reference from a model you also need sentence-transformers and torch:

```bash
# optional: CPU-only torch first, to avoid downloading the CUDA wheels
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "embed-parity[reference] @ git+https://github.com/cb-Sigma/embed-parity"
```

> **Not on PyPI.** This project is not published to PyPI. A different, unrelated project is
> registered there under the name `embed-parity` (see [Related work](#related-work)), so
> `pip install embed-parity` installs that one, not this one. Install from Git as shown above.

Python 3.10 or newer.

## Quick start

### 1. Make a reference (once, on CPU)

```bash
embed-parity reference --model sentence-transformers/all-MiniLM-L6-v2 --out minilm.ref.json
```

```text
wrote minilm.ref.json (272 KiB)
sentence-transformers/all-MiniLM-L6-v2 (sentence-transformers, revision 1110a243fdf4): declares mean pooling, normalised, max 256 tokens, 384 dimensions
self-check ok: declared variant recomputed from token states vs library encode(): cosine min 1.0000000, max|diff| 1.49e-07
variants: 20 tables over 5 input configs
```

The file records what the model declares (pooling module, normalisation, prompts,
`max_seq_length`, truncation side, similarity function, library versions, revision), the
library's own output for 24 probes, and the same probes recomputed under other
implementation choices. `embed-parity inspect minilm.ref.json` prints its contents.
Commit it next to your deployment configuration.

### 2. Check an endpoint

```bash
embed-parity check --base-url http://localhost:8000 --model my-served-name --reference minilm.ref.json
```

`--base-url` is the server root. The default API shape is OpenAI-style `POST /v1/embeddings`;
use `--api tei`, `--api ollama` or `--api infinity` for the others
([server support](#server-support)). Useful options:

| option | purpose |
|---|---|
| `--model-id ID` | compute the reference on the spot instead of loading a file (needs the extra) |
| `--expect pooling=cls,normalize=false` | you deployed differently on purpose: check against that instead |
| `--expect prompt=query` | the server is configured to apply the model's `query` prompt |
| `--dimensions 256` | send `dimensions: 256` and expect the reference truncated and renormalised |
| `--extra-body '{"truncate": true}'` | merge fields into every request |
| `--api-key`, `--header NAME=VALUE` | authentication (`EMBED_PARITY_API_KEY` also works) |
| `--skip invariance` | skip sections: `parity`, `ranking`, `tokens`, `diagnosis`, `invariance`, `rerank` |
| `--json report.json`, `--markdown summary.md` | machine-readable report, CI job summary |

Without `--reference` or `--model-id` only the black-box invariance checks run, which is
still worth doing against any endpoint.

### 3. Use it in CI

Exit code `0` means every check passed, `1` means at least one failed, `2` means the run
itself could not complete (endpoint unreachable, bad arguments, unreadable reference).

```yaml
- name: Embedding parity
  run: |
    pip install git+https://github.com/cb-Sigma/embed-parity
    embed-parity check --base-url "$EMBEDDINGS_URL" --model "$SERVED_MODEL" \
      --reference ci/minilm.ref.json --markdown "$GITHUB_STEP_SUMMARY"
```

A complete workflow is in [examples/github-actions.yml](examples/github-actions.yml).

### Rerankers

```bash
embed-parity reference --model cross-encoder/ms-marco-MiniLM-L6-v2 --kind rerank --out reranker.ref.json
embed-parity check --base-url http://localhost:8000 --model my-reranker --reference reranker.ref.json
```

Scores are compared with the cross-encoder's raw logits and with `sigmoid(logits)`, so the
report says which of the two the endpoint returns, alongside ordering agreement and batch
and order invariance. `--rerank-api cohere` (default; `documents` in, `results[].relevance_score`
out) and `--rerank-api tei` (`texts` in, `[{index, score}]` out) are supported.

<!-- output: embed-parity demo --rerank -->
```text
demo: simulated server, injected faults: rerank-raw

embed-parity 0.1.0
endpoint   http://127.0.0.1:<port> (simulator)  api=cohere  model=toy-rerank
reference  toy-rerank (toy): cross-encoder, declares sigmoid activation

== Rerank parity with the cross-encoder reference ==
  query             max|diff| vs sigmoid probabilities
  q-tides           3.4977
  q-sourdough       3.0510
  q-python-sort     5.4198
  q-photosynthesis  3.3812
  endpoint returns raw logits (max|diff| 0.0000); the model declares sigmoid probabilities
  [FAIL] score parity             scores are raw logits, expected sigmoid probabilities
  [PASS] ordering agreement       top-1 agrees on 4 of 4 queries; Spearman mean 1.0000
  [PASS] batch invariance         max|diff| 0.0000 between scoring a document alone and in a list of 8
  [PASS] order invariance         max|diff| 0.0000 after reversing the document list
  [PASS] repeat determinism       bit-identical
  [PASS] results sorted by score

RESULT: FAIL  (1 failed, 0 warnings, 5 passed; 14 requests)
```
<!-- /output -->

Many cross-encoders declare *no* activation while servers apply a sigmoid by default. That
is a legitimate deployment choice; tell the tool with `--expect activation=sigmoid`.

## The failure modes

These are the classes of silent divergence the tool is built around, with the upstream
reports that motivated them. Each summary below was checked against the linked issue in
November 2026.

| failure mode | example | what catches it |
|---|---|---|
| **Wrong pooling.** A server falls back to an architecture default when it cannot read the model's pooling config. | [vllm-project/vllm#45995](https://github.com/vllm-project/vllm/issues/45995): models saved with sentence-transformers >= 5.4.0 use a compact `1_Pooling/config.json` that vLLM did not parse, so mean-pooled BERT encoders were served with CLS pooling (cosine about 0.89 against the library) with no warning. Closed as fixed. | parity + diagnosis (`endpoint uses CLS pooling, expected mean pooling`) |
| **Missing post-pooling module.** A Dense projection or Normalize step is not applied. | Same issue: for `google/embeddinggemma-300m` the output was close to orthogonal to the library's (cosine about 0.02), consistent with the Dense projector not being applied. | parity + diagnosis (`without the Dense projection`, `un-normalised vectors`) |
| **Attention mask skipped for some batches.** Output depends on what else is in the batch. | [huggingface/text-embeddings-inference#882](https://github.com/huggingface/text-embeddings-inference/issues/882): Qwen3 and Gemma3 candle backends skipped masking when every sequence in a batch had the same length; a real-model reproduction gave cosine 0.14 between a text embedded alone and in an equal-length batch. Padded batches were unaffected. Closed as fixed. | invariance: `batch: same text twice`, `batch: two texts of equal length`, `concurrent requests` |
| **Input construction differs.** Chat template, special tokens or generation prompt differ from the library. | [vllm-project/vllm#33167](https://github.com/vllm-project/vllm/issues/33167) (open): Qwen3-VL-Embedding results from vLLM and offline Transformers disagreed (reported cosine from about 0.60 to 0.95 depending on request settings). Commenters traced parts of it to `add_special_tokens` / `add_generation_prompt` settings and, in one analysis, to which hidden state Transformers returns; the thread records no maintainer-confirmed root cause. | parity + diagnosis for text inputs (`without special tokens`, prompt variants). Image inputs are [not covered yet](#roadmap). |
| **Tokenizer surface differs.** | [huggingface/text-embeddings-inference#525](https://github.com/huggingface/text-embeddings-inference/issues/525): the `/tokenize` route returned the right token ids but mismatched token *text* for a Chinese input. Ids were correct, so embeddings were unaffected; it is listed here as an example of tokenizer-level drift between a server and Transformers. Closed as fixed. | token-count parity from `usage` (counts only; token text is not compared) |
| **Multimodal preprocessing or position handling differs.** | [sgl-project/sglang#35772](https://github.com/sgl-project/sglang/issues/35772): Qwen3-VL vision features in SGLang v0.5.17 diverged from Transformers and vLLM on a grounding task (0 or 1 of 20 hits against 20 of 20). A commenter attributed it to multimodal rotary position handling and the reporter confirmed that a backported patch fixed one of the two models. This is a generation model, not an embedding endpoint; it is cited as evidence that the same class of silent divergence exists for image inputs. | not covered: [roadmap](#roadmap) |

Other things the checks look for, without a specific upstream report behind them: a
dropped or doubled prompt prefix, truncation at a different length or from the other
side, Matryoshka truncation without renormalisation, float16/int8/binary output where
float32 is expected, base64 payloads that disagree with the float encoding, and
non-deterministic output.

## How the diagnosis works

When `reference` runs, it does not only call `model.encode()`. It also takes the model
apart: tokenizer, transformer, the modules after pooling. That lets it recompute the same
probes under other choices:

- **pooling**: CLS, mean, last-token, max
- **prompt**: each declared prompt prefix applied, or none
- **special tokens**: with and without
- **truncation**: from the other side, at half the declared length, at the model's position limit
- **post-pooling**: with and without the Dense projection, if the model has one

Every variant is stored un-normalised. Three properties can then be read independently
from the endpoint's vectors:

1. **Direction** (cosine similarity is unaffected by scaling) identifies the pooling mode
   and input handling: the variant with the highest minimum cosine over all probes wins,
   and it counts as a match only above 0.9995 on every probe.
2. **Norm** identifies normalisation: unit norm, the variant's raw norm, or the tell-tale
   norm of a vector that was normalised and then cut without renormalising.
3. **Length and value set** identify Matryoshka truncation (the endpoint vector is compared
   with the leading dimensions of each variant) and float16 / int8 / binary quantisation.

Because pooling is stored for every input configuration, combined faults (say, CLS pooling
*and* a missing prefix *and* truncated dimensions) are identified together. If the best
variant is close but not exact (cosine between 0.95 and 0.9995) the report says
"approximately matches" and points at numerics (quantised weights, another runtime) rather
than at a recipe. If nothing comes close it says so and lists the probes that fit worst,
instead of guessing.

As a guard, the declared variant recomputed this way is compared with `model.encode()`
when the reference is built (the *self-check* line above). If they disagree, the file says
so and the report repeats the warning.

More detail: [docs/how-diagnosis-works.md](docs/how-diagnosis-works.md). File format:
[docs/reference-format.md](docs/reference-format.md).

## The checks

**Parity** (needs a reference). Each probe is sent in its own request so batching cannot
interfere: cosine similarity, norm ratio and largest absolute difference per probe; rank
agreement on a small retrieval task (4 queries, 8 documents: top-1 agreement and Spearman
correlation of scores, using the model's declared similarity function); and
`usage.prompt_tokens` against the reference tokenizer. Token mismatches are explained when
they line up with a variant ("the counts equal the reference tokenizer without special
tokens").

**Invariance** (no reference needed):

| check | what it catches |
|---|---|
| repeat determinism | sampling, races, uninitialised state |
| same text alone vs twice in a batch, vs with a same-length text | masks skipped for unpadded batches (TEI #882) |
| same text with a shorter text, a longer text, in a mixed batch of 8 | padding leaking into attention or pooling |
| batch order reversed | position-dependent batching bugs |
| 6 concurrent single requests | server-side request coalescing that behaves differently from client batches |
| leading/trailing whitespace | tokenizer stripping differences (warning only) |
| empty string, about 5,200 words | whether the server accepts, truncates or rejects; a 5xx is a failure |
| dimension, finite values, unit norm | wrong size, NaN, missing normalisation |
| base64 vs float | `encoding_format: "base64"` decoding to different numbers |

**Rerank** (needs a reference): score parity against logits and sigmoid(logits), ordering,
one document alone vs in a list, reversed list, determinism, result ordering.

## Thresholds

| option | default | meaning |
|---|---|---|
| `--min-cosine` | `0.999` | every probe's cosine with the reference |
| `--norm-tolerance` | `0.01` | allowed deviation of `norm(endpoint) / norm(reference)` from 1 |
| `--max-abs-diff` | off | largest allowed component difference (reported either way) |
| `--min-rank-correlation` | `0.98` | mean Spearman correlation on the retrieval task; top-1 must also agree |
| `--invariance-min-cosine` | `0.9999` | same text, different batch or request |
| `--token-mismatch` | `warn` | `warn`, `fail` or `ignore` differing `usage.prompt_tokens` |
| `--rerank-max-diff` | `0.02` | largest allowed rerank score difference |
| `--strict` | off | warnings fail the run |

Choosing them:

- **Same weights, float32, different runtime.** Expect cosine above 0.99999. In the
  [validation runs](#validation) a transformers server and sentence-transformers agreed to
  1.6e-07 per component. The defaults are loose for this case; `--min-cosine 0.9999
  --max-abs-diff 1e-4` is reasonable.
- **float16 / bf16 weights on GPU.** Typically 0.9999 or better; the defaults should pass.
  This is an expectation, not something measured in this project (no GPU server was tested).
- **Quantised weights (GGUF Q4/Q8, int8 ONNX).** Expect lower: the one measurement made
  here, a Q4_K_M GGUF of all-MiniLM-L6-v2, gave cosine 0.98 to 0.99 on inputs within the
  context window and Spearman 0.97 on the retrieval task. Decide what you accept, then set `--min-cosine` just below what
  you measure, so that the check guards against *change* rather than against quantisation.
- **Wrong recipe.** Wrong pooling or a missing prefix lands far lower (roughly 0.4 to 0.9 in
  the runs here), so a threshold anywhere between 0.95 and 0.999 separates "numerics" from
  "recipe". Missing normalisation does not change cosine at all, which is why the norm ratio
  is checked separately.

Thresholds are operational defaults, not constants of nature. The report prints every
measured value so you can set your own.

## Server support

"Docs" means the request and response shape was taken from the project's current
documentation or OpenAPI document (sources and dates in
[docs/api-shapes.md](docs/api-shapes.md)). "Tested" means `check` was run end to end
against that software. **No GPU server was available while building this; vLLM, TEI,
SGLang, Infinity, Ollama and llama.cpp's own `llama-server` have not been tested.** If you
run it against one of them, please [report the result](CONTRIBUTING.md), working or not.

| server | embeddings | rerank | status |
|---|---|---|---|
| vLLM | `--api openai` (`/v1/embeddings`) | `--rerank-api cohere` (`/v1/rerank`) | docs only |
| Text Embeddings Inference | `--api tei` (`/embed`) or `--api openai` (`/v1/embeddings`) | `--rerank-api tei` (`/rerank`) | docs only |
| Infinity (Python server) | `--api infinity` (`/embeddings`) | `--rerank-api infinity` (`/rerank`) | docs only |
| SGLang | `--api openai` | n/a | docs only |
| llama.cpp `llama-server` | `--api openai` | `--rerank-api cohere` or `tei` | docs only |
| Ollama | `--api ollama` (`/api/embed`) or `--api openai` | n/a | docs only |
| llama-cpp-python server 0.3.36 | `--api openai` | n/a | **tested** on CPU with a GGUF of all-MiniLM-L6-v2 |
| transformers test server ([examples/](examples/transformers_server.py)) | `--api openai` | `--rerank-api cohere` | **tested** on CPU with real models |
| bundled simulator | all four shapes | all three shapes | **tested** in the test-suite |

Server-specific notes:

- **TEI** rejects over-length input unless `truncate` is set; pass
  `--extra-body '{"truncate": true}'` to compare truncation behaviour, or leave it off and the
  long probe is reported as rejected rather than failed. `/embed` does not report token usage.
- **vLLM** documents `dimensions` for Matryoshka models only and exposes
  `truncate_prompt_tokens`, `add_special_tokens` and `use_activation` as extra request
  fields; all can be set with `--extra-body`.
- **Infinity** publishes its routes without a `/v1` prefix; if your deployment mounts them
  elsewhere, include the prefix in `--base-url` or use `--path`.
- **Ollama** `/api/embed` truncates by default (`truncate: true`).
- The `--path` and `--rerank-path` options override the route for anything else that speaks
  one of these shapes.

## Validation

What was actually run while developing this (CPU only, November 2026; commands and full
output in [docs/validation.md](docs/validation.md)):

- **Variant maths on a real model.** For `sentence-transformers/all-MiniLM-L6-v2`
  (sentence-transformers 6.3.0, transformers 5.22.2, torch 2.15.0+cpu) every stored variant
  was recomputed through sentence-transformers itself, by rebuilding the pipeline with a
  different `Pooling` module, changing `max_seq_length` and flipping the tokenizer's
  truncation side, and matched to cosine above 0.999999. This is
  `tests/test_real_model.py` (9 tests, opt-in because it downloads weights).
- **A correct server passes.** A server written directly on `transformers` (no
  sentence-transformers) serving the same model: 19 of 19 checks passed, largest component
  difference 1.6e-07.
- **Injected faults in that real server are named.** `--pooling cls`: "endpoint matches CLS
  pooling with normalisation". `--no-normalize`: "mean pooling without normalisation".
  `--max-length 128`: "truncating at 128 tokens (the model declares 256)". `--ignore-mask`:
  single-request parity passes and `batch: with a longer text` fails at cosine 0.77.
- **A third-party server.** llama-cpp-python's OpenAI-compatible server 0.3.36 with a
  community float32 GGUF of the same model: directions match the library (cosine 0.999997 or
  better on short inputs), and the report correctly identifies two differences from what
  the model declares: vectors are not normalised, and long inputs are truncated at the
  server's 512-token context rather than the model's 256. It also ignored
  `encoding_format: "base64"`. With a Q4_K_M GGUF the diagnosis becomes "approximately
  matches" at cosine 0.98.
- **Rerank.** `cross-encoder/ms-marco-MiniLM-L6-v2` behind the test server: raw logits pass
  (the model declares no activation); with sigmoid applied the report says "endpoint returns
  sigmoid probabilities; the model declares raw logits".
- **Simulator.** 143 tests in the default suite, including one per injectable fault asserting that the diagnosis
  names that fault.

## Limitations

- **Text only.** Image, audio and video inputs are not checked.
- **Model architectures.** The reference builder handles sentence-transformers models that
  are `Transformer -> Pooling -> (Dense ...) -> (Normalize)` with CLS, mean, last-token or
  max pooling. Static embeddings, sparse (SPLADE) and late-interaction (ColBERT) models,
  weighted-mean or attention pooling, and models whose first module is custom code are
  refused with a clear message. Only one real model has been validated; models with
  prompts, Dense layers, `include_prompt = false` or last-token pooling are exercised only
  through the toy encoder and unit tests.
- **Prompts.** The probe is sent as plain text and compared with what `model.encode(text)`
  returns, which applies the model's *default* prompt only. If your server or client adds a
  query or document prefix, say so with `--expect prompt=NAME`. Task-specific instructions
  that are not declared in the model's `prompts` are outside the variant space.
- **Variants are single deviations in input handling.** Pooling, normalisation, dimension
  and quantisation combine freely with one input-handling deviation; two input-handling
  deviations at once (for example no special tokens *and* left truncation) are reported as
  "no variant matches".
- **Quantisation detection** is based on the values the endpoint returns (all float16-exact,
  all small integers, all 0/1). int8 comparison uses cosine against the float reference, not
  a calibrated re-quantisation.
- **Equal-length batches** are approximated by reordering the words of a sentence, which
  gives the same token count for common tokenizers but is not guaranteed for all.
- **Concurrency.** Six simultaneous requests may or may not be coalesced into one batch by
  a given server; a pass does not prove the coalescing path is correct.
- **24 probes** are a smoke test of the recipe, not a benchmark. Passing does not prove
  equivalence on your data; failing does prove a difference.
- **Reference library versions matter.** The reference reflects the sentence-transformers
  and transformers versions it was built with; they are recorded in the file.

## FAQ

**Why not just compare cosine with a script?** That tells you *that* something differs.
The hours go into finding out *what*. A cosine of 0.89 could be pooling, a prefix or
truncation; the variant match separates them in one run.

**Does a pass mean my deployment is correct?** It means the endpoint reproduces the
library on these probes, alone and in batches. It does not cover your tokenizer edge cases,
your documents' length distribution or your client's own preprocessing.

**The check fails but I configured the server this way on purpose.** Use `--expect`, for
example `--expect normalize=false` or `--expect pooling=last`. The report then verifies the
behaviour you intended.

**My model needs `query:` / `passage:` prefixes.** If the prefixes are declared in the
model's sentence-transformers config, the reference contains a variant for each. Run once
with `--expect prompt=query` if the server applies it, or leave the default if the client is
responsible for prefixes.

**Can I use my own texts?** Yes: `embed-parity probes > probes.jsonl`, edit, then
`embed-parity reference --probes probes.jsonl ...`. Tag lines `query` / `document` to form
the retrieval task. See [examples/probes.jsonl](examples/probes.jsonl).

**How big is a reference file?** About 270 KiB for a 384-dimensional model with variants,
roughly proportional to the dimension. `--no-variants` stores only the declared output
(about 60 KiB) and disables diagnosis.

**Does it need a GPU or network access?** No GPU. Building a reference downloads the model
from the Hugging Face Hub unless it is cached; checking only talks to your endpoint.

**Is the simulated server useful for anything else?** `embed-parity serve-sim --fault ...`
gives you a deliberately broken embeddings endpoint for testing your own monitoring.

## Related work

- **[`embed-parity` on PyPI](https://pypi.org/project/embed-parity/)** (unrelated to this
  repository despite the name) compares sentence-transformers with a TEI deployment:
  cosine and norm statistics, pairwise geometry, nearest-neighbour overlap, controlled
  token lengths and concurrent requests, and it reproduced TEI #882. It is TEI-specific and
  loads the reference model at comparison time; it reports that a prefix "may" explain a
  difference for known model families. If you run TEI, look at it.
- **Per-model parity repositories** such as
  [rkishore/dotnet-embedding-parity](https://github.com/rkishore/dotnet-embedding-parity)
  (three .NET libraries against a sentence-transformers reference for all-MiniLM-L6-v2),
  [jpe90/candle-pytorch-parity-testing](https://github.com/jpe90/candle-pytorch-parity-testing)
  and [polaria-tech/zerank-2-reranker-vllm](https://github.com/polaria-tech/zerank-2-reranker-vllm)
  measure one model on one stack. [jerrythomas/rust-embedding-bench](https://github.com/jerrythomas/rust-embedding-bench)
  validates each backend against a sentence-transformers reference before benchmarking it.
- **Snapshot drift tools** such as
  [bafnayeygian/VectorDriftCheck](https://github.com/bafnayeygian/VectorDriftCheck) compare
  two sets of stored embeddings.
- **OpenAI-compatible conformance suites** such as
  [saylee21/openai-gateway-conformance](https://github.com/saylee21/openai-gateway-conformance)
  and [aitk-org/LLMConform](https://github.com/aitk-org/LLMConform) target LLM gateways and
  chat-style APIs. None of the ones found in this search compare embedding values with a
  reference library.
- **Upstream regression tests.** Servers add their own parity tests as bugs are found (the
  fix discussed in vLLM #45995 came with regression tests and parity validation). Those
  cover the models in the server's own test matrix, not your model, revision and flags.

What this project adds is the offline reference file, coverage of several API shapes, and
the variant-based diagnosis. This list comes from a brief search of GitHub, PyPI and the
web in November 2026 and is certainly incomplete; corrections are welcome.

## Roadmap

- Test against real vLLM, TEI, Infinity, SGLang, Ollama and `llama-server` deployments and
  record the results in the support table.
- Multimodal embeddings (image inputs): reference via the model's processor, with
  resize/normalisation/patching variants. Deliberately left out of this release rather than
  shipped half-done.
- TEI `/tokenize` and vLLM `/tokenize`: compare token ids, not only counts.
- Sparse and late-interaction outputs (`/embed_sparse`, ColBERT-style multi-vector).
- More variants: weighted-mean pooling, pairs of input-handling deviations, calibrated int8.
- A medium-length probe band so truncation at several lengths is separated by more than
  one probe.
- Publication on a package index under a name that does not collide.

## Contributing

Bug reports with a `--json` report attached are the most useful contribution, especially
from servers in the "docs only" rows above. See [CONTRIBUTING.md](CONTRIBUTING.md) for the
development setup and [SECURITY.md](SECURITY.md) for reporting vulnerabilities.

## Independence

This is an independent project. It is not affiliated with, endorsed by or sponsored by
Hugging Face, the vLLM, SGLang, llama.cpp, Ollama, Infinity or Text Embeddings Inference
projects, OpenAI, Cohere or Jina AI. Product names are used only to describe compatibility.

## License

[Apache-2.0](LICENSE). Copyright 2026 The embed-parity Authors.
