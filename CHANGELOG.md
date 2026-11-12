# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-11-12

First release.

### Added

- `embed-parity reference`: compute a portable reference file with sentence-transformers on
  CPU, recording the model's declared configuration, the library's output for a 24-probe
  set, token counts, and variants (CLS/mean/last-token/max pooling; each declared prompt
  prefix or none; with and without special tokens; truncation from the other side and at
  other lengths; with and without a Dense projection). Cross-encoders are supported with
  `--kind rerank`.
- `embed-parity check`: per-probe cosine similarity, norm ratio and maximum absolute
  difference against the reference; rank-order agreement on a small retrieval task;
  `usage.prompt_tokens` parity; exit codes for CI; JSON and Markdown reports.
- Diagnosis: identifies which variant an endpoint matches, and separately its
  normalisation, Matryoshka-style dimension truncation and float16/int8/binary
  quantisation. Near matches are reported as approximate; no match is reported as such.
- Black-box invariance checks that need no reference: repeat determinism, batch
  composition (duplicate, equal-length, shorter, longer, mixed), order, concurrent
  requests, whitespace, empty and very long input, dimension, unit norm, finite values,
  base64 versus float encoding.
- Rerank parity: scores against raw logits and sigmoid(logits) with detection of which is
  returned, ordering agreement, batch and order invariance.
- API shapes: OpenAI-style `/v1/embeddings`, TEI `/embed`, Ollama `/api/embed`, Infinity
  `/embeddings`; rerank in Cohere/Jina shape (`documents` -> `results[].relevance_score`)
  and TEI shape (`texts` -> `[{index, score}]`).
- A bundled simulated server with a deterministic toy encoder and 17 injectable faults
  (`embed-parity demo`, `embed-parity serve-sim`), used by the test-suite to show that the
  diagnosis names each injected fault.
- `examples/transformers_server.py`, a small real-model test server with switchable
  pooling, normalisation, masking and truncation.

### Known limitations

- Text inputs only; image and other modalities are on the roadmap.
- Not tested against vLLM, TEI, SGLang, Infinity, Ollama or `llama-server`: their request
  and response shapes were taken from documentation. See the README support table.

[Unreleased]: https://github.com/cb-Sigma/embed-parity/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/cb-Sigma/embed-parity/releases/tag/v0.1.0
