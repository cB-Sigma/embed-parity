# Validation runs

Everything on this page was run on a 4-core CPU machine in November 2026. No GPU was
available, so **vLLM, Text Embeddings Inference, SGLang, Infinity, Ollama and llama.cpp's
own `llama-server` were not run**. Output is pasted unedited apart from shortened paths.

Versions: Python 3.12, sentence-transformers 6.3.0, transformers 5.22.2, torch 2.15.0+cpu,
tokenizers 0.23.3, numpy 2.5.4, llama-cpp-python 0.3.36.

## 1. Building references

```text
$ embed-parity reference --model sentence-transformers/all-MiniLM-L6-v2 --out ref-minilm.json
wrote ref-minilm.json (272 KiB)
sentence-transformers/all-MiniLM-L6-v2 (sentence-transformers, revision 1110a243fdf4): declares mean pooling, normalised, max 256 tokens, 384 dimensions
self-check ok: declared variant recomputed from token states vs library encode(): cosine min 1.0000000, max|diff| 1.49e-07
variants: 20 tables over 5 input configs

$ embed-parity reference --model cross-encoder/ms-marco-MiniLM-L6-v2 --kind rerank --out ref-ce.json
wrote ref-ce.json (12 KiB)
cross-encoder/ms-marco-MiniLM-L6-v2 (sentence-transformers): cross-encoder, declares identity activation
self-check ok: identity(logits) vs library predict(): max|diff| 0.00e+00
```

## 2. Variant maths against sentence-transformers itself

`tests/test_real_model.py` rebuilds the pipeline with a different `Pooling` module (CLS,
mean, max, last token), changes `max_seq_length` (128, 512) and the tokenizer's truncation
side, encodes the probes through sentence-transformers, and compares with the variants
stored in the reference (cosine above 0.999999 required). It also checks token counts
against the tokenizer and runs the diagnosis on real CLS, truncated and float16 vectors.

```text
$ EMBED_PARITY_REAL_MODEL=1 pytest tests/test_real_model.py -q
.........                                                                [100%]
9 passed
```

## 3. A real model behind an independent server

`examples/transformers_server.py` implements tokenisation, pooling and normalisation on
plain `transformers`, without sentence-transformers.

### Correct server

```text
$ python examples/transformers_server.py --rerank-model cross-encoder/ms-marco-MiniLM-L6-v2 &
$ embed-parity check --base-url http://127.0.0.1:8088 --reference ref-minilm.json
embed-parity 0.1.0
endpoint   http://127.0.0.1:8088  api=openai
reference  sentence-transformers/all-MiniLM-L6-v2 (sentence-transformers, revision 1110a243fdf4): declares mean pooling, normalised, max 256 tokens, 384 dimensions

== Parity with the reference (24 probes, one per request) ==
  probe        cosine    norm ratio  max|diff|  tokens (endpoint/ref)
  s-two        1.000000  1.0000      1.60e-07   4/4
  d-tides      1.000000  1.0000      1.34e-07   39/39
  d-sourdough  1.000000  1.0000      1.49e-07   45/45
  q-sourdough  1.000000  1.0000      1.12e-07   12/12
  s-word       1.000000  1.0000      1.30e-07   3/3
  s-question   1.000000  1.0000      1.13e-07   14/14
  (6 of 24 probes shown, lowest cosine first; --verbose for all)
  [PASS] dimension                          384
  [PASS] cosine similarity                  min 1.000000 (s-two), mean 1.000000; all >= 0.999
  [PASS] norm ratio (endpoint / reference)  min 1.0000, max 1.0000; all within 0.01 of 1
  [INFO] max abs difference                 largest 1.60e-07 (no limit set)

== Retrieval ranking (cosine similarity) ==
  query             reference top-1  endpoint top-1  Spearman
  q-tides           d-tides          d-tides         1.0000
  q-sourdough       d-sourdough      d-sourdough     1.0000
  q-python-sort     d-python-sort    d-python-sort   1.0000
  q-photosynthesis  d-chlorophyll    d-chlorophyll   1.0000
  [PASS] rank-order agreement  top-1 agrees on 4 of 4 queries; Spearman mean 1.0000, min 1.0000 (threshold 0.98)

== Token counts from usage ==
  [PASS] token-count parity  usage.prompt_tokens equals the reference tokenizer on all 24 probes

== Diagnosis ==
  endpoint matches mean pooling with normalisation (cosine min 1.00000), which is what the model declares
  [PASS] variant match  the endpoint behaves as declared

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
  [INFO] empty string                              accepted, norm 1.0000
  [INFO] very long input (~5,200 words)            accepted, usage reports 256 tokens
  [PASS] declared dimension                        384
  [PASS] finite, non-zero vectors
  [PASS] unit norm (model declares normalisation)  norms 1.0000 to 1.0000
  [PASS] base64 vs float encoding                  max|diff| 0.00e+00

RESULT: PASS  (0 failed, 0 warnings, 19 passed; 52 requests)
```

### Faults switched on (failing, warning and diagnosis lines only)

`--pooling cls`:

```text
  [FAIL] cosine similarity                  min 0.358228 (u-arabic), mean 0.518351; 24 of 24 probes below 0.999
  [FAIL] rank-order agreement  top-1 agrees on 4 of 4 queries; Spearman mean 0.7381, min 0.5238 (threshold 0.98)
  endpoint matches CLS pooling with normalisation (cosine min 1.00000); the model declares mean pooling with normalisation
    - pooling: endpoint uses CLS pooling, expected mean pooling
  [FAIL] variant match  1 difference(s) from the expected behaviour identified
RESULT: FAIL  (3 failed, 0 warnings, 16 passed; 52 requests)
```

`--no-normalize`:

```text
  [FAIL] norm ratio (endpoint / reference)  min 2.1245, max 8.3517; 24 of 24 probes off by more than 0.01
  endpoint matches mean pooling without normalisation (cosine min 1.00000); the model declares mean pooling with normalisation
    - normalisation: endpoint returns un-normalised vectors, expected unit norm
  [FAIL] variant match  1 difference(s) from the expected behaviour identified
  [FAIL] unit norm (model declares normalisation)  norms 2.6345 to 7.2031
RESULT: FAIL  (3 failed, 0 warnings, 16 passed; 52 requests)
```

`--ignore-mask` (mean pooling over padding positions; note that single-request parity passes):

```text
  endpoint matches mean pooling with normalisation (cosine min 1.00000), which is what the model declares
  [FAIL] batch: with a longer text                 cosine vs single request 0.771037 - output depends on the batch
  [FAIL] batch: 8 mixed texts                      cosine vs single request 0.771037 - output depends on the batch
RESULT: FAIL  (2 failed, 0 warnings, 17 passed; 52 requests)
```

`--max-length 128`:

```text
  [FAIL] cosine similarity                  min 0.962647 (l-ledger), mean 0.998444; 1 of 24 probes below 0.999
  [WARN] token-count parity  1 of 24 probes differ (endpoint/reference: l-ledger 128/256); the counts equal the reference tokenizer truncating at 128 tokens (the model declares 256)
  endpoint matches mean pooling with normalisation, truncating at 128 tokens (the model declares 256) (cosine min 1.00000); the model declares mean pooling with normalisation
    - input handling: endpoint encodes text truncating at 128 tokens (the model declares 256); expected the model's default input handling
  [FAIL] variant match  1 difference(s) from the expected behaviour identified
RESULT: FAIL  (2 failed, 1 warnings, 16 passed; 52 requests)
```

## 4. Rerank

Server returning raw logits (`--rerank-raw`), which is what this model declares:

```text
$ embed-parity check --base-url http://127.0.0.1:8088 --reference ref-ce.json
embed-parity 0.1.0
endpoint   http://127.0.0.1:8088  api=cohere
reference  cross-encoder/ms-marco-MiniLM-L6-v2 (sentence-transformers): cross-encoder, declares identity activation

== Rerank parity with the cross-encoder reference ==
  query             max|diff| vs raw logits
  q-tides           0.0000
  q-sourdough       0.0000
  q-python-sort     0.0000
  q-photosynthesis  0.0000
  [PASS] score parity             endpoint returns raw logits, max|diff| 0.0000 (limit 0.02)
  [PASS] ordering agreement       top-1 agrees on 4 of 4 queries; Spearman mean 1.0000
  [PASS] batch invariance         max|diff| 0.0000 between scoring a document alone and in a list of 8
  [PASS] order invariance         max|diff| 0.0000 after reversing the document list
  [PASS] repeat determinism       bit-identical
  [PASS] results sorted by score

RESULT: PASS  (0 failed, 0 warnings, 6 passed; 14 requests)
```

Server applying a sigmoid:

```text
embed-parity 0.1.0
endpoint   http://127.0.0.1:8088  api=cohere
reference  cross-encoder/ms-marco-MiniLM-L6-v2 (sentence-transformers): cross-encoder, declares identity activation

== Rerank parity with the cross-encoder reference ==
  query             max|diff| vs raw logits
  q-tides           11.2850
  q-sourdough       11.3951
  q-python-sort     11.4594
  q-photosynthesis  11.2880
  endpoint returns sigmoid probabilities (max|diff| 0.0000); the model declares raw logits
  [FAIL] score parity             scores are sigmoid probabilities, expected raw logits
  [PASS] ordering agreement       top-1 agrees on 4 of 4 queries; Spearman mean 1.0000
  [PASS] batch invariance         max|diff| 0.0000 between scoring a document alone and in a list of 8
  [PASS] order invariance         max|diff| 0.0000 after reversing the document list
  [PASS] repeat determinism       bit-identical
  [PASS] results sorted by score

RESULT: FAIL  (1 failed, 0 warnings, 5 passed; 14 requests)
```

## 5. A third-party server: llama-cpp-python

llama-cpp-python 0.3.36 (built from source for CPU), started as
`python -m llama_cpp.server --model all-MiniLM-L6-v2.F32.gguf --embedding true --n_ctx 512`
with the float32 GGUF from `leliuga/all-MiniLM-L6-v2-GGUF` on the Hugging Face Hub. This is
the Python package's server, not the `llama-server` binary from the llama.cpp repository,
whose documentation says it normalises.

```text
$ embed-parity check --base-url http://127.0.0.1:8090 --model minilm --reference ref-minilm.json
embed-parity 0.1.0
endpoint   http://127.0.0.1:8090  api=openai  model=minilm
reference  sentence-transformers/all-MiniLM-L6-v2 (sentence-transformers, revision 1110a243fdf4): declares mean pooling, normalised, max 256 tokens, 384 dimensions

== Parity with the reference (24 probes, one per request) ==
  probe       cosine    norm ratio  max|diff|  tokens (endpoint/ref)
  l-ledger    0.934159  2.3200      2.52e-01   512/256
  u-arabic    0.999997  2.4175      2.27e-01   41/41
  d-waves     0.999999  3.9817      4.95e-01   30/30
  u-japanese  0.999999  5.3334      6.76e-01   16/16
  u-german    0.999999  4.1515      4.97e-01   21/21
  s-two       0.999999  5.7291      1.09e+00   4/4
  (6 of 24 probes shown, lowest cosine first; --verbose for all)
  [PASS] dimension                          384
  [FAIL] cosine similarity                  min 0.934159 (l-ledger), mean 0.997256; 1 of 24 probes below 0.999
  [FAIL] norm ratio (endpoint / reference)  min 2.1232, max 8.3511; 24 of 24 probes off by more than 0.01
  [INFO] max abs difference                 largest 1.29e+00 (no limit set)

== Retrieval ranking (cosine similarity) ==
  query             reference top-1  endpoint top-1  Spearman
  q-tides           d-tides          d-tides         1.0000
  q-sourdough       d-sourdough      d-sourdough     1.0000
  q-python-sort     d-python-sort    d-python-sort   1.0000
  q-photosynthesis  d-chlorophyll    d-chlorophyll   1.0000
  [PASS] rank-order agreement  top-1 agrees on 4 of 4 queries; Spearman mean 1.0000, min 1.0000 (threshold 0.98)

== Token counts from usage ==
  [WARN] token-count parity  1 of 24 probes differ (endpoint/reference: l-ledger 512/256); the counts equal the reference tokenizer truncating at 512 tokens (the model declares 256)

== Diagnosis ==
  endpoint matches mean pooling without normalisation, truncating at 512 tokens (the model declares 256) (cosine min 0.99980); the model declares mean pooling with normalisation
    - input handling: endpoint encodes text truncating at 512 tokens (the model declares 256); expected the model's default input handling
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
  [INFO] empty string                              accepted, norm 6.3667
  [INFO] very long input (~5,200 words)            accepted, usage reports 512 tokens
  [PASS] declared dimension                        384
  [PASS] finite, non-zero vectors
  [FAIL] unit norm (model declares normalisation)  norms 2.9907 to 7.2040
  [WARN] base64 vs float encoding                  openai response: asked for base64 but received list embeddings

RESULT: FAIL  (4 failed, 2 warnings, 13 passed; 52 requests)
```

Reading: the vectors point the same way as the library's for every input that fits in 256
tokens. Two things differ from what the model declares, and both are named: the server
does not L2-normalise, and it embeds up to its 512-token context where the model's
sentence-transformers configuration truncates at 256. It also returned float arrays when
asked for base64. Whether these matter depends on the application: cosine search is
unaffected by the missing normalisation, dot-product search is not.

The same server with the `Q4_K_M` quantisation of the same model:

```text
  [FAIL] cosine similarity                  min 0.932307 (l-ledger), mean 0.988826; 24 of 24 probes below 0.999
  [FAIL] norm ratio (endpoint / reference)  min 2.1722, max 8.4050; 24 of 24 probes off by more than 0.01
  [FAIL] rank-order agreement  top-1 agrees on 4 of 4 queries; Spearman mean 0.9702, min 0.9286 (threshold 0.98)
  [WARN] token-count parity  1 of 24 probes differ (endpoint/reference: l-ledger 512/256); the counts equal the reference tokenizer truncating at 512 tokens (the model declares 256)
  endpoint approximately matches mean pooling without normalisation, truncating at 512 tokens (the model declares 256) (cosine min 0.97952); the model declares mean pooling with normalisation
    - near match only: cosine min 0.97952, mean 0.99113 (an exact match needs 0.9995). This points to the same recipe computed with different numerics - quantised or reduced-precision weights, another runtime, or a slightly different tokenizer - rather than a different recipe
    - input handling: endpoint encodes text truncating at 512 tokens (the model declares 256); expected the model's default input handling
    - normalisation: endpoint returns un-normalised vectors, expected unit norm
  [FAIL] variant match  closest variant is a near match, not an exact one
  [FAIL] unit norm (model declares normalisation)  norms 3.0018 to 7.0983
  [WARN] base64 vs float encoding                  openai response: asked for base64 but received list embeddings
RESULT: FAIL  (5 failed, 2 warnings, 12 passed; 52 requests)
```

## 6. What was not run

- Any GPU server; any of the servers marked "docs only" in the README.
- `infinity-emb` was not installed (its dependency set risked replacing the CPU-only
  torch on a machine with little free disk).
- Models with prompts, Dense modules, last-token pooling or `include_prompt = false`:
  covered by unit tests with the toy encoder and a fake Dense backend only.
- Operating systems other than Linux.
