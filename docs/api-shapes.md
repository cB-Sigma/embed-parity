# API shapes and where they come from

The client in `src/embed_parity/client.py` was written from the documentation listed here,
read in November 2026. "Documented" means the field appears in that source; it does not
mean this project has exercised it against the real server. See the README support table
for what has actually been run.

## Embeddings

### OpenAI-style `POST /v1/embeddings` (`--api openai`)

Request sent: `{"model": ..., "input": [text, ...]}` plus `encoding_format` and
`dimensions` when asked for. Response read: `data[].embedding` (ordered by `data[].index`),
`usage.prompt_tokens`. A string `embedding` is decoded as base64 little-endian float32.

| server | documented | source |
|---|---|---|
| vLLM | `input` (string, list of strings, token ids), `model`, `encoding_format` (`float` default, `base64`), `dimensions` (Matryoshka models only). Extra fields: `truncate_prompt_tokens`, `truncation_side`, `add_special_tokens` (default true), `use_activation` (normalisation; the older `normalize` parameter was removed), `embed_dtype`, `endianness`. Response has `usage.prompt_tokens` and `usage.total_tokens`. Pooling can be overridden with `--pooler-config '{"pooling_type": "MEAN"}'`. | <https://docs.vllm.ai/en/latest/models/pooling_models/embed/> |
| Text Embeddings Inference | `input`, `model`, `encoding_format` (`float`, `base64`), `dimensions`, `user`. Response requires `usage.prompt_tokens` and `usage.total_tokens`. No `truncate` field on this route. | <https://huggingface.github.io/text-embeddings-inference/openapi.json> |
| SGLang | `model`, `input`, `encoding_format: "base64"` (little-endian float32). Decoder-style embedding models need `--is-embedding`. `dimensions` and `usage` are not described on the page. | <https://docs.sglang.io/docs/basic_usage/openai_api_embeddings> |
| llama.cpp `llama-server` | `input` (string or list), `encoding_format: "float"` in the examples. "Requires that the model uses a pooling different than type `none`"; embeddings "are normalized using the Eucledian norm". `base64` and `dimensions` are not mentioned. | <https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md> |
| Ollama | `model`, `input` (string or list of strings; token arrays unsupported), `encoding_format`, `dimensions`. | <https://docs.ollama.com/api/openai-compatibility> |
| Infinity (Python server) | Same shape at `/embeddings` (no `/v1`): `input`, `model`, `encoding_format` (`float`, `base64`), `dimensions`, `modality`. Response has `usage.prompt_tokens`, `usage.total_tokens`. Use `--api infinity`. | <https://github.com/michaelfeil/infinity/blob/main/docs/assets/openapi.json> |

### TEI `POST /embed` (`--api tei`)

Request sent: `{"inputs": [text, ...]}`. Response read: a bare JSON array of vectors.

Documented fields: `inputs`, `normalize` (default `true`), `truncate` (default `false`),
`truncation_direction` (`Left`/`Right`, default right), `prompt_name`, `dimensions`.
No token usage is returned. With `truncate` false, over-length input is rejected, which
`embed-parity` reports as "rejected" rather than as a failure. Source: the OpenAPI
document above.

### Ollama `POST /api/embed` (`--api ollama`)

Request sent: `{"model": ..., "input": [text, ...]}`. Response read: `embeddings`,
`prompt_eval_count`.

Documented fields: `model`, `input`, `truncate` (default `true`; if false an over-length
input is an error), `dimensions`, `options`, `keep_alive`. The page does not say whether
vectors are normalised. Source: <https://docs.ollama.com/api/embed>.

## Rerank

### Cohere/Jina-style (`--rerank-api cohere`, default route `/v1/rerank`)

Request sent: `{"model": ..., "query": ..., "documents": [text, ...]}`. Response read:
`results[].index` and `results[].relevance_score`; `usage` if present.

| server | documented | source |
|---|---|---|
| vLLM | Routes `/rerank`, `/v1/rerank`, `/v2/rerank`. `query`, `documents`, `top_n`, plus `truncate_prompt_tokens`, `use_activation` and others. Response: `results[]` with `index`, `document.text`, `relevance_score`, sorted by relevance; `usage.total_tokens`. | <https://docs.vllm.ai/en/latest/models/pooling_models/scoring/> |
| Infinity (Python server) | Route `/rerank`. `query`, `documents`, `return_documents`, `raw_scores` (default false), `top_n`, `model`. Response: `results[].relevance_score`, `results[].index`, `usage`. Use `--rerank-api infinity`. | OpenAPI document above |
| llama.cpp `llama-server` | Routes `/reranking`, `/rerank`, `/v1/rerank`, `/v1/reranking`. Sending `documents` selects the Jina response shape, sending `texts` the TEI shape. The score "is the match probability in `[0, 1]`". Needs `--reranking` or `--embedding --pooling rank`. | llama.cpp server README above |

### TEI `POST /rerank` (`--rerank-api tei`)

Request sent: `{"query": ..., "texts": [text, ...]}`. Response read: a bare array of
`{"index", "score"}`.

Documented fields: `query`, `texts`, `raw_scores` (default `false`), `return_text`,
`truncate`, `truncation_direction`. Source: the TEI OpenAPI document above.

## Not verified

- OpenAI's own API reference was not consulted; "OpenAI-style" here means the shape the
  servers above document.
- How each server chooses pooling and normalisation defaults for a given model, beyond the
  sentences quoted above.
- Any behaviour of these servers at run time. None of them was run for this project except
  where the README says "tested".
