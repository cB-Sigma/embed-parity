# Reference file format

A reference is one UTF-8 JSON document. Arrays are little-endian binary, base64-encoded,
with their dtype and shape alongside.

```jsonc
{
  "format": "embed-parity/reference",
  "version": 1,
  "meta": {"tool": "embed-parity", "tool_version": "0.1.0", "created": "2026-11-12T20:06:00Z"},

  "model": {                       // what the model declares
    "model_id": "sentence-transformers/all-MiniLM-L6-v2",
    "backend": "sentence-transformers",
    "kind": "embedding",           // or "rerank"
    "revision": "1110a243...",
    "pooling": "mean",             // cls | mean | last | max
    "normalize": true,
    "include_prompt": true,
    "prompts": {"query": "", "document": ""},
    "default_prompt_name": null,
    "max_seq_length": 256,
    "position_limit": 512,
    "truncation_side": "right",
    "hidden_size": 384,
    "dimension": 384,
    "truncate_dim": null,
    "matryoshka_dims": [],
    "similarity_fn": "cosine",
    "modules": ["Transformer", "Pooling", "Normalize"],
    "post_modules": [],            // e.g. ["Dense(768->3072)", "Dense(3072->768)"]
    "activation": null,            // rerank: "sigmoid" | "identity"
    "library_versions": {"sentence_transformers": "6.3.0", "torch": "2.15.0+cpu", "...": "..."}
  },

  "probe_set": {
    "name": "builtin-v1",
    "probes": [{"id": "q-tides", "text": "what causes ocean tides", "tags": ["query", "short"]}],
    "queries": ["q-tides"],
    "documents": ["d-tides"]
  },

  "input_configs": {               // how each variant's input was built
    "default":     {"prompt": "", "prompt_name": null, "add_special_tokens": true,
                    "max_length": 256, "truncation_side": "right"},
    "special=off": {"...": "..."}
  },

  "library":  {"dtype": "float32", "shape": [24, 384], "data": "<base64>"},

  "variants": {                    // key: "<input config>/<pooling>", un-normalised
    "default/mean":    {"probe_ids": ["q-tides", "..."], "dtype": "float16", "shape": [24, 384], "data": "..."},
    "maxlen=128/mean": {"probe_ids": ["l-ledger"],       "dtype": "float16", "shape": [1, 384],  "data": "..."}
  },

  "tokens":      {"default": {"q-tides": 7}, "special=off": {"q-tides": 5}},
  "full_tokens": {"l-ledger": 1021},  // token count before truncation

  "selfcheck": {"status": "ok", "cosine_min": 1.0, "max_abs_diff": 1.5e-07,
                "description": "declared variant recomputed from token states vs library encode()"},

  "rerank": {                      // only for kind = "rerank"
    "query_ids": ["q-tides"], "document_ids": ["d-tides"],
    "logits":  {"dtype": "float32", "shape": [4, 8], "data": "..."},
    "library": {"dtype": "float32", "shape": [4, 8], "data": "..."}
  }
}
```

Notes:

- `library` is exactly what the reference library returned with default arguments. It is
  the vector an endpoint is compared with unless `--expect` changes the expectation, in
  which case the expected vector is derived from `variants`.
- A variant table may cover only some probes (`probe_ids`). Rows for other probes are taken
  from `default/<pooling>`.
- `--no-variants` omits `variants`; parity still works, diagnosis does not.
- `--variant-dtype float32` doubles the size of `variants` and is rarely needed.
- Readers must reject a `version` higher than they understand. Fields may be added without
  changing the version; unknown fields in `model` are ignored.

## Size

Measured with the built-in probe set: 272 KiB for a 384-dimensional model (20 variant
tables). Size grows linearly with the embedding dimension.
