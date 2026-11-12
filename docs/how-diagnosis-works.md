# How the diagnosis works

The aim is to turn "cosine 0.89" into "the endpoint uses CLS pooling where the model
declares mean pooling". This page describes how, and where the method stops working.

## 1. Taking the model apart

`embed-parity reference` loads the model with sentence-transformers and reads what it
declares: the pooling mode, whether a `Normalize` module follows, any `Dense` modules,
prompts and the default prompt, `max_seq_length`, the tokenizer's truncation side, the
similarity function. It then bypasses `model.encode()` and drives the parts directly:

1. tokenize (prompt prefix + text, with or without special tokens, truncated at a given
   length from a given side);
2. run the transformer and keep the hidden state of every real token;
3. pool in numpy (`cls`, `mean`, `last`, `max`);
4. apply the modules that follow pooling, except normalisation.

Run with the declared settings, this must reproduce `model.encode()`. That comparison is
the **self-check**; it is stored in the reference file and printed when the file is built.
For `sentence-transformers/all-MiniLM-L6-v2` the two agree to 1.5e-07 per component.

## 2. Variants

The same four steps are then repeated with one input-handling setting changed at a time:

| input config | change |
|---|---|
| `default` | what the model declares |
| `prompt=NAME` | a declared prompt other than the default is prepended |
| `prompt=none` | the default prompt is left out (only if the model has one) |
| `special=off` | no special tokens |
| `trunc=left` / `trunc=right` | truncate from the other side |
| `maxlen=N` | truncate at half the declared length, and at the model's position limit |
| `dense=off` | pooled vector before the Dense projection (only if the model has one) |

For each config all four pooling modes are stored, so a pooling difference combined with
an input-handling difference is covered. Truncation configs are stored only for probes
that are long enough to be affected; for the others they are identical to `default`.

Vectors are stored **before** normalisation, in float16 (they are only compared by cosine,
where float16 rounding costs about 1e-7).

## 3. Reading the endpoint's vectors

Three properties are independent of one another, so each is read separately.

**Direction.** Cosine similarity ignores scale. For every stored variant the per-probe
cosine with the endpoint is computed, and variants are ranked by their *minimum* over
probes, then mean, then by how few settings differ from the expectation. Ties therefore
resolve towards "as declared": if the endpoint rejected the long probes, truncation
variants cannot be told apart from the default and the default is reported. A variant is a
match only if every probe reaches 0.9995 (0.99 for int8 output, 0.95 for binary).

**Norm.** With the matching variant known, the endpoint's norms are compared with 1 and
with the variant's un-normalised norms:

- all within 0.002 of 1: normalised;
- all within 2% of the variant's raw norm: not normalised;
- equal to `norm(v[:d]) / norm(v)`: normalised at full size, then cut to `d` dimensions
  without renormalising;
- otherwise: reported as unknown scaling, with the median ratio.

**Length and values.** If the endpoint returns fewer dimensions than the model produces,
it is compared with the leading dimensions of each variant (Matryoshka truncation).
Quantisation is detected from the values themselves: every value exactly representable in
float16; all integers in the int8 or uint8 range; all values in {0, 1} or {-1, 1}; or
integers with one eighth of the model's dimension (packed bits). Binary outputs are
compared by sign agreement.

## 4. What is reported

| status | meaning |
|---|---|
| as expected | best variant is the expected one, norms and dimension as expected |
| variant | an exact match with a different variant; each differing property is listed |
| approximate | best variant reaches 0.95 to 0.9995: the recipe is probably right and the numerics differ (quantised weights, a different runtime, a slightly different tokenizer) |
| no match | nothing reaches 0.95; the closest variant and the worst-fitting probes are listed |

The diagnosis never changes the PASS/FAIL verdict of the parity metrics. It fails only
when parity failed, and is a warning when parity passed but a property differs (for
example float16 output).

## 5. Where it stops working

- **Outside the variant space.** Different weights or revision, a different tokenizer
  vocabulary, a custom pooling (weighted mean, attention), an instruction format that is
  not in the model's declared prompts, or two input-handling deviations at once. These end
  as "approximate" or "no match".
- **Variants that coincide.** If the default prompt is empty, "prompt dropped" does not
  exist as a variant. If no probe exceeds the context window for a model (very long
  context), truncation variants are absent. `embed-parity inspect` shows what a file holds.
- **Batch-dependent bugs** are invisible here by design: parity sends one text per
  request. They are the job of the invariance checks.
- **The reference is only as good as the library.** If the reference library is itself
  wrong for a model (as argued in one analysis in vllm-project/vllm#33167), the tool will
  report the server as different. It measures agreement, not truth.
- **Self-check failures.** If the decomposition does not reproduce `model.encode()` for
  some architecture, parity (which uses the library output) is still valid, while the
  variants may not be. The report says so.
