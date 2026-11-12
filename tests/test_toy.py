from __future__ import annotations

import numpy as np
import pytest

from embed_parity.backends import InputConfig, pool
from embed_parity.toy import (
    CLS,
    SEP,
    TOY_MODELS,
    ToyEncoder,
    ToyReranker,
    _hash_uniform,
    is_toy,
    load_toy,
    tokenize,
)


def test_tokenizer_is_deterministic_and_splits_long_words() -> None:
    assert tokenize("Hello world") == tokenize("hello   WORLD")
    assert len(tokenize("lighthouse")) == 2  # "light" + "##house"
    assert len(tokenize("a, b")) == 3
    assert tokenize("") == []


def test_build_ids_special_tokens_and_truncation() -> None:
    enc = ToyEncoder()
    text = " ".join(f"w{i}" for i in range(100))
    right, full = enc.build_ids(text, InputConfig(max_length=10))
    left, _ = enc.build_ids(text, InputConfig(max_length=10, truncation_side="left"))
    assert full == 102
    assert len(right) == len(left) == 10
    assert right[0] == left[0] == CLS
    assert right[-1] == left[-1] == SEP
    assert right[1:-1] == tokenize(text)[:8]
    assert left[1:-1] == tokenize(text)[-8:]
    bare, _ = enc.build_ids("a b", InputConfig(add_special_tokens=False))
    assert CLS not in bare
    assert SEP not in bare
    assert len(enc.build_ids("", InputConfig(add_special_tokens=False))[0]) == 1


def test_values_are_platform_stable() -> None:
    # Guards the README demo output: parameters come from integer hashing only.
    vec = ToyEncoder().encode(["Hello world"])[0]
    assert vec.shape == (64,)
    assert float(np.linalg.norm(vec)) == pytest.approx(1.0, abs=1e-6)
    assert [round(float(v), 4) for v in vec[:4]] == pytest.approx(
        [-0.1294, -0.1012, 0.0125, -0.2551], abs=2e-4
    )
    assert _hash_uniform((3,), 5).tolist() == [
        -0.7873066168655751,
        -0.3453484715637485,
        -0.6522642680806343,
    ]


def test_masked_batch_equals_single() -> None:
    enc = ToyEncoder()
    a, b = [CLS, 10, 11, SEP], [CLS, 20, 21, 22, 23, 24, SEP]
    alone = enc.forward([a])[0]
    batched = enc.forward([a, b])[0]
    assert np.allclose(alone, batched, atol=1e-6)
    leaked = enc.forward([a, b], respect_mask=False)[0]
    assert leaked.shape[0] == len(b)  # padding rows are kept
    assert not np.allclose(alone, leaked[: len(a)], atol=1e-3)


def test_position_limit_enforced() -> None:
    enc = ToyEncoder()
    with pytest.raises(ValueError, match="position limit"):
        enc.forward([[5] * 200])


@pytest.mark.parametrize("mode", ["cls", "mean", "last", "max"])
def test_pooling_modes(mode: str) -> None:
    states = np.arange(12, dtype=np.float32).reshape(4, 3)
    out = pool(states, mode)
    expected = {
        "cls": states[0],
        "mean": states.mean(axis=0),
        "last": states[-1],
        "max": states.max(axis=0),
    }[mode]
    assert np.allclose(out, expected)


def test_pool_start_and_errors() -> None:
    states = np.arange(12, dtype=np.float32).reshape(4, 3)
    assert np.allclose(pool(states, "mean", start=2), states[2:].mean(axis=0))
    with pytest.raises(ValueError, match="unknown pooling"):
        pool(states, "median")
    with pytest.raises(ValueError, match="cannot pool"):
        pool(np.zeros((0, 3)), "mean")


def test_encode_honours_declared_config() -> None:
    enc = ToyEncoder(TOY_MODELS["toy-embed-raw"])
    out = enc.encode(["a b c", "d"])
    assert not np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-3)
    assert np.allclose(np.linalg.norm(enc.encode(["a b c"], normalize=True), axis=1), 1.0)


def test_reranker_prefers_overlap() -> None:
    rr = ToyReranker()
    logits = rr.logits([("ocean tides", "tides of the ocean"), ("ocean tides", "bread recipe")])
    assert logits[0] > logits[1]
    probs = rr.library_scores([("ocean tides", "tides of the ocean")])
    assert probs is not None
    assert 0 < probs[0] < 1
    assert rr.describe().activation == "sigmoid"


def test_load_toy() -> None:
    assert isinstance(load_toy("toy:toy-embed"), ToyEncoder)
    assert isinstance(load_toy("toy-rerank"), ToyReranker)
    assert is_toy("toy-embed")
    assert not is_toy("sentence-transformers/all-MiniLM-L6-v2")
    with pytest.raises(KeyError, match="unknown toy model"):
        load_toy("toy:nope")
