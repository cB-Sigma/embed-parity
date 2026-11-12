from __future__ import annotations

import numpy as np
import pytest

from embed_parity import mathx


def test_cosine_and_norms() -> None:
    a = np.array([[1.0, 0.0], [1.0, 1.0]])
    b = np.array([[2.0, 0.0], [1.0, -1.0]])
    assert mathx.row_cosine(a, b) == pytest.approx([1.0, 0.0])
    assert mathx.norms(a) == pytest.approx([1.0, np.sqrt(2)])
    assert mathx.norms(mathx.normalize(b)) == pytest.approx([1.0, 1.0])
    assert mathx.cosine(a[0], b[0]) == pytest.approx(1.0)


def test_normalize_keeps_zero_rows() -> None:
    assert np.all(mathx.normalize(np.zeros((1, 3))) == 0)


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        mathx.row_cosine(np.ones((1, 2)), np.ones((1, 3)))
    with pytest.raises(ValueError, match="shape mismatch"):
        mathx.max_abs_diff(np.ones(2), np.ones(3))


def test_rankdata_ties() -> None:
    assert mathx.rankdata(np.array([10.0, 20.0, 10.0, 30.0])).tolist() == [1.5, 3.0, 1.5, 4.0]


def test_spearman() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert mathx.spearman(x, x**3) == pytest.approx(1.0)
    assert mathx.spearman(x, -x) == pytest.approx(-1.0)
    assert mathx.spearman(np.ones(3), np.ones(3)) == 1.0
    assert mathx.spearman(np.array([1.0]), np.array([5.0])) == 1.0


def test_sigmoid_is_stable() -> None:
    out = mathx.sigmoid(np.array([-1000.0, 0.0, 1000.0]))
    assert out.tolist() == [0.0, 0.5, 1.0]


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
def test_array_roundtrip(dtype: str) -> None:
    x = np.linspace(-1, 1, 12).reshape(3, 4)
    back = mathx.decode_array(mathx.encode_array(x, dtype))
    assert back.shape == (3, 4)
    assert np.allclose(back, x, atol=1e-3 if dtype == "float16" else 1e-7)


def test_array_decode_rejects_bad_input() -> None:
    obj = mathx.encode_array(np.ones((2, 2)))
    with pytest.raises(ValueError, match="shape"):
        mathx.decode_array({**obj, "shape": [3, 3]})
    with pytest.raises(ValueError, match="dtype"):
        mathx.decode_array({**obj, "dtype": "int8"})
    with pytest.raises(ValueError, match="dtype"):
        mathx.encode_array(np.ones(2), "int8")


def test_base64_embedding() -> None:
    import base64

    vec = np.array([0.25, -1.5, 3.0], dtype="<f4")
    assert mathx.decode_base64_embedding(base64.b64encode(vec.tobytes()).decode()).tolist() == [
        0.25,
        -1.5,
        3.0,
    ]
    with pytest.raises(ValueError, match="multiple"):
        mathx.decode_base64_embedding(base64.b64encode(b"abcde").decode())


def test_detect_quantization() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 64)).astype(np.float32).astype(np.float64)
    assert mathx.detect_quantization(x) == "float32"
    assert mathx.detect_quantization(x.astype(np.float16).astype(np.float64)) == "float16"
    assert mathx.detect_quantization(np.round(x * 40)) == "int8"
    assert mathx.detect_quantization(np.round(x * 40) + 128) == "uint8"
    assert mathx.detect_quantization((x > 0).astype(float)) == "binary"
    assert mathx.detect_quantization(np.where(x > 0, 1.0, -1.0)) == "binary"
    packed = np.packbits(x > 0, axis=1).astype(np.float64)
    assert mathx.detect_quantization(packed, full_dim=64) == "packed-binary"
    assert mathx.detect_quantization(np.array([[np.nan, 1.0]])) == "float32"


def test_to_signs_unpacks_bits() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(3, 16))
    packed = np.packbits(x > 0, axis=1).astype(np.float64)
    assert np.array_equal(mathx.to_signs(packed, "packed-binary", 16), mathx.to_signs(x, "float"))
    signed = packed.copy()
    signed[signed > 127] -= 256  # int8 view of the same bytes
    assert np.array_equal(mathx.to_signs(signed, "packed-binary", 16), mathx.to_signs(x, "float"))
