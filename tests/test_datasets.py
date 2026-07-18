import json

import numpy as np

from motion.data.dataset_la import _box_to_tokens
from motion.data.dataset_sam3 import _rle_encode


def _rle_decode(rle):
    counts = rle["counts"]
    h, w = rle["size"]
    flat = np.zeros(h * w, np.uint8)
    idx, val = 0, 0
    for c in counts:
        flat[idx : idx + c] = val
        idx += c
        val ^= 1
    return flat.reshape((h, w), order="F")


def test_rle_roundtrip_random():
    rng = np.random.default_rng(0)
    for shape in [(4, 5), (1, 1), (3, 3), (7, 2), (2, 7)]:
        for _ in range(20):
            m = (rng.random(shape) > 0.5).astype(np.uint8)
            rle = _rle_encode(m)
            assert sum(rle["counts"]) == m.size
            assert np.array_equal(m, _rle_decode(rle))


def test_rle_roundtrip_edges():
    for m in [np.ones((3, 3), np.uint8), np.zeros((3, 3), np.uint8)]:
        assert np.array_equal(m, _rle_decode(_rle_encode(m)))
    # starts-with-foreground (the previously-buggy case)
    m = np.array([[1, 0], [0, 0]], np.uint8)
    assert np.array_equal(m, _rle_decode(_rle_encode(m)))


def test_la_box_tokens_normalize():
    # full-width/height box → 0..1000
    assert _box_to_tokens([0, 0, 100, 200], 100, 200) == "<box>0 0 1000 1000</box>"
    # clamps out-of-range
    tok = _box_to_tokens([-10, -10, 500, 500], 100, 100)
    assert json.dumps(tok)  # serializable
    assert "<box>0 0 1000 1000</box>" == tok
