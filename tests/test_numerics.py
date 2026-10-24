import numpy as np
import pytest
from qcenn_qml.quantum import unit_vector, grover_response
from qcenn_qml.mnist import image_amplitudes
from qcenn_qml.smol135 import (
    tokenize, create_vocabulary, encode_tokens,
    build_memory_amplitudes, context_integer,
)


def test_nonzero_and_power_of_two():
    assert np.linalg.norm(unit_vector([3, 4, 0, 0])) == pytest.approx(1)
    with pytest.raises(ValueError):
        unit_vector([0, 0, 0, 0])


def test_grover_polynomial():
    for p in (0, 0.1, 0.3, 0.5, 1):
        assert grover_response(p, 0) == pytest.approx(p)
    assert grover_response(0.25, 1) == pytest.approx(1)


def test_mnist_amplitude_shape():
    img = np.arange(28*28, dtype=np.uint8).reshape(28, 28)
    result = image_amplitudes(img)
    assert result.shape == (64,)
    assert np.linalg.norm(result) == pytest.approx(1)


def test_memory_normalization_and_context():
    tok = tokenize("the cat the dog the cat the dog")
    vocab = create_vocabulary(tok, 4)
    state, bits, transitions = build_memory_amplitudes(tok, vocab, order=1, smoothing=0)
    assert state.shape == (1 << (2*bits),)
    assert np.linalg.norm(state) == pytest.approx(1)
    assert transitions > 0
    code = context_integer(encode_tokens(["the"], vocab), bits)
    p = state.reshape(1 << bits, 1 << bits)[code]**2
    assert p[encode_tokens(["cat"], vocab)[0]] == pytest.approx(p[encode_tokens(["dog"], vocab)[0]])


def test_empty_memory_rejected():
    with pytest.raises(ValueError):
        build_memory_amplitudes(["cat"], ["<unk>", "cat"], smoothing=0)
