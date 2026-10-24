import numpy as np
import pytest
pytest.importorskip("qiskit")
from qiskit.quantum_info import Statevector
from qcenn_qml.mnist import fidelity_circuit
from qcenn_qml.smol135 import (
    memory_circuit, build_memory_amplitudes,
    create_vocabulary, tokenize, encode_tokens, context_integer,
)


def state(circuit):
    circuit = circuit.copy()
    circuit.remove_final_measurements()
    return np.asarray(Statevector.from_instruction(circuit).data)


def test_quantum_fidelity_extremes():
    x = np.array([1, 0, 0, 0], dtype=float)
    y = np.array([0, 1, 0, 0], dtype=float)
    assert abs(state(fidelity_circuit(x, x, 1))[0])**2 == pytest.approx(1)
    assert abs(state(fidelity_circuit(x, y, 1))[0])**2 == pytest.approx(0, abs=1e-10)


def test_coherent_polynomial_one_quarter():
    x = np.array([1, 0, 0, 0], dtype=float)
    y = np.array([0.5, np.sqrt(3)/2, 0, 0], dtype=float)
    assert abs(state(fidelity_circuit(x, y, 0))[0])**2 == pytest.approx(.25)
    assert abs(state(fidelity_circuit(x, y, 1))[0])**2 == pytest.approx(1)


def test_context_amplification_preserves_conditional_distribution():
    tokens = tokenize("the cat the dog the cat the dog")
    vocab = create_vocabulary(tokens, 4)
    amp, bits, _ = build_memory_amplitudes(tokens, vocab, smoothing=0)
    context = context_integer(encode_tokens(["the"], vocab), bits)
    rows = []
    for rounds in (0, 1):
        probabilities = abs(state(memory_circuit(amp, bits, 1, context, rounds)))**2
        selected = probabilities.reshape(1 << bits, 1 << bits)[context, :len(vocab)]
        rows.append(selected / selected.sum())
    assert np.allclose(rows[0], rows[1])
