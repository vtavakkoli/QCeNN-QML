"""Diffusion logic tests: independent channel oracle and quantum circuit checks."""
import numpy as np
import pytest
from qcenn_qml.diffusion.core import (
    VOCAB, MASK, dataset, conditional_distribution, context_number,
    expected_clean_probability, counts_to_distribution
)


def test_corpus_and_conditional_model():
    train, test = dataset(seed=42)
    assert len(train) == 36 and len(test) == 12
    assert len(VOCAB) == 4 and VOCAB[MASK] == "[MASK]"
    t = conditional_distribution(train)
    assert t.shape == (16,4)
    assert np.all(t >= 0)
    np.testing.assert_allclose(t.sum(axis=1),1)
    assert np.all(t[:,MASK]==0)


def test_replacement_channel_oracle():
    distribution = np.array([0.6,0.2,0.2,0])
    assert expected_clean_probability(0,distribution,1,0.5,0)==0
    assert expected_clean_probability(0,distribution,1,1,2)==pytest.approx(0.6)
    assert expected_clean_probability(0,distribution,1,0.5,2)==pytest.approx(0.45)
    assert expected_clean_probability(0,distribution,0,0.5,0)==1
    with pytest.raises(ValueError):
        expected_clean_probability(0,distribution,2,0.5,1)


def test_binary_counts():
    np.testing.assert_allclose(counts_to_distribution({"00":2,"11":6},8),
                               [0.25,0,0,0.75])


def test_quantum_endpoints_and_conditioning():
    pytest.importorskip("qiskit")
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    from qcenn_qml.diffusion.core import denoising_circuit
    train,_=dataset(42)
    table=conditional_distribution(train)
    left,target,right=0,2,0
    uncorrected=denoising_circuit(left,target,right,table=table,beta=1,steps=0)
    repaired=denoising_circuit(left,target,right,table=table,beta=1,eta=1,steps=1)
    backend=AerSimulator(method="matrix_product_state",max_parallel_threads=1)
    result=backend.run(transpile([uncorrected,repaired],backend,optimization_level=0),
                       shots=1024,seed_simulator=17).result()
    assert result.success
    assert result.get_counts(0)=={"11":1024}
    frequencies=counts_to_distribution(result.get_counts(1),1024)
    assert frequencies[MASK]==0
    np.testing.assert_allclose(frequencies,table[context_number(left,right)],atol=.07)
