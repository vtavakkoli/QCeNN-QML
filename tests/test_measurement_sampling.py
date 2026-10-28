"""Born-distribution equivalence and wide readout, not model quality claims."""
import gzip
import json

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qcenn_qml.coherent import execute
from qcenn_qml.measurement_sampling import full_readout, project


def test_final_full_readout_preserves_nontrivial_marginal_and_native_cx(monkeypatch):
    circuit = QuantumCircuit(5, 4)
    circuit.ry(.73, 0); circuit.h(2); circuit.cx(0, 1)
    circuit.cry(.42, 2, 4); circuit.cx(1, 4); circuit.rz(.51, 4)
    state = Statevector.from_instruction(circuit)
    expected = state.probabilities([4, 0])
    circuit.measure(4, 0); circuit.measure(0, 3)
    monkeypatch.setenv('QCENN_MPS_READOUT', 'full_marginal')
    counts, native, costs = execute([circuit], 8192, 42)
    actual = np.zeros(4)
    for bits, n in counts[0].items():
        value = int(bits, 2)
        assert value & 6 == 0  # Original unassigned classical bits remain zero.
        actual[(value & 1) + 2 * ((value >> 3) & 1)] += n / 8192
    assert actual == pytest.approx(expected, abs=.025)
    assert costs['cx_shot_weighted_total'] == 8192 * native[0].count_ops()['cx']
    assert costs['physical_measurements_total'] == 5 * 8192
    assert costs['readout_instruments'][0]['instrument'].startswith('final_full')


def test_wide_241_qubit_readout_samples_without_a_dense_probability_vector(monkeypatch):
    circuit = QuantumCircuit(241, 4)
    circuit.h(0)
    for index in range(1, 241):
        circuit.cx(index - 1, index)
    circuit.measure(240, 0); circuit.measure(0, 3)
    monkeypatch.setenv('QCENN_MPS_READOUT', 'full_marginal')
    counts, _, costs = execute([circuit], 128, 42)
    assert set(counts[0]) == {'0000', '1001'}
    assert sum(counts[0].values()) == 128
    assert costs['qubits_max'] == 241 and costs['mps_sampling_algorithm'] == 'mps_probabilities'


def test_extra_measurements_preserve_declared_readout_noise_and_audit(monkeypatch, tmp_path):
    from qcenn_qml import measurement_sampling
    monkeypatch.setattr(measurement_sampling, '_AUDIT_WRITTEN', False)
    monkeypatch.setenv('QCENN_PROGRESS_FILE', str(tmp_path / 'execution-progress.json'))
    monkeypatch.setenv('QCENN_MPS_READOUT', 'full_marginal')
    circuit = QuantumCircuit(8, 1)
    circuit.x(7); circuit.measure(7, 0)
    circuit.metadata = {'output_qubits': [7]}
    counts, _, costs = execute([circuit], 4096, 42, readout_error=.1)
    assert counts[0]['1'] / 4096 == pytest.approx(.9, abs=.025)
    audit = costs['readout_instruments'][0]['first_full_readout_audit']
    with gzip.open(audit, 'rt') as stream:
        saved = json.load(stream)
    projected = {}
    mapping = {int(bit): qubit for bit, qubit in saved['old_bit_to_measured_qubit'][0].items()}
    for bits, count in saved['full_counts'][0].items():
        key = project(bits, mapping, 1)
        projected[key] = projected.get(key, 0) + count
    assert projected == counts[0] == saved['original_counts'][0]
    assert json.loads((tmp_path / 'execution-progress.json').read_text())['completed_measurements'] == 1


def test_intermediate_or_overwritten_measurements_keep_original_instrument():
    circuit = QuantumCircuit(2, 1)
    circuit.measure(0, 0); circuit.x(1); circuit.measure(1, 0)
    original, mapping = full_readout(circuit)
    assert original is circuit and mapping is None
