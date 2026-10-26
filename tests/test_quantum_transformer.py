"""Full coherent vectors, clean workspace, causal dependence and operator audit."""
import json

import numpy as np
import pytest

from qcenn_qml.coherent import execute
from qcenn_qml.quantum_transformer import block_circuit, block_unitary, reference_block, _pack


def test_full_block_basis_and_superposition_outputs_match_independent_reference():
    inputs = [[1, -1], [2, 1]]
    other = [[-4, 3], [3, -4]]
    circuit = block_circuit(inputs, superposition=other)
    measurements = [i for i, item in enumerate(circuit.data) if item.operation.name == 'measure']
    assert measurements == list(range(measurements[0], len(circuit.data)))
    assert not any(item.operation.name == 'reset' or getattr(item.operation, 'condition', None) is not None for item in circuit.data)
    counts, _, costs = execute([circuit], 32, 7)
    expected = {_pack(x): _pack(reference_block(x)) for x in (inputs, other)}
    seen = set()
    for bits, count in counts[0].items():
        value = int(bits.replace(' ', ''), 2)
        original = value & 4095
        assert (value >> 12) & 4095 == expected[original]
        assert value >> 24 == 0  # Every work qubit returns to zero.
        seen.add(original)
    assert seen == set(expected)
    assert costs['cx_total_per_execution'] > 0 and costs['qubits_max'] == 241


def test_causal_block_first_token_cannot_depend_on_second_token():
    for second in ([0, 0], [1, -1], [-4, 3], [3, 3]):
        assert reference_block([[1, -1], second])[0] == reference_block([[1, -1], [0, 0]])[0]
    assert len({tuple(reference_block([[1, -1], x])[1]) for x in ([0, 0], [1, -1], [-4, 3])}) > 1


def test_block_preserves_coherence_for_functional_phase_kickback():
    from qiskit import QuantumCircuit
    left, right = [[1, -1], [2, 1]], [[-2, 1], [1, -1]]
    unitary = block_unitary()
    circuit = block_circuit(left, superposition=right).remove_final_measurements(inplace=False)
    prep_size = len(circuit.data) - len(unitary.data)
    prepare = QuantumCircuit(unitary.num_qubits)
    for item in circuit.data[:prep_size]:
        prepare.append(item.operation, [circuit.find_bit(q).index for q in item.qubits])
    changed_output = _pack(reference_block(left)) ^ _pack(reference_block(right))
    phase_bit = next(bit for bit in range(12) if changed_output >> bit & 1)
    circuit.z(unitary.metadata['output_qubits'][phase_bit])
    circuit.compose(unitary.inverse(), inplace=True)
    circuit.compose(prepare.inverse(), inplace=True)
    circuit.measure_all()
    counts, _, _ = execute([circuit], 8, 91)
    input_difference = _pack(left) ^ _pack(right)
    pivot = next(bit for bit in range(12) if input_difference >> bit & 1)
    assert counts == [{format(1 << pivot, f'0{unitary.num_qubits}b'): 8}]


def test_scalar_function_tables_are_data_independent_and_no_full_block_lookup_exists():
    metadata = block_unitary().metadata
    names = {row['name'] for row in metadata['universal_scalar_functions']}
    assert names == {'inverse_rms_epsilon_0.25', 'two_entry_scaled_softmax_weight', 'silu'}
    assert all(not row['input_data_dependent'] and row['input_bits'] <= 7
        for row in metadata['universal_scalar_functions'])
    with pytest.raises(MemoryError):
        block_circuit([[0, 0], [0, 0]], max_qubits=20)
    with pytest.raises(ValueError):
        block_circuit([[0, 0], [4, 0]])


def test_block_builder_cannot_use_classical_transformer_predictions(monkeypatch):
    import qcenn_qml.quantum_transformer as module
    monkeypatch.setattr(module, 'reference_block', lambda *args: pytest.fail('CPU reference used during circuit construction'))
    assert module.block_circuit([[1, 0], [-1, 1]]).metadata['classical_hidden_state_updates'] is False


def test_smol_audit_keeps_full_precision_and_pretrained_claims_separate(tmp_path):
    from qcenn_qml.quantum_transformer import OPERATORS
    from qcenn_qml.smol_feasibility import audit
    report = tmp_path / 'block.json'
    report.write_text(json.dumps({'experiment': 'coherent_fixed_point_transformer_block',
        'status': 'VERIFIED', 'operators': OPERATORS,
        'cases': [{'correct_output_fraction': 1., 'clean_workspace_fraction': 1.}]}))
    result = audit(transformer_report=str(report), output_dir=str(tmp_path / 'audit'))
    assert result['complete_small_coherent_block_verified']
    assert all(result['tiny_block_operator_coverage'].values())
    assert not result['coverage']['actual_pretrained_weight_loading']
    assert not result['coverage']['full_precision_qkv_and_output_projections']
    assert result['full_model_quantum_conversion_test'] == 'NOT_IMPLEMENTED_NOT_DEMONSTRATED'
    damaged = json.loads(report.read_text()); damaged['cases'][0]['clean_workspace_fraction'] = .5
    report.write_text(json.dumps(damaged))
    with pytest.raises(ValueError, match='verified complete'):
        audit(transformer_report=str(report), output_dir=str(tmp_path / 'bad'))
