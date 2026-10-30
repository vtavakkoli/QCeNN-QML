"""Safe GPU method bounds plus optional real-hardware transport qualification."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from qcenn_qml.gpu_dispatch import choose_method


def test_gpu_dispatch_never_uses_a_dense_state_for_the_wide_block():
    profile = {'enabled': True, 'verified_methods': ['statevector', 'tensor_network'],
        'memory_mb_by_id': {'0': 24576}}
    assert choose_method(profile, 22) == 'statevector'
    assert choose_method(profile, 64) == 'tensor_network'
    assert choose_method(profile, 241) is None
    assert choose_method({**profile, 'verified_methods': ['statevector']}, 64) is None
    assert choose_method({**profile, 'enabled': False}, 22) is None


@pytest.mark.skipif(not os.environ.get('QCENN_GPU_PROFILE'), reason='Real CUDA GPU/profile required')
def test_hardware_native_memory_counts_noise_and_transport(monkeypatch, tmp_path):
    from qiskit.quantum_info import Statevector
    from qcenn_qml.coherent import _compile_components, execute
    from qcenn_qml.quantum_memory import memory_circuit
    profile = json.loads(Path(os.environ['QCENN_GPU_PROFILE']).read_text())
    if not profile.get('enabled'):
        pytest.skip('GPU preflight did not verify a device')
    monkeypatch.setenv('QCENN_GPU_MIN_QUBITS', '0')
    monkeypatch.setenv('QCENN_GPU_FRACTION', '1')
    native = _compile_components(memory_circuit([.6, .8], [[1, 0], [0, 1], [1, 1], [1, 2]],
        [0, 1, 2, 3], rounds=2, value_bits=2), 42)
    for gpu_id in profile['verified_gpu_ids']:
        for method in profile['verified_methods']:
            method_profile = tmp_path / (method + '.json')
            method_profile.write_text(json.dumps({**profile, 'verified_gpu_ids': [gpu_id], 'verified_methods': [method]}))
            monkeypatch.setenv('QCENN_GPU_PROFILE', str(method_profile))
            for error in (0., .1):
                counts, _, resources = execute([native], 8192, 42, error, precompiled=True)
                assert resources['device'] == 'GPU' and sum(counts[0].values()) == 8192
                # Independent reference is evaluated AFTER execution, never used to
                # build the model gates, quantum probabilities or readout samples.
                measured = [native.find_bit(i.qubits[0]).index for i in native.data if i.operation.name == 'measure']
                ideal = Statevector.from_instruction(native.remove_final_measurements(inplace=False)).probabilities(measured)
                expected = np.zeros_like(ideal)
                for before, probability in enumerate(ideal):
                    for flip in range(4):
                        hamming = flip.bit_count()
                        expected[before ^ flip] += probability * error**hamming * (1-error)**(2-hamming)
                observed = np.zeros_like(ideal)
                for bits, count in counts[0].items():
                    observed[int(bits.replace(' ', ''), 2)] += count / 8192
                assert observed == pytest.approx(expected, abs=.025)
                assert resources['cx_shot_weighted_total'] == 8192 * native.count_ops()['cx']
