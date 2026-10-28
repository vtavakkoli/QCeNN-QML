"""Standalone Aer GPU transport: native QPY in, real shot strings/counts out.

This file runs in a separate GPU-only Python environment, with no project
installation or CPU/GPU Aer wheel overlap. It does not construct model scores.
"""
import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def packages():
    return {name: version(name) for name in ('qiskit', 'qiskit-aer-gpu', 'numpy')}


def probe(output):
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    backend = AerSimulator()
    answer = {'python': sys.executable, 'packages': packages(),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'devices': list(backend.available_devices()), 'methods': list(backend.available_methods()),
        'verified_gpu_ids': [], 'verified_methods': [], 'memory_mb_by_id': {}, 'errors': []}
    if 'GPU' in answer['devices']:
        info = subprocess.run(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10, check=True)
        memory = [int(value.strip()) for value in info.stdout.splitlines() if value.strip().isdigit()]
        visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        count = len(visible.split(',')) if visible and visible not in ('-1', 'none') else len(memory)
        if visible in ('-1', 'none', ''):
            count = 0
        circuit = QuantumCircuit(2, 2)
        circuit.h(0); circuit.cx(0, 1); circuit.measure(range(2), range(2))
        for device in range(count):
            try:
                result = AerSimulator(method='statevector', device='GPU', target_gpus=[device],
                    max_parallel_threads=1, max_parallel_experiments=1, max_parallel_shots=1).run(
                        circuit, shots=128, seed_simulator=42).result()
                counts = result.get_counts()
                if not result.success or sum(counts.values()) != 128 or set(counts) != {'00', '11'}:
                    raise RuntimeError('GPU Bell readout failed')
                answer['verified_gpu_ids'].append(device)
                answer['memory_mb_by_id'][str(device)] = min(memory) if memory else 0
            except Exception as error:
                answer['errors'].append({'gpu': device, 'error': str(error)})
        if answer['verified_gpu_ids']:
            answer['verified_methods'].append('statevector')
            if 'tensor_network' in answer['methods']:
                try:
                    result = AerSimulator(method='tensor_network', device='GPU',
                        target_gpus=[answer['verified_gpu_ids'][0]], max_parallel_threads=1).run(
                            circuit, shots=128, seed_simulator=42).result()
                    if result.success and sum(result.get_counts().values()) == 128:
                        answer['verified_methods'].append('tensor_network')
                except Exception as error:
                    answer['errors'].append({'method': 'tensor_network', 'error': str(error)})
    answer['enabled'] = bool(answer['verified_gpu_ids'])
    save(output, answer)


def execute(request_file):
    from qiskit import qpy
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, ReadoutError
    request = json.loads(Path(request_file).read_text())
    if packages() != request['expected_packages']:
        raise ValueError('GPU package versions changed since preflight')
    raw = Path(request['qpy']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != request['qpy_sha256']:
        raise ValueError('GPU native QPY changed in transport')
    with Path(request['qpy']).open('rb') as stream:
        native = qpy.load(stream)
    noise = None
    if request['readout_error']:
        noise = NoiseModel()
        error = request['readout_error']
        for qubit in request['output_qubits']:
            noise.add_readout_error(ReadoutError([[1-error, error], [error, 1-error]]), [qubit])
    backend = AerSimulator(method=request['method'], device='GPU', target_gpus=[request['gpu_id']],
        max_parallel_threads=1, max_parallel_experiments=1, max_parallel_shots=1,
        max_memory_mb=request['memory_limit_mb'], noise_model=noise)
    start = time.perf_counter()
    result = backend.run(native, shots=request['shots'], seed_simulator=request['seed'],
                         memory=request['memory']).result()
    if not result.success:
        raise RuntimeError(str(result.status))
    counts = [dict(result.get_counts(i)) for i in range(len(native))]
    if any(sum(c.values()) != request['shots'] for c in counts):
        raise RuntimeError('GPU returned an incomplete shot budget')
    answer = {'success': True, 'counts': counts,
        'memory': [result.get_memory(i) for i in range(len(native))] if request['memory'] else None,
        'method': request['method'], 'device': 'GPU', 'gpu_id': request['gpu_id'],
        'packages': packages(), 'qpy_sha256': request['qpy_sha256'],
        'sampling_seconds': time.perf_counter()-start}
    save(request['result'], answer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe')
    parser.add_argument('--request')
    parser.add_argument('--run-root')
    options = parser.parse_args()
    try:
        if options.probe:
            probe(options.probe)
        elif options.request:
            execute(options.request)
        else:
            parser.error('Specify --probe or --request')
    except BaseException:
        traceback.print_exc()
        raise
