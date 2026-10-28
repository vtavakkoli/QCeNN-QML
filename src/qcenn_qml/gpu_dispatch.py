"""Opportunistic, leased GPU execution for suitable native circuits only."""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time

from .resources import acquire_files, execution_progress, inherited_worker_descriptors, thread_environment


def choose_method(profile, qubits):
    if not profile.get('enabled') or qubits < int(os.environ.get('QCENN_GPU_MIN_QUBITS', '20')):
        return None
    memory = min(profile['memory_mb_by_id'].values()) * 1024 ** 2 * .7
    dense_limit = min(int(os.environ.get('QCENN_GPU_MAX_STATEVECTOR_QUBITS', '30')),
                      math.floor(math.log2(max(16, memory) / 16)))
    if 'statevector' in profile['verified_methods'] and qubits <= dense_limit:
        return 'statevector'
    if 'tensor_network' in profile['verified_methods'] and qubits <= int(os.environ.get('QCENN_GPU_MAX_TENSOR_QUBITS', '128')):
        return 'tensor_network'
    return None  # Never allocate a dense statevector for a wide MPS/block.


class GPUResult:
    def __init__(self, answer, native):
        self.answer = answer
        self.success, self.status = True, 'COMPLETED'
        self.qcenn_backend = {key: answer[key] for key in ('method', 'device', 'gpu_id', 'packages', 'qpy_sha256', 'sampling_seconds')}
        self.qcenn_readout = {'instrument': 'original_partial_readout',
            'mps_sampling_algorithm': None,
            'physical_measurements_per_shot': [sum(i.operation.name == 'measure' for i in c.data) for c in native],
            'execution_depth_max': max(c.depth() for c in native)}

    def get_counts(self, index=0):
        return self.answer['counts'][index]

    def get_memory(self, index=0):
        return self.answer['memory'][index]


def try_execute(native, shots, seed, readout_error, memory=False):
    filename = os.environ.get('QCENN_GPU_PROFILE')
    if not filename or os.environ.get('QCENN_FORCE_CPU') == '1':
        return None
    profile = json.loads(Path(filename).read_text())
    method = choose_method(profile, max(c.num_qubits for c in native))
    if method is None:
        return None
    from .study_utils import circuit_digest
    fraction = float(os.environ.get('QCENN_GPU_FRACTION', '.025'))
    if not 0 <= fraction <= 1:
        raise ValueError('QCENN_GPU_FRACTION must be between zero and one')
    decision = hashlib.sha256((':'.join(circuit_digest(c) for c in native) + ':' + str(seed)).encode()).digest()
    if int.from_bytes(decision[:8], 'big') / 2**64 >= fraction:
        return None
    directory = Path(os.environ.get('QCENN_GPU_LOCK_DIR', f'/tmp/qcenn-gpu-slots-{os.getuid()}'))
    ids = profile['verified_gpu_ids']
    offset = os.getpid() % len(ids)
    order = ids[offset:] + ids[:offset]
    paths = [directory / f'gpu-{device}.lock' for device in order]
    lease = acquire_files(paths)
    deadline = time.monotonic() + float(os.environ.get('QCENN_GPU_WAIT_TIMEOUT_SECONDS', '300'))
    execution_progress('gpu_wait', backend=f'GPU/{method}', shots=shots)
    while lease is None and time.monotonic() < deadline:
        time.sleep(.1)
        lease = acquire_files(paths)
    if lease is None:
        raise TimeoutError('Verified GPU slots remain busy; no backend or quantum counts were silently substituted')
    device = int(lease.slots[0].removeprefix('gpu-').removesuffix('.lock'))
    process = None
    try:
        with tempfile.TemporaryDirectory(prefix='qcenn-gpu-') as temporary:
            root = Path(temporary)
            from qiskit import qpy
            path = root / 'native.qpy'
            with path.open('wb') as stream:
                qpy.dump(native, stream)
            request = {'qpy': str(path), 'qpy_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'expected_packages': profile['packages'], 'gpu_id': device, 'method': method,
                'shots': shots, 'seed': seed, 'readout_error': readout_error, 'memory': memory,
                'memory_limit_mb': int(profile['memory_mb_by_id'][str(device)] * .7),
                'output_qubits': (native[0].metadata or {}).get('output_qubits', []),
                'result': str(root / 'result.json')}
            request_file = root / 'request.json'; request_file.write_text(json.dumps(request))
            execution_progress('gpu_sampling', backend=f'GPU/{method}', gpu_id=device, shots=shots)
            started = time.perf_counter()
            # Remain in the quantum worker's process group for watchdog cleanup.
            process = subprocess.Popen([profile['python'], str(Path(__file__).with_name('gpu_worker.py')),
                '--request', str(request_file), '--run-root', os.environ.get('QCENN_RUN_OUTPUT', temporary)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, env={**os.environ, **thread_environment(1)},
                pass_fds=[*lease.descriptors, *inherited_worker_descriptors()])
            while True:
                try:
                    output, _ = process.communicate(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    print(f'GPU {device}/{method} still running; elapsed={time.perf_counter()-started:.0f}s', flush=True)
            if process.returncode:
                raise RuntimeError(f'GPU {device}/{method} failed: {output[-4000:]}')
            answer = json.loads((root / 'result.json').read_text())
            answer['dispatch_fraction'] = fraction
            if any(sum(c.values()) != shots for c in answer['counts']):
                raise RuntimeError('GPU measured-shot accounting failed')
            return GPUResult(answer, native)
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
        lease.close()
