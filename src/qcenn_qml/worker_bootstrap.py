"""Bind leased CPUs before NumPy/Qiskit imports and first memory touch."""
import os


if __name__ == '__main__':
    assigned = os.environ.get('QCENN_ASSIGNED_CPUS')
    if assigned and hasattr(os, 'sched_setaffinity'):
        selected = {int(cpu) for cpu in assigned.split(',')}
        if not selected <= os.sched_getaffinity(0):
            raise RuntimeError('Assigned worker CPUs are outside the process allocation')
        os.sched_setaffinity(0, selected)
    from .study_utils import main
    main()
