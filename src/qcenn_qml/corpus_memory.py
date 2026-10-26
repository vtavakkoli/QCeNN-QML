"""Raw-record quantum conditional memory and coherent token replacement."""
import numpy as np
from .coherent import load_basis, clean_pattern, qrom, uniform_records


def conditional_memory(contexts, values, query, *, context_bits, value_bits,
                       amplify=1):
    from qiskit import QuantumCircuit
    if not 1 <= len(contexts) == len(values) <= 256 or not 0 <= amplify <= 3:
        raise ValueError("Invalid raw corpus memory/amplification budget")
    index_bits = max(1,(len(contexts)-1).bit_length())
    loaded_width = index_bits+context_bits+value_bits
    query_register=list(range(loaded_width,loaded_width+context_bits)) if query is None else []
    work_start=loaded_width+len(query_register)
    scratch=list(range(work_start,work_start+max(0,context_bits-2,index_bits-2)))
    width = work_start+len(scratch)+1
    circuit = QuantumCircuit(width)
    index = list(range(index_bits))
    context = list(range(index_bits,index_bits+context_bits))
    output = list(range(index_bits+context_bits,loaded_width))
    load = QuantumCircuit(loaded_width)
    qrom(load,index,context,contexts)
    qrom(load,index,output,values)
    prepare = uniform_records(len(contexts))
    circuit.compose(prepare,index,inplace=True)
    circuit.compose(load,range(loaded_width),inplace=True)
    def query_phase():
        for a,b in zip(query_register,context,strict=False):
            circuit.cx(a,b)
        clean_pattern(circuit,context,0 if query is None else int(query),None,scratch,phase=True)
        for a,b in zip(query_register,context,strict=False):
            circuit.cx(a,b)
    for _ in range(amplify):
        query_phase()
        circuit.compose(load.inverse(),range(loaded_width),inplace=True)
        circuit.compose(prepare.inverse(),index,inplace=True)
        clean_pattern(circuit,index,0,None,scratch,phase=True)
        circuit.compose(prepare,index,inplace=True)
        circuit.compose(load,range(loaded_width),inplace=True)
    flag = width-1
    for a,b in zip(query_register,context,strict=False):
        circuit.cx(a,b)
    clean_pattern(circuit,context,0 if query is None else int(query),flag,scratch)
    for a,b in zip(query_register,context,strict=False):
        circuit.cx(a,b)
    circuit.metadata={"query_qubits":query_register}
    return circuit, output, flag


def denoising_memory_circuit(contexts, values, query, *, context_bits,
        value_bits, mask, original, steps=3, beta=1., eta=.65, amplify=1,
        max_qubits=96):
    from qiskit import QuantumCircuit
    if not 0 <= beta <= 1 or not 0 <= eta <= 1 or not 0 <= steps <= 8:
        raise ValueError("Invalid corruption/replacement configuration")
    memory, source, success = conditional_memory(contexts,values,query,
        context_bits=context_bits,value_bits=value_bits,amplify=amplify)
    # Retain the environment coherently. Source basis-value copies are
    # entangled CNOT copies, not cloning of arbitrary quantum states.
    initial_environment = value_bits+1 if beta != 1 else 0
    total = memory.num_qubits+value_bits+initial_environment+steps*(value_bits+1)
    if total > max_qubits:
        raise MemoryError(f"Circuit needs {total} qubits, budget {max_qubits}")
    circuit=QuantumCircuit(total,value_bits+1)
    circuit.compose(memory,range(memory.num_qubits),inplace=True)
    target=list(range(memory.num_qubits,memory.num_qubits+value_bits))
    offset=memory.num_qubits+value_bits
    if beta == 1:
        # Fully masked inference circuit is independent of the held-out truth.
        load_basis(circuit,target,mask)
    else:
        load_basis(circuit,target,original)
        environment=list(range(offset,offset+value_bits)); flag=offset+value_bits
        load_basis(circuit,environment,mask)
        circuit.ry(2*np.arcsin(np.sqrt(beta)),flag)
        for a,b in zip(target,environment,strict=True):
            circuit.cswap(flag,a,b)
        offset+=value_bits+1
    for _ in range(steps):
        environment=list(range(offset,offset+value_bits)); flag=offset+value_bits
        for a,b in zip(source,environment,strict=True):
            circuit.cx(a,b)
        circuit.ry(2*np.arcsin(np.sqrt(eta)),flag)
        for a,b in zip(target,environment,strict=True):
            circuit.cswap(flag,a,b)
        offset+=value_bits+1
    circuit.measure(target,range(value_bits))
    circuit.measure(success,value_bits)
    circuit.metadata={"output_qubits":target,"raw_corpus_records":len(contexts),
        "classical_optimizer":False,"classical_conditional_table_used":False,
        "heldout_truth_in_reverse_gates":False,"steps":steps,
        "entire_environment_retained":True}
    return circuit
