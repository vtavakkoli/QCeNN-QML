"""Signed ternary dot-product/softmax attention evaluated inside a circuit.

The circuit computes scores reversibly, applies a DATA-INDEPENDENT exp-function
rotation, and postselects a flag to normalize attention quantum mechanically.
It samples weighted values. It does not output a coherent full-precision Smol
hidden vector, import pretrained weights, or implement a transformer block.
"""
import argparse
from pathlib import Path
import numpy as np
from .coherent import CORE_DISCLOSURE, execute, load_basis, qrom, uniform_records
from .recurrent_mnist import provenance, write_json


def add_power(circuit, control, register, power, inverse=False):
    """Reversible controlled addition/subtraction of 2**power modulo 2**n."""
    from qiskit import QuantumCircuit
    gate = QuantumCircuit(circuit.num_qubits)
    for bit in range(len(register)-1, power, -1):
        gate.mcx([control,*register[power:bit]],register[bit])
    gate.cx(control,register[power])
    circuit.compose(gate.inverse() if inverse else gate,inplace=True)


def dot_product_gate(features, score_bits):
    from qiskit import QuantumCircuit
    circuit=QuantumCircuit(4*features+score_bits+1,name="signed_reversible_dot")
    q=list(range(2*features)); k=list(range(2*features,4*features))
    score=list(range(4*features,4*features+score_bits)); product=circuit.num_qubits-1
    for feature in range(features):
        for a in range(2):
            for b in range(2):
                circuit.ccx(q[2*feature+a],k[2*feature+b],product)
                add_power(circuit,product,score,a+b,inverse=(a==1) != (b==1))
                circuit.ccx(q[2*feature+a],k[2*feature+b],product)
    return circuit


def packed_ternary(row):
    return sum((int(value)&3) << (2*i) for i,value in enumerate(row))


def attention_circuit(query, keys, values, *, max_qubits=48):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UCRYGate
    query, keys, values = np.asarray(query),np.asarray(keys),np.asarray(values)
    if keys.ndim!=2 or query.shape!=keys.shape[1:] or values.ndim!=2 or len(values)!=len(keys):
        raise ValueError("Attention input shapes disagree")
    if not 1<=len(keys)<=8 or not 1<=len(query)<=4 or not 1<=values.shape[1]<=2:
        raise ValueError("Tiny attention dimension budget exceeded")
    if any(not np.isin(x,[-1,0,1]).all() for x in (query,keys,values)):
        raise ValueError("This verified attention component uses signed ternary data")
    features=len(query); index_bits=max(1,(len(keys)-1).bit_length())
    score_bits=max(3,(2*features).bit_length())
    index=list(range(index_bits)); q=list(range(index_bits,index_bits+2*features))
    k=list(range(q[-1]+1,q[-1]+1+2*features))
    score=list(range(k[-1]+1,k[-1]+1+score_bits))
    product=score[-1]+1; flag=product+1
    out=list(range(flag+1,flag+1+2*values.shape[1]))
    width=out[-1]+1
    if width>max_qubits:
        raise MemoryError("Attention qubit budget exceeded")
    circuit=QuantumCircuit(width,index_bits+len(out)+1)
    circuit.compose(uniform_records(len(keys)),index,inplace=True)
    load_basis(circuit,q,packed_ternary(query))
    loader=QuantumCircuit(index_bits+2*features)
    qrom(loader,range(index_bits),range(index_bits,index_bits+2*features),
         [packed_ternary(row) for row in keys])
    circuit.compose(loader,[*index,*k],inplace=True)
    dot=dot_product_gate(features,score_bits)
    circuit.compose(dot,[*q,*k,*score,product],inplace=True)
    # Universal function approximation table, independent of query/key values.
    # Valid scores lie in [-features,features]. Extra basis states are unreachable.
    angles=[]
    for code in range(1<<score_bits):
        signed=code if code<1<<(score_bits-1) else code-(1<<score_bits)
        probability=np.exp(min(0.,(signed-features)/np.sqrt(features)))
        angles.append(float(2*np.arcsin(np.sqrt(probability))))
    circuit.append(UCRYGate(angles),[flag,*score])
    circuit.compose(dot.inverse(),[*q,*k,*score,product],inplace=True)
    circuit.compose(loader.inverse(),[*index,*k],inplace=True)
    qrom(circuit,index,out,[packed_ternary(row) for row in values])
    circuit.measure(index,range(index_bits))
    circuit.measure(out,range(index_bits,index_bits+len(out)))
    circuit.measure(flag,index_bits+len(out))
    circuit.metadata={"output_qubits":[*index,*out],"index_bits":index_bits,
        "value_dimensions":values.shape[1],"score_bits":score_bits,
        "classical_attention_scores":False,"classical_softmax":False,
        "score_register_uncomputed":True,"keys_uncomputed":True,
        "output":"postselected attention samples and measured value expectation"}
    return circuit


def decode_attention(counts,key_count,index_bits,value_dimensions):
    weights=np.zeros(key_count,dtype=int); means=np.zeros(value_dimensions)
    value_width=2*value_dimensions; accepted=0
    for binary,count in counts.items():
        code=int(binary.replace(" ",""),2)
        if not code>>(index_bits+value_width)&1:
            continue
        index=code&((1<<index_bits)-1)
        if index>=key_count:
            raise RuntimeError("Invalid attended index")
        weights[index]+=count; accepted+=count
        packed=code>>index_bits
        for dim in range(value_dimensions):
            value=(packed>>(2*dim))&3
            means[dim]+=count*(value if value<2 else value-4)
    return {"accepted_shots":accepted,"accepted_fraction":accepted/sum(counts.values()),
        "attention_weights":(weights/max(accepted,1)).tolist(),
        "weighted_value_mean":(means/max(accepted,1)).tolist(),"counts":counts}


def run(shots=8192,seed=42,output_dir="results/quantum-attention"):
    from qiskit import qpy
    directory=Path(output_dir)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("Use a fresh attention output directory")
    directory.mkdir(parents=True,exist_ok=True)
    examples=[([1,-1],[[1,-1],[-1,1],[0,1]],[[1,0],[0,1],[-1,1]]),
              ([-1,0],[[1,0],[-1,0],[0,1]],[[1,-1],[-1,1],[0,0]])]
    rows=[]
    for query,keys,values in examples:
        circuit=attention_circuit(query,keys,values)
        counts,native,cost=execute([circuit],shots,seed)
        observed=decode_attention(counts[0],len(keys),circuit.metadata['index_bits'],len(values[0]))
        # Independent reference AFTER measured inference, never circuit input.
        scores=np.asarray(keys)@np.asarray(query)/np.sqrt(len(query))
        ideal=np.exp(scores-scores.max()); ideal/=ideal.sum()
        rows.append({"query":query,"keys":keys,"values":values,**observed,
            "classical_reference_attention":ideal.tolist(),
            "classical_reference_value":(ideal@np.asarray(values)).tolist(),
            "attention_max_absolute_error":float(np.max(abs(np.asarray(observed['attention_weights'])-ideal))),
            "resources":cost})
        with (directory/f"attention-{len(rows)}.qpy").open('wb') as stream:
            qpy.dump(native[0],stream)
    report={"experiment":"signed_ternary_quantum_dot_softmax_attention",**CORE_DISCLOSURE,
        "model_has_neural_training":False,"classical_scores_or_softmax_in_prediction":False,
        "universal_exp_rotation_table_data_independent":True,
        "full_smol_attention_block":False,"coherent_weighted_sum_vector_output":False,
        "shots":shots,"seed":seed,"cases":rows,"source":provenance()}
    write_json(directory/"report.json",report)
    print(f"attention report: {directory/'report.json'}",flush=True)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shots',type=int,default=8192)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--output-dir',default='results/quantum-attention')
    return run(**vars(parser.parse_args(argv)))


if __name__=='__main__':
    main()
