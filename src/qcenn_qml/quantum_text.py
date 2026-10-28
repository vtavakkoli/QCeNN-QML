"""Coherent raw-corpus token memory; every generated token measured at the end.

No teacher, classical transition table, fitted weights, or intermediate CPU
token-state update. This remains a small nonparametric sequence memory, with
postselection bias, rather than the pretrained SmolLM2 transformer.
"""
import argparse
import hashlib
from pathlib import Path
import re
import numpy as np
from .coherent import CORE_DISCLOSURE, execute, load_basis
from .corpus_memory import conditional_memory
from .recurrent_mnist import write_json, provenance


TEXT='the cat sat on the mat . the dog ran on the road . the cat ran on the road . the dog sat on the mat .'


def sequence_circuit(contexts,values,prompt,*,bits,order,new_tokens,amplify=1,max_qubits=96):
    from qiskit import QuantumCircuit
    if order not in (1,2) or new_tokens<1 or len(prompt)!=order:
        raise ValueError("Invalid finite-context sequence budget")
    block,source,flag=conditional_memory(contexts,values,None,context_bits=order*bits,
        value_bits=bits,amplify=amplify)
    shared=block.metadata['query_qubits']
    fresh_width=block.num_qubits-len(shared)
    width=order*bits+new_tokens*fresh_width
    if width>max_qubits:
        raise MemoryError(f"Coherent generation requires {width} qubits, budget {max_qubits}")
    circuit=QuantumCircuit(width,new_tokens*(bits+1))
    past=[list(range(i*bits,(i+1)*bits)) for i in range(order)]
    for register,value in zip(past,prompt,strict=True):
        load_basis(circuit,register,value)
    offset=order*bits; tokens=[]; flags=[]
    for _ in range(new_tokens):
        mapping=[]; next_fresh=offset; query_wires=[q for reg in past[-order:] for q in reg]
        shared_map=dict(zip(shared,query_wires,strict=True))
        for i in range(block.num_qubits):
            if i in shared_map:
                mapping.append(shared_map[i])
            else:
                mapping.append(next_fresh); next_fresh+=1
        circuit.compose(block,mapping,inplace=True)
        output=[mapping[q] for q in source]
        tokens.append(output); flags.append(mapping[flag]); past.append(output)
        offset+=fresh_width
    for i,register in enumerate(tokens):
        circuit.measure(register,range(i*bits,(i+1)*bits))
    circuit.measure(flags,range(new_tokens*bits,new_tokens*(bits+1)))
    circuit.metadata={'output_qubits':[q for reg in tokens for q in reg],
        'intermediate_measurements':0,'classical_token_state_updates':0,
        'postselection_bias':'Final conditioning on all retrieval flags can reweight earlier tokens'}
    return circuit


def generate(prompt='the cat',corpus_file=None,order=1,new_tokens=3,shots=1024,
        seed=42,amplify=1,output_dir='results/quantum-text'):
    from qiskit import qpy
    text=Path(corpus_file).read_text(encoding='utf8') if corpus_file else TEXT
    tokens=re.findall(r'[a-z]+|[.,!?]',text.lower())
    vocab=['<unk>']+sorted(set(tokens))
    if len(vocab)>32 or not order<len(tokens)<=256:
        raise ValueError("Tiny raw-corpus memory budget: <=32 vocabulary entries, <=256 tokens")
    mapping={w:i for i,w in enumerate(vocab)}; ids=[mapping[w] for w in tokens]
    bits=max(1,(len(vocab)-1).bit_length())
    contexts=[sum(ids[j]<<(bits*(j-i+order)) for j in range(i-order,i)) for i in range(order,len(ids))]
    values=ids[order:]
    initial=([0]*order+[mapping.get(w,0) for w in re.findall(r'[a-z]+|[.,!?]',prompt.lower())])[-order:]
    directory=Path(output_dir)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("Use a fresh text output directory")
    directory.mkdir(parents=True,exist_ok=True)
    source=provenance()
    (directory/'corpus.txt').write_text(text,encoding='utf8')
    write_json(directory/'progress.json',{'status':'running','phase':'constructing_circuit',
        'new_tokens':new_tokens,'shots':shots})
    print(f'Quantum text: constructing a coherent {new_tokens}-token circuit',flush=True)
    circuit=sequence_circuit(contexts,values,initial,bits=bits,order=order,new_tokens=new_tokens,amplify=amplify)
    write_json(directory/'progress.json',{'status':'running','phase':'executing_circuit',
        'qubits':circuit.num_qubits,'shots':shots})
    samples,native,cost=execute([circuit],shots,seed)
    write_json(directory/'counts.json',{'counts':samples[0],'resources':cost,'source':source})
    histogram={}; accepted=0
    for binary,count in samples[0].items():
        code=int(binary.replace(' ',''),2)
        if code>>(new_tokens*bits)!=(1<<new_tokens)-1:
            continue
        sequence=tuple((code>>(bits*i))&((1<<bits)-1) for i in range(new_tokens))
        if any(i>=len(vocab) for i in sequence):
            raise RuntimeError('Invalid quantum sequence token')
        words=' '.join(vocab[i] for i in sequence)
        histogram[words]=histogram.get(words,0)+count; accepted+=count
    generated=max(histogram,key=histogram.get) if histogram else None
    with (directory/'generation-native.qpy').open('wb') as stream:
        qpy.dump(native[0],stream)
    report={'experiment':'coherent_raw_corpus_sequence_memory',**CORE_DISCLOSURE,
        'corpus_sha256':hashlib.sha256(text.encode('utf8')).hexdigest(),
        'configuration':{'prompt':prompt,'corpus_file':corpus_file,'order':order,
            'new_tokens':new_tokens,'shots':shots,'seed':seed,'amplify':amplify},
        'actual_SmolLM2_weights_on_quantum_circuit':False,'actual_SmolLM2_transformer_implemented':False,
        'classical_teacher_used':False,'intermediate_measurements':0,'classical_token_state_updates':0,
        'vocabulary':vocab,'order':order,'prompt':prompt,'generated_text':generated,
        'sequence_counts':histogram,'counts':samples[0],'accepted_shots':accepted,
        'accepted_shot_fraction':accepted/shots,'qubits':circuit.num_qubits,'resources':cost,
        'raw_context_records':contexts,'raw_value_records':values,'source':source,
        'limitations':['Nonparametric corpus memory, not pretrained Smol weights or quantum neural training',
            'Postselection of complete sequences reweights their distribution',
            'Classical input encoding, compilation and final sequence count decoding remain']}
    write_json(directory/'report.json',report)
    write_json(directory/'progress.json',{'status':'complete','report':'report.json'})
    print(f'quantum sequence: {generated}; accepted={accepted}/{shots}',flush=True)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompt',default='the cat')
    parser.add_argument('--corpus-file')
    parser.add_argument('--order',type=int,choices=[1,2],default=1)
    parser.add_argument('--new-tokens',type=int,default=3)
    parser.add_argument('--shots',type=int,default=1024)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--amplify',type=int,default=1)
    parser.add_argument('--output-dir',default='results/quantum-text')
    return generate(**vars(parser.parse_args(argv)))


if __name__=='__main__':
    main()
