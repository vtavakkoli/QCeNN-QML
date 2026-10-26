"""Raw quantum corpus memory, classical references, and paired readout noise.

There are no fitted classical conditional probabilities in the quantum gates.
The classical conditional-table baseline is evaluated separately for comparison.
"""
import argparse
import hashlib
from collections import Counter
from pathlib import Path
import re
import numpy as np

from ..coherent import CORE_DISCLOSURE, decode_postselection, execute
from ..corpus_memory import denoising_memory_circuit
from ..recurrent_mnist import provenance, write_json
from .core import dataset as toy_dataset, VOCAB


TEXT="""the cat sat on the mat.
the dog sat on the mat.
the cat ran to the door.
the dog ran to the door.
the bird sat in the tree.
the cat slept in the room.
the dog slept in the room.
the bird flew to the tree.
the child sat on the mat.
the child ran to the door.
the cat sat in the room.
the dog sat in the room.
the child slept in the room.
the bird sat on the mat.
the cat ran to the tree.
the dog ran to the tree.
"""


def corpus(corpus_kind="text",corpus_file=None,seed=42):
    if corpus_kind=="toy" and corpus_file is None:
        train,test=toy_dataset(seed)
        return list(VOCAB),3,train,test,{"source":"legacy_three_token_toy","split":"seeded_records"}
    text=Path(corpus_file).read_text(encoding='utf8') if corpus_file else TEXT
    units=[re.findall(r"[a-z]+",s.lower()) for s in re.split(r"[.!?\n]+",text)]
    units=[u for u in units if len(u)>=3]
    if len(units)<4:
        raise ValueError("Provide at least four sentences of three or more words")
    order=np.random.default_rng(seed).permutation(len(units))
    boundary=max(1,min(len(units)-1,int(.75*len(units))))
    training=[units[i] for i in order[:boundary]]
    heldout=[units[i] for i in order[boundary:]]
    vocab=['<unk>','[MASK]']+sorted({w for s in training for w in s})
    if len(vocab)>64:
        raise ValueError("Tiny corpus-memory vocabulary budget is 64; use a smaller corpus")
    mapping={w:i for i,w in enumerate(vocab)}
    def windows(sentences):
        return [tuple(mapping.get(w,0) for w in sentence[i:i+3])
                for sentence in sentences for i in range(len(sentence)-2)]
    train,test=windows(training),windows(heldout)
    if len(train)>256:
        raise ValueError("Raw quantum corpus memory is limited to 256 records")
    return vocab,1,train,test,{"source":str(corpus_file) if corpus_file else "bundled_small_English_text",
        "corpus_sha256":hashlib.sha256(text.encode('utf8')).hexdigest(),
        "split":"whole_sentences_before_windowing","train_sentence_ids":order[:boundary].tolist(),
        "heldout_sentence_ids":order[boundary:].tolist(),"heldout_vocabulary_from_training_only":True}


def execute_benchmark(shots=1024,beta=1.,eta=.65,steps=(0,1,3),examples=8,
        seed=42,readout_error=.05,corpus_kind="text",corpus_file=None,amplify=1,
        output_dir="results/diffusion-matched"):
    from qiskit import qpy
    if shots<1 or examples<1 or not 0<=readout_error<.5 or not steps:
        raise ValueError("Invalid diffusion experiment budget")
    directory=Path(output_dir)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("Use a fresh output directory")
    vocab,mask,train,test,split=corpus(corpus_kind,corpus_file,seed)
    selected=np.random.default_rng(seed+100).permutation(len(test))[:examples]
    cases=[test[int(i)] for i in selected]
    if not cases:
        raise ValueError("No held-out evaluation cases")
    bits=max(1,(len(vocab)-1).bit_length())
    contexts=[int(a)+(int(c)<<bits) for a,b,c in train]; values=[int(b) for a,b,c in train]
    directory.mkdir(parents=True,exist_ok=True)
    write_json(directory/'splits.json',{'vocabulary':vocab,'training_records':train,
        'heldout_records':test,'evaluated_records':cases,'evaluated_window_indices':selected.tolist(),**split})
    np.savez_compressed(directory/'raw-memory.npz',contexts=contexts,values=values)
    # Independent classical REFERENCE ONLY. These arrays never enter a circuit.
    counts=Counter((a,c,b) for a,b,c in train)
    table={}
    for a,_,c in cases:
        q=np.array([counts[(a,c,b)] for b in range(len(vocab))],dtype=float)
        table[(a,c)]=q/q.sum() if q.sum() else q
    baseline={'correct_token_probability_mean':float(np.mean([table[(a,c)][b] for a,b,c in cases])),
        'top1_accuracy':float(np.mean([int(table[(a,c)].argmax())==b if table[(a,c)].sum() else False for a,b,c in cases])),
        'unseen_context_cases':sum(not table[(a,c)].sum() for a,b,c in cases),
        'smoothing':0.,'used_to_compile_quantum_model':False}
    modes=[('noiseless',0.)]+([('noisy',readout_error)] if readout_error else [])
    all_results={mode:{} for mode,_ in modes}; rows=[]
    for step in steps:
        if not isinstance(step,int) or not 0<=step<=8:
            raise ValueError("Denoising steps must be 0..8")
        circuits=[denoising_memory_circuit(contexts,values,a+(c<<bits),context_bits=2*bits,
            value_bits=bits,mask=mask,original=b,steps=step,beta=beta,eta=eta,
            amplify=amplify) for a,b,c in cases]
        native=None
        for mode,error in modes:
            samples,compiled,cost=execute(circuits if native is None else native,
                shots,seed+step,readout_error=error,precompiled=native is not None)
            if native is None:
                native=compiled
                with (directory/f'steps-{step}-native.qpy').open('wb') as stream:
                    qpy.dump(native[0],stream)
            observed=[]; expected=[]; accepted=0; top=[]
            for sample,(a,b,c) in zip(samples,cases,strict=True):
                decoded=decode_postselection(sample,bits,1,len(vocab))
                distribution=np.asarray(decoded['histogram'],float)/max(1,decoded['accepted_shots'])
                q=table[(a,c)]
                ideal=((1-eta)**step*(1-beta)+(1-(1-eta)**step)*q[b]) if q.sum() else 0.
                observed.append(float(distribution[b])); expected.append(float(ideal))
                accepted+=decoded['accepted_shots']; top.append(decoded['prediction']==b)
                rows.append({'mode':mode,'steps':step,'context':[vocab[a],vocab[c]],
                    'truth':vocab[b],**decoded,'quantum_distribution':distribution.tolist(),
                    'analytic_ideal_target_probability':float(ideal),
                    'quantum_target_frequency':float(distribution[b])})
            all_results[mode][str(step)]={'measured_correct_token_probability':float(np.mean(observed)),
                'independent_ideal_probability_mean':float(np.mean(expected)),
                'sequence_top1_accuracy':float(np.mean(top)),'samples':len(cases),
                'total_shots':shots*len(cases),'accepted_shots':accepted,
                'accepted_shot_fraction':accepted/(shots*len(cases)),
                'compiled_depth_min':min(c.depth() for c in compiled),
                'compiled_depth_max':max(c.depth() for c in compiled),
                'compiled_max_qubits':max(c.num_qubits for c in compiled),
                'classical_matched_replacement_probability_mean':float(np.mean(expected)),
                'resources':cost}
            print(f'diffusion steps={step} {mode}: correct-token={np.mean(observed):.4f}, accepted={accepted}',flush=True)
            write_json(directory/'progress.json',{'status':'in_progress',
                'matched_results':all_results,'cases':rows})
    selected='noisy' if readout_error else 'noiseless'
    report={'name':'Raw-corpus quantum-memory masked denoising',**CORE_DISCLOSURE,
        'configuration':{'shots':shots,'beta':beta,'eta':eta,'steps':list(steps),
            'examples':examples,'seed':seed,'readout_error':readout_error,
            'corpus_kind':corpus_kind,'corpus_file':corpus_file,'amplify':amplify},
        'status':'nonparametric_quantum_memory_not_a_trained_diffusion_LLM',
        'backend':'Qiskit Aer numerical MPS gate simulation (truncation 1e-20)','vocabulary':vocab,'mask_token':vocab[mask],
        'train_count':len(train),'heldout_count':len(test),'evaluation_count':len(cases),
        'unique_evaluation_contexts':len({(a,c) for a,b,c in cases}),
        'forward_mask_probability_beta':beta,'reverse_replacement_probability_eta':eta,
        'readout_bitflip_probability':readout_error,'shots_per_case':shots,'steps':list(steps),
        'seeds':{'data':seed,'simulator':seed},'classical_conditional_baseline':baseline,
        'same_native_circuits_and_cases_for_noise_comparison':True,
        'matched_results':all_results,'results':all_results[selected],'cases':rows,
        'classical_conditional_tables_in_quantum_gates':False,
        'actual_smol135_weights_on_quantum_device':False,'source':provenance(),
        'uncertainty_scope':'Raw counts and acceptance are preserved; repeated contexts and tiny corpus prevent language-generalization claims',
        'limitations':['Quantum learning here is raw example-memory loading, not neural weight training',
            'Conditioning on an unseen context produces abstention, not a classical prediction fallback',
            'Classical input encoding, gate compilation and final counting remain',
            'Only output readout-bit flips are simulated; no physical QPU/gate noise claim']}
    write_json(directory/'report.json',report)
    write_json(directory/'progress.json',{'status':'complete','report':'report.json'})
    lines=['# Quantum-memory denoising comparison','',
        'Gate-only numerical model; raw-record loading; no classical optimizer or fitted conditional-table gates.','',
        f'Corpus: {split["source"]}. Training records: {len(train)}; held-out records: {len(test)}; evaluated: {len(cases)}.',
        f'Classical direct-table top-1: {baseline["top1_accuracy"]:.4f}; unseen contexts: {baseline["unseen_context_cases"]}.','',
        '| Steps | Noiseless correct-token | Noisy correct-token | Classical matched channel |',
        '|---:|---:|---:|---:|']
    for step in steps:
        n=all_results['noiseless'][str(step)]
        noisy=all_results.get('noisy',{}).get(str(step),{})
        lines.append(f'| {step} | {n["measured_correct_token_probability"]:.4f} | '
            +(f'{noisy["measured_correct_token_probability"]:.4f}' if noisy else 'not requested')
            +f' | {n["independent_ideal_probability_mean"]:.4f} |')
    lines+=['','Postselection, abstentions, repeated contexts and circuit resources are retained in JSON. ',
        'This is neither a full trained diffusion language model nor a full quantum host pipeline.','']
    (directory/'report.md').write_text('\n'.join(lines),encoding='utf8')
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shots',type=int,default=1024)
    parser.add_argument('--beta',type=float,default=1.)
    parser.add_argument('--eta',type=float,default=.65)
    parser.add_argument('--steps',type=int,nargs='+',default=[0,1,3])
    parser.add_argument('--examples',type=int,default=8)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--readout-error',type=float,default=.05)
    parser.add_argument('--corpus-kind',choices=['toy','text'],default='text')
    parser.add_argument('--corpus-file')
    parser.add_argument('--amplify',type=int,default=1)
    parser.add_argument('--output-dir',default='results/diffusion-matched')
    return execute_benchmark(**vars(parser.parse_args(argv)))


if __name__=='__main__':
    main()
