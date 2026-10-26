import json
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qcenn_qml.coherent import qrom, controlled_amplitudes, execute, decode_postselection
from qcenn_qml.quantum_memory import memory_circuit, reference_distribution
from qcenn_qml.quantum_attention import attention_circuit, dot_product_gate, packed_ternary
from qcenn_qml.corpus_memory import denoising_memory_circuit


def marginal(circuit):
    clean=circuit.remove_final_measurements(inplace=False)
    return Statevector.from_instruction(clean).probabilities(range(circuit.num_clbits))


def test_raw_record_loader_isometry_and_inverse():
    circuit=QuantumCircuit(4)
    circuit.h([0,1])
    before=circuit.copy()
    loader=QuantumCircuit(4)
    qrom(loader,[0,1],[2,3],[3,1,2,0])
    circuit.compose(loader,inplace=True)
    state=Statevector.from_instruction(circuit).data
    expected=np.zeros(16)
    for i,value in enumerate([3,1,2,0]): expected[i+(value<<2)]=.5
    np.testing.assert_allclose(state,expected,atol=1e-13)
    circuit.compose(loader.inverse(),inplace=True)
    np.testing.assert_allclose(Statevector.from_instruction(circuit).data,
        Statevector.from_instruction(before).data,atol=1e-13)


@pytest.mark.parametrize('controls',[3,4,6])
def test_clean_toffoli_ladder_restores_all_work_qubits(controls):
    from qcenn_qml.coherent import mcx_clean,load_basis
    for value in range(1<<controls):
        circuit=QuantumCircuit(2*controls-1)
        load_basis(circuit,range(controls),value)
        mcx_clean(circuit,range(controls),controls,list(range(controls+1,2*controls-1)))
        p=Statevector.from_instruction(circuit).probabilities()
        expected=value+((1<<controls) if value==(1<<controls)-1 else 0)
        assert p[expected]==pytest.approx(1.,abs=1e-12)


def test_multiplexed_raw_amplitudes_no_similarity_table():
    records=np.random.default_rng(5).random((3,4))
    loader=controlled_amplitudes(records)
    circuit=QuantumCircuit(4); circuit.h([0,1]); circuit.compose(loader,inplace=True)
    state=Statevector.from_instruction(circuit).data.reshape(4,4)
    for index in range(3):
        np.testing.assert_allclose(state[:,index],records[index]/np.linalg.norm(records[index])/2,atol=1e-13)


@pytest.mark.parametrize('encoding',['amplitude','angle'])
@pytest.mark.parametrize('rounds',[1,2,3])
def test_coherent_recurrence_matches_independent_kernel(encoding,rounds):
    query=np.array([.4,.8,.2,.1]) if encoding=='amplitude' else np.array([.4,.8])
    keys=np.array([[.1,.9,.3,.2],[.8,.1,.2,.1]]) if encoding=='amplitude' else np.array([[.1,.9],[.8,.1]])
    circuit=memory_circuit(query,keys,[0,1],rounds=rounds,encoding=encoding,value_bits=1)
    clean=circuit.remove_final_measurements(inplace=False)
    state=Statevector.from_instruction(clean)
    # Derive measured qubit order from actual measurement instructions.
    measured=[circuit.find_bit(e.qubits[0]).index for e in circuit.data if e.operation.name=='measure']
    p=state.probabilities(measured)
    selected=p[((1<<rounds)-1)<<1:]
    expected,acceptance=reference_distribution(query,keys,[0,1],rounds=rounds,encoding=encoding,values=2)
    assert selected.sum()==pytest.approx(acceptance,abs=2e-12)
    np.testing.assert_allclose(selected/selected.sum(),expected,atol=2e-12)


def test_quantum_memory_measured_tiny_training_retrieval():
    keys=np.eye(4)[:2]
    circuits=[memory_circuit(q,keys,[0,1],rounds=2,value_bits=1) for q in keys]
    counts,_,_=execute(circuits,512,4)
    assert [decode_postselection(c,1,2,2)['prediction'] for c in counts]==[0,1]


def test_coherent_reset_control_loses_accumulated_filter_but_charges_acceptance():
    query=np.array([.3,.7,.2,.1]); keys=np.array([[.2,.9,.1,.1],[.7,.2,.1,.3]])
    circuit=memory_circuit(query,keys,[0,1],rounds=2,value_bits=1,reset_memory=True)
    clean=circuit.remove_final_measurements(inplace=False)
    measured=[circuit.find_bit(e.qubits[0]).index for e in circuit.data if e.operation.name=='measure']
    p=Statevector.from_instruction(clean).probabilities(measured)[6:]
    expected,a=reference_distribution(query,keys,[0,1],rounds=1,values=2)
    assert p.sum()==pytest.approx(a*a,abs=1e-12)
    np.testing.assert_allclose(p/p.sum(),expected,atol=1e-12)


def test_readout_noise_invalid_tokens_remain_in_probability_denominator():
    result=decode_postselection({'111':6,'100':4},2,1,3)
    assert result['accepted_shots']==10
    assert result['invalid_value_shots']==6
    assert result['prediction']==-1
    assert result['histogram']==[4,0,0]


def test_reversible_signed_dot_product_all_ternary_pairs():
    from qcenn_qml.coherent import load_basis
    gate=dot_product_gate(2,3)
    rows=[(a,b) for a in (-1,0,1) for b in (-1,0,1)]
    for q in rows:
        for k in rows:
            circuit=QuantumCircuit(gate.num_qubits)
            load_basis(circuit,range(4),packed_ternary(q))
            load_basis(circuit,range(4,8),packed_ternary(k))
            circuit.compose(gate,inplace=True)
            probability=Statevector.from_instruction(circuit).probabilities()
            expected=packed_ternary(q)+(packed_ternary(k)<<4)+((int(np.dot(q,k))&7)<<8)
            assert probability[expected]==pytest.approx(1.,abs=1e-12)


def test_quantum_softmax_attention_exact_distribution_and_value():
    query=[1,-1]; keys=[[1,0],[0,1]]; values=[[1],[-1]]
    circuit=attention_circuit(query,keys,values)
    clean=circuit.remove_final_measurements(inplace=False)
    measured=[circuit.find_bit(e.qubits[0]).index for e in circuit.data if e.operation.name=='measure']
    probabilities=Statevector.from_instruction(clean).probabilities(measured)
    # index bit, two signed value bits, success bit.
    selected=probabilities[8:]; index=np.array([selected[::2].sum(),selected[1::2].sum()])
    scores=np.array(keys)@np.array(query)/np.sqrt(2)
    expected=np.exp(scores); expected/=expected.sum()
    np.testing.assert_allclose(index/index.sum(),expected,atol=2e-12)
    assert selected.sum()==pytest.approx(float(np.exp((scores-2/np.sqrt(2))).mean()),abs=2e-12)


def test_fully_masked_reverse_gates_do_not_depend_on_truth():
    options=dict(context_bits=1,value_bits=2,mask=3,steps=2,beta=1.,eta=.6,amplify=0)
    first=denoising_memory_circuit([0,0,1],[0,1,1],0,original=0,**options)
    second=denoising_memory_circuit([0,0,1],[0,1,1],0,original=2,**options)
    # Decompose autogenerated loader names before comparing actual gates.
    assert first.decompose(reps=8)==second.decompose(reps=8)
    assert all(e.operation.name not in ('measure','reset') for e in first.data[:-3])


def test_coherent_denoising_matches_classical_replacement_channel():
    circuit=denoising_memory_circuit([0,0,1],[0,1,1],0,context_bits=1,value_bits=2,
        mask=3,original=0,steps=2,beta=1.,eta=.5,amplify=0)
    clean=circuit.remove_final_measurements(inplace=False)
    measured=[circuit.find_bit(e.qubits[0]).index for e in circuit.data if e.operation.name=='measure']
    p=Statevector.from_instruction(clean).probabilities(measured)[4:]
    assert p.sum()==pytest.approx(2/3,abs=1e-12)
    np.testing.assert_allclose(p/p.sum(),[.375,.375,0,.25],atol=2e-12)


def test_coherent_generation_uses_quantum_previous_tokens():
    from qcenn_qml.quantum_text import sequence_circuit
    circuit=sequence_circuit([0,1],[1,0],[0],bits=1,order=1,new_tokens=2,amplify=0)
    counts,_,_=execute([circuit],512,3)
    accepted={k:v for k,v in counts[0].items() if int(k,2)>>2==3}
    assert sum(accepted.values())>0
    assert set(accepted)=={'1101'}  # generated 1 then 0, flags both successful
    assert all(e.operation.name!='measure' for e in circuit.data[:-4])


def test_text_split_precedes_windows_and_has_broader_vocabulary():
    from qcenn_qml.diffusion.matched import corpus
    vocab,mask,train,test,meta=corpus('text',seed=42)
    assert len(vocab)>10 and train and test
    assert not set(meta['train_sentence_ids'])&set(meta['heldout_sentence_ids'])
    assert all(mask not in row for row in train)


def test_matched_diffusion_reuses_native_circuits_and_classical_reference(tmp_path):
    from qcenn_qml.diffusion.matched import execute_benchmark
    r=execute_benchmark(shots=128,steps=[0,1],examples=2,corpus_kind='toy',
        readout_error=.05,output_dir=str(tmp_path/'diffusion'))
    assert r['classical_conditional_tables_in_quantum_gates'] is False
    assert r['same_native_circuits_and_cases_for_noise_comparison'] is True
    assert set(r['matched_results'])=={'noiseless','noisy'}
    assert r['matched_results']['noisy']['1']['resources']['compilation_seconds']>=0
    assert r['matched_results']['noiseless']['1']['classical_matched_replacement_probability_mean']==pytest.approx(.65)


def test_tiny_fit_failure_blocks_validation_and_test(monkeypatch,tmp_path):
    import qcenn_qml.quantum_mnist as module
    called=[]
    def fake(images,labels,*args,**kwargs):
        called.append(len(images)); return {'accuracy':0.}
    monkeypatch.setattr(module,'measure_split',fake)
    r=module.evaluate(dataset='digits-proxy',side=4,shots=32,output_dir=str(tmp_path/'mnist'))
    assert len(called)==1 and r['test_accuracy_mean'] is None
    assert r['runs'][0]['test_evaluated'] is False


def test_legacy_hybrid_is_not_default_cli(monkeypatch):
    import qcenn_qml.quantum_mnist as quantum
    import qcenn_qml.recurrent_mnist as old
    monkeypatch.setattr(quantum,'main',lambda argv: {'quantum':True,'argv':argv})
    assert old.main(['--shots','32'])['quantum'] is True


def test_smol_audit_never_calls_toy_attention_full_model(tmp_path):
    from qcenn_qml.smol_feasibility import audit
    path=tmp_path/'attention.json'
    path.write_text(json.dumps({'experiment':'signed_ternary_quantum_dot_softmax_attention','cases':[{}]}))
    r=audit(str(path),output_dir=str(tmp_path/'audit'))
    assert r['parameters_from_shapes']==134515008
    assert r['chosen_basis_encoding_weight_qubits']==2152240128
    assert r['coverage']['ternary_signed_dot_product_attention_sampling'] is True
    assert r['full_model_quantum_conversion_test']=='NOT_IMPLEMENTED_NOT_DEMONSTRATED'
    assert r['actual_smol_weights_executed'] is False


def test_combined_report_uses_paired_noise_reference_not_another_corpus(tmp_path):
    from qcenn_qml.report import build
    entry=lambda p: {'measured_correct_token_probability':p,'independent_ideal_probability_mean':p}
    unrelated=tmp_path/'toy.json'
    paired=tmp_path/'text.json'
    unrelated.write_text(json.dumps({'results':{'1':entry(.9)}}))
    paired.write_text(json.dumps({'results':{'1':entry(.4)},
        'matched_results':{'noiseless':{'1':entry(.5)}}}))
    target=tmp_path/'combined'
    build(str(tmp_path/'missing-mnist'),str(tmp_path/'missing-smol'),str(unrelated),str(target),
        diffusion_noisy=str(paired))
    assert '| 1 | 0.4000 | 0.5000 |' in (target/'combined.md').read_text()
