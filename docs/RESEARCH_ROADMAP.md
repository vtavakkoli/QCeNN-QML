# Quantum research roadmap

## What exists and what does not

Version 0.3 defaults are documented in [QUANTUM_CORE.md](QUANTUM_CORE.md).
They use raw-record quantum memories with no fitted classical parameters,
require tiny training retrieval before validation/test, retain k=0 as a
measured quantum reference, and compare denoising with direct classical and
matched noiseless/noisy references on sentence-held-out small text. A signed
ternary reversible softmax-attention component and full-Smol resource/coverage
audit are implemented. Quantum neural-weight training, coherent full-precision
transformer blocks and actual SmolLM2 weight execution remain unimplemented.
Classical input loading/compilation/readout are declared; no entire-program
purity, physical-QPU result or quantum advantage is claimed.

Version 0.2 adds [recurrent ten-class MNIST](RECURRENT_MNIST.md): four retained
memory qubits, a reset/reused environment, input-conditioned local collision
channels and trained quantum readout gates. Classical SPSA-Adam training uses
an independently circuit-checked reduced-density simulator; final predictions
come only from explicit finite-shot circuits. Seeded train/validation/test
comparisons, exact classical references, measured overlap baselines and a
separately trained last-patch control are implemented. This is a local
QCeNN-inspired channel model, distinct from the matrix-inversion residual
channel below. Larger accuracy studies remain necessary.

The source [QCeNN-MI](https://github.com/vtavakkoli/QCeNN-MI) has dense SVD-compiled recurrent collision channels that encode solutions of linear systems into quantum states, as well as polynomial and Chebyshev filtering experiments. Those implementations are NOT quantum classifiers. An input-independent attracting fixed point will erase classification information.

The current MNIST experiment builds an explicit quantum overlap circuit with real amplitude state preparation for each input and labeled training exemplar. One or more Grover iterations induce p(k) = sin^2((2k+1) arcsin(sqrt(p))), where p is the ideal unfiltered overlap probability. The polynomial is nonmonotonic on [0,1]; do not imply amplification necessarily improves discrimination. Qiskit Aer finite-shot measurements supply scores, class averaging/argmax run on CPU, and classical state-preparation cost is not neglected.

The current language demonstration creates joint quantum memory
sum_(context,next) sqrt((count(context,next)+alpha)/Z) |context>|next>.
Qiskit prepares the state, optionally Grover-amplifies the query context, then measures context/output and postselects the desired context. No classical runtime transition-table lookup makes the prediction. However the joint memory is compiled classically, all text is tokenized classically, between-token sampling is classical, and only a tiny 8-token vocabulary is supported. This is not an autoregressive transformer.

The actual SmolLM2-135M transformer has 30 layers, hidden width 576, and 49,152 vocabulary entries in its Instruct config. Its 135M weights are not represented or executed by these circuits; the optional Hugging Face teacher simply creates an offline text corpus.

## Future research tests

1. Independently verify Grover exact-prediction polynomial and quantum phase-flip correctness for all qubit subsets. Test shot-noise error, state-preparation scaling, depth, qubits.
2. Use the new recurrent comparison runner on larger real-MNIST splits and multiple seeds. Compare fidelity against classical exemplar overlap and properly budgeted quantum polynomial scoring. Report resource differences and improve beyond tiny smoke studies.
3. Extend the local input-conditioned collision model after its measured accuracy is established. Its CPTP proof, retained-history tests and input-separated stationary subfamily are documented; universal ten-class separation at convergence is not proved. An original matrix-inversion-residual classifier and compatible polynomial filtering remain future work.
4. Create quantum recurrent token-state model with learnable quantum gates and a documented *hybrid classical optimization* training strategy, or specify a physically quantum-only optimizer if essential.
5. Compare held-out language log loss/perplexity and accepted shots to classical n-gram baselines before attributing any language competence.
6. Quantify cost of encoding a single actual SmolLM2 operator; prohibit claims of end-to-end quantum 135M execution without resource proof and hardware traces.

Sources: https://github.com/vtavakkoli/QCeNN-MI ; https://huggingface.co/HuggingFaceTB/SmolLM2-135M ; https://qiskit.github.io/qiskit-aer/
