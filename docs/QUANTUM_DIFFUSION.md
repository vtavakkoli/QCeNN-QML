# QCeNN-DiffusionLM — toy controlled-noise masked language experiment

Implemented in src/qcenn_qml/diffusion/. This is a feasibility prototype, not a full diffusion language model, quantum-trained neural network, or replication of QCeNN-MI's matrix-inversion channel.

## Physics

A three-token sentence consists of visible left and right tokens and a middle target, each encoded as a two-qubit computational-basis register. Vocabulary: cat, dog, sat, [MASK]. Six qubits hold the sentence; two environment qubits and one flag enable a 9-qubit dissipative collision circuit.

Forward mask corruption:
N_beta(rho)=(1-beta) rho + beta |MASK><MASK|.
It is implemented by preparing a mask environment, applying a quantum-controlled SWAP with probability beta, and tracing/resetting environment and flag. When beta=1, the target starts entirely masked and contains no label information.

The inverse/recurrent channel depends on both visible context tokens c and a quantum-prepared conditional distribution |psi_c> computed from classical toy triples:
D_eta,c(rho)=(1-eta)rho + eta |psi_c><psi_c|.
It is implemented with a quantum context-controlled loader, ancilla rotation, controlled SWAP, and ancillary reset. There is no measurement or classical reconstruction of the target register between recurrence steps. Gates are compiled from the training count distribution (classical synthesis cost) and are shared across examples. Conditional probabilities arise from classical frequency counts and are not quantum-trained weights. Device-level ancilla reset is a dissipative operation.

For clean target y, after t reverse steps with fixed c:
P(y;t) = (1-eta)^t (1-beta) + [1-(1-eta)^t] P_model(y|c)
(assuming no physical-gate/readout error). This closed-form formula is an independent audit and never a shortcut used for predictions.

The reported "noise" is controlled quantum masking and collision reset; the optional --readout-error inserts a separate measurement bit-flip noise model. A positive effect from denoising does NOT establish that uncontrolled hardware noise aids language modeling. Denoising is directed toward a conditional distribution known from classical counts, and thus the experiment does not demonstrate a learned generative quantum model.

## Commands

    python -m pip install -e '.[test]'
    python -m qcenn_qml.diffusion.benchmark --shots 1024 --examples 8 --beta 1 --eta 0.65 --steps 0 1 3 --output-dir results/diffusion/ideal
    python -m qcenn_qml.diffusion.benchmark --shots 1024 --examples 8 --beta 1 --eta 0.65 --steps 0 1 3 --readout-error 0.05 --output-dir results/diffusion/noisy

Compare both runs at the same seeds/shots. Outputs: report.json and report.md. Qiskit Aer simulates all gates, resets and measurements with declared noise settings. This is not physical QPU execution. Large steps/depth can be expensive because each 16-context loader uses controlled state-preparation gates.

## Acceptance criteria for extending

- Add a configurable multi-token masking schedule and learned reversible/conditional quantum denoiser with held-out teacher forcing and conditional likelihood.
- Compare to matched classical conditional-frequency, quantum associative-memory, and ordinary quantum circuit baselines; disclose the cost of classical loading and compilation.
- Derive an actual QCeNN-MI-inspired residual channel with a context-dependent fixed point. Prove CPTP, fixed-point and convergence properties; the present replacement channel is only an analog of recurrent dissipation.
- Test train/validation/test separation, realistic vocabulary scale, arbitrary mask positions, and true hardware noise; report parameter count, shots, actual compiled native depth, runtime and hardware jobs.
- Never describe this as SmolLM2-135M running on quantum hardware or as quantum advantage.
