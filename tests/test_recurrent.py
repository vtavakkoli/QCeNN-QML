import numpy as np
import pytest

from qcenn_qml.recurrent import (apply_damping, cell_density, decode_counts, evolve,
    initial_parameters, probabilities, recurrent_circuit)
from qcenn_qml.recurrent_mnist import (classification_metrics, exemplar_scores,
    patches_from_images, train_model)
from qcenn_qml.mnist_data import read_idx, split_manifest, stratified_indices


def aer_density(circuit):
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    backend = AerSimulator(method="density_matrix", max_parallel_threads=1)
    circuit = circuit.copy()
    circuit.save_density_matrix(list(range(4)), label="memory")
    compiled = transpile(circuit, backend, optimization_level=0, num_processes=1)
    result = backend.run(compiled, shots=1, seed_simulator=7).result()
    assert result.success
    return np.asarray(result.data(0)["memory"])


@pytest.mark.parametrize("damping", [-6., -1., 2.])
def test_complete_collision_circuit_matches_independent_reduced_state(damping):
    rng = np.random.default_rng(92)
    patches = rng.random((2, 3, 4))
    theta = rng.normal(0, .8, 19)
    theta[6] = damping
    batched = evolve(patches, theta)
    for i in range(2):
        actual = aer_density(recurrent_circuit(patches[i], theta, measure=False))
        np.testing.assert_allclose(actual, batched[i], atol=3e-13, rtol=0)
        assert np.trace(batched[i]) == pytest.approx(1)
        assert np.linalg.eigvalsh(batched[i]).min() >= -1e-13


def test_local_collision_is_cptp_and_preserves_correlations():
    from qiskit.quantum_info import DensityMatrix, Kraus
    rng = np.random.default_rng(3)
    matrix = rng.normal(size=(16, 16)) + 1j*rng.normal(size=(16, 16))
    rho = matrix @ matrix.conj().T
    rho /= np.trace(rho)
    gamma = .6
    k0 = np.diag([1., np.cos(gamma)])
    k1 = np.array([[0., np.sin(gamma)], [0., 0.]])
    np.testing.assert_allclose(k0.T @ k0 + k1.T @ k1, np.eye(2), atol=1e-14)
    for qubit in range(4):
        expected = DensityMatrix(rho).evolve(Kraus([k0, k1]), qargs=[qubit]).data
        actual = apply_damping(rho[None], gamma, qubit)[0]
        np.testing.assert_allclose(actual, expected, atol=1e-14)
    updated = cell_density(rho[None], np.array([[.1, .3, .8, 1.]]), initial_parameters(8))[0]
    assert np.trace(updated) == pytest.approx(1)
    assert np.linalg.eigvalsh(updated).min() >= -1e-13


def test_input_driven_fixed_points_remain_distinct():
    theta = np.zeros(19)
    theta[1] = .7
    # gamma=pi/4 gives nontrivial dissipation. All other shared interactions zero.
    inputs = np.array([[0., 0., 0., 0.], [.4, .2, .7, .1]])
    short = probabilities(np.repeat(inputs[:, None], 80, axis=1), theta)
    long = probabilities(np.repeat(inputs[:, None], 160, axis=1), theta)
    np.testing.assert_allclose(short, long, atol=1e-10)
    assert np.linalg.norm(long[0, :10]-long[1, :10]) > .1


def test_memory_carries_information_from_earlier_patches():
    patches = np.array([[[0., .2, .3, .7], [.2, .1, .8, .4]],
                        [[1., .7, .8, .3], [.2, .1, .8, .4]]])
    theta = initial_parameters(42)
    recurrent = probabilities(patches, theta)
    reset = probabilities(patches, theta, retain_memory=False)
    assert np.linalg.norm(recurrent[0]-recurrent[1]) > .01
    np.testing.assert_allclose(reset[0], reset[1], atol=1e-14)
    actual = aer_density(recurrent_circuit(patches[0], theta, retain_memory=False, measure=False))
    np.testing.assert_allclose(actual, evolve(patches[:1], theta, retain_memory=False)[0], atol=1e-13)


def test_measured_readout_has_no_ideal_fallback():
    result = decode_counts({"1111": 17, "1010": 15})
    assert result["prediction"] == -1
    assert result["accepted_shots"] == 0
    result = decode_counts({"1001": 9, "0000": 2, "1111": 5})
    assert result["prediction"] == 9
    assert result["accepted_shot_fraction"] == pytest.approx(11/16)
    metrics = classification_metrics([9, 0], [9, -1])
    assert metrics["accuracy"] == .5
    assert metrics["rejected_examples"] == 1
    assert metrics["confusion_matrix"][0][-1] == 1


def test_final_shots_agree_with_born_probabilities():
    from qcenn_qml.recurrent_mnist import run_circuits
    patches = np.random.default_rng(8).random((1, 2, 4))
    theta = initial_parameters(20)
    counts, compiled, _ = run_circuits([recurrent_circuit(patches[0], theta)], 8192, 33)
    decoded = decode_counts(counts[0])
    observed = np.asarray(decoded["counts"])/8192
    np.testing.assert_allclose(observed, probabilities(patches, theta)[0], atol=.025)
    assert compiled[0].num_qubits == 5
    original = recurrent_circuit(patches[0], theta)
    operations = [entry.operation.name for entry in original.data]
    assert operations.count("measure") == 4
    assert operations.count("reset") == 8
    assert operations[-4:] == ["measure"]*4


def test_seeded_stratification_no_train_validation_overlap():
    train_labels = np.tile(np.arange(10), 9)
    test_labels = np.tile(np.arange(10), 5)
    one = stratified_indices(train_labels, test_labels, 3, 2, 2, 11)
    two = stratified_indices(train_labels, test_labels, 3, 2, 2, 11)
    different = stratified_indices(train_labels, test_labels, 3, 2, 2, 12)
    for first, repeated in zip(one, two, strict=True):
        np.testing.assert_array_equal(first, repeated)
    assert not set(one[0]) & set(one[1])
    assert not np.array_equal(one[0], different[0])
    np.testing.assert_array_equal(np.bincount(train_labels[one[0]]), np.full(10, 3))
    with pytest.raises(ValueError, match="Not enough"):
        stratified_indices(train_labels, test_labels, 8, 2, 2, 11)
    manifest = split_manifest("digits-proxy", 11, *one, np.zeros((90, 8, 8)),
                               train_labels, np.zeros((50, 8, 8)), test_labels)
    assert manifest["is_real_mnist"] is False
    assert manifest["splits"]["validation"]["source"] == "training_source"


def test_patch_order_and_idx_validation():
    import struct
    image = np.arange(16, dtype=np.uint8).reshape(4, 4)
    patches = patches_from_images(image[None], side=4)
    np.testing.assert_allclose(patches[0, 0], np.array([0, 1, 4, 5])/255)
    np.testing.assert_allclose(patches[0, -1], np.array([10, 11, 14, 15])/255)
    raw = b"\x00\x00\x08\x03" + struct.pack(">III", 1, 4, 4) + image.tobytes()
    np.testing.assert_array_equal(read_idx(raw), image[None])
    with pytest.raises(ValueError, match="length"):
        read_idx(raw[:-1])
    with pytest.raises(ValueError, match="header"):
        read_idx(b"corrupt")


def test_hybrid_training_is_deterministic_and_validation_selects_checkpoint():
    rng = np.random.default_rng(3)
    train, validation = rng.random((10, 2, 4)), rng.random((10, 2, 4))
    options = dict(seed=17, steps=3, batch_size=4, validation_interval=1)
    first, first_log = train_model(train, np.arange(10), validation, np.arange(10), **options)
    second, second_log = train_model(train, np.arange(10), validation, np.arange(10), **options)
    np.testing.assert_array_equal(first, second)
    assert first_log["history"] == second_log["history"]
    assert first_log["selected_validation_nll"] == min(row["validation_nll"] for row in first_log["history"])
    assert first_log["quantum_training_shots"] == 0


def test_exact_exemplar_comparison_uses_same_score_aggregation():
    prototypes = np.eye(16)[:10]
    scores = exemplar_scores(prototypes, prototypes, np.arange(10), 0)
    np.testing.assert_allclose(scores, np.eye(10), atol=1e-14)


def test_ten_class_proxy_comparison_pipeline(tmp_path):
    from qiskit import qpy
    from qcenn_qml.recurrent_mnist import evaluate
    report = evaluate(dataset="digits-proxy", seeds=[3], train_per_class=1,
        validation_per_class=1, test_per_class=1, exemplars_per_class=1,
        side=4, training_steps=1, batch_size=4, validation_interval=1,
        shots=64, overlap_iterations=[0, 1], include_reset_control=True,
        output_dir=str(tmp_path / "run"))
    assert report["classes"] == list(range(10))
    assert report["is_real_mnist"] is False
    assert report["training_is_hybrid"] is True
    methods = report["runs"][0]["methods"]
    assert methods["recurrent"]["resources"]["total_shots"] == 10*64
    for k in (0, 1):
        assert methods[f"quantum_overlap_k{k}"]["resources"]["total_shots"] == 100*64
        assert methods[f"classical_overlap_k{k}"]["n"] == 10
    assert all(sum(row["counts"]) == 64 for row in methods["recurrent"]["predictions"])
    assert (tmp_path / "run/seed-3/recurrent/model.json").exists()
    with (tmp_path / "run/seed-3/recurrent/example-native.qpy").open("rb") as stream:
        assert qpy.load(stream)[0].num_qubits == 5
    with pytest.raises(FileExistsError, match="fresh"):
        evaluate(dataset="digits-proxy", output_dir=str(tmp_path / "run"))
