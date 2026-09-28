"""Independent algebra, fitting, force-chain and immutable-model checks."""

import json

import numpy as np
import pytest

from molsimflow.postprocess.conditional_path import (
    SCHEMA,
    audit_total_bias,
    evaluate_log_normalizer,
    fit_log_normalizer,
    load_model,
    log_mixture,
    main,
    plumed_correction,
    save_model,
)
from molsimflow.postprocess.pimd_fes import (
    frame_log_weights,
    normalized_log_weights,
    total_bias_energy,
)


def model():
    return {
        "schema": SCHEMA,
        "domain": [-2.0, 2.0],
        "coefficients": [-0.8, 0.3, 0.1],
        "epsilon": 0.1,
        "fit_diagnostics": {},
        "provenance": {},
    }


@pytest.mark.parametrize("coupling", [0, 0.25, 0.9, 1 - 1e-12])
def test_mixture_and_full_chain_rule_against_independent_energy(coupling):
    x = np.array([-0.6, 0.4, 1.1, 0.2])

    def potential(path):
        c = np.mean(path) ** 2
        a = 0.1 + np.mean(np.exp(-(path**2) / 2))
        # Independent direct polynomial and mixture, not evaluate_log_normalizer.
        m = np.exp(-0.8 + 0.3 * c / 2 + 0.1 * (c / 2) ** 2)
        return 0.7 * c * c - 2.3 * np.log(1 - coupling + coupling * a / m)

    c = np.mean(x) ** 2
    logm, dm = evaluate_log_normalizer(model(), c)
    h = np.exp(-x * x / 2)
    a = 0.1 + np.mean(h)
    _, chi = log_mixture(np.log(a), logm, coupling)
    dc = 2 * np.mean(x) / len(x)
    force = -1.4 * c * dc + 2.3 * chi * ((-x * h / len(x)) / a - dm * dc)
    numerical = []
    for bead in range(len(x)):
        dx = np.zeros_like(x)
        dx[bead] = 1e-5
        numerical.append(-(potential(x + dx) - potential(x - dx)) / 2e-5)
    np.testing.assert_allclose(force, numerical, atol=2e-10)
    if coupling > 0:
        assert np.max(np.abs(force + 2.3 * chi * dm * dc - numerical)) > 1e-4


@pytest.mark.parametrize("delta", [-1000.0, -20.0, 0.0, 20.0, 1000.0])
def test_extreme_log_ratios_and_matching_normalizer(delta):
    ratio, chi = log_mixture(delta, 0.0, 0.5)
    assert np.isfinite(ratio) and 0 <= chi <= 1
    if delta == 0:
        assert ratio == pytest.approx(0)
    if abs(delta) == 1000:
        assert ratio == pytest.approx(max(delta, 0) - np.log(2))
    zero, derivative = log_mixture(delta, 0.0, 0)
    assert zero == 0 and derivative == 0


@pytest.mark.parametrize("coupling", [-0.1, 1, np.nan, np.inf, True])
def test_invalid_coupling(coupling):
    with pytest.raises(ValueError):
        log_mixture(0.0, 0.0, coupling)


def test_domain_model_validation_and_derivative():
    m = model()
    c = np.array([-1.8, -0.1, 1.9])
    _, d = evaluate_log_normalizer(m, c)
    plus, _ = evaluate_log_normalizer(m, c + 1e-6)
    minus, _ = evaluate_log_normalizer(m, c - 1e-6)
    np.testing.assert_allclose(d, (plus - minus) / 2e-6, atol=1e-10)
    for bad in [2.001, np.nan, np.inf, [1 + 1j]]:
        with pytest.raises(ValueError):
            evaluate_log_normalizer(m, bad)
    m["coefficients"] = [float("nan")]
    with pytest.raises(ValueError):
        evaluate_log_normalizer(m, 0)


def pilot():
    centers = np.array([-0.75, -0.25, 0.25, 0.75])
    c = np.tile(centers, 4)
    fraction = np.exp(-1 + 0.2 * c + 0.1 * c * c) - 0.1
    return {
        "conditioning": c,
        "region_fraction": fraction,
        "log_weights": np.zeros(16),
        "bin_edges": np.linspace(-1, 1, 5),
        "block_ids": np.repeat(np.arange(4), 4),
        "epsilon": 0.1,
        "degree": 2,
        "min_frame_ess": 3,
        "stationary_pilot": True,
        "provenance": {"fixture": "exact log quadratic"},
    }


def test_fitter_recovers_known_function_and_preserves_whole_frame_counts():
    m = fit_log_normalizer(**pilot())
    np.testing.assert_allclose(m["coefficients"], [-1, 0.2, 0.1], atol=1e-14)
    diagnostics = m["fit_diagnostics"]["conditional_statistics"]
    assert diagnostics["frame_count"] == 16
    assert diagnostics["block_count"] == 4
    assert all(v["frame_ess"] == 4 for v in diagnostics["cells"])
    # Correlated beads have already been averaged; no P factor enters fitting.
    assert m["fit_diagnostics"]["held_out_validation"] == "NOT_ASSESSED"


@pytest.mark.parametrize(
    "change",
    [
        {"stationary_pilot": False},
        {"min_frame_ess": 5},
        {"degree": 6},
        {"epsilon": 0},
        {"bin_edges": [-0.5, 0, 0.5]},
        {"block_ids": np.arange(16) % 2},
        {"region_fraction": np.ones(16) * 1.1},
    ],
)
def test_fitting_rejects_unsupported_or_ambiguous_inputs(change):
    args = pilot()
    args.update(change)
    with pytest.raises(ValueError):
        fit_log_normalizer(**args)


def test_hash_lock_no_overwrite_and_export_cli(tmp_path):
    m = fit_log_normalizer(**pilot())
    path = tmp_path / "model.json"
    digest = save_model(m, path)
    assert load_model(path, expected_sha256=digest) == m
    with pytest.raises(FileExistsError):
        save_model(m, path)
    output = tmp_path / "correction.dat"
    assert (
        main(
            [
                "export",
                "--model",
                str(path),
                "--model-sha256",
                digest,
                "--output",
                str(output),
                "--centroid-label",
                "c",
                "--score-label",
                "a",
                "--coupling",
                "0.5",
                "--kbt",
                "2.5",
            ]
        )
        == 0
    )
    graph = output.read_text()
    assert "CONDITIONAL_PATH ARG=c,conditional_loga,conditional_logm" in graph
    assert "BIASVALUE ARG=conditional_energy" in graph and digest in graph
    changed = json.loads(path.read_text())
    changed["coefficients"][0] += 0.1
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_model(path, expected_sha256=digest)
    with pytest.raises(ValueError, match="identifiers"):
        plumed_correction(
            m, centroid_label="c\nPRINT ARG=*", score_label="a", prefix="b", coupling=0.5, kbt=1
        )


def test_single_state_reweighting_recovers_known_path_distribution():
    # Two conditioning cells, two distinct paths per cell, exact conditional m.
    target = np.array([0.1, 0.3, 0.2, 0.4])
    a = np.array([0.2, 0.8, 0.5, 1.0])
    normalizer = np.repeat([np.dot(target[:2], a[:2]) / 0.4, np.dot(target[2:], a[2:]) / 0.6], 2)
    bc = np.repeat([0.3, -0.2], 2)
    ratio, _ = log_mixture(np.log(a), np.log(normalizer), 0.6)
    total = bc - ratio
    biased = target * np.exp(-total)
    biased /= biased.sum()
    baseline = target * np.exp(-bc)
    baseline /= baseline.sum()
    np.testing.assert_allclose(biased.reshape(2, 2).sum(1), baseline.reshape(2, 2).sum(1))
    energy = total_bias_energy("centroid_conditioned", sampling_bias_energy=total)
    logs = frame_log_weights("fixed_bias", bias_energy=energy, kbt=1)
    recovered = np.exp(normalized_log_weights(np.log(biased) + logs))
    np.testing.assert_allclose(recovered, target, atol=1e-15)
    assert (
        audit_total_bias(total, bc, np.log(a), np.log(normalizer), coupling=0.6, kbt=1, atol=1e-14)
        == 0
    )
    with pytest.raises(ValueError, match="does not match"):
        audit_total_bias(bc, bc, np.log(a), np.log(normalizer), coupling=0.6, kbt=1, atol=1e-10)
    with pytest.raises(ValueError):
        total_bias_energy("centroid_conditioned", bead_bias_energies=np.ones((4, 32)))


@pytest.mark.parametrize(
    "centroid,score",
    [
        ("c", "c"),
        ("conditional_logm", "a"),
        ("c", "conditional_bias.bias"),
    ],
)
def test_export_rejects_label_collisions(centroid, score):
    with pytest.raises(ValueError, match="input labels"):
        plumed_correction(
            model(),
            centroid_label=centroid,
            score_label=score,
            prefix="conditional",
            coupling=0.5,
            kbt=1,
        )


def test_export_rejects_unrepresentable_domain_and_boolean_temperature():
    m = model()
    m["domain"] = [0.0, np.nextafter(0.0, 1.0)]
    with pytest.raises(ValueError, match="domain width"):
        evaluate_log_normalizer(m, 0.0)
    with pytest.raises(ValueError, match="kbt"):
        plumed_correction(
            model(),
            centroid_label="c",
            score_label="a",
            prefix="conditional",
            coupling=0.5,
            kbt=True,
        )
