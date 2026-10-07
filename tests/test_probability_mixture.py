import itertools

import numpy as np
import pytest

from molsimflow.postprocess.probability_mixture import (
    arithmetic_bias,
    audit_record,
    centroid_mixture,
    export_plumed,
    log_mean_exp,
    probability_ratio_bias,
    validate_manifest,
)


def manifest(mode="bead_probability_mixture"):
    return {"schema": "probability-mixture-v1", "frozen": True, "mode": mode,
            "expected_beads": 2, "kbt": 1.0, "energy_unit": "reduced",
            "field_sha256": "a" * 64, "centroid_field_sha256": "b" * 64,
            "coupling": 0.4, "log_normalizer": 0.2}


@pytest.mark.parametrize("bias_function", [probability_ratio_bias, arithmetic_bias])
@pytest.mark.parametrize("beads", [1, 2, 8, 32])
def test_gradient_permutation_equal_beads(beads, bias_function):
    values = np.linspace(-3, 2, beads)
    energy, alpha = bias_function(values, kbt=0.7)
    assert np.isclose(alpha.sum(), 1)
    for b in range(beads):
        for step in [1e-4, 1e-5, 1e-6]:
            plus, minus = values.copy(), values.copy()
            plus[b] += step
            minus[b] -= step
            derivative = (bias_function(plus, kbt=0.7)[0]
                          - bias_function(minus, kbt=0.7)[0]) / (2 * step)
            assert derivative == pytest.approx(alpha[b], abs=1e-8, rel=1e-6)
    shifted, weights = bias_function(values + 123, kbt=0.7)
    assert shifted == pytest.approx(energy + 123)
    np.testing.assert_allclose(weights, alpha)
    permuted, weights = bias_function(values[::-1], kbt=0.7)
    assert permuted == pytest.approx(energy)
    np.testing.assert_allclose(weights, alpha[::-1])
    energy, alpha = bias_function(np.full(beads, 2.3), kbt=0.7)
    assert energy == pytest.approx(2.3)
    np.testing.assert_allclose(alpha, 1 / beads)


def test_extreme_logs_and_input_rejection():
    value, alpha = log_mean_exp([-10000, 10000])
    assert value == pytest.approx(10000 - np.log(2))
    np.testing.assert_array_equal(alpha, [0, 1])
    for values in ([], 1, [np.nan], [np.inf]):
        with pytest.raises(ValueError):
            log_mean_exp(values)
    for thermal in (0, -1, np.inf, np.nan):
        with pytest.raises(ValueError):
            arithmetic_bias([1], kbt=thermal)


@pytest.mark.parametrize("eta", [0, 0.01, 0.5, 0.999])
def test_mixed_derivatives_and_gauge(eta):
    vc, va, thermal, log_c = 1.2, -0.8, 0.7, 1.3
    energy, chi = centroid_mixture(vc, va, kbt=thermal, coupling=eta,
                                  log_normalizer=log_c)
    for arg, expected in [(0, 1 - chi), (1, chi)]:
        for step in [1e-4, 1e-5, 1e-6]:
            plus, minus = [vc, va], [vc, va]
            plus[arg] += step
            minus[arg] -= step
            ep = centroid_mixture(*plus, kbt=thermal, coupling=eta, log_normalizer=log_c)[0]
            em = centroid_mixture(*minus, kbt=thermal, coupling=eta, log_normalizer=log_c)[0]
            assert (ep - em) / (2 * step) == pytest.approx(expected, abs=1e-8, rel=1e-6)
    shifted, shifted_chi = centroid_mixture(vc + 3, va - 2, kbt=thermal,
                                          coupling=eta, log_normalizer=log_c + 5 / thermal)
    assert shifted == pytest.approx(energy + 3)
    assert shifted_chi == pytest.approx(chi)
    if eta == 0:
        assert centroid_mixture(vc, np.nan, kbt=thermal, coupling=0, log_normalizer=0)[0] == vc


def test_exact_path_target_and_uniform_observable():
    p, target = np.array([0.8, 0.2]), np.array([0.3, 0.7])
    paths = np.array(list(itertools.product(range(2), repeat=3)))
    p0 = p[paths].prod(axis=1)
    ratio = target / p
    va, alpha = arithmetic_bias(-np.log(ratio[paths]), kbt=1)
    qa = p0 * np.exp(-va)
    assert qa.sum() == pytest.approx(1)
    uniform = (paths == 1).mean(axis=1)
    assert np.dot(qa, uniform) == pytest.approx(target[1] / 3 + p[1] * 2 / 3)
    recovered = qa * np.exp(va)
    assert np.dot(recovered, uniform) / recovered.sum() == pytest.approx(p[1])
    # Force coefficients do not define the physical uniform-bead observable.
    assert np.dot(qa, np.sum(alpha * (paths == 1), axis=1)) == pytest.approx(target[1])
    vc = 0.3 * paths.sum(axis=1) ** 2
    zc = np.dot(p0, np.exp(-vc))
    mixed, _ = centroid_mixture(vc, va, kbt=1, coupling=0.4, log_normalizer=-np.log(zc))
    np.testing.assert_allclose(p0 * np.exp(-mixed) / zc, 0.6 * p0 * np.exp(-vc) / zc + 0.4 * qa)


def test_manifest_audit_and_export():
    spec = manifest()
    values = np.array([[0.1, 0.5], [0.2, -1]])
    total, _ = arithmetic_bias(values, kbt=1)
    assert audit_record(spec, values, total, field_sha256="a" * 64)["frames"] == 2
    with pytest.raises(ValueError, match="total bias"):
        audit_record(spec, values, values.mean(axis=1), field_sha256="a" * 64)
    with pytest.raises(ValueError, match="every bead"):
        audit_record(spec, values[:, :1], total, field_sha256="a" * 64)
    with pytest.raises(ValueError, match="identity"):
        audit_record(spec, values, total, field_sha256="c" * 64)
    with pytest.raises(ValueError):
        validate_manifest({**spec, "expected_beads": True})
    graph = export_plumed(spec, field_arg="opes.bias", active_field_bias=True)
    assert "EXPECTED_REPLICAS=2" in graph and "FUNC=x-y" in graph
    spec = manifest("centroid_probability_mixture")
    spec["coupling"] = 0
    assert audit_record(spec, None, [0.2], field_sha256="a" * 64, centroid_bias=[0.2])["inactive_arithmetic_field"]
    assert export_plumed(spec, field_arg="unused", centroid_arg="vc") == "pm_bias: BIASVALUE ARG=vc\n"
    with pytest.raises(ValueError):
        export_plumed(spec, field_arg="opes.bias", centroid_arg="vc", active_field_bias=True)


@pytest.mark.parametrize("eta", [-0.1, 1, np.nan, np.inf])
def test_invalid_coupling(eta):
    with pytest.raises(ValueError):
        centroid_mixture(0, 0, kbt=1, coupling=eta, log_normalizer=0)
