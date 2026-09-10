import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from molsimflow.postprocess.pimd_fes import (
    assemble_bead_frames,
    frame_log_weights,
    quantum_fes_1d,
    quantum_histogram_masses,
    restart_unique_indices,
    total_bias_energy,
    validate_bias_mode,
)
from molsimflow.postprocess.pimd_reweight import (
    KB_EV_PER_K,
    analysis_profile,
    aligned_time_indices,
    analyze,
    adapt_reference_source,
    artifact_basename,
    compute_surfaces,
    cumulative_weight_diagnostics,
    cv_column_names,
    diagnostic_cv_spec,
    estimator_plot_labels,
    inverse_piecewise_logdistance,
    normalized_log_weights,
    piecewise_derived_coordinate_spec,
    piecewise_logdistance,
    piecewise_logdistance_jacobian,
    portable_artifact_path,
    reconstruction_within_tolerance,
    ring_polymer_spread,
    sampling_protocol_label,
    sha256,
    soft_voronoi_occupancies,
    surface_difference_metrics,
    threshold_run_rows,
    time_window_mask,
    transform_piecewise_logdistance_fes,
    validate_piecewise_logdistance_printed,
    verify_manifest_inputs,
)
from molsimflow.postprocess.pimd_reweight_compare import (
    aligned_surface_difference,
    comparison_report_title,
)


def test_normalized_log_weights_are_finite_and_sum_to_one():
    log_weights = normalized_log_weights(np.array([-900.0, -10.0, 0.0, 3.0]))
    weights = np.exp(log_weights)
    assert np.isfinite(log_weights).all()
    assert np.isclose(np.sum(weights), 1.0)


def test_fixed_and_quasi_static_opes_frame_weights_use_total_bias_only():
    bias = np.array([-0.2, 0.0, 0.3])
    expected = bias / 0.025
    assert np.allclose(
        frame_log_weights("fixed_bias", bias_energy=bias, kbt=0.025), expected
    )
    assert np.allclose(
        frame_log_weights(
            "quasi_static_opes",
            bias_energy=bias,
            kbt=0.025,
            quasi_static=True,
        ),
        expected,
    )
    try:
        frame_log_weights(
            "quasi_static_opes",
            bias_energy=bias,
            kbt=0.025,
            quasi_static=False,
        )
    except ValueError as exc:
        assert "declared explicitly" in str(exc)
    else:
        raise AssertionError("adaptive OPES was accepted without a quasi-static declaration")


def test_all_three_bias_modes_and_total_path_energies_are_explicit():
    assert validate_bias_mode("centroid_coord") == "centroid_coord"
    assert validate_bias_mode("bead_mean") == "bead_mean"
    assert validate_bias_mode("bead_density_shared") == "bead_density_shared"
    sampling = np.array([-0.3, 0.2])
    assert np.array_equal(
        total_bias_energy("centroid_coord", sampling_bias_energy=sampling), sampling
    )
    local = np.array([[-0.8, -0.4, 0.0, 0.4], [0.1, 0.2, 0.3, 0.4]])
    assert np.allclose(
        total_bias_energy("bead_density_shared", bead_bias_energies=local),
        [-0.2, 0.25],
    )


def test_estimator_plot_labels_use_equation_numbers_only_for_centroid():
    centroid = estimator_plot_labels("centroid_coord")
    assert centroid["probability_mean"] == "Quantum FES (Lamaire Eq. 8)"
    assert centroid["logmean"] == "Bead-logmean diagnostic (Lamaire Eq. 10)"
    for mode in ("bead_mean", "bead_density_shared"):
        labels = estimator_plot_labels(mode)
        assert labels["probability_mean"] == "Quantum FES"
        assert labels["logmean"] == "Bead-logmean diagnostic"
        assert "Eq." not in " ".join(labels.values())


def test_analysis_profile_and_protocol_labels_are_explicit():
    assert analysis_profile({}) == "water_ionization_opes"
    assert analysis_profile({"analysis_profile": "core"}) == "core"
    assert sampling_protocol_label({"weight_kind": "fixed_bias"}) == "fixed bias"
    assert sampling_protocol_label(
        {"weight_kind": "precomputed", "protocol_label": "WTMetaD reweighting"}
    ) == "WTMetaD reweighting"


def test_bead_density_rejects_a_single_sampling_bias_energy():
    try:
        total_bias_energy(
            "bead_density_shared", sampling_bias_energy=np.array([-0.3, 0.2])
        )
    except ValueError as exc:
        assert "bead-local" in str(exc)
    else:
        raise AssertionError("bead-density accepted a single-bead bias energy")


def test_direct_and_conditional_histograms_match_for_all_bias_modes():
    bead_cv = np.array(
        [
            [-0.8, -0.6, -0.4],
            [-0.3, -0.1, 0.1],
            [0.0, 0.2, 0.4],
            [0.5, 0.7, 0.9],
            [0.8, 0.9, 1.0],
        ]
    )
    conditions = {
        "centroid_coord": np.array([-0.7, -0.2, 0.1, 0.6, 0.9]),
        "bead_mean": np.mean(bead_cv, axis=1),
        "bead_density_shared": np.mean(bead_cv, axis=1),
    }
    for mode, condition in conditions.items():
        validate_bias_mode(mode)
        log_weights = frame_log_weights(
            "fixed_bias", bias_energy=0.2 * condition**2, kbt=0.6
        )
        result = quantum_histogram_masses(
            bead_cv,
            log_weights,
            np.linspace(-1.0, 1.0, 9),
            conditioning=condition,
            conditioning_edges=np.linspace(-1.0, 1.0, 6),
        )
        assert np.allclose(result["direct"], result["conditional"], atol=1e-15)


def test_p1_eq8_and_eq10_are_identical():
    bead_cv = np.array([[-0.75], [-0.25], [0.25], [0.75]])
    result = quantum_fes_1d(
        bead_cv,
        frame_log_weights("precomputed", precomputed=[0.0, 0.2, -0.1, 0.3]),
        np.linspace(-1.0, 1.0, 5),
        kbt=0.6,
    )
    assert np.allclose(result["eq8"][result["support"]], result["eq10"][result["support"]])
    assert np.array_equal(result["probability_mean"], result["eq8"])
    assert np.array_equal(result["logmean_diagnostic"], result["eq10"])


def test_long_bead_table_rejects_a_missing_bead():
    frame_ids = [0, 0, 1]
    bead_ids = [1, 2, 1]
    try:
        assemble_bead_frames(frame_ids, bead_ids, [0.1, 0.2, 0.3], expected_beads=2)
    except ValueError as exc:
        assert "missing beads" in str(exc)
    else:
        raise AssertionError("an incomplete ring-polymer frame was accepted")


def test_restart_seam_deduplication_keeps_the_predecessor_endpoint():
    indices, duplicates = restart_unique_indices([0, 1, 1, 2], policy="keep_first")
    assert indices.tolist() == [0, 1, 3]
    assert duplicates == 1
    try:
        restart_unique_indices([0, 1, 1, 2], policy="error")
    except ValueError as exc:
        assert "restart-seam" in str(exc)
    else:
        raise AssertionError("a duplicate restart seam was accepted under error policy")


def test_representation_specific_cv_columns_default_and_override():
    logical = ("logdistance", "ionization")
    assert cv_column_names({}, "sampling_cv_names", logical) == logical
    assert cv_column_names(
        {"sampling_cv_names": ["mean.logdistance", "mean.ionization"]},
        "sampling_cv_names",
        logical,
    ) == ("mean.logdistance", "mean.ionization")


def test_direct_diagnostic_cv_is_validated_without_becoming_a_fes_coordinate():
    spec = diagnostic_cv_spec(
        {
            "name": "iondistance",
            "sampling_column": "mean.iondistance",
            "bead_column": "iondistance",
            "label": "iondistance (A)",
        }
    )
    assert spec is not None
    assert spec["name"] == "iondistance"
    assert spec["mean_tolerance"] == 1e-12


def test_cumulative_weight_diagnostics_are_scale_invariant():
    ess_fraction, maximum_share = cumulative_weight_diagnostics([1.0, 1.0, 2.0])
    assert np.allclose(ess_fraction, [1.0, 1.0, 8.0 / 9.0])
    assert np.allclose(maximum_share, [1.0, 0.5, 0.5])
    scaled = cumulative_weight_diagnostics([10.0, 10.0, 20.0])
    assert np.allclose(scaled[0], ess_fraction)
    assert np.allclose(scaled[1], maximum_share)


def test_identical_beads_make_eq8_and_eq10_identical():
    centroid = np.array(
        [
            [-1.0, 0.1],
            [-0.6, 0.2],
            [-0.2, 0.4],
            [0.2, 0.7],
            [0.6, 0.9],
        ]
    )
    beads = np.repeat(centroid[:, None, :], 4, axis=1)
    surfaces = compute_surfaces(
        beads,
        centroid,
        np.linspace(-1.0, 1.0, len(centroid)),
        np.linspace(-1.5, 1.0, 13),
        np.linspace(-0.2, 1.2, 11),
        (0.2, 0.1),
        0.025852,
    )
    assert np.allclose(surfaces["eq8"], surfaces["eq10"], atol=1e-12)
    assert np.allclose(surfaces["centroid"], surfaces["eq10"], atol=1e-12)
    assert np.min(surfaces["raw_eq10"] - surfaces["raw_eq8"]) >= -1e-12


def test_reference_grid_patch_is_minimal_and_explicit():
    source = "before\nx,y=np.meshgrid(grid_cv_x,grid_cv_y)\nafter\n"
    adapted = adapt_reference_source(source)
    assert adapted == "before\nx,y=np.meshgrid(grid_cv_x,grid_cv_y,indexing='ij')\nafter\n"


def test_output_artifact_paths_are_portable_and_basenames_are_safe():
    with TemporaryDirectory() as directory:
        temporary = Path(directory)
        output = temporary / ".staging-result-123"
        assert portable_artifact_path(output / "qc" / "reference.dat", output) == (
            "{output}/qc/reference.dat"
        )
        external = temporary / "driver.py"
        assert portable_artifact_path(external, output) == str(external)
        assert artifact_basename({}, "filtered_colvar_name", "COLVAR.after-50ps") == (
            "COLVAR.after-50ps"
        )
        assert artifact_basename(
            {"filtered_colvar_name": "COLVAR.selected-1to6ns"},
            "filtered_colvar_name",
            "COLVAR.after-50ps",
        ) == "COLVAR.selected-1to6ns"
        for invalid in ("", "../COLVAR", "inputs/COLVAR"):
            try:
                artifact_basename(
                    {"filtered_colvar_name": invalid},
                    "filtered_colvar_name",
                    "COLVAR.after-50ps",
                )
            except ValueError:
                pass
            else:
                raise AssertionError("unsafe basename was accepted")


def test_surface_difference_uses_only_shared_support():
    reference = np.array([[0.0, 1.0], [2.0, 3.0]])
    current = np.array([[0.0, 2.0], [102.0, 103.0]])
    count, rmse, maximum = surface_difference_metrics(
        reference, current, np.array([[True, True], [False, False]])
    )
    assert count == 2
    assert np.isclose(rmse, np.sqrt(0.5))
    assert np.isclose(maximum, 1.0)


def test_soft_voronoi_occupancies_recover_neutral_and_pair_like_counts():
    types = np.array([2, 2, 1, 1, 1, 1])
    neutral = np.array(
        [[1.0, 1.0, 1.0], [7.0, 1.0, 1.0], [1.8, 1.0, 1.0],
         [1.0, 1.8, 1.0], [7.8, 1.0, 1.0], [7.0, 1.8, 1.0]]
    )
    occupancies, hard = soft_voronoi_occupancies(
        neutral, types, np.array([10.0, 10.0, 10.0]), 2, 1, 20.0
    )
    assert np.allclose(occupancies, [2.0, 2.0], atol=1e-12)
    assert hard.tolist() == [2, 2]
    pair_like = neutral.copy()
    pair_like[4] = [1.0, 2.0, 1.0]
    occupancies, hard = soft_voronoi_occupancies(
        pair_like, types, np.array([10.0, 10.0, 10.0]), 2, 1, 20.0
    )
    assert np.allclose(occupancies, [3.0, 1.0], atol=1e-12)
    assert hard.tolist() == [3, 1]


def test_threshold_runs_report_contiguous_duration():
    rows = threshold_run_rows(
        np.arange(6) * 0.1,
        {"centroid": np.array([0.0, 0.6, 0.7, 0.0, 0.8, 0.0])},
        [0.5],
    )
    assert rows[0]["frames_at_or_above"] == 3
    assert rows[0]["contiguous_runs"] == 2
    assert np.isclose(rows[0]["longest_run_ps"], 0.2)


def test_aligned_surface_difference_removes_free_energy_offset_on_support():
    reference = np.array([[0.0, 1.0], [2.0, 3.0]])
    current = reference + 7.5
    difference, offset, rmse, maximum = aligned_surface_difference(
        reference, current, np.array([[True, True], [False, False]])
    )
    assert np.isclose(offset, 7.5)
    assert np.allclose(difference, 0.0)
    assert np.isclose(rmse, 0.0)
    assert np.isclose(maximum, 0.0)


def test_comparison_report_title_is_explicit_or_representation_neutral():
    assert comparison_report_title({}) == "PIMD OPES comparison"
    assert comparison_report_title({"report_title": "NMPIMD vs PIMD"}) == (
        "NMPIMD vs PIMD"
    )


def test_reconstruction_tolerance_can_be_enforced_or_measure_only():
    assert reconstruction_within_tolerance(1e-8, 1e-7)
    assert not reconstruction_within_tolerance(1e-6, 1e-7)
    assert reconstruction_within_tolerance(1.0, None)


def test_piecewise_logdistance_roundtrip_jacobian_and_fes_transform():
    iondistance = np.array([-0.02, 0.0, 0.5, 0.999, 1.0, 2.0])
    logdistance = piecewise_logdistance(iondistance)
    assert np.allclose(inverse_piecewise_logdistance(logdistance), iondistance, atol=1e-12)
    jacobian = piecewise_logdistance_jacobian(iondistance)
    assert np.allclose(jacobian[:4], 1.0 / (iondistance[:4] + 0.03))
    assert np.allclose(jacobian[4:], 1.0)

    kbt = 0.596161
    source_fes = np.zeros((2, len(logdistance)))
    target_grid, target_jacobian, transformed = transform_piecewise_logdistance_fes(
        logdistance, source_fes, kbt
    )
    expected = -kbt * np.log(target_jacobian)
    expected -= np.min(expected)
    assert np.allclose(target_grid, iondistance, atol=1e-12)
    assert transformed.shape == source_fes.shape
    assert np.allclose(transformed[0], expected)
    assert np.allclose(transformed[1], expected)

    source_surface = np.zeros((len(logdistance), 2))
    _, _, transformed_first_axis = transform_piecewise_logdistance_fes(
        logdistance, source_surface, kbt, axis=0
    )
    assert transformed_first_axis.shape == source_surface.shape
    assert np.allclose(transformed_first_axis[:, 0], expected)
    assert np.allclose(transformed_first_axis[:, 1], expected)


def test_printed_piecewise_coordinate_validation_passes_and_fails_closed():
    iondistance = np.array([-0.02, 0.0, 0.5, 1.0, 2.0])
    logdistance = piecewise_logdistance(iondistance)
    result = validate_piecewise_logdistance_printed(
        logdistance, iondistance, tolerance=1e-12
    )
    assert result["maximum_absolute_error"] <= 1e-12

    inconsistent = iondistance.copy()
    inconsistent[2] += 1e-3
    try:
        validate_piecewise_logdistance_printed(
            logdistance, inconsistent, tolerance=1e-12
        )
    except ValueError as exc:
        assert "transform mismatch" in str(exc)
    else:
        raise AssertionError("inconsistent printed coordinate was accepted")


def test_derived_coordinate_contract_is_optional_and_not_an_independent_cv():
    cv_names = ("logdistance", "ionization")
    assert piecewise_derived_coordinate_spec(None, cv_names) is None
    spec = piecewise_derived_coordinate_spec(
        {
            "kind": "piecewise_logdistance",
            "source": "logdistance",
            "target": "iondistance",
            "printed_column": "iondistance",
            "switch": 1.0,
            "offset": 0.03,
            "linear_shift": 0.9704412,
            "printed_transform_tolerance": 1e-12,
            "label": "iondistance (A)",
        },
        cv_names,
    )
    assert spec is not None
    assert spec["source"] == "logdistance"
    assert spec["target"] == "iondistance"
    assert spec["sampling_printed_column"] == "iondistance"

    mapped = piecewise_derived_coordinate_spec(
        {
            "kind": "piecewise_logdistance",
            "source": "logdistance",
            "target": "iondistance",
            "printed_column": "iondistance",
            "sampling_printed_column": "mean.iondistance",
        },
        cv_names,
    )
    assert mapped is not None
    assert mapped["sampling_printed_column"] == "mean.iondistance"

    try:
        piecewise_derived_coordinate_spec(
            {
                "kind": "piecewise_logdistance",
                "source": "logdistance",
                "target": "ionization",
                "printed_column": "iondistance",
            },
            cv_names,
        )
    except ValueError as exc:
        assert "independent biased CV" in str(exc)
    else:
        raise AssertionError("derived coordinate was accepted as a biased CV")


def test_time_window_mask_is_closed_and_does_not_leak_later_kernel_rows():
    times = np.array([1000.0, 1001.0, 1051.0, 1051.0001, 6001.0])
    selected = time_window_mask(times, 1001.0, 1051.0)
    assert selected.tolist() == [False, True, True, False, False]


def test_time_alignment_maps_a_selected_grid_and_rejects_missing_frames():
    source = np.arange(0.0, 2.01, 0.5)
    target = np.array([0.5, 1.5, 2.0])
    assert aligned_time_indices(source, target).tolist() == [1, 3, 4]
    try:
        aligned_time_indices(source, np.array([0.5, 1.25]))
    except ValueError as exc:
        assert "absent" in str(exc)
    else:
        raise AssertionError("missing target frame was accepted")


def test_ring_polymer_spread_uses_only_requested_trajectory_steps():
    def dump_text(offset):
        frames = []
        for step in range(3):
            frames.append(
                "\n".join(
                    [
                        "ITEM: TIMESTEP",
                        str(step),
                        "ITEM: NUMBER OF ATOMS",
                        "2",
                        "ITEM: BOX BOUNDS pp pp pp",
                        "0 10",
                        "0 10",
                        "0 10",
                        "ITEM: ATOMS id type x y z",
                        f"1 1 {offset:.6f} 0 0",
                        f"2 2 {1.0 + offset:.6f} 0 0",
                    ]
                )
            )
        return "\n".join(frames) + "\n"

    with TemporaryDirectory() as directory:
        root = Path(directory)
        paths = [root / "bead-1.dump", root / "bead-2.dump"]
        paths[0].write_text(dump_text(0.0), encoding="utf-8")
        paths[1].write_text(dump_text(0.2), encoding="utf-8")
        rows = ring_polymer_spread(paths, [0, 2], {1: "H", 2: "O"})
        assert [int(row["step"]) for row in rows] == [0, 2]


@pytest.mark.parametrize(
    "weight_kind, declaration, rejected",
    [
        ("precomputed", None, False),
        ("fixed_bias", None, False),
        ("quasi_static_opes", True, False),
        ("quasi_static_opes", None, True),
        ("quasi_static_opes", False, True),
        ("quasi_static_opes", "false", True),
        ("quasi_static_opes", 1, True),
    ],
)
def test_core_profile_runs_one_generic_cv_with_declared_weights(
    weight_kind, declaration, rejected,
):
    def write_plumed(path, fields, rows):
        body = ["#! FIELDS " + " ".join(fields)]
        body.extend(" ".join(f"{value:.12g}" for value in row) for row in rows)
        path.write_text("\n".join(body) + "\n", encoding="utf-8")

    with TemporaryDirectory() as directory:
        root = Path(directory)
        times = np.arange(12, dtype=float)
        sampling_times = times * 4.0
        bead_times = times * 2.0
        # A biased three-state sample with exactly known target probabilities.
        centers = np.array([-0.75, 0.0, 0.75])
        counts = np.array([6, 3, 3])
        probabilities = np.array([0.2, 0.3, 0.5])
        sampling_values = np.repeat(centers, counts)
        bead_values = (sampling_values - 0.12, sampling_values + 0.12)
        log_weights = np.log(np.repeat(probabilities / counts, counts))
        temperature = 300.0
        supplied_weights = log_weights
        if weight_kind != "precomputed":
            supplied_weights = log_weights * KB_EV_PER_K * temperature
        write_plumed(
            root / "sampling.colvar",
            ("time", "mean.coordination", "logw"),
            zip(sampling_times, sampling_values, supplied_weights),
        )
        for bead, values in enumerate(bead_values):
            write_plumed(
                root / f"bead-{bead}.colvar",
                ("time", "coordination"),
                zip(bead_times, values),
            )
        manifest = root / "RAW-SHA256SUMS"
        manifest.write_text("".join(
            f"{sha256(root / name)}  {name}\n"
            for name in ["sampling.colvar", "bead-0.colvar", "bead-1.colvar"]
        ), encoding="utf-8")
        contract = {
            "analysis_profile": "core",
            "source": {
                "run_root": str(root),
                "raw_manifest": manifest.name,
                "raw_manifest_sha256": sha256(manifest),
                "sampling_colvar": "sampling.colvar",
                "bead_colvars": ["bead-0.colvar", "bead-1.colvar"],
                "sampling_label": "Coordination mean",
                "sampling_slug": "coordination_mean",
                "sampling_time_scale_to_fs": 0.25,
                "bead_time_scale_to_fs": 0.5,
            },
            "selection": {
                "first_time_ps": 0.0,
                "last_time_ps": 0.011,
                "timestep_fs": 1.0,
                "expected_frames": len(times),
            },
            "reweight": {
                "cv_names": ["coordination"],
                "sampling_cv_names": ["mean.coordination"],
                "bead_cv_names": ["coordination"],
                "bias_mode": "bead_mean",
                "weight_kind": "precomputed",
                "log_weight_column": "logw",
                "protocol_label": "generic enhanced-sampling weights",
                "temperature_K": temperature,
                "kbt_eV": KB_EV_PER_K * temperature,
                "grid": {"coordination": [-1.5, 1.5, 61]},
                "bandwidth_variants": {"primary": [0.25], "wide": [0.35]},
                "primary_bandwidth": "primary",
                "relative_density_support": 1e-8,
                "uncertainty": {"block_frames": 3, "reference_grid_index": [30]},
                "blocks": 2,
                "plot_max_kcal_mol": 12.0,
            },
            "plots": {"cv_labels": {"coordination": "Coordination number"}},
        }
        contract["reweight"]["weight_kind"] = weight_kind
        if weight_kind != "precomputed":
            contract["reweight"]["bias_column"] = "logw"
        if declaration is not None:
            contract["reweight"]["quasi_static"] = declaration
        contract_path = root / "contract.json"
        contract_path.write_text(json.dumps(contract), encoding="utf-8")
        output = root / "analysis"
        if rejected:
            with pytest.raises(ValueError, match="quasi-static OPES must be declared explicitly"):
                analyze(contract_path, output)
            return
        summary = analyze(contract_path, output)
        assert summary["status"] == "PASS"
        assert summary["analysis_profile"] == "core"
        assert summary["fes"]["dimensions"] == 1
        uncertainty = np.genfromtxt(output / "blocks" / "quantum-fes-uncertainty.csv", delimiter=",", names=True)
        assert len(uncertainty) == 61
        assert uncertainty["standard_error_eV"][30] == 0
        assert json.loads((output / "blocks" / "quantum-fes-uncertainty.json").read_text())["blocks"] == 4
        assert summary["fes"]["probability_mean_label"] == "Quantum FES"
        assert summary["reference_crosscheck"] is None
        assert (output / "figures" / "fes1d-coordination.png").is_file()
        assert (output / "figures" / "cv-time-series.png").is_file()

        # Closed-form Gaussian mixtures, independent of production KDE,
        # log-weight normalization, and eV-to-kcal conversion helpers.
        table = np.genfromtxt(
            output / "fes1d" / "coordination.csv", delimiter=",", names=True
        )
        grid = table["coordination"]
        sigma = 0.25
        sampling_density = sum(
            probability * np.exp(-0.5 * ((grid - center) / sigma) ** 2)
            for center, probability in zip(centers, probabilities)
        )
        bead_densities = [
            sum(
                probability
                * np.exp(-0.5 * ((grid - center - offset) / sigma) ** 2)
                for center, probability in zip(centers, probabilities)
            )
            for offset in (-0.12, 0.12)
        ]
        # The common Gaussian prefactor cancels in the FES reference shift.
        rt_kcal_mol = 8.31446261815324 * temperature / 4184.0
        expected = {
            "F_sampling_kcal_mol": -rt_kcal_mol * np.log(sampling_density),
            "F_quantum_probability_mean_kcal_mol": (
                -rt_kcal_mol * np.log(np.mean(bead_densities, axis=0))
            ),
            "F_bead_logmean_diagnostic_kcal_mol": (
                -rt_kcal_mol * np.mean(np.log(bead_densities), axis=0)
            ),
        }
        for column, free_energy in expected.items():
            np.testing.assert_allclose(
                table[column], free_energy - np.min(free_energy),
                rtol=1e-9, atol=1e-9,
            )
        frames = np.genfromtxt(
            output / "tables" / "frame-series.csv", delimiter=",", names=True
        )
        np.testing.assert_allclose(
            frames["normalized_weight"],
            np.repeat(probabilities / counts, counts),
            rtol=1e-9, atol=1e-12,
        )
        np.testing.assert_allclose(frames["time_ps"], times / 1000.0)


@pytest.mark.parametrize("bias_mode", ["centroid_coord", "bead_mean", "bead_density_shared"])
def test_core_2d_recovers_analytic_mixture_and_frame_ess(tmp_path, bias_mode):
    # Unequal axes, bandwidths, and correlated centers expose axis swaps.
    centers = np.array([[-0.7, 0.4], [0.1, -0.5], [0.65, 0.8]])
    counts = np.array([6, 3, 3])
    probabilities = np.array([0.2, 0.3, 0.5])
    samples = np.repeat(centers, counts, axis=0)
    weights = np.repeat(probabilities / counts, counts)
    offsets = np.array([[-0.12, 0.2], [0.12, -0.2]])
    times = np.arange(len(samples), dtype=float)

    def write_colvar(name, fields, values):
        np.savetxt(
            tmp_path / name, values, fmt="%.17g",
            header="FIELDS " + " ".join(fields), comments="#! ",
        )

    write_colvar(
        "sampling.colvar", ["time", "x", "y", "logw"],
        np.column_stack([times, samples, np.log(weights)]),
    )
    for bead, offset in enumerate(offsets):
        write_colvar(
            f"bead-{bead}.colvar", ["time", "x", "y"],
            np.column_stack([times, samples + offset]),
        )
    # A real manifest is retained, without asserting entry verification here.
    manifest = tmp_path / "RAW-SHA256SUMS"
    manifest.write_text("".join(
        f"{sha256(tmp_path / name)}  {name}\n"
        for name in ["sampling.colvar", "bead-0.colvar", "bead-1.colvar"]
    ), encoding="utf-8")
    temperature = 300.0
    bandwidth = np.array([0.25, 0.4])
    contract = {
        "analysis_profile": "core",
        "source": {
            "run_root": str(tmp_path),
            "raw_manifest": manifest.name,
            "raw_manifest_sha256": sha256(manifest),
            "sampling_colvar": "sampling.colvar",
            "bead_colvars": ["bead-0.colvar", "bead-1.colvar"],
            "sampling_label": "Sampling coordinates",
            "sampling_slug": "sampling",
        },
        "selection": {
            "first_time_ps": 0.0, "last_time_ps": 0.011,
            "timestep_fs": 1.0, "expected_frames": len(times),
        },
        "reweight": {
            "cv_names": ["x", "y"],
            "bias_mode": bias_mode,
            "weight_kind": "precomputed", "log_weight_column": "logw",
            "temperature_K": temperature, "kbt_eV": KB_EV_PER_K * temperature,
            "grid": {"x": [-1.2, 1.2, 23], "y": [-1.3, 1.5, 17]},
            "bandwidth_variants": {"primary": bandwidth.tolist()},
            "primary_bandwidth": "primary", "relative_density_support": 1e-8,
            "uncertainty": {"block_frames": 3, "reference_grid_index": [11, 8]},
            "blocks": 2, "plot_max_kcal_mol": 12.0,
            "difference_max_kcal_mol": 2.0,
        },
        "plots": {"cv_labels": {"x": "First coordinate", "y": "Second coordinate"}},
    }
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    output = tmp_path / "analysis"
    summary = analyze(contract_path, output)
    assert summary["status"] == "PASS"
    assert summary["fes"]["dimensions"] == 2
    uncertainty = np.genfromtxt(output / "blocks" / "quantum-fes-uncertainty.csv", delimiter=",", names=True)
    assert len(uncertainty) == 23 * 17
    assert uncertainty["standard_error_eV"][8 * 23 + 11] == 0

    table = np.genfromtxt(output / "fes2d" / "primary.csv", delimiter=",", names=True)
    assert len(table) == 23 * 17
    assert len(np.unique(table["x"])) == 23
    assert len(np.unique(table["y"])) == 17
    points = np.column_stack([table["x"], table["y"]])

    # Independently reconstruct the four deleted-block KDE ratios.
    uncertainty_points = np.column_stack([uncertainty["x"], uncertainty["y"]])
    deletion_curves = []
    for deleted_block in range(4):
        keep = np.arange(len(samples)) // 3 != deleted_block
        kernels = np.exp(-0.5 * np.sum(
            ((uncertainty_points[:, None, None, :]
              - samples[keep][None, :, None, :] - offsets[None, None, :, :])
             / bandwidth) ** 2, axis=-1
        ))
        density = np.sum(kernels * weights[keep][None, :, None], axis=(1, 2))
        deletion_curves.append(
            -KB_EV_PER_K * temperature * np.log(density / density[8 * 23 + 11])
        )
    deletion_curves = np.array(deletion_curves)
    expected_error = np.sqrt(3 / 4 * np.sum(
        (deletion_curves - deletion_curves.mean(axis=0)) ** 2, axis=0
    ))
    supported = uncertainty["support"].astype(bool)
    np.testing.assert_allclose(
        uncertainty["standard_error_eV"][supported], expected_error[supported],
        rtol=1e-9, atol=1e-12,
    )


    def density(offset):
        # Closed-form finite Gaussian mixture; no production estimator helpers.
        return sum(
            probability * np.exp(-0.5 * np.sum(
                ((points - center - offset) / bandwidth) ** 2, axis=1
            ))
            for center, probability in zip(centers, probabilities)
        )

    bead_densities = np.array([density(offset) for offset in offsets])
    rt = 8.31446261815324 * temperature / 4184.0
    expected = {
        "F_centroid_kcal_mol": -rt * np.log(density(np.zeros(2))),
        "F_eq8_kcal_mol": -rt * np.log(np.mean(bead_densities, axis=0)),
        "F_eq10_kcal_mol": -rt * np.mean(np.log(bead_densities), axis=0),
    }
    for column, free_energy in expected.items():
        np.testing.assert_allclose(
            table[column], free_energy - np.min(free_energy), rtol=1e-9, atol=1e-9
        )
    assert summary["reweighting"]["ess"] == pytest.approx(1.0 / sum(weights**2))
    assert summary["reweighting"]["ess_fraction"] == pytest.approx(
        1.0 / sum(weights**2) / len(times)
    )
    blocks = np.genfromtxt(
        output / "blocks" / "block-diagnostics.csv", delimiter=",", names=True
    )
    np.testing.assert_array_equal(blocks["frames"], [6, 6])
    for row, block_weights in zip(blocks, np.array_split(weights, 2)):
        normalized = block_weights / sum(block_weights)
        assert row["ess"] == pytest.approx(1.0 / sum(normalized**2))
        assert row["max_weight"] == pytest.approx(max(normalized))


@pytest.mark.parametrize(
    "alias", ["bead.colvar", "./bead.colvar", "bead-link.colvar", "bead-hardlink.colvar", "bead-copy.colvar"]
)
def test_analyze_checks_bead_file_identity(tmp_path, alias):
    bead = tmp_path / "bead.colvar"
    bead.write_text("#! FIELDS time x\n0 0\n1 0.2\n2 0.4\n3 0.6\n")
    if alias == "bead-link.colvar":
        (tmp_path / alias).symlink_to(bead.name)
    if alias == "bead-copy.colvar":
        (tmp_path / alias).write_bytes(bead.read_bytes())
    if alias == "bead-hardlink.colvar":
        os.link(bead, tmp_path / alias)
    sampling = tmp_path / "sampling.colvar"
    sampling.write_text("#! FIELDS time x logw\n0 0 0\n1 0.2 0\n2 0.4 0\n3 0.6 0\n")
    manifest = tmp_path / "RAW-SHA256SUMS"
    names = [sampling.name, bead.name]
    if alias == "bead-copy.colvar":
        names.append(alias)
    manifest.write_text("".join(
        f"{sha256(tmp_path / name)}  {name}\n" for name in names
    ))
    contract = {
        "analysis_profile": "core",
        "source": {
            "run_root": str(tmp_path),
            "raw_manifest": manifest.name,
            "raw_manifest_sha256": sha256(manifest),
            "sampling_colvar": sampling.name,
            "bead_colvars": [bead.name, alias],
        },
        "selection": {
            "first_time_ps": 0.0, "last_time_ps": 0.003,
            "timestep_fs": 1.0, "expected_frames": 4,
        },
        "reweight": {
            "cv_names": ["x"], "bias_mode": "bead_mean",
            "weight_kind": "precomputed", "log_weight_column": "logw",
            "temperature_K": 300.0, "kbt_eV": KB_EV_PER_K * 300.0,
            "grid": {"x": [-1.0, 1.0, 21]},
            "bandwidth_variants": {"primary": [0.3]},
            "primary_bandwidth": "primary", "relative_density_support": 1e-8,
            "blocks": 2, "plot_max_kcal_mol": 12.0,
        },
        "plots": {"cv_labels": {"x": "Coordinate"}},
    }
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract))
    if alias == "bead-copy.colvar":
        assert analyze(path, tmp_path / "analysis")["status"] == "PASS"
    else:
        with pytest.raises(ValueError, match="duplicate bead input file"):
            analyze(path, tmp_path / "analysis")


def test_analyze_rejects_changed_input_before_parsing(tmp_path):
    data = tmp_path / "sample.colvar"
    data.write_text("#! FIELDS time x logw\n0 1 0\n")
    manifest = tmp_path / "RAW-SHA256SUMS"
    manifest.write_text(f"{sha256(data)}  {data.name}\n")
    contract = {
        "analysis_profile": "core",
        "source": {
            "run_root": str(tmp_path), "raw_manifest": manifest.name,
            "raw_manifest_sha256": sha256(manifest),
            "sampling_colvar": data.name, "bead_colvars": [data.name],
        },
        "selection": {}, "reweight": {},
    }
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract))
    data.write_text("#! FIELDS time x logw\n0 2 0\n")
    with pytest.raises(ValueError, match="input hash mismatch"):
        analyze(path, tmp_path / "analysis")


@pytest.mark.parametrize("name", ["plain", "with spaces", "back\\slash", "line\nbreak"])
@pytest.mark.parametrize("binary", [False, True])
def test_manifest_verifies_consumed_inputs_only(tmp_path, name, binary):
    data = tmp_path / name
    data.write_text("immutable data\n")
    encoded = name.replace("\\", "\\\\").replace("\n", "\\n")
    prefix = "\\" if encoded != name else ""
    marker = "*" if binary else " "
    manifest = tmp_path / "SHA256SUMS"
    manifest.write_text(
        f"{prefix}{sha256(data)} {marker}{encoded}\n"
        + "0" * 64 + "  unconsumed-missing-trajectory\n"
    )
    verified = verify_manifest_inputs(manifest, tmp_path, [data, data])
    assert verified == {str(data.resolve()): sha256(data)}


@pytest.mark.parametrize(
    "kind, message",
    [
        ("missing", "input missing from SHA256SUMS"),
        ("changed", "input hash mismatch"),
        ("duplicate", "duplicate SHA256SUMS entry"),
        ("malformed", "invalid SHA256SUMS record"),
        ("empty", "empty SHA256SUMS manifest"),
        ("escape", "invalid SHA256SUMS filename escape"),
    ],
)
def test_manifest_rejects_invalid_inputs(tmp_path, kind, message):
    data = tmp_path / "data"
    data.write_text("original")
    record = f"{sha256(data)}  data\n"
    if kind == "missing":
        record = record.replace("data", "different")
    elif kind == "changed":
        data.write_text("modified")
    elif kind == "duplicate":
        record += record.replace("data", "./data")
    elif kind == "malformed":
        record = "not a checksum record\n"
    elif kind == "empty":
        record = ""
    elif kind == "escape":
        record = "\\" + record.replace("data", r"bad\q")
    manifest = tmp_path / "SHA256SUMS"
    manifest.write_text(record)
    with pytest.raises(ValueError, match=message):
        verify_manifest_inputs(manifest, tmp_path, [data])
