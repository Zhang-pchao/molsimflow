import csv
import json
from pathlib import Path

from molsimflow.cli import build_parser
from molsimflow.postprocess.sphere_interface_compare import (
    collect_comparison,
    load_cases,
    resolve_results,
)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def _write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _case(root: Path, offset: float) -> None:
    def results(module: str) -> Path:
        path = root / module / "latest" / "results"
        path.mkdir(parents=True)
        return path

    core = results("droplet_spreading")
    _write_json(
        core / "summary.json",
        {"status": "PASS", "analyzed_frames": 2, "first_step": 1, "last_step": 2, "timestep_fs": 0.5},
    )
    _write_csv(
        core / "droplet_spreading.csv",
        [
            {
                "step": 1,
                "droplet_center_surface_dz_A": 4 + offset,
                "droplet_lateral_radius_p90_A": 5 + offset,
                "droplet_height_q05_q95_A": 6 + offset,
                "footprint_convex_hull_area_A2": 7 + offset,
            },
            {
                "step": 2,
                "droplet_center_surface_dz_A": 5 + offset,
                "droplet_lateral_radius_p90_A": 6 + offset,
                "droplet_height_q05_q95_A": 7 + offset,
                "footprint_convex_hull_area_A2": 8 + offset,
            },
        ],
    )
    lateral = results("lateral_motion")
    _write_json(lateral / "summary.json", {"status": "PASS", "net_displacement_A": 1, "maximum_displacement_A": 2, "cumulative_path_length_A": 3})
    angle = results("contact_angle_density")
    _write_json(angle / "summary.json", {"status": "PASS", "phase_label": "liquid_water", "block_dense_phase_angle_mean_deg": 30 + offset, "block_dense_phase_angle_std_deg": 2, "primary_dense_phase_contact_angle_deg": 31 + offset, "valid_block_count": 10, "threshold_angle_span_deg": 1})
    line = results("contact_line")
    _write_json(line / "summary.json", {"status": "PASS", "valid_contact_line_frames": 2, "analyzed_frames": 2, "jump_candidate_count": 0})
    _write_csv(line / "contact_line_blocks.csv", [{"start_time_ns": 0, "end_time_ns": 1, "mean_equivalent_radius_A": 4, "std_equivalent_radius_A": 0.2}])
    site = results("surface_site_enrichment")
    _write_json(site / "summary.json", {"status": "PASS", "surface_ch3_fraction": 0.5, "mean_footprint_ch3_enrichment": 0.1, "mean_tpcl_ch3_enrichment": -0.1, "mapped_frames": 2})
    density = results("interfacial_water_density")
    _write_json(density / "summary.json", {"status": "PASS", "mean_hydration_areal_density_A-2": {"footprint": 0.1, "tpcl": 0.2, "far_field": 0.3}})
    _write_csv(density / "water_density_profiles.csv", [{"region": "tpcl", "z_A": 1, "oxygen_number_density_A-3": 0.02}])
    orientation = results("interfacial_water_orientation")
    _write_json(orientation / "summary.json", {"status": "PASS", "mean_cos_theta": {"tpcl_dipole": -0.2}, "orientation_sample_counts": {"tpcl_dipole": 20}})
    _write_csv(orientation / "orientation_profiles.csv", [{"region": "tpcl", "observable": "dipole", "cos_theta": 0, "probability_density": 0.5}])
    hbond = results("interfacial_water_hbond")
    _write_json(hbond / "summary.json", {"status": "PASS", "mean_metrics": {"tpcl_water_water_hbond_degree": 2.0, "tpcl_surface_water_hbond_per_h2o": 0.4}})
    proton = results("surface_proton_transfer")
    _write_json(proton / "summary.json", {"status": "PASS", "analyzed_frames": 2, "frames_with_h3o_candidate": 1, "frames_with_oh_candidate": 0, "frames_with_surface_site_candidate": 1})


def test_manifest_labels_and_collection(tmp_path):
    roots = [tmp_path / "first", tmp_path / "second"]
    for index, root in enumerate(roots):
        _case(root, float(index))
    manifest = tmp_path / "cases.tsv"
    manifest.write_text(
        "case_id\tanalysis_root\tch3_sites\toh_sites\n"
        f"ch3_36_oh_0\t{roots[0]}\t36\t0\n"
        f"ch3_0_oh_36\t{roots[1]}\t0\t36\n"
    )
    cases = load_cases(manifest)
    assert [case.label for case in cases] == ["1.00", "0.00"]
    assert [case.composition_label for case in cases] == [
        "CH₃:OH = 36:0",
        "CH₃:OH = 0:36",
    ]

    output = tmp_path / "output"
    summary = collect_comparison(cases, "nanodroplet", output, make_plots=False)
    assert summary["status"] == "PASS"
    assert summary["case_count"] == 2
    assert (output / "case_summary.csv").is_file()
    rows = list(csv.DictReader((output / "case_summary.csv").open()))
    assert [row["legend_label"] for row in rows] == ["1.00", "0.00"]
    assert [float(row["contact_line_coverage_ns"]) for row in rows] == [1.0, 1.0]
    assert "CH₃:OH = 36:0" in (output / "case_summary.csv").read_text()
    assert json.loads((output / "summary.json").read_text())["kind"] == "nanodroplet"


def test_failed_result_is_read_without_publishing_latest(tmp_path):
    module = tmp_path / "contact_angle_density"
    results = module / "run" / "7" / "results"
    results.mkdir(parents=True)
    (module / "ANALYSIS-RESULT.txt").write_text(f"status=FAILED\nresults={results}\n")
    resolved, status, mode = resolve_results(tmp_path, "contact_angle_density")
    assert resolved == results.resolve()
    assert status == "FAILED"
    assert mode == "recorded_failed_or_unpublished"


def test_cli_exposes_sphere_interface_compare(tmp_path):
    args = build_parser().parse_args(
        [
            "postprocess",
            "sphere-interface-compare",
            "--manifest",
            str(tmp_path / "cases.tsv"),
            "--kind",
            "nanobubble",
            "--output-dir",
            str(tmp_path / "out"),
            "--no-plots",
        ]
    )
    assert args.kind == "nanobubble"
    assert args.block_frames == 10
