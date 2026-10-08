import csv
import gzip
import json
import math
from pathlib import Path

import numpy as np

from molsimflow.postprocess.nanobubble_ion_fields import (
    AVOGADRO,
    main,
    shell_volume_above_plane,
    sphere_volume_above_plane,
)


def test_sphere_and_shell_volume_above_plane():
    full = 4.0 * math.pi / 3.0
    assert np.isclose(sphere_volume_above_plane(1.0, 2.0), full)
    assert np.isclose(sphere_volume_above_plane(1.0, 0.0), full / 2.0)
    assert sphere_volume_above_plane(1.0, -2.0) == 0.0
    assert np.isclose(shell_volume_above_plane(0.0, 1.0, 0.0), full / 2.0)


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    run = tmp_path / "ion_run"
    results = run / "results"
    results.mkdir(parents=True)
    (run / "ANALYSIS-RESULT.txt").write_text("status=PASS\n")
    (run / "VALIDATION.json").write_text(json.dumps({"status": "PASS"}))
    frames = []
    contacts = []
    for step in (0, 20, 40):
        frames.append(
            {
                "step": step,
                "box_x_A": 10,
                "box_y_A": 10,
                "box_z_A": 20,
                "bubble_center_from_top_si_A": 5,
                "largest_cluster_n2_count": 300,
            }
        )
        contacts.append(
            {
                "step": step,
                "bubble_contact_n2_count": 5,
                "min_bubble_surface_distance_A": 2,
            }
        )
    with (results / "frame_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=frames[0])
        writer.writeheader()
        writer.writerows(frames)
    samples = []
    for step in (0, 20, 40):
        samples.append(
            {
                "stage": "late",
                "step": step,
                "time_ns": step * 0.5e-6,
                "species": "Na_plus",
                "atom_id": 1,
                "hydrogen_ids": "",
                "surface_origin_hydrogen_ids": "",
                "surface_origin_donor_ids": "",
                "z_from_top_si_A": 0.5,
                "r_from_bubble_center_A": 0.5,
                "nearest_main_n2_center_A": 1.0,
            }
        )
        samples.append(
            {
                "stage": "late",
                "step": step,
                "time_ns": step * 0.5e-6,
                "species": "H3O_plus_candidate",
                "atom_id": 10,
                "hydrogen_ids": "30;31;32",
                "surface_origin_hydrogen_ids": "31",
                "surface_origin_donor_ids": "7",
                "unused_source_column": "kept upstream, ignored in selected export",
                "z_from_top_si_A": 0.5,
                "r_from_bubble_center_A": 0.5,
                "nearest_main_n2_center_A": 1.0,
            }
        )
    with gzip.open(results / "ion_samples.csv.gz", "wt", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=(*samples[0], "unused_source_column")
        )
        writer.writeheader()
        writer.writerows(samples)
    contact = tmp_path / "contact.csv"
    with contact.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=contacts[0])
        writer.writeheader()
        writer.writerows(contacts)
    manifest = tmp_path / "cases.tsv"
    manifest.write_text(
        "case_id\tsurface\tcondition\tion_run\tcontact_metrics\n"
        f"case\toh\tnaoh\t{run}\t{contact}\n"
    )
    return manifest, run


def test_analyze_case_writes_density_molarity_and_provenance(tmp_path: Path):
    manifest, _ = _write_fixture(tmp_path)
    output = tmp_path / "out"
    assert (
        main(
            [
                "analyze-case",
                "--case-manifest",
                str(manifest),
                "--case-id",
                "case",
                "--output-dir",
                str(output),
                "--start-step",
                "0",
                "--end-step",
                "40",
                "--step-stride",
                "20",
                "--radial-max-A",
                "1",
                "--z-max-A",
                "1",
                "--block-steps",
                "20",
            ]
        )
        == 0
    )
    radial = list(csv.DictReader((output / "radial_number_density.csv").open()))
    na = next(row for row in radial if row["species"] == "Na_plus")
    expected_volume_nm3 = 3 * 4.0 * math.pi / 3.0 / 1000.0
    assert np.isclose(float(na["number_density_nm3"]), 3 / expected_volume_nm3)
    z_rows = list(csv.DictReader((output / "z_molar_concentration.csv").open()))
    na_z = next(row for row in z_rows if row["species"] == "Na_plus")
    assert np.isclose(float(na_z["molar_concentration_M"]), 3e27 / (AVOGADRO * 300))
    proton = list(csv.DictReader((output / "proton_transfer_candidates.csv").open()))
    assert len(proton) == 3
    assert proton[0]["surface_origin_hydrogen_ids"] == "31"
    episodes = list(csv.DictReader((output / "species_episodes.csv").open()))
    h3o = next(row for row in episodes if row["species"] == "H3O_plus_candidate")
    assert h3o["sample_count"] == "3"
    assert h3o["surface_origin_donor_ids"] == "7"
    summary = json.loads((output / "summary.json").read_text())
    assert summary["analysis_tier"] == "PRIMARY_STABLE"


def test_assemble_requires_passed_case_results(tmp_path: Path):
    manifest, _ = _write_fixture(tmp_path)
    result = tmp_path / "case_result"
    main(
        [
            "analyze-case",
            "--case-manifest",
            str(manifest),
            "--case-id",
            "case",
            "--output-dir",
            str(result),
            "--start-step",
            "0",
            "--end-step",
            "40",
            "--step-stride",
            "20",
            "--radial-max-A",
            "1",
            "--z-max-A",
            "1",
            "--block-steps",
            "20",
        ]
    )
    run_manifest = tmp_path / "results.tsv"
    run_manifest.write_text(f"case_id\tresult_dir\ncase\t{result}\n")
    output = tmp_path / "combined"
    assert main(["assemble", "--result-manifest", str(run_manifest), "--output-dir", str(output)]) == 0
    summary = json.loads((output / "summary.json").read_text())
    assert summary["case_count"] == 1
    assert summary["primary_stable_case_count"] == 1
