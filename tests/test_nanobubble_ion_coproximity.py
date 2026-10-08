import csv
import gzip
import json
from pathlib import Path

from molsimflow.postprocess.nanobubble_ion_coproximity import main


def write_case_run(path: Path, species: str) -> None:
    results = path / "results"
    results.mkdir(parents=True)
    (path / "ANALYSIS-RESULT.txt").write_text("status=PASS\n")
    (path / "VALIDATION.json").write_text(
        json.dumps({"status": "PASS", "main_n2_integrity_fraction": 1.0, "min_main_n2_count": 300})
    )
    with gzip.open(results / "ion_samples.csv.gz", "wt", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("species", "z_from_terminal_plane_A", "nearest_main_n2_center_A"),
        )
        writer.writeheader()
        writer.writerows(
            (
                {"species": species, "z_from_terminal_plane_A": 1.0, "nearest_main_n2_center_A": 2.0},
                {"species": species, "z_from_terminal_plane_A": 3.0, "nearest_main_n2_center_A": 3.0},
                {"species": "OH_minus_candidate", "z_from_terminal_plane_A": 1.0, "nearest_main_n2_center_A": 1.0},
            )
        )


def test_joint_fixed_ion_coproximity_excludes_geometric_candidates(tmp_path: Path):
    hcl = tmp_path / "hcl"
    naoh = tmp_path / "naoh"
    write_case_run(hcl, "Cl_minus")
    write_case_run(naoh, "Na_plus")
    manifest = tmp_path / "cases.tsv"
    manifest.write_text(
        "case_id\tsurface\tcondition\trun_dir\n"
        f"case_hcl\toh\thcl\t{hcl}\n"
        f"case_naoh\toh\tnaoh\t{naoh}\n"
    )
    output = tmp_path / "output"
    assert main(
        [
            "--case-manifest",
            str(manifest),
            "--output-dir",
            str(output),
            "--surface-max-A",
            "4",
            "--gas-max-A",
            "4",
            "--bin-width-A",
            "1",
            "--surface-threshold-A",
            "2",
            "--gas-threshold-A",
            "2",
        ]
    ) == 0
    coverage = list(csv.DictReader((output / "coordinate_coverage.csv").open()))
    assert len(coverage) == 2
    assert {row["species"] for row in coverage} == {"Cl_minus", "Na_plus"}
    thresholds = list(csv.DictReader((output / "co_proximity_thresholds.csv").open()))
    assert {float(row["co_proximity_fraction"]) for row in thresholds} == {0.5}
    assert (output / "figures" / "01_joint_fixed_ion_proximity_maps.png").is_file()
    assert (output / "figures" / "02_fixed_ion_co_proximity_threshold_sweep.png").is_file()
    assert (output / "figures" / "03_joint_coordinate_coverage.png").is_file()
