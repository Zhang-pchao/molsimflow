import csv
import gzip
import json
from pathlib import Path

from molsimflow.postprocess.nanobubble_ion_partition import main


def write_case_run(path: Path, species: str, condition: str) -> None:
    results = path / "results"
    results.mkdir(parents=True)
    (path / "ANALYSIS-RESULT.txt").write_text("status=PASS\n")
    (path / "VALIDATION.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "main_n2_integrity_fraction": 1.0,
                "min_main_n2_count": 300,
                "max_main_n2_count": 300,
            }
        )
    )
    with (results / "frame_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("step", "time_ns"))
        writer.writeheader()
        writer.writerows(({"step": 1, "time_ns": 8.0}, {"step": 2, "time_ns": 8.001}))
    with gzip.open(results / "ion_samples.csv.gz", "wt", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("species", "z_from_terminal_plane_A", "nearest_main_n2_center_A"),
        )
        writer.writeheader()
        writer.writerows(
            (
                {
                    "species": species,
                    "z_from_terminal_plane_A": 1.0,
                    "nearest_main_n2_center_A": 2.0,
                },
                {
                    "species": species,
                    "z_from_terminal_plane_A": 2.0,
                    "nearest_main_n2_center_A": 3.0,
                },
                {
                    "species": "OH_minus_candidate",
                    "z_from_terminal_plane_A": 4.0,
                    "nearest_main_n2_center_A": 5.0,
                },
            )
        )


def test_partition_aggregate_writes_fixed_ion_profiles_and_figures(tmp_path: Path):
    hcl = tmp_path / "hcl"
    naoh = tmp_path / "naoh"
    write_case_run(hcl, "Cl_minus", "hcl")
    write_case_run(naoh, "Na_plus", "naoh")
    manifest = tmp_path / "cases.tsv"
    manifest.write_text(
        "case_id\tsurface\tcondition\trun_dir\n"
        f"case_hcl\toh\thcl\t{hcl}\n"
        f"case_naoh\toh\tnaoh\t{naoh}\n"
    )
    output = tmp_path / "output"
    assert main(["--case-manifest", str(manifest), "--output-dir", str(output)]) == 0
    coverage = list(csv.DictReader((output / "case_coverage.csv").open()))
    assert len(coverage) == 2
    profiles = list(csv.DictReader((output / "surface_distance_profiles.csv").open()))
    assert {row["species"] for row in profiles} == {"Na_plus", "Cl_minus"}
    assert (output / "figures" / "01_fixed_ion_surface_distance_profiles.png").is_file()
    assert (output / "figures" / "02_fixed_ion_gas_proximity_profiles.png").is_file()
    assert (output / "figures" / "03_fixed_ion_gas_integrity_coverage.png").is_file()
