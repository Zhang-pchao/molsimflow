"""Aggregate fixed-ion spatial profiles from immutable nanobubble case runs."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FIXED_SPECIES = {"Na_plus", "Cl_minus"}


@dataclass(frozen=True)
class CaseRun:
    case_id: str
    surface: str
    condition: str
    run_dir: Path


def read_tsv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def load_cases(path: Path) -> list[CaseRun]:
    rows = read_tsv(path)
    required = {"case_id", "surface", "condition", "run_dir"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("case manifest requires case_id, surface, condition, and run_dir")
    cases = []
    seen = set()
    for row in rows:
        case_id = row["case_id"].strip()
        if not case_id or case_id in seen:
            raise ValueError("case IDs must be present and unique")
        run_dir = Path(row["run_dir"]).resolve()
        terminal = (run_dir / "ANALYSIS-RESULT.txt").read_text(encoding="utf-8")
        validation = json.loads((run_dir / "VALIDATION.json").read_text(encoding="utf-8"))
        if "status=PASS" not in terminal or validation.get("status") != "PASS":
            raise ValueError(f"case is not terminally validated: {case_id}")
        if not (run_dir / "results" / "ion_samples.csv.gz").is_file():
            raise ValueError(f"missing fixed-ion samples: {case_id}")
        cases.append(CaseRun(case_id, row["surface"], row["condition"], run_dir))
        seen.add(case_id)
    return cases


def read_fixed_samples(case: CaseRun) -> list[dict[str, str]]:
    path = case.run_dir / "results" / "ion_samples.csv.gz"
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row["species"] in FIXED_SPECIES]


def histogram_density(
    values: Iterable[float], minimum: float, maximum: float, width: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not (np.isfinite([minimum, maximum, width]).all() and width > 0 and maximum > minimum):
        raise ValueError("histogram bounds must be finite and ordered")
    edges = np.arange(minimum, maximum + width * 0.5, width)
    values_array = np.asarray(list(values), dtype=float)
    values_array = values_array[np.isfinite(values_array)]
    counts, _ = np.histogram(values_array, bins=edges)
    density = np.zeros_like(counts, dtype=float)
    if len(values_array):
        density = counts / (len(values_array) * width)
    return 0.5 * (edges[:-1] + edges[1:]), counts, density


def write_csv(path: Path, rows: Sequence[dict], fields: Sequence[str]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def collect_profiles(
    cases: Sequence[CaseRun],
    field: str,
    minimum: float,
    maximum: float,
    width: float,
) -> tuple[list[dict], list[dict]]:
    profile_rows: list[dict] = []
    coverage_rows: list[dict] = []
    for case in cases:
        samples = read_fixed_samples(case)
        by_species: dict[str, list[float]] = defaultdict(list)
        for row in samples:
            by_species[row["species"]].append(float(row[field]))
        validation = json.loads((case.run_dir / "VALIDATION.json").read_text(encoding="utf-8"))
        frames = read_tsv(case.run_dir / "results" / "frame_summary.csv")
        coverage_rows.append(
            {
                "case_id": case.case_id,
                "surface": case.surface,
                "condition": case.condition,
                "frame_count": len(frames),
                "fixed_ion_sample_count": len(samples),
                "main_n2_integrity_fraction": validation["main_n2_integrity_fraction"],
                "min_main_n2_count": validation["min_main_n2_count"],
                "max_main_n2_count": validation["max_main_n2_count"],
            }
        )
        for species, values in sorted(by_species.items()):
            centers, counts, density = histogram_density(values, minimum, maximum, width)
            for center, count, probability_density in zip(centers, counts, density):
                profile_rows.append(
                    {
                        "case_id": case.case_id,
                        "surface": case.surface,
                        "condition": case.condition,
                        "species": species,
                        "coordinate": field,
                        "bin_center_A": center,
                        "bin_count": int(count),
                        "probability_density_per_A": probability_density,
                        "sample_count": len(values),
                    }
                )
    return profile_rows, coverage_rows


def plot_profiles(
    rows: Sequence[dict], path: Path, xlabel: str, title: str
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surfaces = sorted({row["surface"] for row in rows})
    figure, axes = plt.subplots(1, len(surfaces), figsize=(4.1 * len(surfaces), 3.6), squeeze=False)
    for axis, surface in zip(axes[0], surfaces):
        grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
        for row in rows:
            if row["surface"] == surface:
                grouped[(row["case_id"], row["condition"], row["species"])].append(row)
        for (_, condition, species), profile in sorted(grouped.items()):
            ordered = sorted(profile, key=lambda row: float(row["bin_center_A"]))
            axis.plot(
                [float(row["bin_center_A"]) for row in ordered],
                [float(row["probability_density_per_A"]) for row in ordered],
                label=f"{condition}: {species}",
            )
        axis.set_title(surface)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Probability density (A-1)")
        axis.set_ylim(bottom=0)
        axis.legend(fontsize=7)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def plot_integrity(rows: Sequence[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = sorted(rows, key=lambda row: (row["surface"], row["condition"]))
    labels = [f"{row['surface']}\n{row['condition']}" for row in ordered]
    values = [float(row["main_n2_integrity_fraction"]) for row in ordered]
    figure, axis = plt.subplots(figsize=(8.2, 3.5))
    axis.bar(range(len(values)), values)
    axis.set_xticks(range(len(values)), labels, rotation=25, ha="right")
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Fraction of frames with main N2 >= threshold")
    axis.set_title("Gas-integrity coverage for fixed-ion spatial coordinates")
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def run_analysis(args: argparse.Namespace) -> dict:
    cases = load_cases(args.case_manifest)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    surface_rows, coverage = collect_profiles(
        cases,
        "z_from_terminal_plane_A",
        args.surface_min_A,
        args.surface_max_A,
        args.bin_width_A,
    )
    gas_rows, _ = collect_profiles(
        cases,
        "nearest_main_n2_center_A",
        args.gas_min_A,
        args.gas_max_A,
        args.bin_width_A,
    )
    write_csv(
        output / "surface_distance_profiles.csv",
        surface_rows,
        (
            "case_id",
            "surface",
            "condition",
            "species",
            "coordinate",
            "bin_center_A",
            "bin_count",
            "probability_density_per_A",
            "sample_count",
        ),
    )
    write_csv(
        output / "gas_proximity_profiles.csv",
        gas_rows,
        (
            "case_id",
            "surface",
            "condition",
            "species",
            "coordinate",
            "bin_center_A",
            "bin_count",
            "probability_density_per_A",
            "sample_count",
        ),
    )
    write_csv(
        output / "case_coverage.csv",
        coverage,
        (
            "case_id",
            "surface",
            "condition",
            "frame_count",
            "fixed_ion_sample_count",
            "main_n2_integrity_fraction",
            "min_main_n2_count",
            "max_main_n2_count",
        ),
    )
    figures = output / "figures"
    figures.mkdir()
    plot_profiles(
        surface_rows,
        figures / "01_fixed_ion_surface_distance_profiles.png",
        "Distance from dynamic terminal plane (A)",
        "Fixed-ion distribution relative to the dynamic surface",
    )
    plot_profiles(
        gas_rows,
        figures / "02_fixed_ion_gas_proximity_profiles.png",
        "Nearest main-N2 center distance (A)",
        "Fixed-ion proximity to the main N2 cluster",
    )
    plot_integrity(coverage, figures / "03_fixed_ion_gas_integrity_coverage.png")
    summary = {
        "status": "PASS",
        "case_count": len(cases),
        "fixed_species": sorted(FIXED_SPECIES),
        "surface_profile_rows": len(surface_rows),
        "gas_profile_rows": len(gas_rows),
        "claim_boundary": [
            "Profiles are within-trajectory spatial associations, not ion adsorption free energies or pH causality.",
            "Gas-integrity coverage labels the coordinate context and does not convert a dispersed-gas state into a TPCL observable.",
            "No H3O+ or OH- geometric candidates enter the fixed-ion profile figures.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--surface-min-A", type=float, default=0.0)
    parser.add_argument("--surface-max-A", type=float, default=40.0)
    parser.add_argument("--gas-min-A", type=float, default=0.0)
    parser.add_argument("--gas-max-A", type=float, default=40.0)
    parser.add_argument("--bin-width-A", type=float, default=0.5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(run_analysis(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
