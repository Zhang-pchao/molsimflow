"""Build stable-nanobubble ion number-density and molarity profiles."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

AVOGADRO = 6.02214076e23
SPECIES = (
    "Na_plus",
    "Cl_minus",
    "H3O_plus_candidate",
    "OH_minus_candidate",
)


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    surface: str
    condition: str
    ion_run: Path
    contact_metrics: Path


def read_rows(path: Path, *, delimiter: str = ",") -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def write_rows(path: Path, rows: Sequence[dict], fields: Sequence[str] | None = None) -> None:
    if fields is None:
        fields = tuple(rows[0]) if rows else ()
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_cases(path: Path) -> list[CaseSpec]:
    rows = read_rows(path, delimiter="\t")
    required = {"case_id", "surface", "condition", "ion_run", "contact_metrics"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"case manifest requires {sorted(required)}")
    cases = []
    seen = set()
    for row in rows:
        case_id = row["case_id"].strip()
        if not case_id or case_id in seen:
            raise ValueError("case IDs must be non-empty and unique")
        cases.append(
            CaseSpec(
                case_id,
                row["surface"].strip(),
                row["condition"].strip(),
                Path(row["ion_run"]).resolve(),
                Path(row["contact_metrics"]).resolve(),
            )
        )
        seen.add(case_id)
    return cases


def sphere_volume_above_plane(radius_A: float, center_height_A: float) -> float:
    """Volume of a sphere lying above z=0, with center at ``center_height_A``."""

    radius = float(radius_A)
    height = float(center_height_A)
    if radius < 0 or not np.isfinite([radius, height]).all():
        raise ValueError("sphere radius and center height must be finite; radius cannot be negative")
    if radius == 0 or height <= -radius:
        return 0.0
    if height >= radius:
        return 4.0 * math.pi * radius**3 / 3.0
    cap_height = radius + height
    return math.pi * cap_height**2 * (2.0 * radius - height) / 3.0


def shell_volume_above_plane(inner_A: float, outer_A: float, center_height_A: float) -> float:
    if not 0 <= inner_A < outer_A:
        raise ValueError("shell radii must satisfy 0 <= inner < outer")
    return sphere_volume_above_plane(outer_A, center_height_A) - sphere_volume_above_plane(
        inner_A, center_height_A
    )


def bin_edges(minimum: float, maximum: float, width: float) -> np.ndarray:
    if not (np.isfinite([minimum, maximum, width]).all() and minimum >= 0 and maximum > minimum and width > 0):
        raise ValueError("bin bounds must be finite and satisfy 0 <= min < max and width > 0")
    bins = round((maximum - minimum) / width)
    if bins <= 0 or not math.isclose(minimum + bins * width, maximum, abs_tol=1.0e-9):
        raise ValueError("bin range must be an integer multiple of width")
    return np.linspace(minimum, maximum, bins + 1)


def _validated_run(run: Path) -> None:
    terminal = (run / "ANALYSIS-RESULT.txt").read_text(encoding="utf-8")
    validation = json.loads((run / "VALIDATION.json").read_text(encoding="utf-8"))
    if "status=PASS" not in terminal or validation.get("status") != "PASS":
        raise ValueError(f"ion run is not accepted: {run}")


def _expected_steps(start_step: int, end_step: int, stride: int) -> list[int]:
    if start_step < 0 or end_step < start_step or stride <= 0:
        raise ValueError("invalid common-grid step contract")
    if (end_step - start_step) % stride:
        raise ValueError("common-grid interval is not divisible by stride")
    return list(range(start_step, end_step + 1, stride))


def _load_samples(path: Path, selected_steps: set[int]) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        rows = []
        for row in csv.DictReader(handle):
            step = int(row["step"])
            if step in selected_steps and row["species"] in SPECIES:
                rows.append(row)
        return rows


def _block_statistics(
    values: np.ndarray,
    steps: np.ndarray,
    frame_values: dict[int, float],
    edges: np.ndarray,
    *,
    start_step: int,
    end_step: int,
    block_steps: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    n_blocks = max(1, math.ceil((end_step - start_step) / block_steps))
    densities = []
    for block in range(n_blocks):
        low = start_step + block * block_steps
        high = end_step if block == n_blocks - 1 else low + block_steps - 1
        frame_steps = [step for step in frame_values if low <= step <= high]
        if not frame_steps:
            continue
        mask = (steps >= low) & (steps <= high)
        counts, _ = np.histogram(values[mask], bins=edges)
        volumes = np.asarray(
            [
                sum(
                    shell_volume_above_plane(edges[index], edges[index + 1], frame_values[step])
                    for step in frame_steps
                )
                / 1000.0
                for index in range(len(edges) - 1)
            ]
        )
        densities.append(np.divide(counts, volumes, out=np.zeros_like(volumes), where=volumes > 0))
    matrix = np.asarray(densities, dtype=float)
    mean = np.mean(matrix, axis=0)
    std = np.std(matrix, axis=0, ddof=1) if len(matrix) > 1 else np.zeros(matrix.shape[1])
    return mean, std, len(matrix)


def radial_profiles(
    samples: Sequence[dict[str, str]],
    frames: Sequence[dict[str, str]],
    *,
    case: CaseSpec,
    cohort: str,
    edges: np.ndarray,
    start_step: int,
    end_step: int,
    block_steps: int,
) -> list[dict]:
    frame_heights = {int(row["step"]): float(row["bubble_center_from_top_si_A"]) for row in frames}
    frame_steps = set(frame_heights)
    volumes_A3 = np.asarray(
        [
            sum(
                shell_volume_above_plane(edges[index], edges[index + 1], height)
                for height in frame_heights.values()
            )
            for index in range(len(edges) - 1)
        ]
    )
    full_A3 = len(frames) * 4.0 * math.pi * (edges[1:] ** 3 - edges[:-1] ** 3) / 3.0
    rows = []
    for species in SPECIES:
        selected = [row for row in samples if row["species"] == species and int(row["step"]) in frame_steps]
        values = np.asarray([float(row["r_from_bubble_center_A"]) for row in selected])
        steps = np.asarray([int(row["step"]) for row in selected], dtype=int)
        counts, _ = np.histogram(values, bins=edges)
        accessible_nm3 = volumes_A3 / 1000.0
        density = np.divide(counts, accessible_nm3, out=np.zeros_like(accessible_nm3), where=accessible_nm3 > 0)
        block_mean, block_std, block_count = _block_statistics(
            values,
            steps,
            frame_heights,
            edges,
            start_step=start_step,
            end_step=end_step,
            block_steps=block_steps,
        )
        for index, count in enumerate(counts):
            rows.append(
                {
                    "case_id": case.case_id,
                    "surface": case.surface,
                    "condition": case.condition,
                    "cohort": cohort,
                    "species": species,
                    "r_inner_A": edges[index],
                    "r_outer_A": edges[index + 1],
                    "r_center_A": 0.5 * (edges[index] + edges[index + 1]),
                    "frame_count": len(frames),
                    "bin_count": int(count),
                    "full_shell_volume_nm3": full_A3[index] / 1000.0,
                    "solid_accessible_shell_volume_nm3": accessible_nm3[index],
                    "number_density_nm3": density[index],
                    "block_mean_number_density_nm3": block_mean[index],
                    "block_std_number_density_nm3": block_std[index],
                    "block_count": block_count,
                }
            )
    return rows


def z_profiles(
    samples: Sequence[dict[str, str]],
    frames: Sequence[dict[str, str]],
    *,
    case: CaseSpec,
    cohort: str,
    edges: np.ndarray,
) -> list[dict]:
    frame_steps = {int(row["step"]) for row in frames}
    area_sum_A2 = sum(float(row["box_x_A"]) * float(row["box_y_A"]) for row in frames)
    rows = []
    for species in SPECIES:
        values = np.asarray(
            [
                float(row["z_from_top_si_A"])
                for row in samples
                if row["species"] == species and int(row["step"]) in frame_steps
            ]
        )
        counts, _ = np.histogram(values, bins=edges)
        for index, count in enumerate(counts):
            volume_A3 = area_sum_A2 * (edges[index + 1] - edges[index])
            concentration = count * 1.0e27 / (AVOGADRO * volume_A3)
            rows.append(
                {
                    "case_id": case.case_id,
                    "surface": case.surface,
                    "condition": case.condition,
                    "cohort": cohort,
                    "species": species,
                    "z_inner_A": edges[index],
                    "z_outer_A": edges[index + 1],
                    "z_center_A": 0.5 * (edges[index] + edges[index + 1]),
                    "frame_count": len(frames),
                    "bin_count": int(count),
                    "planar_volume_A3": volume_A3,
                    "molar_concentration_M": concentration,
                }
            )
    return rows


def species_episodes(
    samples: Sequence[dict[str, str]], *, case: CaseSpec, stride: int, timestep_fs: float
) -> list[dict]:
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in samples:
        if row["species"] in {"H3O_plus_candidate", "OH_minus_candidate"}:
            grouped[(row["species"], int(row["atom_id"]))].append(row)
    episodes = []
    for (species, atom_id), values in sorted(grouped.items()):
        values.sort(key=lambda row: int(row["step"]))
        chunks = []
        chunk = [values[0]]
        for row in values[1:]:
            if int(row["step"]) == int(chunk[-1]["step"]) + stride:
                chunk.append(row)
            else:
                chunks.append(chunk)
                chunk = [row]
        chunks.append(chunk)
        for episode_index, part in enumerate(chunks):
            first_step = int(part[0]["step"])
            last_step = int(part[-1]["step"])
            surface_h = sorted(
                {
                    token
                    for row in part
                    for token in row.get("surface_origin_hydrogen_ids", "").split(";")
                    if token
                },
                key=int,
            )
            donor_ids = sorted(
                {
                    token
                    for row in part
                    for token in row.get("surface_origin_donor_ids", "").split(";")
                    if token
                },
                key=int,
            )
            episodes.append(
                {
                    "case_id": case.case_id,
                    "surface": case.surface,
                    "condition": case.condition,
                    "species": species,
                    "atom_id": atom_id,
                    "episode_index": episode_index,
                    "start_step": first_step,
                    "end_step": last_step,
                    "sample_count": len(part),
                    "observed_span_ps": (last_step - first_step) * timestep_fs / 1000.0,
                    "surface_origin_hydrogen_ids": ";".join(surface_h),
                    "surface_origin_donor_ids": ";".join(donor_ids),
                }
            )
    return episodes


def _plot_case(radial: Sequence[dict], z_rows: Sequence[dict], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output / "figures"
    figures.mkdir()
    for rows, x_key, y_key, xlabel, ylabel, filename in (
        (
            radial,
            "r_center_A",
            "number_density_nm3",
            "Distance from bubble center (A)",
            "Number density (nm$^{-3}$)",
            "radial_number_density.png",
        ),
        (
            z_rows,
            "z_center_A",
            "molar_concentration_M",
            "Distance from dynamic top-Si plane (A)",
            "Planar concentration (M)",
            "z_molar_concentration.png",
        ),
    ):
        figure, axis = plt.subplots(figsize=(6.4, 4.2))
        for species in SPECIES:
            profile = [row for row in rows if row["species"] == species]
            axis.plot([row[x_key] for row in profile], [row[y_key] for row in profile], label=species)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_ylim(bottom=0)
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(figures / filename, dpi=220)
        plt.close(figure)


def analyze_case(case: CaseSpec, output: Path, args: argparse.Namespace) -> dict:
    _validated_run(case.ion_run)
    expected = _expected_steps(args.start_step, args.end_step, args.step_stride)
    expected_set = set(expected)
    frame_rows = read_rows(case.ion_run / "results" / "frame_summary.csv")
    frame_by_step = {int(row["step"]): row for row in frame_rows if int(row["step"]) in expected_set}
    contact_rows = read_rows(case.contact_metrics)
    contact_by_step = {int(row["step"]): row for row in contact_rows if int(row["step"]) in expected_set}
    if set(frame_by_step) != expected_set or set(contact_by_step) != expected_set:
        raise ValueError(f"{case.case_id}: incomplete common 10-ps grid")
    admission = []
    for step in expected:
        frame = frame_by_step[step]
        contact = contact_by_step[step]
        cluster_ok = int(float(frame["largest_cluster_n2_count"])) >= args.minimum_cluster_n2
        contact_ok = int(float(contact["bubble_contact_n2_count"])) >= args.minimum_contact_n2
        distance_ok = float(contact["min_bubble_surface_distance_A"]) <= args.contact_cutoff_A
        accepted = cluster_ok and contact_ok and distance_ok
        admission.append(
            {
                "case_id": case.case_id,
                "surface": case.surface,
                "condition": case.condition,
                "step": step,
                "time_ns": step * args.timestep_fs / 1.0e6,
                "largest_cluster_n2_count": int(float(frame["largest_cluster_n2_count"])),
                "bubble_contact_n2_count": int(float(contact["bubble_contact_n2_count"])),
                "min_bubble_surface_distance_A": float(contact["min_bubble_surface_distance_A"]),
                "cluster_ok": cluster_ok,
                "contact_count_ok": contact_ok,
                "contact_distance_ok": distance_ok,
                "stable_interface_frame": accepted,
            }
        )
    strict = all(row["stable_interface_frame"] for row in admission)
    accepted_steps = {row["step"] for row in admission if row["stable_interface_frame"]}
    samples = _load_samples(case.ion_run / "results" / "ion_samples.csv.gz", expected_set)
    output.mkdir(parents=True, exist_ok=False)
    write_rows(output / "frame_admission.csv", admission)
    selected_frames = [frame_by_step[step] for step in expected]
    radial_edges = bin_edges(args.radial_min_A, args.radial_max_A, args.radial_bin_A)
    z_edges = bin_edges(args.z_min_A, args.z_max_A, args.z_bin_A)
    radial = radial_profiles(
        samples,
        selected_frames,
        case=case,
        cohort="late_common_grid",
        edges=radial_edges,
        start_step=args.start_step,
        end_step=args.end_step,
        block_steps=args.block_steps,
    )
    z_rows = z_profiles(
        samples,
        selected_frames,
        case=case,
        cohort="late_common_grid",
        edges=z_edges,
    )
    if accepted_steps and not strict:
        stable_frames = [frame_by_step[step] for step in expected if step in accepted_steps]
        radial.extend(
            radial_profiles(
                samples,
                stable_frames,
                case=case,
                cohort="stable_contact_only",
                edges=radial_edges,
                start_step=args.start_step,
                end_step=args.end_step,
                block_steps=args.block_steps,
            )
        )
        z_rows.extend(
            z_profiles(
                samples,
                stable_frames,
                case=case,
                cohort="stable_contact_only",
                edges=z_edges,
            )
        )
    episodes = species_episodes(
        samples, case=case, stride=args.step_stride, timestep_fs=args.timestep_fs
    )
    proton_rows = [
        row
        for row in samples
        if row["species"] == "H3O_plus_candidate"
        and row.get("surface_origin_hydrogen_ids", "")
    ]
    for row in proton_rows:
        row.update({"case_id": case.case_id, "surface": case.surface, "condition": case.condition})
    write_rows(output / "radial_number_density.csv", radial)
    write_rows(output / "z_molar_concentration.csv", z_rows)
    write_rows(output / "species_episodes.csv", episodes)
    proton_fields = (
        "case_id",
        "surface",
        "condition",
        "stage",
        "step",
        "time_ns",
        "species",
        "atom_id",
        "hydrogen_ids",
        "surface_origin_hydrogen_ids",
        "surface_origin_donor_ids",
        "z_from_top_si_A",
        "r_from_bubble_center_A",
        "nearest_main_n2_center_A",
    )
    write_rows(output / "proton_transfer_candidates.csv", proton_rows, proton_fields)
    _plot_case(
        [row for row in radial if row["cohort"] == "late_common_grid"],
        [row for row in z_rows if row["cohort"] == "late_common_grid"],
        output,
    )
    summary = {
        "status": "PASS",
        "case_id": case.case_id,
        "surface": case.surface,
        "condition": case.condition,
        "analysis_tier": "PRIMARY_STABLE" if strict else "SECONDARY_INTERMITTENT_CONTACT",
        "common_grid_frame_count": len(expected),
        "stable_interface_frame_count": len(accepted_steps),
        "stable_interface_fraction": len(accepted_steps) / len(expected),
        "reactive_species_are_geometric_candidates": True,
        "molarity_is_full_xy_planar_average": True,
        "radial_volume_correction_excludes_solid_half_space_only": True,
        "surface_origin_h3o_sample_count": len(proton_rows),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def assemble(manifest: Path, output: Path) -> dict:
    cases = read_rows(manifest, delimiter="\t")
    required = {"case_id", "result_dir"}
    if not cases or not required.issubset(cases[0]):
        raise ValueError("result manifest requires case_id and result_dir")
    output.mkdir(parents=True, exist_ok=False)
    summaries = []
    files = (
        "frame_admission.csv",
        "radial_number_density.csv",
        "z_molar_concentration.csv",
        "species_episodes.csv",
        "proton_transfer_candidates.csv",
    )
    combined: dict[str, list[dict[str, str]]] = {name: [] for name in files}
    for row in cases:
        result = Path(row["result_dir"]).resolve()
        summary = json.loads((result / "summary.json").read_text(encoding="utf-8"))
        if summary.get("status") != "PASS" or summary.get("case_id") != row["case_id"]:
            raise ValueError(f"unaccepted case result: {row['case_id']}")
        summaries.append(summary)
        for name in files:
            combined[name].extend(read_rows(result / name))
    for name, rows in combined.items():
        write_rows(output / name, rows)
    write_rows(output / "case_summary.csv", summaries)
    summary = {
        "status": "PASS",
        "case_count": len(summaries),
        "primary_stable_case_count": sum(
            row["analysis_tier"] == "PRIMARY_STABLE" for row in summaries
        ),
        "secondary_case_count": sum(
            row["analysis_tier"] != "PRIMARY_STABLE" for row in summaries
        ),
        "claim_boundary": [
            "Reactive H3O/OH labels are geometric coordination candidates.",
            "Profiles are within-trajectory spatial associations, not adsorption free energies.",
            "Block variation is not independent-replicate uncertainty.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    analyze = subparsers.add_parser("analyze-case")
    analyze.add_argument("--case-manifest", type=Path, required=True)
    analyze.add_argument("--case-id", required=True)
    analyze.add_argument("--output-dir", type=Path, required=True)
    analyze.add_argument("--start-step", type=int, default=16_000_000)
    analyze.add_argument("--end-step", type=int, default=20_000_000)
    analyze.add_argument("--step-stride", type=int, default=20_000)
    analyze.add_argument("--timestep-fs", type=float, default=0.5)
    analyze.add_argument("--minimum-cluster-n2", type=int, default=250)
    analyze.add_argument("--minimum-contact-n2", type=int, default=3)
    analyze.add_argument("--contact-cutoff-A", type=float, default=4.0)
    analyze.add_argument("--radial-min-A", type=float, default=0.0)
    analyze.add_argument("--radial-max-A", type=float, default=35.0)
    analyze.add_argument("--radial-bin-A", type=float, default=1.0)
    analyze.add_argument("--z-min-A", type=float, default=0.0)
    analyze.add_argument("--z-max-A", type=float, default=80.0)
    analyze.add_argument("--z-bin-A", type=float, default=1.0)
    analyze.add_argument("--block-steps", type=int, default=400_000)
    combine = subparsers.add_parser("assemble")
    combine.add_argument("--result-manifest", type=Path, required=True)
    combine.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.operation == "assemble":
        result = assemble(args.result_manifest, args.output_dir)
    else:
        cases = {case.case_id: case for case in load_cases(args.case_manifest)}
        if args.case_id not in cases:
            raise ValueError(f"unknown case ID: {args.case_id}")
        result = analyze_case(cases[args.case_id], args.output_dir, args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
