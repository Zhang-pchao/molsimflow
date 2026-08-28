"""Align density-fit contact angles with contact-line radii on identical step blocks."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np


def read_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {path}")
        return list(reader)


def parse_labeled_path(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("Contact-line input must be LABEL=PATH")
    label, path = text.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("Contact-line input must be LABEL=PATH")
    return label, Path(path)


def align_blocks(
    angle_rows: Sequence[dict],
    line_rows: Sequence[dict],
    *,
    configuration: str,
    angle_column: str,
    radius_column: str,
    radius_stability: float,
    angle_change: float,
) -> list[dict]:
    steps = np.asarray([int(row["step"]) for row in line_rows])
    radii = np.asarray([float(row[radius_column]) for row in line_rows])
    output = []
    for index, angle_row in enumerate(angle_rows):
        first_step, last_step = int(angle_row["first_step"]), int(angle_row["last_step"])
        selected = radii[(steps >= first_step) & (steps <= last_step) & np.isfinite(radii)]
        if len(selected) < 3:
            raise ValueError(f"Only {len(selected)} contact-line frames in block {index}")
        row = {
            "configuration": configuration,
            "block_index": index,
            "first_step": first_step,
            "last_step": last_step,
            "start_time_ns": float(angle_row["start_time_ns"]),
            "end_time_ns": float(angle_row["end_time_ns"]),
            "frame_count": len(selected),
            "contact_angle_deg": float(angle_row[angle_column]),
            "contact_angle_fit_rmse_A": float(angle_row.get("fit_rmse_A", math.nan)),
            "mean_contact_line_radius_A": float(np.mean(selected)),
            "std_contact_line_radius_A": float(np.std(selected, ddof=1)),
            "delta_contact_angle_deg": math.nan,
            "delta_contact_line_radius_A": math.nan,
            "pinning_candidate": False,
        }
        if output:
            row["delta_contact_angle_deg"] = row["contact_angle_deg"] - output[-1]["contact_angle_deg"]
            row["delta_contact_line_radius_A"] = (
                row["mean_contact_line_radius_A"] - output[-1]["mean_contact_line_radius_A"]
            )
            row["pinning_candidate"] = (
                abs(row["delta_contact_line_radius_A"]) <= radius_stability
                and abs(row["delta_contact_angle_deg"]) >= angle_change
            )
        output.append(row)
    return output


def write_plot(rows: Sequence[dict], output: Path, font_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager
    from matplotlib import pyplot as plt

    font_manager.fontManager.addfont(font_path)
    properties = font_manager.FontProperties(fname=font_path)
    matplotlib.rcParams["font.family"] = properties.get_name()
    figure, axes = plt.subplots(2, 1, figsize=(7.2, 6.5), sharex=True)
    configurations = list(dict.fromkeys(row["configuration"] for row in rows))
    baseline = [row for row in rows if row["configuration"] == configurations[0]]
    time = [0.5 * (row["start_time_ns"] + row["end_time_ns"]) for row in baseline]
    axes[0].plot(time, [row["contact_angle_deg"] for row in baseline], "o-")
    for configuration in configurations:
        selected = [row for row in rows if row["configuration"] == configuration]
        axes[1].plot(
            [0.5 * (row["start_time_ns"] + row["end_time_ns"]) for row in selected],
            [row["mean_contact_line_radius_A"] for row in selected],
            "o-",
            label=configuration,
        )
    axes[0].set_ylabel("Contact angle (deg)")
    axes[1].set_ylabel("Contact-line radius (Å)")
    axes[1].set_xlabel("Time (ns)")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout(); figure.savefig(output, dpi=300); plt.close(figure)


def run_analysis(args: argparse.Namespace) -> dict:
    angle_rows = read_rows(args.contact_angle_blocks)
    all_rows = []
    input_paths = {}
    for configuration, path in args.contact_line:
        input_paths[configuration] = str(path.resolve())
        all_rows.extend(
            align_blocks(
                angle_rows,
                read_rows(path),
                configuration=configuration,
                angle_column=args.angle_column,
                radius_column=args.radius_column,
                radius_stability=args.radius_stability_A,
                angle_change=args.angle_change_deg,
            )
        )
    configurations = list(input_paths)
    candidates = {
        configuration: [
            row["block_index"]
            for row in all_rows
            if row["configuration"] == configuration and row["pinning_candidate"]
        ]
        for configuration in configurations
    }
    common = sorted(set.intersection(*(set(values) for values in candidates.values())))
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=False)
    with (output / "aligned_contact_angle_line.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader(); writer.writerows(all_rows)
    summary = {
        "status": "PASS",
        "configuration_count": len(configurations),
        "block_count_per_configuration": len(angle_rows),
        "radius_stability_A": args.radius_stability_A,
        "angle_change_deg": args.angle_change_deg,
        "candidate_blocks_by_configuration": candidates,
        "common_candidate_blocks": common,
        "pinning_is_threshold_screen_not_proof": True,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "contact_angle_blocks": str(args.contact_angle_blocks.resolve()),
        "contact_line_inputs": input_paths,
        "angle_column": args.angle_column,
        "radius_column": args.radius_column,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_plot(all_rows, output / "contact_angle_line_alignment.png", args.font_path)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contact-angle-blocks", type=Path, required=True)
    parser.add_argument("--contact-line", type=parse_labeled_path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--angle-column", default="dense_phase_contact_angle_deg")
    parser.add_argument("--radius-column", default="contact_line_equivalent_radius_A")
    parser.add_argument("--radius-stability-A", type=float, default=1.0)
    parser.add_argument("--angle-change-deg", type=float, default=3.0)
    parser.add_argument("--font-path", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if min(args.radius_stability_A, args.angle_change_deg) <= 0:
        raise ValueError("Stability and angle-change thresholds must be positive")
    print(json.dumps(run_analysis(args), indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
