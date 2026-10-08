"""Compute PBC-unwrapped planar motion and time-origin-averaged MSD from a CSV."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np


def unwrap_planar(x: np.ndarray, y: np.ndarray, box_x: float, box_y: float) -> np.ndarray:
    wrapped = np.column_stack((x, y))
    lengths = np.array([box_x, box_y], dtype=float)
    unwrapped = np.empty_like(wrapped)
    unwrapped[0] = wrapped[0]
    for index in range(1, len(wrapped)):
        delta = wrapped[index] - wrapped[index - 1]
        delta -= lengths * np.round(delta / lengths)
        unwrapped[index] = unwrapped[index - 1] + delta
    return unwrapped


def time_origin_msd(coords: np.ndarray) -> np.ndarray:
    values = np.zeros(len(coords))
    for lag in range(1, len(coords)):
        delta = coords[lag:] - coords[:-lag]
        values[lag] = float(np.mean(np.sum(delta * delta, axis=1)))
    return values


def read_columns(path: Path, step_column: str, x_column: str, y_column: str) -> tuple[np.ndarray, ...]:
    with Path(path).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    for column in (step_column, x_column, y_column):
        if column not in rows[0]:
            raise ValueError(f"Missing CSV column: {column}")
    return tuple(np.array([float(row[column]) for row in rows]) for column in (step_column, x_column, y_column))


def write_plot(time_ns: np.ndarray, displacement: np.ndarray, msd: np.ndarray, output: Path, font: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager
    from matplotlib import pyplot as plt

    font_manager.fontManager.addfont(font)
    properties = font_manager.FontProperties(fname=font)
    font_manager.findfont(properties, fallback_to_default=False)
    matplotlib.rcParams["font.family"] = properties.get_name()
    figure, axes = plt.subplots(2, 1, figsize=(7.2, 6.0))
    axes[0].plot(time_ns, displacement)
    axes[0].set_xlabel("Time (ns)"); axes[0].set_ylabel("Planar displacement (Å)")
    axes[1].plot(time_ns - time_ns[0], msd)
    axes[1].set_xlabel("Lag time (ns)"); axes[1].set_ylabel(r"Planar MSD (Å$^2$)")
    for axis in axes: axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout(); figure.savefig(output, dpi=300); plt.close(figure)


def run_analysis(args: argparse.Namespace) -> dict:
    step, x, y = read_columns(args.input, args.step_column, args.x_column, args.y_column)
    coords = unwrap_planar(x, y, args.box_x_A, args.box_y_A)
    delta = coords - coords[0]
    displacement = np.linalg.norm(delta, axis=1)
    increments = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    msd = time_origin_msd(coords)
    time_ns = step * args.timestep_fs / 1.0e6
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "planar_motion.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["step", "time_ns", "x_unwrapped_A", "y_unwrapped_A", "displacement_A", "displacement_squared_A2"]
        )
        for values in zip(step.astype(int), time_ns, coords[:, 0], coords[:, 1], displacement, displacement**2):
            writer.writerow(values)
    with (output / "planar_msd.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["lag_frames", "lag_time_ns", "msd_A2"])
        for index, value in enumerate(msd):
            writer.writerow([index, time_ns[index] - time_ns[0], value])
    summary = {
        "status": "PASS", "frames": len(step),
        "net_displacement_A": float(displacement[-1]),
        "maximum_displacement_A": float(displacement.max()),
        "cumulative_path_length_A": float(increments.sum()),
        "msd_is_time_origin_averaged_not_an_equilibrium_diffusion_fit": True,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_plot(time_ns, displacement, msd, output / "planar_motion.png", args.font_path)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--step-column", default="step")
    parser.add_argument("--x-column", required=True)
    parser.add_argument("--y-column", required=True)
    parser.add_argument("--box-x-A", type=float, required=True)
    parser.add_argument("--box-y-A", type=float, required=True)
    parser.add_argument("--timestep-fs", type=float, default=0.5)
    parser.add_argument("--font-path", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    print(json.dumps(run_analysis(build_parser().parse_args(argv)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
