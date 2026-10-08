"""Analyze work, thermostat energy removal, and pressure in constant-force MD."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from molsimflow.postprocess.constant_force_events import stitch_motion_tables

THERMO_REQUIRED = ("Step", "TotEng", "f_BATH")
MOTION_REQUIRED = (
    "TimeStep",
    "v_drivework",
    "v_drivepower",
    "f_BATH",
    "v_pxx_box",
    "v_pyy_box",
    "v_pzz_box",
    "v_pxy_box",
    "v_pxz_box",
    "v_pyz_box",
)
PRESSURE_COLUMNS = (
    "v_pxx_box",
    "v_pyy_box",
    "v_pzz_box",
    "v_pxy_box",
    "v_pxz_box",
    "v_pyz_box",
)


def _numeric(fields: Sequence[str]) -> list[float] | None:
    try:
        return [float(value) for value in fields]
    except ValueError:
        return None


def read_lammps_thermo_block(
    path: Path, required: Sequence[str] = THERMO_REQUIRED
) -> tuple[list[str], np.ndarray]:
    """Return the longest complete thermo block containing the required columns."""

    candidates: list[tuple[list[str], np.ndarray]] = []
    header: list[str] | None = None
    rows: list[list[float]] = []

    def finish() -> None:
        nonlocal header, rows
        if header is not None and rows:
            data = np.asarray(rows, dtype=float)
            if np.all(np.diff(data[:, header.index("Step")]) > 0):
                candidates.append((header, data))
        header, rows = None, []

    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        fields = raw.split()
        if fields and fields[0] == "Step":
            finish()
            if not set(required).difference(fields):
                header = fields
            continue
        if header is None or len(fields) != len(header):
            continue
        numeric = _numeric(fields)
        if numeric is None or not numeric[0].is_integer():
            continue
        if rows and numeric[0] <= rows[-1][0]:
            finish()
            continue
        rows.append(numeric)
    finish()
    if not candidates:
        raise ValueError(f"{path}: no monotonic thermo block contains {tuple(required)}")
    return max(candidates, key=lambda item: len(item[1]))


def stitch_thermo_tables(paths: Sequence[Path]) -> tuple[list[str], np.ndarray]:
    """Join production thermo blocks at restart endpoints without duplicate rows."""

    if not paths:
        raise ValueError("At least one thermo log is required")
    columns: list[str] | None = None
    segments: list[np.ndarray] = []
    previous_step: int | None = None
    for path in paths:
        current_columns, data = read_lammps_thermo_block(path)
        if columns is None:
            columns = current_columns
        elif columns != current_columns:
            raise ValueError(f"Thermo columns differ in {path}")
        step_index = current_columns.index("Step")
        if previous_step is not None:
            first_step = int(round(data[0, step_index]))
            if first_step < previous_step:
                raise ValueError(f"Thermo segment overlaps before the shared endpoint in {path}")
            if first_step == previous_step:
                data = data[1:]
        if len(data):
            segments.append(data)
            previous_step = int(round(data[-1, step_index]))
    assert columns is not None
    if not segments:
        raise ValueError("Thermo logs contain no production rows")
    combined = np.vstack(segments)
    if np.any(np.diff(combined[:, columns.index("Step")]) <= 0):
        raise ValueError("Stitched thermo timesteps are not strictly increasing")
    return columns, combined


def _write_table(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def analyze_energy_balance(
    motion_paths: Sequence[Path],
    thermo_paths: Sequence[Path],
    *,
    timestep_fs: float,
    block_ns: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    """Build aligned time series, block balances, and an endpoint summary."""

    if timestep_fs <= 0 or block_ns <= 0:
        raise ValueError("timestep_fs and block_ns must be positive")
    motion_columns, motion = stitch_motion_tables(
        motion_paths, displacement_columns=("v_drivework",)
    )
    thermo_columns, thermo = stitch_thermo_tables(thermo_paths)
    missing_motion = set(MOTION_REQUIRED).difference(motion_columns)
    missing_thermo = set(THERMO_REQUIRED).difference(thermo_columns)
    if missing_motion or missing_thermo:
        raise ValueError(
            f"Missing energy columns: motion={sorted(missing_motion)}, "
            f"thermo={sorted(missing_thermo)}"
        )
    motion_index = {name: index for index, name in enumerate(motion_columns)}
    thermo_index = {name: index for index, name in enumerate(thermo_columns)}
    motion_steps = motion[:, motion_index["TimeStep"]].astype(np.int64)
    thermo_steps = thermo[:, thermo_index["Step"]].astype(np.int64)
    if not np.array_equal(motion_steps, thermo_steps):
        raise ValueError("Motion and thermo tables do not have identical stitched timesteps")

    work = motion[:, motion_index["v_drivework"]]
    work = work - work[0]
    bath = thermo[:, thermo_index["f_BATH"]]
    thermostat_removed = bath - bath[0]
    total_energy = thermo[:, thermo_index["TotEng"]]
    delta_energy = total_energy - total_energy[0]
    residual = delta_energy - work + thermostat_removed
    relative_time_ns = (motion_steps - motion_steps[0]) * timestep_fs / 1.0e6

    series_rows: list[dict[str, object]] = []
    for row_index, step in enumerate(motion_steps):
        row: dict[str, object] = {
            "step": int(step),
            "relative_time_ns": float(relative_time_ns[row_index]),
            "drive_work_eV": float(work[row_index]),
            "thermostat_removed_eV": float(thermostat_removed[row_index]),
            "delta_total_energy_eV": float(delta_energy[row_index]),
            "closure_residual_eV": float(residual[row_index]),
            "drive_power_eV_per_ps": float(
                motion[row_index, motion_index["v_drivepower"]]
            ),
        }
        row.update(
            {
                name.removeprefix("v_").removesuffix("_box") + "_bar": float(
                    motion[row_index, motion_index[name]]
                )
                for name in PRESSURE_COLUMNS
            }
        )
        series_rows.append(row)

    duration_ns = float(relative_time_ns[-1])
    block_edges = np.arange(0.0, duration_ns + block_ns * 0.5, block_ns)
    if len(block_edges) < 2 or not np.isclose(block_edges[-1], duration_ns):
        raise ValueError("Trajectory duration is not an integer number of requested blocks")
    block_rows: list[dict[str, object]] = []
    for block_index, (start, stop) in enumerate(zip(block_edges[:-1], block_edges[1:])):
        left = int(np.searchsorted(relative_time_ns, start, side="left"))
        right = int(np.searchsorted(relative_time_ns, stop, side="left"))
        if right >= len(relative_time_ns) or not np.isclose(relative_time_ns[right], stop):
            raise ValueError(f"Missing exact block endpoint at {stop} ns")
        sample = slice(left, right + 1)
        row = {
            "block_index": block_index,
            "start_ns": float(start),
            "end_ns": float(stop),
            "start_step": int(motion_steps[left]),
            "end_step": int(motion_steps[right]),
            "drive_work_eV": float(work[right] - work[left]),
            "thermostat_removed_eV": float(
                thermostat_removed[right] - thermostat_removed[left]
            ),
            "delta_total_energy_eV": float(delta_energy[right] - delta_energy[left]),
            "closure_residual_eV": float(residual[right] - residual[left]),
            "mean_drive_power_eV_per_ps": float(
                np.mean(motion[sample, motion_index["v_drivepower"]])
            ),
        }
        row.update(
            {
                "mean_" + name.removeprefix("v_").removesuffix("_box") + "_bar": float(
                    np.mean(motion[sample, motion_index[name]])
                )
                for name in PRESSURE_COLUMNS
            }
        )
        block_rows.append(row)

    summary: dict[str, object] = {
        "status": "PASS",
        "samples": len(series_rows),
        "first_step": int(motion_steps[0]),
        "last_step": int(motion_steps[-1]),
        "duration_ns": duration_ns,
        "drive_work_eV": float(work[-1]),
        "thermostat_removed_eV": float(thermostat_removed[-1]),
        "delta_total_energy_eV": float(delta_energy[-1]),
        "closure_residual_eV": float(residual[-1]),
        "closure_residual_rms_eV": float(np.sqrt(np.mean(residual**2))),
        "mean_drive_power_eV_per_ps": float(
            np.mean(motion[:, motion_index["v_drivepower"]])
        ),
        "thermostat_sign_convention": (
            "positive f_BATH increment is energy removed from the physical system; "
            "LAMMPS econserve equals etotal plus ecouple"
        ),
    }
    return series_rows, block_rows, summary


def _write_plot(rows: Sequence[dict[str, object]], path: Path, font_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager
    from matplotlib import pyplot as plt

    font_manager.fontManager.addfont(font_path)
    matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
    time = np.asarray([row["relative_time_ns"] for row in rows], dtype=float)
    figure, axes = plt.subplots(2, 1, figsize=(7.2, 7.2), sharex=True)
    for name, label in (
        ("drive_work_eV", "Drive work"),
        ("thermostat_removed_eV", "Thermostat removal"),
        ("delta_total_energy_eV", "Total-energy change"),
    ):
        axes[0].plot(time, [row[name] for row in rows], label=label)
    axes[0].set_ylabel("Energy (eV)")
    axes[0].legend(frameon=False)
    axes[1].plot(time, [row["closure_residual_eV"] for row in rows], color="black")
    axes[1].axhline(0.0, color="0.7", linewidth=0.8)
    axes[1].set_xlabel("Relative time (ns)")
    axes[1].set_ylabel("Closure residual (eV)")
    figure.tight_layout()
    figure.savefig(path, dpi=300)
    plt.close(figure)


def run_analysis(args: argparse.Namespace) -> dict[str, object]:
    series, blocks, summary = analyze_energy_balance(
        args.motion,
        args.thermo,
        timestep_fs=args.timestep_fs,
        block_ns=args.block_ns,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    _write_table(output / "energy_timeseries.tsv", series)
    _write_table(output / "energy_blocks.tsv", blocks)
    _write_plot(series, output / "energy_balance.png", args.font_path)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    manifest = {
        "motion": [str(Path(path).resolve()) for path in args.motion],
        "thermo": [str(Path(path).resolve()) for path in args.thermo],
        "output_dir": str(output.resolve()),
        "timestep_fs": args.timestep_fs,
        "block_ns": args.block_ns,
        "font_path": str(args.font_path.resolve()),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, action="append", required=True)
    parser.add_argument("--thermo", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timestep-fs", type=float, default=0.5)
    parser.add_argument("--block-ns", type=float, default=1.0)
    parser.add_argument("--font-path", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if len(args.motion) != len(args.thermo):
        raise ValueError("--motion and --thermo must specify the same number of segments")
    print(json.dumps(run_analysis(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
