"""Block diagnostics for LAMMPS thermo, pressure, and TPCL geometry tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from molsimflow.postprocess.tpcl_pinning_slip import _configure_matplotlib, _save_figure

THERMO_FIELDS = ("Temp", "Density", "PotEng", "KinEng", "TotEng", "Volume", "Press")
PRESSURE_FIELDS = ("Pxx", "Pyy", "Pzz", "Pxy", "Pxz", "Pyz")
DEFAULT_FRAME_FIELDS = (
    "largest_cluster_size",
    "contact_line_area_A2",
    "contact_line_mean_radius_A",
    "contact_line_circularity",
    "phase_center_lateral_displacement_A",
    "decomposed_mean_radius_A",
    "shape_mode_1_amplitude_A",
    "shape_mode_2_amplitude_A",
)
UNITS = {
    "Temp": "K",
    "Density": "g cm-3",
    "PotEng": "eV",
    "KinEng": "eV",
    "TotEng": "eV",
    "Volume": "A3",
    "Press": "bar",
    "Pxx": "bar",
    "Pyy": "bar",
    "Pzz": "bar",
    "Pxy": "bar",
    "Pxz": "bar",
    "Pyz": "bar",
    "normal_minus_tangential_bar": "bar",
    "largest_cluster_size": "molecules",
    "contact_line_area_A2": "A2",
    "contact_line_mean_radius_A": "A",
    "contact_line_circularity": "1",
    "phase_center_lateral_displacement_A": "A",
    "decomposed_mean_radius_A": "A",
    "shape_mode_1_amplitude_A": "A",
    "shape_mode_2_amplitude_A": "A",
}


@dataclass(frozen=True)
class StateSource:
    """One case routed entirely by its manifest row."""

    case_id: str
    frame_metrics: Path
    thermo_log: Path
    global_stress: Path


def read_sources(path: Path) -> list[StateSource]:
    """Read a tab-separated, path-explicit state-diagnostics manifest."""

    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = ("case_id", "frame_metrics", "thermo_log", "global_stress")
    if not rows or set(required).difference(rows[0]):
        raise ValueError(f"{path}: missing state source rows or columns")
    sources = [
        StateSource(
            row["case_id"],
            Path(row["frame_metrics"]),
            Path(row["thermo_log"]),
            Path(row["global_stress"]),
        )
        for row in rows
    ]
    if any(not source.case_id for source in sources) or len(
        {source.case_id for source in sources}
    ) != len(sources):
        raise ValueError(f"{path}: case_id values must be nonempty and unique")
    return sources


def _numeric(fields: Sequence[str]) -> list[float] | None:
    try:
        return [float(value) for value in fields]
    except ValueError:
        return None


def _as_series(
    rows: dict[str, list[tuple[float, float]]],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        name: (np.asarray([item[0] for item in values]), np.asarray([item[1] for item in values]))
        for name, values in rows.items()
        if values
    }


def _canonical_pressure_field(name: str) -> str:
    """Normalize standard and user-variable pressure labels without case routing."""

    candidate = name.removeprefix("v_").removeprefix("c_").removesuffix("_check")
    return {field.lower(): field for field in PRESSURE_FIELDS}.get(candidate.lower(), candidate)


def read_lammps_thermo(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Read non-overlapping standard LAMMPS thermo tables from a log file."""

    header: list[str] | None = None
    rows: dict[str, list[tuple[float, float]]] = {}
    last_step: int | None = None
    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        fields = raw.split()
        if fields and fields[0] == "Step" and "Time" in fields:
            header = fields
            continue
        if header is None or len(fields) != len(header):
            continue
        numeric = _numeric(fields)
        if numeric is None:
            continue
        step = int(numeric[0])
        if numeric[0] != step:
            continue
        if last_step is not None and step <= last_step:
            raise ValueError(f"{path}: thermo timestep is not strictly increasing at {step}")
        last_step = step
        time_ns = numeric[header.index("Time")] / 1000.0
        for name, value in zip(header, numeric):
            if name not in {"Step", "Time"}:
                rows.setdefault(name, []).append((time_ns, value))
    if not rows:
        raise ValueError(f"{path}: no LAMMPS thermo table found")
    return _as_series(rows)


def read_global_stress(path: Path, timestep_fs: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Read an ave/time pressure table and derive normal-minus-tangential pressure."""

    if timestep_fs <= 0:
        raise ValueError("timestep_fs must be positive")
    header: list[str] | None = None
    rows: dict[str, list[tuple[float, float]]] = {}
    last_step: int | None = None
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if stripped.startswith("# TimeStep"):
            header = stripped[2:].split()
            continue
        if not stripped or stripped.startswith("#") or header is None:
            continue
        fields = stripped.split()
        if len(fields) != len(header):
            raise ValueError(f"{path}: malformed ave/time row")
        numeric = _numeric(fields)
        if numeric is None:
            raise ValueError(f"{path}: non-numeric ave/time row")
        step = int(numeric[0])
        if numeric[0] != step or (last_step is not None and step <= last_step):
            raise ValueError(f"{path}: non-monotonic stress timestep")
        last_step = step
        for name, value in zip(header[1:], numeric[1:]):
            rows.setdefault(_canonical_pressure_field(name), []).append(
                (step * timestep_fs / 1.0e6, value)
            )
    if {"Pxx", "Pyy", "Pzz"}.difference(rows):
        raise ValueError(f"{path}: missing required pressure components")
    if any(
        time_x != time_y or time_x != time_z
        for (time_x, _), (time_y, _), (time_z, _) in zip(rows["Pxx"], rows["Pyy"], rows["Pzz"])
    ):
        raise ValueError(f"{path}: pressure components have mismatched timesteps")
    rows["normal_minus_tangential_bar"] = [
        (time, pzz - 0.5 * (pxx + pyy))
        for (time, pxx), (_, pyy), (_, pzz) in zip(rows["Pxx"], rows["Pyy"], rows["Pzz"])
    ]
    return _as_series(rows)


def read_frame_metrics(
    path: Path, fields: Sequence[str]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Read selected finite numeric TPCL frame fields on the native time grid."""

    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "time_ns" not in reader.fieldnames:
            raise ValueError(f"{path}: missing time_ns")
        missing = set(fields).difference(reader.fieldnames)
        if missing:
            raise ValueError(f"{path}: missing frame fields {sorted(missing)}")
        rows = {field: [] for field in fields}
        last_time: float | None = None
        for row in reader:
            time_ns = float(row["time_ns"])
            if last_time is not None and time_ns <= last_time:
                raise ValueError(f"{path}: frame time is not strictly increasing")
            last_time = time_ns
            for field in fields:
                value = float(row[field])
                if not math.isfinite(value):
                    raise ValueError(f"{path}: non-finite {field} at {time_ns:g} ns")
                rows[field].append((time_ns, value))
    return _as_series(rows)


def _blocks(time_ns: np.ndarray, values: np.ndarray, block_ps: float) -> tuple[list[dict], dict]:
    if len(time_ns) < 2 or block_ps <= 0 or not np.all(np.diff(time_ns) > 0):
        raise ValueError("a series needs increasing time, two rows, and a positive block size")
    block_ns = block_ps / 1000.0
    start, end = float(time_ns[0]), float(time_ns[-1])
    count = math.ceil((end - start) / block_ns)
    records, means = [], []
    for index in range(count):
        left, right = start + index * block_ns, min(end, start + (index + 1) * block_ns)
        mask = (time_ns >= left) & ((time_ns <= right) if index + 1 == count else (time_ns < right))
        chunk = values[mask]
        if not len(chunk):
            continue
        mean = float(np.mean(chunk))
        means.append(mean)
        records.append(
            {
                "block_index": index,
                "block_start_ns": left,
                "block_end_ns": right,
                "sample_count": len(chunk),
                "mean": mean,
                "std": float(np.std(chunk, ddof=1)) if len(chunk) > 1 else math.nan,
            }
        )
    slope = float(np.polyfit(time_ns, values, 1)[0])
    lag1, effective = math.nan, math.nan
    if len(values) > 2 and np.std(values[:-1]) > 0 and np.std(values[1:]) > 0:
        lag1 = float(np.corrcoef(values[:-1], values[1:])[0, 1])
        if math.isfinite(lag1):
            effective = float(max(1.0, min(len(values), len(values) * (1.0 - lag1) / (1.0 + lag1))))
    return records, {
        "sample_count": len(values),
        "block_count": len(records),
        "time_start_ns": start,
        "time_end_ns": end,
        "time_mean": float(np.mean(values)),
        "time_std": float(np.std(values, ddof=1)),
        "linear_slope_per_ns": slope,
        "first_block_mean": means[0],
        "last_block_mean": means[-1],
        "last_minus_first_block_mean": means[-1] - means[0],
        "lag1_autocorrelation": lag1,
        "ar1_effective_sample_size": effective,
    }


def summarize_sources(
    sources: Sequence[StateSource],
    *,
    timestep_fs: float,
    block_ps: float,
    frame_fields: Sequence[str],
) -> tuple[list[dict], list[dict], dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]]:
    """Return per-block and per-series diagnostics for every manifest condition."""

    block_rows, summary_rows = [], []
    plotted: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for source in sources:
        groups = {
            "thermo": read_lammps_thermo(source.thermo_log),
            "global_stress": read_global_stress(source.global_stress, timestep_fs),
            "tpcl_geometry": read_frame_metrics(source.frame_metrics, frame_fields),
        }
        plotted[source.case_id] = {
            f"{group}:{field}": series
            for group, data in groups.items()
            for field, series in data.items()
        }
        for group, data in groups.items():
            for field, (time_ns, values) in data.items():
                records, summary = _blocks(time_ns, values, block_ps)
                for record in records:
                    block_rows.append(
                        {
                            "case_id": source.case_id,
                            "source": group,
                            "metric": field,
                            "unit": UNITS.get(field, "native"),
                            **record,
                        }
                    )
                summary_rows.append(
                    {
                        "case_id": source.case_id,
                        "source": group,
                        "metric": field,
                        "unit": UNITS.get(field, "native"),
                        **summary,
                    }
                )
    return block_rows, summary_rows, plotted


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_figures(
    plotted: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    output: Path,
    font_path: Path,
    frame_fields: Sequence[str],
) -> None:
    """Write raw time-series figures traceable to the tabular outputs."""

    _configure_matplotlib(font_path)
    from matplotlib import pyplot as plt

    figures = Path(output) / "figures"
    figures.mkdir()
    for filename, source, fields in (
        ("01_thermo", "thermo", THERMO_FIELDS),
        (
            "02_global_pressure_diagnostic",
            "global_stress",
            (*PRESSURE_FIELDS, "normal_minus_tangential_bar"),
        ),
        ("03_tpcl_geometry", "tpcl_geometry", frame_fields),
    ):
        available = [
            field
            for field in fields
            if any(f"{source}:{field}" in data for data in plotted.values())
        ]
        figure, axes = plt.subplots(
            len(available), 1, figsize=(8.4, 2.2 * len(available)), sharex=True
        )
        for axis, field in zip(np.atleast_1d(axes), available):
            for case_id, data in plotted.items():
                series = data.get(f"{source}:{field}")
                if series is not None:
                    axis.plot(series[0], series[1], linewidth=0.7, label=case_id)
            axis.set_ylabel(f"{field}\n[{UNITS.get(field, 'native')}]")
            axis.grid(alpha=0.25)
        np.atleast_1d(axes)[0].legend(fontsize=7, ncol=2)
        np.atleast_1d(axes)[-1].set_xlabel("Time [ns]")
        figure.suptitle(filename.replace("_", " "))
        figure.tight_layout()
        _save_figure(figure, figures / filename)
        plt.close(figure)


def run_summary(
    sources_path: Path,
    output_dir: Path,
    *,
    timestep_fs: float,
    block_ps: float,
    font_path: Path,
    frame_fields: Sequence[str] = DEFAULT_FRAME_FIELDS,
    make_plots: bool = True,
) -> dict:
    """Create auditable state diagnostics without any project-specific path in code."""

    sources = read_sources(sources_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    block_rows, summary_rows, plotted = summarize_sources(
        sources, timestep_fs=timestep_fs, block_ps=block_ps, frame_fields=frame_fields
    )
    _write_csv(output / "block_summary.csv", block_rows)
    _write_csv(output / "important_data.csv", summary_rows)
    Path(output / "report.md").write_text(
        "# TPCL state diagnostics\n\n"
        f"Cases: {', '.join(source.case_id for source in sources)}. "
        f"Fixed block length: {block_ps:g} ps.\n\n"
        "These are numerical and geometric stationarity diagnostics, not an equilibrium verdict. "
        "The pressure tensor and normal-minus-tangential pressure are global-box "
        "diagnostics, not nanodroplet mechanical surface tension. The AR(1) effective "
        "sample size is a serial-correlation proxy, not an independent replicate count.\n",
        encoding="utf-8",
    )
    if make_plots:
        write_figures(plotted, output, font_path, frame_fields)
    summary = {
        "status": "PASS",
        "case_count": len(sources),
        "block_ps": block_ps,
        "series_count": len(summary_rows),
        "scientific_boundary": (
            "numerical diagnostics; no equilibrium or local surface-tension claim"
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timestep-fs", type=float, required=True)
    parser.add_argument("--block-ps", type=float, default=500.0)
    parser.add_argument("--font-path", type=Path, required=True)
    parser.add_argument("--frame-field", action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_summary(
        args.sources,
        args.output_dir,
        timestep_fs=args.timestep_fs,
        block_ps=args.block_ps,
        font_path=args.font_path,
        frame_fields=tuple(args.frame_field) or DEFAULT_FRAME_FIELDS,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
