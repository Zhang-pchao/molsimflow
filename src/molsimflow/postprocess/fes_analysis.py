"""Reusable free-energy and barrier analysis utilities."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


DEFAULT_BARRIER_WINDOWS: Tuple[Tuple[str, float, float], ...] = (
    ("all", -math.inf, math.inf),
    ("nucleation_s_lt_210", -math.inf, 210.0),
    ("dissolution_s_gt_50", 50.0, math.inf),
)


@dataclass(frozen=True)
class FesCurveSpec:
    """Input metadata for one 1D FES curve."""

    path: Path
    label: str
    group: str = "default"
    dataset_key: str = ""


@dataclass
class FesCurve:
    """Loaded 1D FES curve data."""

    spec: FesCurveSpec
    cv: np.ndarray
    free_energy: np.ndarray
    uncertainty: np.ndarray


@dataclass(frozen=True)
class FesConvergenceSpec:
    """Input metadata for one FES convergence/robustness dataset."""

    path: Path
    label: str
    group: str = "default"
    dataset_key: str = ""
    series: str = "default"
    chemistry: str = ""
    block_paths: Tuple[Path, ...] = ()
    cumulative_paths: Tuple[Path, ...] = ()


@dataclass(frozen=True)
class Fes2DGrid:
    """Regular 2D FES grid loaded from a PLUMED-style table."""

    source_path: Path
    fields: Tuple[str, ...]
    metadata: Dict[str, str]
    x_name: str
    y_name: str
    z_name: str
    uncertainty_name: Optional[str]
    x_values: np.ndarray
    y_values: np.ndarray
    free_energy: np.ndarray
    uncertainty: Optional[np.ndarray]


@dataclass(frozen=True)
class Fes2DBatchCaseSpec:
    """Input metadata for one 2D FES batch case."""

    bias_dir: Path
    case_label: str
    family: str = ""
    dataset_key: str = ""
    safe_label: str = ""
    colvar_path: Optional[Path] = None
    run_dir: Optional[Path] = None
    fes_file: Optional[Path] = None
    output_dir: Optional[Path] = None


@dataclass(frozen=True)
class Fes2DBatchConfig:
    """Path and plotting defaults for 2D FES batch manifest generation."""

    colvar_name: str = "COLVAR_tmp"
    run_subdir: str = "fes2D/bins50"
    fes_name: str = "fes-rew.dat"
    output_subdir: str = "fes2d_plot"
    prefix_template: str = "{safe_label}_fes2d"
    x_low: float = 20.0
    x_high: float = 52.0
    y_low: float = 50.0
    y_high: float = 380.0
    max_fes: float = 200.0
    smooth_sigma: float = 0.8
    smooth_valid_threshold: float = 0.25
    contour_levels: int = 16
    dpi: int = 300
    zero_scope: str = "window"
    missing_plot_value: str = "max"
    write_plots: bool = False
    write_comparison: bool = False


@dataclass(frozen=True)
class FesCumulativeReweightSpec:
    """Input metadata for one cumulative reweight dataset."""

    system: str
    workdir: Path
    colvar: Path
    sample_size: int
    output_dir: Optional[Path] = None
    group: str = "default"


@dataclass(frozen=True)
class FesCumulativeReweightConfig:
    """Command defaults for cumulative-prefix FES reweight planning."""

    driver: Path
    output_root: Path
    fractions: Tuple[float, ...] = (0.60, 0.80, 1.00)
    output_prefix: str = "fes-cum"
    python_executable: str = "python"
    cv: str = "d3d_all"
    cv_min: float = 5.0
    cv_max: float = 52.0
    delta_f_at: float = 45.0
    sigma: float = 0.06
    skiprows: int = 50000
    blocks: int = 3
    temperature: float = 330.0


@dataclass(frozen=True)
class PlumedTable:
    """A numeric PLUMED table with its first FIELDS header."""

    path: Path
    fields: Tuple[str, ...]
    data: np.ndarray


@dataclass(frozen=True)
class BandwidthChoice:
    """Smoothing width selected for one projected CV."""

    cv_name: str
    value: Optional[float]
    sigma_bins: float
    source: str


@dataclass(frozen=True)
class ReweightProjectionResult:
    """Output metadata for one reweighted FES projection."""

    kind: str
    cvs: Tuple[str, ...]
    safe_label: str
    fes_path: Path
    plot_path: Optional[Path]
    sample_size: int
    effective_sample_size: float
    finite_points: int
    bandwidth_sources: Tuple[str, ...]


def _as_float(value: str) -> float:
    text = str(value).strip()
    if text.lower() in {"inf", "+inf", "infinity", "+infinity"}:
        return math.inf
    if text.lower() in {"-inf", "-infinity"}:
        return -math.inf
    return float(text)


def parse_window(raw: str) -> Tuple[str, float, float]:
    """Parse `name:low:high` window syntax."""

    parts = str(raw).split(":")
    if len(parts) != 3:
        raise ValueError("Barrier windows must use name:low:high syntax")
    name = parts[0].strip()
    if not name:
        raise ValueError("Barrier window name cannot be empty")
    low = _as_float(parts[1])
    high = _as_float(parts[2])
    if high <= low:
        raise ValueError(f"Invalid barrier window bounds for {name}: high <= low")
    return name, low, high


def parse_curve_spec(values: Sequence[str]) -> FesCurveSpec:
    """Parse a CLI curve tuple: PATH LABEL GROUP."""

    if len(values) != 3:
        raise ValueError("--curve requires PATH LABEL GROUP")
    path, label, group = values
    dataset_key = Path(path).stem
    return FesCurveSpec(path=Path(path), label=label, group=group, dataset_key=dataset_key)


def load_curve_manifest(path: Path) -> List[FesCurveSpec]:
    """Load curve specs from a CSV manifest.

    Required columns are `path` and `label`.  Optional columns are `group` and
    `dataset_key`.
    """

    specs: List[FesCurveSpec] = []
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {path}")
        missing = [column for column in ["path", "label"] if column not in reader.fieldnames]
        if missing:
            raise ValueError("Manifest missing required columns: " + ", ".join(missing))
        for index, row in enumerate(reader):
            raw_path = str(row.get("path", "")).strip()
            label = str(row.get("label", "")).strip()
            if not raw_path or not label:
                continue
            group = str(row.get("group") or "default")
            dataset_key = str(row.get("dataset_key") or Path(raw_path).stem or f"curve_{index}")
            specs.append(FesCurveSpec(path=Path(raw_path), label=label, group=group, dataset_key=dataset_key))
    if not specs:
        raise ValueError(f"No curves found in manifest: {path}")
    return specs


def _manifest_path(raw: object, base_dir: Path) -> Optional[Path]:
    text = str(raw or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path


def _split_manifest_list(raw: object) -> List[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    normalized = text.replace("|", ";").replace(",", ";")
    return [item.strip() for item in normalized.split(";") if item.strip()]


def _shell_quote(value: object) -> str:
    return shlex.quote(str(value))


def _manifest_path_list(raw: object, base_dir: Path) -> Tuple[Path, ...]:
    paths: List[Path] = []
    for item in _split_manifest_list(raw):
        path = _manifest_path(item, base_dir)
        if path is not None:
            paths.append(path)
    return tuple(paths)


def infer_block_paths(final_path: Path, block_count: int = 3) -> Tuple[Path, ...]:
    """Infer existing block FES paths next to a final FES file."""

    final_path = Path(final_path)
    candidates = [
        final_path.parent / f"{final_path.stem}_{index}{final_path.suffix}"
        for index in range(1, int(block_count) + 1)
    ]
    return tuple(path for path in candidates if path.exists())


def sort_cumulative_paths(paths: Iterable[Path]) -> Tuple[Path, ...]:
    """Sort cumulative FES paths by the last integer in each filename."""

    def sort_key(path: Path) -> Tuple[int, str]:
        matches = re.findall(r"\d+", path.stem)
        return (int(matches[-1]) if matches else 10**9, path.name)

    return tuple(sorted((Path(path) for path in paths), key=sort_key))


def discover_cumulative_paths(root: Path, pattern: str = "fes-cum_*.dat") -> Tuple[Path, ...]:
    """Discover cumulative FES profiles in a directory."""

    root = Path(root)
    if not root.exists():
        return ()
    return sort_cumulative_paths(root.glob(pattern))


def load_fes_convergence_manifest(
    path: Path,
    infer_blocks: bool = False,
    block_count: int = 3,
    cumulative_glob: str = "fes-cum_*.dat",
) -> List[FesConvergenceSpec]:
    """Load FES convergence inputs from a CSV manifest.

    Required column: `path`. Optional columns: `label`, `group`,
    `dataset_key`, `series`, `chemistry`, `block_paths`, `cumulative_paths`,
    and `cumulative_dir`. Relative paths are resolved against the manifest
    directory.
    """

    manifest_path = Path(path)
    base_dir = manifest_path.parent
    specs: List[FesConvergenceSpec] = []
    with manifest_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {path}")
        if "path" not in reader.fieldnames:
            raise ValueError("Manifest missing required column: path")
        for index, row in enumerate(reader):
            input_path = _manifest_path(row.get("path"), base_dir)
            if input_path is None:
                continue
            dataset_key = str(row.get("dataset_key") or input_path.stem or f"dataset_{index}").strip()
            label = str(row.get("label") or dataset_key).strip()
            group = str(row.get("group") or "default").strip()
            series = str(row.get("series") or group or "default").strip()
            chemistry = str(row.get("chemistry") or "").strip()

            block_paths = _manifest_path_list(row.get("block_paths"), base_dir)
            if not block_paths and infer_blocks:
                block_paths = infer_block_paths(input_path, block_count=block_count)

            cumulative_paths = _manifest_path_list(row.get("cumulative_paths"), base_dir)
            cumulative_dir = _manifest_path(row.get("cumulative_dir"), base_dir)
            if not cumulative_paths and cumulative_dir is not None:
                cumulative_paths = discover_cumulative_paths(cumulative_dir, pattern=cumulative_glob)

            specs.append(
                FesConvergenceSpec(
                    path=input_path,
                    label=label,
                    group=group,
                    dataset_key=dataset_key,
                    series=series,
                    chemistry=chemistry,
                    block_paths=tuple(block_paths),
                    cumulative_paths=sort_cumulative_paths(cumulative_paths),
                )
            )
    if not specs:
        raise ValueError(f"No FES convergence datasets found in manifest: {path}")
    return specs


def safe_file_label(label: str) -> str:
    """Return a filesystem-friendly label for generated FES outputs."""

    text = str(label).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "case"


def infer_case_label_from_path(path: Path, family: str = "") -> str:
    """Infer a generic case label from a path when no explicit label is given."""

    path = Path(path)
    parts = [part for part in path.parts if part]
    base = path.name or (parts[-1] if parts else "case")
    if base.startswith("bias_") and len(parts) >= 2:
        base = parts[-2]
    label = safe_file_label(base).replace("_", "-")
    return f"{label}-{family}" if family else label


def read_fes2d_case_path_list(path: Path, family: str = "") -> List[Fes2DBatchCaseSpec]:
    """Read a plain text file with one bias directory per non-comment line."""

    specs: List[Fes2DBatchCaseSpec] = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        bias_dir = Path(line).expanduser()
        label = infer_case_label_from_path(bias_dir, family=family)
        specs.append(
            Fes2DBatchCaseSpec(
                bias_dir=bias_dir,
                case_label=label,
                family=family,
                dataset_key=safe_file_label(label),
            )
        )
    return specs


def load_fes2d_batch_case_manifest(path: Path) -> List[Fes2DBatchCaseSpec]:
    """Load 2D FES batch cases from a CSV manifest.

    Required column: `bias_dir`. Optional columns are `case_label`, `family`,
    `dataset_key`, `safe_label`, `colvar_path`, `run_dir`, `fes_file`, and
    `output_dir`. Relative paths are resolved against the manifest directory.
    """

    manifest_path = Path(path)
    base_dir = manifest_path.parent
    specs: List[Fes2DBatchCaseSpec] = []
    with manifest_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {path}")
        if "bias_dir" not in reader.fieldnames:
            raise ValueError("2D FES batch manifest missing required column: bias_dir")
        for index, row in enumerate(reader):
            bias_dir = _manifest_path(row.get("bias_dir"), base_dir)
            if bias_dir is None:
                continue
            family = str(row.get("family") or "").strip()
            label = str(row.get("case_label") or "").strip() or infer_case_label_from_path(bias_dir, family=family)
            dataset_key = str(row.get("dataset_key") or safe_file_label(label) or f"case_{index}").strip()
            safe_label = str(row.get("safe_label") or safe_file_label(label)).strip()
            specs.append(
                Fes2DBatchCaseSpec(
                    bias_dir=bias_dir,
                    case_label=label,
                    family=family,
                    dataset_key=dataset_key,
                    safe_label=safe_label,
                    colvar_path=_manifest_path(row.get("colvar_path"), base_dir),
                    run_dir=_manifest_path(row.get("run_dir"), base_dir),
                    fes_file=_manifest_path(row.get("fes_file"), base_dir),
                    output_dir=_manifest_path(row.get("output_dir"), base_dir),
                )
            )
    if not specs:
        raise ValueError(f"No 2D FES batch cases found in manifest: {path}")
    return specs


def parse_case_path_list_spec(raw: str) -> Tuple[str, Path]:
    """Parse `FAMILY:PATH` or `PATH` syntax for plain case-path lists."""

    text = str(raw).strip()
    if not text:
        raise ValueError("Empty case-path-list spec")
    if ":" in text and not re.match(r"^[A-Za-z]:[\\/]", text):
        family, path = text.split(":", 1)
        return family.strip(), Path(path.strip())
    return "", Path(text)


def build_fes2d_grid_command(row: Mapping[str, object], executable: str = "molsimflow") -> str:
    """Build a shell-safe command string for an already prepared manifest row."""

    parts = [
        executable,
        "postprocess",
        "fes2d-grid",
        "--fes-file",
        str(row["fes_file"]),
        "--output-dir",
        str(row["plot_dir"]),
        "--x-range",
        str(row["x_low"]),
        str(row["x_high"]),
        "--y-range",
        str(row["y_low"]),
        str(row["y_high"]),
        "--max-fes",
        str(row["max_fes"]),
        "--smooth-sigma",
        str(row["smooth_sigma"]),
        "--smooth-valid-threshold",
        str(row["smooth_valid_threshold"]),
        "--contour-levels",
        str(row["contour_levels"]),
        "--dpi",
        str(row["dpi"]),
        "--prefix",
        str(row["prefix"]),
        "--title",
        str(row["case_label"]),
        "--zero-scope",
        str(row["zero_scope"]),
        "--missing-plot-value",
        str(row["missing_plot_value"]),
    ]
    if int(row.get("write_plots", 0)):
        parts.append("--write-plots")
    if int(row.get("write_comparison", 0)):
        parts.append("--write-comparison")
    return " ".join(_shell_quote(part) for part in parts)


def prepare_fes2d_batch_manifest(
    cases: Sequence[Fes2DBatchCaseSpec],
    output_manifest: Path,
    *,
    config: Fes2DBatchConfig = Fes2DBatchConfig(),
    command_executable: str = "molsimflow",
    create_dirs: bool = False,
) -> List[Dict[str, object]]:
    """Prepare a path-explicit 2D FES batch manifest and command table."""

    rows: List[Dict[str, object]] = []
    for case in cases:
        safe_label = case.safe_label or safe_file_label(case.case_label)
        run_dir = case.run_dir or (case.bias_dir / config.run_subdir)
        colvar_path = case.colvar_path or (case.bias_dir / config.colvar_name)
        fes_file = case.fes_file or (run_dir / config.fes_name)
        plot_dir = case.output_dir or (run_dir / config.output_subdir)
        prefix = config.prefix_template.format(
            safe_label=safe_label,
            dataset_key=case.dataset_key or safe_label,
            family=case.family,
            case_label=case.case_label,
        )
        if create_dirs:
            run_dir.mkdir(parents=True, exist_ok=True)
            plot_dir.mkdir(parents=True, exist_ok=True)
        row: Dict[str, object] = {
            "dataset_key": case.dataset_key or safe_label,
            "family": case.family,
            "case_label": case.case_label,
            "safe_label": safe_label,
            "bias_dir": str(case.bias_dir),
            "run_dir": str(run_dir),
            "colvar_tmp": str(colvar_path),
            "fes_file": str(fes_file),
            "plot_dir": str(plot_dir),
            "prefix": prefix,
            "x_low": config.x_low,
            "x_high": config.x_high,
            "y_low": config.y_low,
            "y_high": config.y_high,
            "max_fes": config.max_fes,
            "smooth_sigma": config.smooth_sigma,
            "smooth_valid_threshold": config.smooth_valid_threshold,
            "contour_levels": config.contour_levels,
            "dpi": config.dpi,
            "zero_scope": config.zero_scope,
            "missing_plot_value": config.missing_plot_value,
            "write_plots": int(config.write_plots),
            "write_comparison": int(config.write_comparison),
            "bias_dir_exists": int(case.bias_dir.exists()),
            "colvar_tmp_exists": int(colvar_path.exists()),
            "fes_file_exists": int(fes_file.exists()),
        }
        row["fes2d_grid_command"] = build_fes2d_grid_command(row, executable=command_executable)
        rows.append(row)

    fieldnames = [
        "dataset_key",
        "family",
        "case_label",
        "safe_label",
        "bias_dir",
        "run_dir",
        "colvar_tmp",
        "fes_file",
        "plot_dir",
        "prefix",
        "x_low",
        "x_high",
        "y_low",
        "y_high",
        "max_fes",
        "smooth_sigma",
        "smooth_valid_threshold",
        "contour_levels",
        "dpi",
        "zero_scope",
        "missing_plot_value",
        "write_plots",
        "write_comparison",
        "bias_dir_exists",
        "colvar_tmp_exists",
        "fes_file_exists",
        "fes2d_grid_command",
    ]
    _write_csv_with_fields(Path(output_manifest), rows, fieldnames)
    return rows


def load_fes_cumulative_reweight_manifest(path: Path) -> List[FesCumulativeReweightSpec]:
    """Load cumulative reweight planning inputs from a CSV manifest.

    Required columns are `system`, `workdir`, `colvar`, and `sample_size`.
    Optional columns are `output_dir` and `group`. Relative paths are resolved
    against the manifest directory.
    """

    manifest_path = Path(path)
    base_dir = manifest_path.parent
    specs: List[FesCumulativeReweightSpec] = []
    with manifest_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {path}")
        required = ["system", "workdir", "colvar", "sample_size"]
        missing = [column for column in required if column not in reader.fieldnames]
        if missing:
            raise ValueError("Cumulative reweight manifest missing required columns: " + ", ".join(missing))
        for row in reader:
            system = str(row.get("system") or "").strip()
            if not system:
                continue
            workdir = _manifest_path(row.get("workdir"), base_dir)
            colvar = _manifest_path(row.get("colvar"), base_dir)
            if workdir is None or colvar is None:
                continue
            try:
                sample_size = int(float(str(row.get("sample_size", "")).strip()))
            except ValueError as exc:
                raise ValueError(f"Invalid sample_size for {system}: {row.get('sample_size')!r}") from exc
            if sample_size <= 0:
                raise ValueError(f"sample_size must be positive for {system}")
            specs.append(
                FesCumulativeReweightSpec(
                    system=system,
                    workdir=workdir,
                    colvar=colvar,
                    sample_size=sample_size,
                    output_dir=_manifest_path(row.get("output_dir"), base_dir),
                    group=str(row.get("group") or "default").strip(),
                )
            )
    if not specs:
        raise ValueError(f"No cumulative reweight datasets found in manifest: {path}")
    return specs


def normalize_reweight_fractions(fractions: Sequence[float]) -> Tuple[float, ...]:
    """Validate and sort cumulative trajectory fractions."""

    values = tuple(sorted({float(value) for value in fractions}))
    if not values:
        raise ValueError("At least one cumulative fraction is required")
    for value in values:
        if value <= 0.0 or value > 1.0:
            raise ValueError("Cumulative fractions must be in (0, 1]")
    return values


def build_cumulative_reweight_command(row: Mapping[str, object], config: FesCumulativeReweightConfig) -> str:
    """Build a shell-safe reweight command for one planned cumulative output."""

    parts = [
        config.python_executable,
        str(config.driver),
        "-o",
        str(row["output_path"]),
        "-f",
        str(row["colvar"]),
        "--cv",
        config.cv,
        "--min",
        str(config.cv_min),
        "--max",
        str(config.cv_max),
        "--deltaFat",
        str(config.delta_f_at),
        "--sigma",
        str(config.sigma),
        "--skiprows",
        str(config.skiprows),
        "--skipfoot",
        str(row["skipfoot"]),
        "--blocks",
        str(config.blocks),
        "--temp",
        str(config.temperature),
    ]
    return " ".join(_shell_quote(part) for part in parts)


def prepare_fes_cumulative_reweight_manifest(
    specs: Sequence[FesCumulativeReweightSpec],
    output_manifest: Path,
    *,
    config: FesCumulativeReweightConfig,
    create_dirs: bool = False,
) -> List[Dict[str, object]]:
    """Prepare a command table for cumulative-prefix FES reweighting."""

    fractions = normalize_reweight_fractions(config.fractions)
    rows: List[Dict[str, object]] = []
    for spec in specs:
        out_dir = spec.output_dir or (config.output_root / safe_file_label(spec.system))
        if create_dirs:
            out_dir.mkdir(parents=True, exist_ok=True)
        for fraction in fractions:
            keep_after_skip = int(round(float(spec.sample_size) * fraction))
            skipfoot = max(0, int(spec.sample_size) - keep_after_skip)
            percent = int(round(fraction * 100.0))
            output_path = out_dir / f"{config.output_prefix}_{percent:03d}.dat"
            row: Dict[str, object] = {
                "system": spec.system,
                "group": spec.group,
                "workdir": str(spec.workdir),
                "colvar": str(spec.colvar),
                "sample_size": int(spec.sample_size),
                "trajectory_fraction": fraction,
                "keep_after_skip": keep_after_skip,
                "skipfoot": skipfoot,
                "output_dir": str(out_dir),
                "output_path": str(output_path),
                "driver": str(config.driver),
                "cv": config.cv,
                "cv_min": config.cv_min,
                "cv_max": config.cv_max,
                "delta_f_at": config.delta_f_at,
                "sigma": config.sigma,
                "skiprows": config.skiprows,
                "blocks": config.blocks,
                "temperature": config.temperature,
                "workdir_exists": int(spec.workdir.exists()),
                "colvar_exists": int(spec.colvar.exists()),
                "driver_exists": int(config.driver.exists()),
                "output_exists": int(output_path.exists()),
            }
            row["reweight_command"] = build_cumulative_reweight_command(row, config)
            rows.append(row)
    fieldnames = [
        "system",
        "group",
        "workdir",
        "colvar",
        "sample_size",
        "trajectory_fraction",
        "keep_after_skip",
        "skipfoot",
        "output_dir",
        "output_path",
        "driver",
        "cv",
        "cv_min",
        "cv_max",
        "delta_f_at",
        "sigma",
        "skiprows",
        "blocks",
        "temperature",
        "workdir_exists",
        "colvar_exists",
        "driver_exists",
        "output_exists",
        "reweight_command",
    ]
    _write_csv_with_fields(Path(output_manifest), rows, fieldnames)
    return rows


def load_fes_curve(
    spec: FesCurveSpec,
    cv_column: int = 0,
    free_energy_column: int = 1,
    uncertainty_column: Optional[int] = 2,
) -> FesCurve:
    """Load a whitespace FES file with comments ignored."""

    if not spec.path.exists():
        raise FileNotFoundError(spec.path)
    data = np.loadtxt(spec.path, comments="#", ndmin=2)
    if data.ndim != 2 or data.shape[1] <= max(cv_column, free_energy_column):
        raise ValueError(f"FES file does not contain requested columns: {spec.path}")
    cv = np.asarray(data[:, cv_column], dtype=float)
    free_energy = np.asarray(data[:, free_energy_column], dtype=float)
    if uncertainty_column is not None and data.shape[1] > uncertainty_column:
        uncertainty = np.asarray(data[:, uncertainty_column], dtype=float)
    else:
        uncertainty = np.full_like(cv, np.nan, dtype=float)
    return FesCurve(spec=spec, cv=cv, free_energy=free_energy, uncertainty=uncertainty)


def parse_fes_header(path: Path) -> Tuple[Tuple[str, ...], Dict[str, str]]:
    """Parse PLUMED FES header fields and `#! SET` metadata."""

    fields: Tuple[str, ...] = ()
    metadata: Dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("#!"):
                break
            content = line[2:].strip()
            if not content:
                continue
            parts = content.split(None, 2)
            if parts[0] == "FIELDS":
                fields = tuple(content.split()[1:])
            elif parts[0] == "SET" and len(parts) >= 3:
                metadata[parts[1]] = parts[2]
    return fields, metadata


def infer_fes2d_column_names(fields: Sequence[str], n_cols: int) -> Tuple[str, str, str, Optional[str]]:
    """Infer x/y/free-energy/uncertainty column names from a FES header."""

    if len(fields) >= 3:
        return fields[0], fields[1], fields[2], fields[3] if len(fields) >= 4 else None
    if n_cols < 3:
        raise ValueError("Expected at least three numeric columns: x, y, free_energy")
    return "x", "y", "free_energy", "uncertainty" if n_cols >= 4 else None


def load_fes2d_grid(path: Path) -> Fes2DGrid:
    """Load a complete regular 2D FES grid from a whitespace table."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    fields, metadata = parse_fes_header(path)
    data = np.loadtxt(path, comments="#", ndmin=2)
    if data.shape[1] < 3:
        raise ValueError(f"Expected at least three numeric columns in {path}")

    x_name, y_name, z_name, uncertainty_name = infer_fes2d_column_names(fields, data.shape[1])
    x_raw = np.asarray(data[:, 0], dtype=float)
    y_raw = np.asarray(data[:, 1], dtype=float)
    z_raw = np.asarray(data[:, 2], dtype=float)
    uncertainty_raw = np.asarray(data[:, 3], dtype=float) if data.shape[1] >= 4 else None

    x_values = np.unique(x_raw)
    y_values = np.unique(y_raw)
    nx = int(x_values.size)
    ny = int(y_values.size)
    if nx * ny != data.shape[0]:
        raise ValueError(
            f"Input is not a complete regular grid: rows={data.shape[0]}, "
            f"unique_x={nx}, unique_y={ny}"
        )

    x_index = {float(value): index for index, value in enumerate(x_values)}
    y_index = {float(value): index for index, value in enumerate(y_values)}
    free_energy = np.full((ny, nx), np.nan, dtype=float)
    uncertainty = np.full((ny, nx), np.nan, dtype=float) if uncertainty_raw is not None else None

    for row_index in range(data.shape[0]):
        ix = x_index[float(x_raw[row_index])]
        iy = y_index[float(y_raw[row_index])]
        free_energy[iy, ix] = z_raw[row_index]
        if uncertainty is not None and uncertainty_raw is not None:
            uncertainty[iy, ix] = uncertainty_raw[row_index]

    return Fes2DGrid(
        source_path=path,
        fields=fields,
        metadata=metadata,
        x_name=x_name,
        y_name=y_name,
        z_name=z_name,
        uncertainty_name=uncertainty_name,
        x_values=x_values,
        y_values=y_values,
        free_energy=free_energy,
        uncertainty=uncertainty,
    )


def clean_series(values: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Return the input series and its finite-value mask without imputation."""

    arr = np.asarray(values, dtype=float).copy()
    finite = np.isfinite(arr)
    return arr, finite


def window_mask(cv: np.ndarray, low: float, high: float) -> np.ndarray:
    """Return a mask selecting `low <= cv < high`, with infinite bounds allowed."""

    mask = np.ones_like(cv, dtype=bool)
    if not math.isinf(low):
        mask &= cv >= float(low)
    if not math.isinf(high):
        mask &= cv < float(high)
    return mask


def min_in_window(cv: np.ndarray, values: np.ndarray, low: float, high: float) -> float:
    """Return finite minimum in a CV window, falling back to global finite min."""

    mask = window_mask(cv, low, high) & np.isfinite(values)
    if np.any(mask):
        return float(np.min(values[mask]))
    finite = values[np.isfinite(values)]
    return float(np.min(finite)) if finite.size else 0.0


def shift_to_reference_window(
    cv: np.ndarray,
    free_energy: np.ndarray,
    reference_low: float = -math.inf,
    reference_high: float = math.inf,
) -> Tuple[np.ndarray, float]:
    """Shift a FES curve by the minimum in a reference CV window."""

    clean, _ = clean_series(free_energy)
    shift = min_in_window(np.asarray(cv, dtype=float), clean, reference_low, reference_high)
    return clean - shift, shift


def effective_window_length(n_points: int, requested_window: int) -> Optional[int]:
    """Return an odd smoothing window length or None for too-short arrays."""

    if requested_window <= 1 or n_points < 3:
        return None
    window = min(int(requested_window), int(n_points))
    if window % 2 == 0:
        window -= 1
    return window if window >= 3 else None


def moving_average_smooth(values: Sequence[float], window_length: int = 1, passes: int = 1) -> np.ndarray:
    """Smooth finite contiguous segments with an edge-padded moving average.

    This is intentionally dependency-light.  It is not a direct Savitzky-Golay
    replacement.  Non-finite values remain non-finite, and smoothing never
    crosses a missing-data boundary.
    """

    clean, finite = clean_series(values)
    out = clean.copy()
    start = 0
    while start < clean.size:
        if not finite[start]:
            start += 1
            continue
        stop = start + 1
        while stop < clean.size and finite[stop]:
            stop += 1
        segment = clean[start:stop]
        window = effective_window_length(segment.size, window_length)
        if window is not None:
            kernel = np.ones(window, dtype=float) / float(window)
            pad = window // 2
            smoothed = segment.copy()
            for _ in range(max(1, int(passes))):
                padded = np.pad(smoothed, pad_width=pad, mode="edge")
                smoothed = np.convolve(padded, kernel, mode="valid")
            out[start:stop] = smoothed
        start = stop
    return out


def coordinate_mask(values: np.ndarray, bounds: Optional[Tuple[float, float]]) -> np.ndarray:
    """Return an inclusive coordinate mask for optional low/high bounds."""

    arr = np.asarray(values, dtype=float)
    mask = np.ones(arr.shape, dtype=bool)
    if bounds is None:
        return mask
    low, high = float(bounds[0]), float(bounds[1])
    if high < low:
        raise ValueError(f"Invalid coordinate range: high < low ({low}, {high})")
    mask &= arr >= low
    mask &= arr <= high
    return mask


def finite_grid_min(values: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Return the finite minimum from a 2D grid, optionally restricted by mask."""

    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if mask is not None:
        finite &= np.asarray(mask, dtype=bool)
    if not np.any(finite):
        raise ValueError("No finite FES values found for zeroing")
    return float(np.min(arr[finite]))


def gaussian_kernel1d(sigma: float, truncate: float = 4.0) -> np.ndarray:
    """Build a normalized 1D Gaussian kernel in grid-bin units."""

    sigma = float(sigma)
    if sigma <= 0:
        return np.array([1.0], dtype=float)
    radius = max(1, int(float(truncate) * sigma + 0.5))
    points = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (points / sigma) ** 2)
    return kernel / float(np.sum(kernel))


def _convolve_along_axis(values: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    radius = int(kernel.size // 2)
    if radius == 0:
        return np.asarray(values, dtype=float).copy()
    pad_width = [(0, 0)] * values.ndim
    pad_width[int(axis)] = (radius, radius)
    padded = np.pad(values, pad_width=pad_width, mode="constant", constant_values=0.0)
    return np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), int(axis), padded)


def normalized_gaussian_smooth_2d(
    values: np.ndarray,
    sigma: float,
    valid_threshold: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray]:
    """Smooth a 2D FES grid while normalizing by finite-value support."""

    arr = np.asarray(values, dtype=float)
    valid = np.isfinite(arr)
    if float(sigma) <= 0:
        return arr.copy(), valid.astype(float)

    kernel = gaussian_kernel1d(float(sigma))
    weights = valid.astype(float)
    filled = np.where(valid, arr, 0.0)
    numerator = _convolve_along_axis(_convolve_along_axis(filled, kernel, axis=1), kernel, axis=0)
    denominator = _convolve_along_axis(_convolve_along_axis(weights, kernel, axis=1), kernel, axis=0)

    out = np.full_like(arr, np.nan, dtype=float)
    keep = denominator >= float(valid_threshold)
    out[keep] = numerator[keep] / denominator[keep]
    return out, denominator


def plot_values_for_fes2d(values: np.ndarray, max_fes: float, missing_plot_value: str = "max") -> np.ndarray:
    """Clip finite FES values to plotting range and handle missing points."""

    if missing_plot_value not in {"max", "nan"}:
        raise ValueError("missing_plot_value must be 'max' or 'nan'")
    out = np.where(np.isfinite(values), np.clip(values, 0.0, float(max_fes)), np.nan)
    if missing_plot_value == "max":
        out = np.where(np.isfinite(out), out, float(max_fes))
    return out


def percentile_summary(values: np.ndarray) -> Dict[str, Optional[float]]:
    """Return a compact percentile summary for finite values."""

    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    quantiles = (0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100)
    if finite.size == 0:
        return {str(q): None for q in quantiles}
    return {str(q): float(np.percentile(finite, q)) for q in quantiles}


def write_fes2d_grid_csv(
    path: Path,
    x_values: np.ndarray,
    y_values: np.ndarray,
    raw_zeroed: np.ndarray,
    raw_plot: np.ndarray,
    smooth_zeroed: np.ndarray,
    smooth_plot: np.ndarray,
    smooth_support: np.ndarray,
    x_name: str = "x",
    y_name: str = "y",
) -> None:
    """Write a long-form 2D FES grid table for plotting or downstream analysis."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                x_name,
                y_name,
                "free_energy_raw_zeroed_kj_mol",
                "free_energy_raw_plot_kj_mol",
                "free_energy_smooth_zeroed_kj_mol",
                "free_energy_smooth_plot_kj_mol",
                "smooth_support_weight",
                "raw_finite",
                "smooth_finite",
            ]
        )
        for iy, y_value in enumerate(y_values):
            for ix, x_value in enumerate(x_values):
                raw_value = raw_zeroed[iy, ix]
                raw_plot_value = raw_plot[iy, ix]
                smooth_value = smooth_zeroed[iy, ix]
                smooth_plot_value = smooth_plot[iy, ix]
                support_value = smooth_support[iy, ix]
                writer.writerow(
                    [
                        f"{x_value:.10g}",
                        f"{y_value:.10g}",
                        f"{raw_value:.10g}" if np.isfinite(raw_value) else "",
                        f"{raw_plot_value:.10g}" if np.isfinite(raw_plot_value) else "",
                        f"{smooth_value:.10g}" if np.isfinite(smooth_value) else "",
                        f"{smooth_plot_value:.10g}" if np.isfinite(smooth_plot_value) else "",
                        f"{support_value:.10g}" if np.isfinite(support_value) else "",
                        int(np.isfinite(raw_value)),
                        int(np.isfinite(smooth_value)),
                    ]
                )


def _require_matplotlib():
    try:  # pragma: no cover - plotting is smoke-tested only when matplotlib exists.
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for --write-plots") from exc
    return plt


def plot_fes2d_panel(
    output_path: Path,
    x_values: np.ndarray,
    y_values: np.ndarray,
    values: np.ndarray,
    max_fes: float = 200.0,
    contour_levels: int = 16,
    dpi: int = 300,
    title: str = "",
    x_label: str = "x",
    y_label: str = "y",
    cmap: str = "viridis",
) -> None:
    """Write one contour plot from a processed 2D FES grid."""

    plt = _require_matplotlib()
    levels = np.linspace(0.0, float(max_fes), int(contour_levels) + 1)
    fig, ax = plt.subplots(figsize=(7.5, 6.0), dpi=int(dpi), constrained_layout=True)
    filled = ax.contourf(x_values, y_values, values, levels=levels, cmap=cmap, extend="max")
    ax.contour(x_values, y_values, values, levels=levels, colors="k", linewidths=0.35)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    cbar = fig.colorbar(filled, ax=ax)
    cbar.set_label("Delta F (kJ/mol)")
    fig.savefig(output_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def plot_fes2d_comparison(
    output_path: Path,
    x_values: np.ndarray,
    y_values: np.ndarray,
    raw_plot: np.ndarray,
    smooth_plot: np.ndarray,
    max_fes: float = 200.0,
    contour_levels: int = 16,
    dpi: int = 300,
    x_label: str = "x",
    y_label: str = "y",
    cmap: str = "viridis",
) -> None:
    """Write a raw-vs-smoothed 2D FES comparison figure."""

    plt = _require_matplotlib()
    levels = np.linspace(0.0, float(max_fes), int(contour_levels) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), dpi=int(dpi), constrained_layout=True)
    mappable = None
    for ax, values, title in zip(axes, [raw_plot, smooth_plot], ["Raw clipped", "Smoothed"]):
        mappable = ax.contourf(x_values, y_values, values, levels=levels, cmap=cmap, extend="max")
        ax.contour(x_values, y_values, values, levels=levels, colors="k", linewidths=0.3)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
    if mappable is not None:
        cbar = fig.colorbar(mappable, ax=axes, orientation="horizontal", shrink=0.85, pad=0.08)
        cbar.set_label("Delta F (kJ/mol)")
    fig.savefig(output_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def _metadata_range(bounds: Optional[Tuple[float, float]]) -> Optional[List[float]]:
    if bounds is None:
        return None
    return [float(bounds[0]), float(bounds[1])]


def process_fes2d_grid(
    fes_file: Path,
    output_dir: Path,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    max_fes: float = 200.0,
    smooth_sigma: float = 0.8,
    valid_threshold: float = 0.25,
    prefix: str = "fes2d",
    zero_scope: str = "window",
    missing_plot_value: str = "max",
    write_grid: bool = True,
    write_plots: bool = False,
    write_comparison: bool = False,
    contour_levels: int = 16,
    dpi: int = 300,
    title: str = "",
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    cmap: str = "viridis",
) -> Dict[str, Path]:
    """Process a 2D FES grid and write path-explicit outputs."""

    if zero_scope not in {"window", "all"}:
        raise ValueError("zero_scope must be 'window' or 'all'")
    grid = load_fes2d_grid(Path(fes_file))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_mask = coordinate_mask(grid.x_values, x_range)
    y_mask = coordinate_mask(grid.y_values, y_range)
    if not np.any(x_mask) or not np.any(y_mask):
        raise ValueError("The requested x/y range selects no FES grid points")
    selected_mask = np.outer(y_mask, x_mask)

    raw = np.asarray(grid.free_energy, dtype=float).copy()
    raw[~np.isfinite(raw)] = np.nan
    zero_mask = selected_mask if zero_scope == "window" else None
    raw_zero_value = finite_grid_min(raw, zero_mask)
    raw_zeroed = raw - raw_zero_value

    raw_for_smoothing = np.where(np.isfinite(raw_zeroed), np.clip(raw_zeroed, 0.0, float(max_fes)), np.nan)
    smooth, support = normalized_gaussian_smooth_2d(
        raw_for_smoothing,
        sigma=float(smooth_sigma),
        valid_threshold=float(valid_threshold),
    )
    try:
        smooth_zero_value = finite_grid_min(smooth, zero_mask)
    except ValueError:
        smooth_zero_value = finite_grid_min(smooth)
    smooth_zeroed = smooth - smooth_zero_value

    raw_plot = plot_values_for_fes2d(raw_zeroed, max_fes=max_fes, missing_plot_value=missing_plot_value)
    smooth_plot = plot_values_for_fes2d(smooth_zeroed, max_fes=max_fes, missing_plot_value=missing_plot_value)

    x_crop = grid.x_values[x_mask]
    y_crop = grid.y_values[y_mask]
    raw_zeroed_crop = raw_zeroed[np.ix_(y_mask, x_mask)]
    raw_plot_crop = raw_plot[np.ix_(y_mask, x_mask)]
    smooth_zeroed_crop = smooth_zeroed[np.ix_(y_mask, x_mask)]
    smooth_plot_crop = smooth_plot[np.ix_(y_mask, x_mask)]
    support_crop = support[np.ix_(y_mask, x_mask)]

    outputs: Dict[str, Path] = {}
    if write_grid:
        grid_csv = output_dir / f"{prefix}_plot_grid.csv"
        write_fes2d_grid_csv(
            grid_csv,
            x_crop,
            y_crop,
            raw_zeroed_crop,
            raw_plot_crop,
            smooth_zeroed_crop,
            smooth_plot_crop,
            support_crop,
            x_name=grid.x_name,
            y_name=grid.y_name,
        )
        outputs["plot_grid_csv"] = grid_csv

    figure_x_label = x_label or grid.x_name
    figure_y_label = y_label or grid.y_name
    if write_plots or write_comparison:
        smooth_png = output_dir / f"{prefix}_smooth.png"
        plot_fes2d_panel(
            smooth_png,
            x_crop,
            y_crop,
            smooth_plot_crop,
            max_fes=max_fes,
            contour_levels=contour_levels,
            dpi=dpi,
            title=title,
            x_label=figure_x_label,
            y_label=figure_y_label,
            cmap=cmap,
        )
        outputs["smooth_png"] = smooth_png
        if write_comparison:
            comparison_png = output_dir / f"{prefix}_raw_vs_smooth.png"
            plot_fes2d_comparison(
                comparison_png,
                x_crop,
                y_crop,
                raw_plot_crop,
                smooth_plot_crop,
                max_fes=max_fes,
                contour_levels=contour_levels,
                dpi=dpi,
                x_label=figure_x_label,
                y_label=figure_y_label,
                cmap=cmap,
            )
            outputs["raw_vs_smooth_png"] = comparison_png

    metadata = {
        "source_path": str(grid.source_path),
        "fields": list(grid.fields),
        "header_metadata": grid.metadata,
        "x_name": grid.x_name,
        "y_name": grid.y_name,
        "z_name": grid.z_name,
        "uncertainty_name": grid.uncertainty_name,
        "x_range": _metadata_range(x_range),
        "y_range": _metadata_range(y_range),
        "x_bins_total": int(grid.x_values.size),
        "y_bins_total": int(grid.y_values.size),
        "x_bins_selected": int(x_crop.size),
        "y_bins_selected": int(y_crop.size),
        "finite_points_total": int(np.count_nonzero(np.isfinite(raw))),
        "finite_points_selected": int(np.count_nonzero(np.isfinite(raw_zeroed_crop))),
        "zero_scope": zero_scope,
        "raw_zero_value_kj_mol": raw_zero_value,
        "smooth_zero_value_kj_mol": smooth_zero_value,
        "max_fes_kj_mol": float(max_fes),
        "smooth_sigma_bins": float(smooth_sigma),
        "smooth_valid_threshold": float(valid_threshold),
        "missing_plot_value": missing_plot_value,
        "raw_zeroed_selected_percentiles_kj_mol": percentile_summary(raw_zeroed_crop),
        "smooth_zeroed_selected_percentiles_kj_mol": percentile_summary(smooth_zeroed_crop),
        "outputs": {name: str(path) for name, path in outputs.items()},
    }
    metadata_path = output_dir / f"{prefix}_metadata.json"
    metadata["outputs"]["metadata_json"] = str(metadata_path)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    outputs["metadata_json"] = metadata_path
    return outputs


def read_plumed_numeric_table(
    path: Path,
    *,
    skiprows: int = 0,
    skip_last_data_line: bool = False,
    deduplicate_time: bool = True,
) -> PlumedTable:
    """Read a PLUMED-style numeric table using the first FIELDS header."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    fields: Tuple[str, ...] = ()
    data_lines: List[Tuple[int, str]] = []
    rows: List[List[float]] = []
    bad_lines: List[int] = []
    data_seen = 0
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#!"):
            parts = line.split()
            if len(parts) >= 3 and parts[1] == "FIELDS" and not fields:
                fields = tuple(parts[2:])
            continue
        if line.startswith("#"):
            continue
        if not fields:
            raise ValueError(f"No FIELDS header found before data in {path}")
        data_seen += 1
        if data_seen <= int(skiprows):
            continue
        data_lines.append((line_number, line))
    if not fields:
        raise ValueError(f"No FIELDS header found in {path}")
    if skip_last_data_line and data_lines:
        data_lines = data_lines[:-1]
    for line_number, line in data_lines:
        parts = line.split()
        if len(parts) != len(fields):
            bad_lines.append(line_number)
            continue
        try:
            row = [float(value) for value in parts]
        except ValueError:
            bad_lines.append(line_number)
            continue
        if not all(math.isfinite(value) for value in row):
            bad_lines.append(line_number)
            continue
        rows.append(row)
    if bad_lines:
        raise ValueError(
            f"Skipped malformed/non-finite data lines in {path}: "
            + ",".join(str(value) for value in bad_lines[:8])
        )
    if not rows:
        raise ValueError(f"No numeric data rows left in {path} after skiprows={skiprows}")
    data = np.asarray(rows, dtype=float)
    if deduplicate_time and "time" in fields:
        time_index = fields.index("time")
        _, first_indices = np.unique(data[:, time_index], return_index=True)
        first_indices.sort()
        data = data[first_indices]
    return PlumedTable(path=path, fields=fields, data=data)


def read_plumed_numeric_tables(
    paths: Sequence[Path],
    *,
    skiprows: int = 0,
    skip_last_data_line_per_file: bool = False,
    deduplicate_time: bool = True,
) -> PlumedTable:
    """Read and concatenate multiple PLUMED tables with matching fields."""

    resolved_paths = tuple(Path(path) for path in paths)
    if not resolved_paths:
        raise ValueError("At least one PLUMED table path is required")
    tables = [
        read_plumed_numeric_table(
            path,
            skiprows=0,
            skip_last_data_line=bool(skip_last_data_line_per_file),
            deduplicate_time=False,
        )
        for path in resolved_paths
    ]
    fields = tables[0].fields
    for table in tables[1:]:
        if table.fields != fields:
            raise ValueError(
                "Cannot concatenate PLUMED tables with different FIELDS headers: "
                f"{tables[0].path} has {fields}; {table.path} has {table.fields}"
            )
    data = np.vstack([table.data for table in tables])
    if deduplicate_time and "time" in fields:
        time_index = fields.index("time")
        _unique_times, first_indices = np.unique(data[:, time_index], return_index=True)
        first_indices.sort()
        data = data[first_indices]
    if int(skiprows) > 0:
        data = data[int(skiprows) :]
    if data.size == 0:
        raise ValueError(f"No numeric data rows left after concatenating {len(resolved_paths)} tables")
    return PlumedTable(path=resolved_paths[0], fields=fields, data=data)


def _resolve_run_file(run_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(run_dir) / path
    return path


def _field_index(fields: Sequence[str], name: str) -> int:
    try:
        return tuple(fields).index(name)
    except ValueError as exc:
        raise ValueError(f"Column not found in PLUMED table: {name}") from exc


def _select_bias_indices(fields: Sequence[str], bias_names: Sequence[str]) -> Tuple[int, ...]:
    if not bias_names:
        bias_names = (".bias",)
    if len(bias_names) == 1 and str(bias_names[0]).upper() == "NO":
        return ()
    indices: List[int] = []
    for name in bias_names:
        if name in {".bias", ".rbias"}:
            matches = [
                index
                for index, field in enumerate(fields)
                if ".bias" in field or ".rbias" in field
            ]
            indices.extend(matches)
            continue
        indices.append(_field_index(fields, str(name)))
    unique: List[int] = []
    for index in indices:
        if index not in unique:
            unique.append(index)
    return tuple(unique)


def _sum_bias_columns(table: PlumedTable, bias_names: Sequence[str]) -> Tuple[np.ndarray, Tuple[str, ...]]:
    indices = _select_bias_indices(table.fields, bias_names)
    if not indices:
        return np.zeros(table.data.shape[0], dtype=float), ()
    bias = np.zeros(table.data.shape[0], dtype=float)
    names: List[str] = []
    for index in indices:
        bias += table.data[:, index]
        names.append(table.fields[index])
    return bias, tuple(names)


def boltzmann_kbt(temperature: float) -> float:
    """Return kBT in kJ/mol for a temperature in Kelvin."""

    return float(temperature) * 0.0083144621


def reweight_weights(bias_kj_mol: np.ndarray, temperature: float) -> Tuple[np.ndarray, float]:
    """Return numerically stable OPES reweighting weights and Neff."""

    if bias_kj_mol.size == 0:
        raise ValueError("No bias values available for reweighting")
    kbt = boltzmann_kbt(float(temperature))
    scaled = np.asarray(bias_kj_mol, dtype=float) / kbt
    shifted = scaled - float(np.max(scaled))
    weights = np.exp(shifted)
    denom = float(np.sum(weights**2))
    effective = float(np.sum(weights) ** 2 / denom) if denom > 0.0 else math.nan
    return weights, effective


def parse_named_float_map(values: Optional[Sequence[Sequence[str]]]) -> Dict[str, float]:
    """Parse repeated `NAME VALUE` pairs from argparse."""

    parsed: Dict[str, float] = {}
    for item in values or []:
        if len(item) != 2:
            raise ValueError("Expected NAME VALUE")
        parsed[str(item[0])] = float(item[1])
    return parsed


def parse_named_ranges(values: Optional[Sequence[Sequence[str]]]) -> Dict[str, Tuple[float, float]]:
    """Parse repeated `CV LOW HIGH` ranges from argparse."""

    parsed: Dict[str, Tuple[float, float]] = {}
    for item in values or []:
        if len(item) != 3:
            raise ValueError("Expected CV LOW HIGH")
        low = float(item[1])
        high = float(item[2])
        if high <= low:
            raise ValueError(f"Invalid range for {item[0]}: high <= low")
        parsed[str(item[0])] = (low, high)
    return parsed


def infer_table_ranges(
    table: PlumedTable,
    cv_names: Sequence[str],
    *,
    explicit_ranges: Optional[Mapping[str, Tuple[float, float]]] = None,
    padding_fraction: float = 0.05,
) -> Dict[str, Tuple[float, float]]:
    """Infer plotting/reweight ranges for CVs not specified explicitly."""

    explicit_ranges = explicit_ranges or {}
    ranges: Dict[str, Tuple[float, float]] = {}
    for cv_name in cv_names:
        if cv_name in explicit_ranges:
            ranges[cv_name] = explicit_ranges[cv_name]
            continue
        values = table.data[:, _field_index(table.fields, cv_name)]
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            raise ValueError(f"No finite values for CV {cv_name}")
        low = float(np.min(finite))
        high = float(np.max(finite))
        width = high - low
        if width <= 0.0:
            pad = max(1.0, abs(low) * 0.01)
        else:
            pad = width * float(padding_fraction)
        ranges[cv_name] = (low - pad, high + pad)
    return ranges


def infer_hills_bandwidths(path: Optional[Path]) -> Dict[str, float]:
    """Infer last finite OPES sigma values from a HILLS file."""

    if path is None or not Path(path).exists():
        return {}
    try:
        table = read_plumed_numeric_table(path, deduplicate_time=False)
    except Exception:
        return {}
    bandwidths: Dict[str, float] = {}
    for index, field in enumerate(table.fields):
        if not field.startswith("sigma_"):
            continue
        cv_name = field[len("sigma_") :]
        values = table.data[:, index]
        finite = values[np.isfinite(values) & (values > 0.0)]
        if finite.size:
            bandwidths[cv_name] = float(finite[-1])
    return bandwidths


def infer_hills_bandwidths_from_paths(paths: Sequence[Path]) -> Dict[str, float]:
    """Infer sigma values from one or more HILLS files, with later files winning."""

    merged: Dict[str, float] = {}
    for path in paths:
        merged.update(infer_hills_bandwidths(Path(path)))
    return merged


def choose_bandwidth(
    cv_name: str,
    *,
    ranges: Mapping[str, Tuple[float, float]],
    bins: int,
    explicit_bandwidths: Mapping[str, float],
    hills_bandwidths: Mapping[str, float],
    default_smooth_bins: float,
) -> BandwidthChoice:
    """Choose histogram-smoothing bandwidth for one CV."""

    value: Optional[float] = None
    source = "default_bins"
    if cv_name in explicit_bandwidths:
        value = float(explicit_bandwidths[cv_name])
        source = "explicit"
    elif cv_name in hills_bandwidths:
        value = float(hills_bandwidths[cv_name])
        source = "hills"
    low, high = ranges[cv_name]
    width = (float(high) - float(low)) / max(1, int(bins))
    if value is not None and width > 0.0:
        sigma_bins = max(0.0, float(value) / width)
    else:
        sigma_bins = max(0.0, float(default_smooth_bins))
    return BandwidthChoice(cv_name=cv_name, value=value, sigma_bins=sigma_bins, source=source)


def _smooth_hist1d(counts: np.ndarray, sigma_bins: float) -> np.ndarray:
    kernel = gaussian_kernel1d(float(sigma_bins))
    return np.convolve(np.asarray(counts, dtype=float), kernel, mode="same")


def _smooth_hist2d(counts: np.ndarray, sigma_bins_x: float, sigma_bins_y: float) -> np.ndarray:
    out = np.asarray(counts, dtype=float)
    if float(sigma_bins_x) > 0.0:
        out = _convolve_along_axis(out, gaussian_kernel1d(float(sigma_bins_x)), axis=0)
    if float(sigma_bins_y) > 0.0:
        out = _convolve_along_axis(out, gaussian_kernel1d(float(sigma_bins_y)), axis=1)
    return out


def _fes_from_probability(probability: np.ndarray, temperature: float) -> np.ndarray:
    kbt = boltzmann_kbt(float(temperature))
    prob = np.asarray(probability, dtype=float)
    fes = np.full(prob.shape, np.nan, dtype=float)
    positive = np.isfinite(prob) & (prob > 0.0)
    fes[positive] = -kbt * np.log(prob[positive])
    if np.any(np.isfinite(fes)):
        fes -= float(np.nanmin(fes))
    return fes


def _uncertainty_from_blocks(block_fes: Sequence[np.ndarray], shape: Tuple[int, ...]) -> np.ndarray:
    if len(block_fes) <= 1:
        return np.full(shape, np.nan, dtype=float)
    stack = np.asarray(block_fes, dtype=float)
    flat = stack.reshape((stack.shape[0], -1))
    uncertainty = np.full(flat.shape[1], np.nan, dtype=float)
    for index in range(flat.shape[1]):
        values = flat[:, index]
        finite = values[np.isfinite(values)]
        if finite.size > 1:
            uncertainty[index] = float(np.std(finite, ddof=1))
    return uncertainty.reshape(shape)


def _format_float(value: float) -> str:
    if not np.isfinite(float(value)):
        return "nan"
    return f"{float(value): .10g}"


def write_fes1d_file(
    path: Path,
    cv_name: str,
    centers: np.ndarray,
    free_energy: np.ndarray,
    uncertainty: Optional[np.ndarray],
    *,
    sample_size: int,
    effective_sample_size: float,
    cv_range: Tuple[float, float],
    bandwidth: BandwidthChoice,
    temperature: float,
    bias_columns: Sequence[str],
) -> None:
    """Write a PLUMED-style 1D reweighted FES table."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    has_uncertainty = uncertainty is not None
    with path.open("w", encoding="utf-8") as handle:
        fields = f"#! FIELDS {cv_name} file.free"
        if has_uncertainty:
            fields += " uncertainty"
        handle.write(fields + "\n")
        handle.write(f"#! SET sample_size {int(sample_size)}\n")
        handle.write(f"#! SET effective_sample_size {_format_float(effective_sample_size)}\n")
        handle.write(f"#! SET temperature_K {_format_float(float(temperature))}\n")
        handle.write(f"#! SET bias_columns {','.join(bias_columns) if bias_columns else 'NO'}\n")
        handle.write(f"#! SET min_{cv_name} {_format_float(cv_range[0])}\n")
        handle.write(f"#! SET max_{cv_name} {_format_float(cv_range[1])}\n")
        handle.write(f"#! SET nbins_{cv_name} {int(centers.size)}\n")
        handle.write(f"#! SET bandwidth_{cv_name} {_format_float(bandwidth.value) if bandwidth.value is not None else 'nan'}\n")
        handle.write(f"#! SET bandwidth_source_{cv_name} {bandwidth.source}\n")
        handle.write(f"#! SET bandwidth_bins_{cv_name} {_format_float(bandwidth.sigma_bins)}\n")
        for index, (cv_value, fes_value) in enumerate(zip(centers, free_energy)):
            line = f"{_format_float(cv_value)}  {_format_float(fes_value)}"
            if has_uncertainty and uncertainty is not None:
                line += f" {_format_float(float(uncertainty[index]))}"
            handle.write(line + "\n")


def write_fes2d_file(
    path: Path,
    x_name: str,
    y_name: str,
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    free_energy: np.ndarray,
    uncertainty: Optional[np.ndarray],
    *,
    sample_size: int,
    effective_sample_size: float,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    x_bandwidth: BandwidthChoice,
    y_bandwidth: BandwidthChoice,
    temperature: float,
    bias_columns: Sequence[str],
) -> None:
    """Write a PLUMED-style 2D reweighted FES table."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    has_uncertainty = uncertainty is not None
    with path.open("w", encoding="utf-8") as handle:
        fields = f"#! FIELDS {x_name} {y_name} file.free"
        if has_uncertainty:
            fields += " uncertainty"
        handle.write(fields + "\n")
        handle.write(f"#! SET sample_size {int(sample_size)}\n")
        handle.write(f"#! SET effective_sample_size {_format_float(effective_sample_size)}\n")
        handle.write(f"#! SET temperature_K {_format_float(float(temperature))}\n")
        handle.write(f"#! SET bias_columns {','.join(bias_columns) if bias_columns else 'NO'}\n")
        handle.write(f"#! SET min_{x_name} {_format_float(x_range[0])}\n")
        handle.write(f"#! SET max_{x_name} {_format_float(x_range[1])}\n")
        handle.write(f"#! SET nbins_{x_name} {int(x_centers.size)}\n")
        handle.write(f"#! SET min_{y_name} {_format_float(y_range[0])}\n")
        handle.write(f"#! SET max_{y_name} {_format_float(y_range[1])}\n")
        handle.write(f"#! SET nbins_{y_name} {int(y_centers.size)}\n")
        handle.write(f"#! SET bandwidth_{x_name} {_format_float(x_bandwidth.value) if x_bandwidth.value is not None else 'nan'}\n")
        handle.write(f"#! SET bandwidth_source_{x_name} {x_bandwidth.source}\n")
        handle.write(f"#! SET bandwidth_bins_{x_name} {_format_float(x_bandwidth.sigma_bins)}\n")
        handle.write(f"#! SET bandwidth_{y_name} {_format_float(y_bandwidth.value) if y_bandwidth.value is not None else 'nan'}\n")
        handle.write(f"#! SET bandwidth_source_{y_name} {y_bandwidth.source}\n")
        handle.write(f"#! SET bandwidth_bins_{y_name} {_format_float(y_bandwidth.sigma_bins)}\n")
        for ix, x_value in enumerate(x_centers):
            for iy, y_value in enumerate(y_centers):
                line = (
                    f"{_format_float(float(x_value))} "
                    f"{_format_float(float(y_value))}  "
                    f"{_format_float(float(free_energy[ix, iy]))}"
                )
                if has_uncertainty and uncertainty is not None:
                    line += f" {_format_float(float(uncertainty[ix, iy]))}"
                handle.write(line + "\n")
            handle.write("\n")


def _histogram_fes_1d(
    values: np.ndarray,
    weights: np.ndarray,
    cv_range: Tuple[float, float],
    bins: int,
    sigma_bins: float,
    temperature: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.linspace(float(cv_range[0]), float(cv_range[1]), int(bins) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    counts, _edges = np.histogram(values, bins=edges, weights=weights)
    smooth_counts = _smooth_hist1d(counts, sigma_bins=sigma_bins)
    return centers, _fes_from_probability(smooth_counts, temperature=temperature), smooth_counts


def _histogram_fes_2d(
    x_values: np.ndarray,
    y_values: np.ndarray,
    weights: np.ndarray,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    bins: Tuple[int, int],
    sigma_bins: Tuple[float, float],
    temperature: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_edges = np.linspace(float(x_range[0]), float(x_range[1]), int(bins[0]) + 1)
    y_edges = np.linspace(float(y_range[0]), float(y_range[1]), int(bins[1]) + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    counts, _x_edges, _y_edges = np.histogram2d(
        x_values,
        y_values,
        bins=[x_edges, y_edges],
        weights=weights,
    )
    smooth_counts = _smooth_hist2d(
        counts,
        sigma_bins_x=float(sigma_bins[0]),
        sigma_bins_y=float(sigma_bins[1]),
    )
    return x_centers, y_centers, _fes_from_probability(smooth_counts, temperature=temperature), smooth_counts


def _split_block_indices(n_rows: int, blocks: int) -> List[np.ndarray]:
    blocks = max(1, int(blocks))
    if blocks <= 1 or n_rows < blocks:
        return []
    return [part for part in np.array_split(np.arange(n_rows), blocks) if part.size > 0]


def _write_reweight_summary(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = (
        "kind",
        "cvs",
        "safe_label",
        "fes_path",
        "plot_path",
        "sample_size",
        "effective_sample_size",
        "finite_points",
        "bandwidth_sources",
    )
    _write_csv_with_fields(path, rows, fields)


def plot_fes1d_curve(
    output_path: Path,
    cv_name: str,
    centers: np.ndarray,
    free_energy: np.ndarray,
    uncertainty: Optional[np.ndarray],
    *,
    max_fes: float = 200.0,
    dpi: int = 180,
) -> None:
    """Write one 1D FES PNG."""

    plt = _require_matplotlib()
    y_values = np.asarray(free_energy, dtype=float)
    y_plot = np.where(np.isfinite(y_values), np.clip(y_values, 0.0, float(max_fes)), np.nan)
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=int(dpi), constrained_layout=True)
    ax.plot(centers, y_plot, lw=1.8)
    if uncertainty is not None:
        unc = np.asarray(uncertainty, dtype=float)
        valid = np.isfinite(y_plot) & np.isfinite(unc)
        if np.any(valid):
            ax.fill_between(
                np.asarray(centers)[valid],
                np.clip(y_plot[valid] - unc[valid], 0.0, float(max_fes)),
                np.clip(y_plot[valid] + unc[valid], 0.0, float(max_fes)),
                alpha=0.22,
                linewidth=0,
            )
    ax.set_xlabel(cv_name)
    ax.set_ylabel("Delta F (kJ/mol)")
    ax.set_ylim(bottom=0.0)
    fig.savefig(output_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def reweight_plumed_colvar(
    run_dir: Path,
    *,
    colvar_name: str = "COLVAR",
    hills_name: str = "HILLS",
    colvar_names: Optional[Sequence[str]] = None,
    hills_names: Optional[Sequence[str]] = None,
    output_root: Optional[Path] = None,
    cv_names: Sequence[str] = (),
    pairs: Sequence[Tuple[str, str]] = (),
    bias_names: Sequence[str] = (".bias",),
    ranges: Optional[Mapping[str, Tuple[float, float]]] = None,
    bandwidths: Optional[Mapping[str, float]] = None,
    temperature: float = 330.0,
    skiprows: int = 0,
    blocks: int = 3,
    bins: int = 160,
    pair_bins: Tuple[int, int] = (90, 90),
    default_smooth_bins: float = 1.5,
    range_padding_fraction: float = 0.05,
    skip_last_data_line: bool = False,
    skip_last_data_line_per_file: bool = False,
    deduplicate_time: bool = True,
    max_fes: float = 200.0,
    write_plots: bool = False,
    write_comparison: bool = False,
    contour_levels: int = 16,
    dpi: int = 180,
    cmap: str = "viridis",
) -> Dict[str, Path]:
    """Run OPES-style reweighting for 1D CVs and 2D CV pairs from COLVAR."""

    run_dir = Path(run_dir)
    output_root = Path(output_root) if output_root is not None else run_dir
    raw_colvar_names = tuple(colvar_names) if colvar_names is not None else (colvar_name,)
    raw_hills_names = tuple(hills_names) if hills_names is not None else (hills_name,)
    colvar_paths = tuple(_resolve_run_file(run_dir, name) for name in raw_colvar_names)
    hills_paths = tuple(_resolve_run_file(run_dir, name) for name in raw_hills_names)
    table = read_plumed_numeric_tables(
        colvar_paths,
        skiprows=int(skiprows),
        skip_last_data_line_per_file=bool(skip_last_data_line or skip_last_data_line_per_file),
        deduplicate_time=bool(deduplicate_time),
    )
    hills_bandwidths = infer_hills_bandwidths_from_paths(hills_paths)
    bias, bias_columns = _sum_bias_columns(table, bias_names)
    weights, effective_sample_size = reweight_weights(bias, temperature=temperature)

    pair_list = [(str(x), str(y)) for x, y in pairs]
    requested_cvs: List[str] = []
    for cv_name in cv_names:
        if cv_name not in requested_cvs:
            requested_cvs.append(str(cv_name))
    for x_name, y_name in pair_list:
        for cv_name in (x_name, y_name):
            if cv_name not in requested_cvs:
                requested_cvs.append(cv_name)
    if not requested_cvs:
        requested_cvs = [
            field
            for field in table.fields
            if field != "time" and field not in bias_columns and not field.startswith("sigma_")
        ]
    for cv_name in requested_cvs:
        _field_index(table.fields, cv_name)

    explicit_ranges = dict(ranges or {})
    all_ranges = infer_table_ranges(
        table,
        requested_cvs,
        explicit_ranges=explicit_ranges,
        padding_fraction=float(range_padding_fraction),
    )
    explicit_bandwidths = dict(bandwidths or {})

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "fes_reweight").mkdir(parents=True, exist_ok=True)
    (output_root / "fes1D").mkdir(parents=True, exist_ok=True)
    (output_root / "fes2D").mkdir(parents=True, exist_ok=True)

    results: List[ReweightProjectionResult] = []
    manifest_rows: List[Dict[str, object]] = []
    block_indices = _split_block_indices(table.data.shape[0], blocks)

    for cv_name in cv_names:
        cv_name = str(cv_name)
        safe = safe_file_label(cv_name)
        values = table.data[:, _field_index(table.fields, cv_name)]
        bandwidth = choose_bandwidth(
            cv_name,
            ranges=all_ranges,
            bins=int(bins),
            explicit_bandwidths=explicit_bandwidths,
            hills_bandwidths=hills_bandwidths,
            default_smooth_bins=float(default_smooth_bins),
        )
        centers, fes, _counts = _histogram_fes_1d(
            values,
            weights,
            all_ranges[cv_name],
            bins=int(bins),
            sigma_bins=bandwidth.sigma_bins,
            temperature=float(temperature),
        )
        block_fes: List[np.ndarray] = []
        reweight_dir = output_root / "fes_reweight" / safe
        reweight_dir.mkdir(parents=True, exist_ok=True)
        for block_number, indices in enumerate(block_indices, start=1):
            _centers, block_curve, _block_counts = _histogram_fes_1d(
                values[indices],
                weights[indices],
                all_ranges[cv_name],
                bins=int(bins),
                sigma_bins=bandwidth.sigma_bins,
                temperature=float(temperature),
            )
            block_fes.append(block_curve)
            write_fes1d_file(
                reweight_dir / f"fes-rew_{block_number}.dat",
                cv_name,
                centers,
                block_curve,
                None,
                sample_size=int(indices.size),
                effective_sample_size=math.nan,
                cv_range=all_ranges[cv_name],
                bandwidth=bandwidth,
                temperature=float(temperature),
                bias_columns=bias_columns,
            )
        uncertainty = _uncertainty_from_blocks(block_fes, fes.shape)
        fes_path = reweight_dir / "fes-rew.dat"
        write_fes1d_file(
            fes_path,
            cv_name,
            centers,
            fes,
            uncertainty,
            sample_size=int(table.data.shape[0]),
            effective_sample_size=effective_sample_size,
            cv_range=all_ranges[cv_name],
            bandwidth=bandwidth,
            temperature=float(temperature),
            bias_columns=bias_columns,
        )
        plot_path: Optional[Path] = None
        if write_plots:
            plot_path = output_root / "fes1D" / f"{safe}_fes1d.png"
            plot_fes1d_curve(
                plot_path,
                cv_name,
                centers,
                fes,
                uncertainty,
                max_fes=float(max_fes),
                dpi=int(dpi),
            )
        result = ReweightProjectionResult(
            kind="1d",
            cvs=(cv_name,),
            safe_label=safe,
            fes_path=fes_path,
            plot_path=plot_path,
            sample_size=int(table.data.shape[0]),
            effective_sample_size=effective_sample_size,
            finite_points=int(np.count_nonzero(np.isfinite(fes))),
            bandwidth_sources=(f"{cv_name}:{bandwidth.source}",),
        )
        results.append(result)
        manifest_rows.append(
            {
                "kind": result.kind,
                "cvs": ",".join(result.cvs),
                "safe_label": result.safe_label,
                "fes_path": str(result.fes_path),
                "plot_path": str(result.plot_path) if result.plot_path is not None else "",
                "sample_size": result.sample_size,
                "effective_sample_size": result.effective_sample_size,
                "finite_points": result.finite_points,
                "bandwidth_sources": ";".join(result.bandwidth_sources),
            }
        )

    for x_name, y_name in pair_list:
        safe = f"{safe_file_label(x_name)}__{safe_file_label(y_name)}"
        x_values = table.data[:, _field_index(table.fields, x_name)]
        y_values = table.data[:, _field_index(table.fields, y_name)]
        x_bandwidth = choose_bandwidth(
            x_name,
            ranges=all_ranges,
            bins=int(pair_bins[0]),
            explicit_bandwidths=explicit_bandwidths,
            hills_bandwidths=hills_bandwidths,
            default_smooth_bins=float(default_smooth_bins),
        )
        y_bandwidth = choose_bandwidth(
            y_name,
            ranges=all_ranges,
            bins=int(pair_bins[1]),
            explicit_bandwidths=explicit_bandwidths,
            hills_bandwidths=hills_bandwidths,
            default_smooth_bins=float(default_smooth_bins),
        )
        x_centers, y_centers, fes2d, _counts2d = _histogram_fes_2d(
            x_values,
            y_values,
            weights,
            all_ranges[x_name],
            all_ranges[y_name],
            bins=(int(pair_bins[0]), int(pair_bins[1])),
            sigma_bins=(x_bandwidth.sigma_bins, y_bandwidth.sigma_bins),
            temperature=float(temperature),
        )
        block_fes2d: List[np.ndarray] = []
        pair_dir = output_root / "fes2D" / safe
        pair_dir.mkdir(parents=True, exist_ok=True)
        for block_number, indices in enumerate(block_indices, start=1):
            _x, _y, block_grid, _block_counts = _histogram_fes_2d(
                x_values[indices],
                y_values[indices],
                weights[indices],
                all_ranges[x_name],
                all_ranges[y_name],
                bins=(int(pair_bins[0]), int(pair_bins[1])),
                sigma_bins=(x_bandwidth.sigma_bins, y_bandwidth.sigma_bins),
                temperature=float(temperature),
            )
            block_fes2d.append(block_grid)
            write_fes2d_file(
                pair_dir / f"fes-rew_{block_number}.dat",
                x_name,
                y_name,
                x_centers,
                y_centers,
                block_grid,
                None,
                sample_size=int(indices.size),
                effective_sample_size=math.nan,
                x_range=all_ranges[x_name],
                y_range=all_ranges[y_name],
                x_bandwidth=x_bandwidth,
                y_bandwidth=y_bandwidth,
                temperature=float(temperature),
                bias_columns=bias_columns,
            )
        uncertainty2d = _uncertainty_from_blocks(block_fes2d, fes2d.shape)
        fes_path = pair_dir / "fes-rew.dat"
        write_fes2d_file(
            fes_path,
            x_name,
            y_name,
            x_centers,
            y_centers,
            fes2d,
            uncertainty2d,
            sample_size=int(table.data.shape[0]),
            effective_sample_size=effective_sample_size,
            x_range=all_ranges[x_name],
            y_range=all_ranges[y_name],
            x_bandwidth=x_bandwidth,
            y_bandwidth=y_bandwidth,
            temperature=float(temperature),
            bias_columns=bias_columns,
        )
        plot_path = None
        if write_plots or write_comparison:
            plot_outputs = process_fes2d_grid(
                fes_path,
                pair_dir / "plot",
                max_fes=float(max_fes),
                smooth_sigma=0.0,
                valid_threshold=0.0,
                prefix=safe,
                zero_scope="all",
                missing_plot_value="max",
                write_grid=True,
                write_plots=True,
                write_comparison=bool(write_comparison),
                contour_levels=int(contour_levels),
                dpi=int(dpi),
                title=f"{x_name} vs {y_name}",
                x_label=x_name,
                y_label=y_name,
                cmap=cmap,
            )
            plot_path = plot_outputs.get("smooth_png")
        result = ReweightProjectionResult(
            kind="2d",
            cvs=(x_name, y_name),
            safe_label=safe,
            fes_path=fes_path,
            plot_path=plot_path,
            sample_size=int(table.data.shape[0]),
            effective_sample_size=effective_sample_size,
            finite_points=int(np.count_nonzero(np.isfinite(fes2d))),
            bandwidth_sources=(f"{x_name}:{x_bandwidth.source}", f"{y_name}:{y_bandwidth.source}"),
        )
        results.append(result)
        manifest_rows.append(
            {
                "kind": result.kind,
                "cvs": ",".join(result.cvs),
                "safe_label": result.safe_label,
                "fes_path": str(result.fes_path),
                "plot_path": str(result.plot_path) if result.plot_path is not None else "",
                "sample_size": result.sample_size,
                "effective_sample_size": result.effective_sample_size,
                "finite_points": result.finite_points,
                "bandwidth_sources": ";".join(result.bandwidth_sources),
            }
        )

    summary_path = output_root / "fes_reweight" / "reweight_summary.csv"
    _write_reweight_summary(summary_path, manifest_rows)
    metadata = {
        "run_dir": str(run_dir),
        "colvar": str(colvar_paths[0]),
        "hills": str(hills_paths[0]),
        "colvar_paths": [str(path) for path in colvar_paths],
        "hills_paths": [str(path) for path in hills_paths],
        "output_root": str(output_root),
        "fields": list(table.fields),
        "skiprows": int(skiprows),
        "rows_after_filters": int(table.data.shape[0]),
        "temperature_K": float(temperature),
        "kbt_kj_mol": boltzmann_kbt(float(temperature)),
        "bias_columns": list(bias_columns),
        "effective_sample_size": effective_sample_size,
        "ranges": {name: [float(low), float(high)] for name, (low, high) in all_ranges.items()},
        "hills_bandwidths": hills_bandwidths,
        "outputs": [str(result.fes_path) for result in results],
    }
    metadata_path = output_root / "fes_reweight" / "reweight_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    report_path = output_root / "fes_reweight" / "analysis_report.md"
    report_lines = [
        "# OPES FES reweighting report",
        "",
        f"- Run directory: `{run_dir}`",
        f"- COLVAR inputs: {', '.join(f'`{path}`' for path in colvar_paths)}",
        f"- HILLS inputs: {', '.join(f'`{path}`' for path in hills_paths)}",
        f"- Rows after filters: {table.data.shape[0]}",
        f"- Temperature: {float(temperature):.3f} K",
        f"- Bias columns: {', '.join(bias_columns) if bias_columns else 'NO'}",
        f"- Effective sample size: {effective_sample_size:.6g}",
        "",
        "## Projections",
        "",
    ]
    for row in manifest_rows:
        report_lines.append(
            f"- {row['kind']} `{row['cvs']}` -> `{row['fes_path']}` "
            f"(finite points: {row['finite_points']}, bandwidth: {row['bandwidth_sources']})"
        )
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return {
        "summary": summary_path,
        "metadata": metadata_path,
        "report": report_path,
    }


def zero_curve(
    cv: np.ndarray,
    free_energy: np.ndarray,
    zero_low: float = -math.inf,
    zero_high: float = math.inf,
) -> Tuple[np.ndarray, float]:
    """Shift a curve so the minimum in `zero_low:zero_high` is zero."""

    clean, _ = clean_series(free_energy)
    zero = min_in_window(np.asarray(cv, dtype=float), clean, zero_low, zero_high)
    return clean - zero, zero


def numeric_fes_metadata(path: Path) -> Dict[str, float]:
    """Return numeric `#! SET` metadata values from a FES header."""

    _fields, metadata = parse_fes_header(path)
    numeric: Dict[str, float] = {}
    for key, value in metadata.items():
        try:
            numeric[key] = float(value)
        except (TypeError, ValueError):
            continue
    return numeric


def smooth_fes_curve(
    values: Sequence[float],
    smooth_window: int = 1,
    smooth_passes: int = 1,
) -> np.ndarray:
    """Optionally smooth a FES curve without imputing or bridging observations."""

    return moving_average_smooth(values, window_length=smooth_window, passes=smooth_passes)


def zero_values(values: Sequence[float]) -> Tuple[np.ndarray, float]:
    """Zero a sequence by its finite minimum."""

    arr, _finite = clean_series(values)
    finite = arr[np.isfinite(arr)]
    zero = float(np.min(finite)) if finite.size else 0.0
    return arr - zero, zero


def delta_f_window(cv: np.ndarray, free_energy_zeroed: np.ndarray, low: float, high: float) -> float:
    """Compute max-min Delta-F in an inclusive CV window."""

    mask = (np.asarray(cv, dtype=float) >= float(low)) & (np.asarray(cv, dtype=float) <= float(high))
    mask &= np.isfinite(free_energy_zeroed)
    if not np.any(mask):
        return math.nan
    values = np.asarray(free_energy_zeroed, dtype=float)[mask]
    return float(np.max(values) - np.min(values))


def _finite_array(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def _finite_mean(values: Sequence[float]) -> float:
    finite = _finite_array(values)
    return float(np.mean(finite)) if finite.size else math.nan


def _finite_sd(values: Sequence[float]) -> float:
    finite = _finite_array(values)
    return float(np.std(finite, ddof=1)) if finite.size > 1 else math.nan


def _finite_min_value(values: Sequence[float]) -> float:
    finite = _finite_array(values)
    return float(np.min(finite)) if finite.size else math.nan


def _finite_max_value(values: Sequence[float]) -> float:
    finite = _finite_array(values)
    return float(np.max(finite)) if finite.size else math.nan


def _finite_range(values: Sequence[float]) -> float:
    finite = _finite_array(values)
    return float(np.max(finite) - np.min(finite)) if finite.size else math.nan


def _value_or_nan(mapping: Mapping[str, float], key: str) -> float:
    value = mapping.get(key, math.nan)
    return float(value) if np.isfinite(value) else math.nan


def _processed_convergence_curve(
    path: Path,
    spec: FesConvergenceSpec,
    curve_type: str,
    window_low: float,
    window_high: float,
    smooth_window: int,
    smooth_passes: int,
    block_index: Optional[int] = None,
    cumulative_index: Optional[int] = None,
    trajectory_fraction: Optional[float] = None,
) -> Tuple[Dict[str, object], List[Dict[str, object]], Dict[str, float]]:
    curve = load_fes_curve(
        FesCurveSpec(
            path=Path(path),
            label=spec.label,
            group=spec.group,
            dataset_key=spec.dataset_key,
        )
    )
    metadata = numeric_fes_metadata(path)
    raw_zeroed, _raw_zero = zero_values(curve.free_energy)
    smooth_values = smooth_fes_curve(
        curve.free_energy,
        smooth_window=smooth_window,
        smooth_passes=smooth_passes,
    )
    smooth_zeroed, _smooth_zero = zero_values(smooth_values)
    raw_delta = delta_f_window(curve.cv, raw_zeroed, low=window_low, high=window_high)
    smooth_delta = delta_f_window(curve.cv, smooth_zeroed, low=window_low, high=window_high)

    metrics: Dict[str, object] = {
        "dataset_key": spec.dataset_key or spec.path.stem,
        "label": spec.label,
        "group": spec.group,
        "series": spec.series,
        "chemistry": spec.chemistry,
        "curve_type": curve_type,
        "block_index": block_index if block_index is not None else math.nan,
        "cumulative_index": cumulative_index if cumulative_index is not None else math.nan,
        "trajectory_fraction": trajectory_fraction if trajectory_fraction is not None else math.nan,
        "delta_f_win_smooth_kj_mol": smooth_delta,
        "delta_f_win_raw_kj_mol": raw_delta,
        "sample_size": _value_or_nan(metadata, "sample_size"),
        "effective_sample_size": _value_or_nan(metadata, "effective_sample_size"),
        "blocks_effective_num": _value_or_nan(metadata, "blocks_effective_num"),
        "source_path": str(path),
    }
    curve_rows: List[Dict[str, object]] = []
    for point_index, (cv, raw, smooth, uncertainty) in enumerate(
        zip(curve.cv, raw_zeroed, smooth_zeroed, curve.uncertainty)
    ):
        curve_rows.append(
            {
                "dataset_key": spec.dataset_key or spec.path.stem,
                "label": spec.label,
                "group": spec.group,
                "series": spec.series,
                "chemistry": spec.chemistry,
                "curve_type": curve_type,
                "block_index": block_index if block_index is not None else math.nan,
                "cumulative_index": cumulative_index if cumulative_index is not None else math.nan,
                "trajectory_fraction": trajectory_fraction if trajectory_fraction is not None else math.nan,
                "point_index": point_index,
                "cv": float(cv),
                "free_energy_raw_zeroed_kj_mol": float(raw),
                "free_energy_smooth_zeroed_kj_mol": float(smooth),
                "uncertainty_kj_mol": float(uncertainty) if np.isfinite(uncertainty) else math.nan,
                "source_path": str(path),
            }
        )
    return metrics, curve_rows, metadata


def _rank_rows(rows: List[Dict[str, object]], group_keys: Sequence[str], value_key: str, output_key: str) -> None:
    groups: Dict[Tuple[object, ...], List[Dict[str, object]]] = {}
    for row in rows:
        key = tuple(row.get(group_key, "") for group_key in group_keys)
        groups.setdefault(key, []).append(row)
    for group_rows in groups.values():
        finite_rows = [
            row for row in group_rows if np.isfinite(float(row.get(value_key, math.nan)))
        ]
        sorted_rows = sorted(finite_rows, key=lambda row: float(row.get(value_key, math.nan)), reverse=True)
        last_value: Optional[float] = None
        last_rank = 0
        for position, row in enumerate(sorted_rows, start=1):
            value = float(row.get(value_key, math.nan))
            if last_value is None or not math.isclose(value, last_value, rel_tol=0.0, abs_tol=0.0):
                last_rank = position
                last_value = value
            row[output_key] = last_rank
        for row in group_rows:
            row.setdefault(output_key, math.nan)


CONVERGENCE_SUMMARY_FIELDS: Tuple[str, ...] = (
    "dataset_key",
    "label",
    "group",
    "series",
    "chemistry",
    "window_low",
    "window_high",
    "delta_f_win_final_smooth_kj_mol",
    "delta_f_win_final_raw_kj_mol",
    "raw_minus_smooth_kj_mol",
    "rank_final_smooth_within_series",
    "n_blocks_available",
    "delta_f_win_block_mean_smooth_kj_mol",
    "delta_f_win_block_sd_smooth_kj_mol",
    "delta_f_win_block_min_smooth_kj_mol",
    "delta_f_win_block_max_smooth_kj_mol",
    "delta_f_win_block_range_smooth_kj_mol",
    "delta_f_win_block_mean_raw_kj_mol",
    "block_effective_sample_size_mean",
    "n_cumulative_profiles_available",
    "delta_f_win_cumulative_first_smooth_kj_mol",
    "delta_f_win_cumulative_penultimate_smooth_kj_mol",
    "delta_f_win_cumulative_last_smooth_kj_mol",
    "delta_f_win_cumulative_tail_range_smooth_kj_mol",
    "delta_f_win_final_minus_cumulative_last_smooth_kj_mol",
    "sample_size",
    "effective_sample_size",
    "blocks_effective_num",
    "source_path",
)


CONVERGENCE_VALUE_FIELDS: Tuple[str, ...] = (
    "dataset_key",
    "label",
    "group",
    "series",
    "chemistry",
    "curve_type",
    "block_index",
    "cumulative_index",
    "trajectory_fraction",
    "rank_cumulative_smooth_within_series",
    "delta_f_win_smooth_kj_mol",
    "delta_f_win_raw_kj_mol",
    "sample_size",
    "effective_sample_size",
    "blocks_effective_num",
    "source_path",
)


CONVERGENCE_CURVE_FIELDS: Tuple[str, ...] = (
    "dataset_key",
    "label",
    "group",
    "series",
    "chemistry",
    "curve_type",
    "block_index",
    "cumulative_index",
    "trajectory_fraction",
    "point_index",
    "cv",
    "free_energy_raw_zeroed_kj_mol",
    "free_energy_smooth_zeroed_kj_mol",
    "uncertainty_kj_mol",
    "source_path",
)


def analyze_fes_convergence(
    specs: Sequence[FesConvergenceSpec],
    output_dir: Path,
    window_low: float = 20.0,
    window_high: float = 52.0,
    smooth_window: int = 1,
    smooth_passes: int = 1,
) -> Dict[str, Path]:
    """Analyze final/block/cumulative FES Delta-F convergence."""

    if not specs:
        raise ValueError("At least one FES convergence dataset is required")
    if float(window_high) <= float(window_low):
        raise ValueError("window_high must be greater than window_low")

    summary_rows: List[Dict[str, object]] = []
    block_rows: List[Dict[str, object]] = []
    cumulative_rows: List[Dict[str, object]] = []
    curve_rows: List[Dict[str, object]] = []
    manifest_rows: List[Dict[str, object]] = []

    for spec in specs:
        final_metrics, final_curve_rows, _final_metadata = _processed_convergence_curve(
            spec.path,
            spec,
            curve_type="final",
            window_low=window_low,
            window_high=window_high,
            smooth_window=smooth_window,
            smooth_passes=smooth_passes,
        )
        curve_rows.extend(final_curve_rows)

        block_smooth_values: List[float] = []
        block_raw_values: List[float] = []
        block_neff_values: List[float] = []
        for block_index, block_path in enumerate(spec.block_paths, start=1):
            block_metrics, block_curve_rows, _block_metadata = _processed_convergence_curve(
                block_path,
                spec,
                curve_type="block",
                window_low=window_low,
                window_high=window_high,
                smooth_window=smooth_window,
                smooth_passes=smooth_passes,
                block_index=block_index,
            )
            block_rows.append(block_metrics)
            curve_rows.extend(block_curve_rows)
            block_smooth_values.append(float(block_metrics["delta_f_win_smooth_kj_mol"]))
            block_raw_values.append(float(block_metrics["delta_f_win_raw_kj_mol"]))
            if np.isfinite(float(block_metrics["effective_sample_size"])):
                block_neff_values.append(float(block_metrics["effective_sample_size"]))

        cumulative_records: List[Tuple[Path, Dict[str, object], List[Dict[str, object]], Dict[str, float]]] = []
        for cumulative_index, cumulative_path in enumerate(spec.cumulative_paths, start=1):
            metrics, rows, metadata = _processed_convergence_curve(
                cumulative_path,
                spec,
                curve_type="cumulative",
                window_low=window_low,
                window_high=window_high,
                smooth_window=smooth_window,
                smooth_passes=smooth_passes,
                cumulative_index=cumulative_index,
            )
            cumulative_records.append((cumulative_path, metrics, rows, metadata))
        sample_sizes = [
            float(metadata.get("sample_size", math.nan))
            for _path, _metrics, _rows, metadata in cumulative_records
            if np.isfinite(float(metadata.get("sample_size", math.nan)))
        ]
        max_sample_size = max(sample_sizes, default=math.nan)
        cumulative_smooth_values: List[float] = []
        cumulative_raw_values: List[float] = []
        for index, (_path, metrics, rows, _metadata) in enumerate(cumulative_records, start=1):
            sample_size = float(metrics.get("sample_size", math.nan))
            if np.isfinite(sample_size) and np.isfinite(max_sample_size) and max_sample_size > 0:
                fraction = sample_size / max_sample_size
            else:
                fraction = index / max(1, len(cumulative_records))
            metrics["trajectory_fraction"] = fraction
            for row in rows:
                row["trajectory_fraction"] = fraction
            cumulative_rows.append(metrics)
            curve_rows.extend(rows)
            cumulative_smooth_values.append(float(metrics["delta_f_win_smooth_kj_mol"]))
            cumulative_raw_values.append(float(metrics["delta_f_win_raw_kj_mol"]))

        cumulative_arr = np.asarray(cumulative_smooth_values, dtype=float)
        cumulative_tail = cumulative_arr[-2:] if cumulative_arr.size >= 2 else np.array([], dtype=float)
        cumulative_first = float(cumulative_arr[0]) if cumulative_arr.size else math.nan
        cumulative_penultimate = float(cumulative_arr[-2]) if cumulative_arr.size >= 2 else math.nan
        cumulative_last = float(cumulative_arr[-1]) if cumulative_arr.size else math.nan
        final_smooth = float(final_metrics["delta_f_win_smooth_kj_mol"])
        final_raw = float(final_metrics["delta_f_win_raw_kj_mol"])
        summary_rows.append(
            {
                "dataset_key": spec.dataset_key or spec.path.stem,
                "label": spec.label,
                "group": spec.group,
                "series": spec.series,
                "chemistry": spec.chemistry,
                "window_low": float(window_low),
                "window_high": float(window_high),
                "delta_f_win_final_smooth_kj_mol": final_smooth,
                "delta_f_win_final_raw_kj_mol": final_raw,
                "raw_minus_smooth_kj_mol": final_raw - final_smooth,
                "n_blocks_available": len(spec.block_paths),
                "delta_f_win_block_mean_smooth_kj_mol": _finite_mean(block_smooth_values),
                "delta_f_win_block_sd_smooth_kj_mol": _finite_sd(block_smooth_values),
                "delta_f_win_block_min_smooth_kj_mol": _finite_min_value(block_smooth_values),
                "delta_f_win_block_max_smooth_kj_mol": _finite_max_value(block_smooth_values),
                "delta_f_win_block_range_smooth_kj_mol": _finite_range(block_smooth_values),
                "delta_f_win_block_mean_raw_kj_mol": _finite_mean(block_raw_values),
                "block_effective_sample_size_mean": _finite_mean(block_neff_values),
                "n_cumulative_profiles_available": len(cumulative_records),
                "delta_f_win_cumulative_first_smooth_kj_mol": cumulative_first,
                "delta_f_win_cumulative_penultimate_smooth_kj_mol": cumulative_penultimate,
                "delta_f_win_cumulative_last_smooth_kj_mol": cumulative_last,
                "delta_f_win_cumulative_tail_range_smooth_kj_mol": _finite_range(cumulative_tail),
                "delta_f_win_final_minus_cumulative_last_smooth_kj_mol": (
                    final_smooth - cumulative_last if np.isfinite(cumulative_last) else math.nan
                ),
                "sample_size": final_metrics["sample_size"],
                "effective_sample_size": final_metrics["effective_sample_size"],
                "blocks_effective_num": final_metrics["blocks_effective_num"],
                "source_path": str(spec.path),
            }
        )
        manifest_rows.append(
            {
                "dataset_key": spec.dataset_key or spec.path.stem,
                "label": spec.label,
                "group": spec.group,
                "series": spec.series,
                "chemistry": spec.chemistry,
                "path": str(spec.path),
                "n_block_paths": len(spec.block_paths),
                "n_cumulative_paths": len(spec.cumulative_paths),
            }
        )

    _rank_rows(
        summary_rows,
        group_keys=("series",),
        value_key="delta_f_win_final_smooth_kj_mol",
        output_key="rank_final_smooth_within_series",
    )
    _rank_rows(
        cumulative_rows,
        group_keys=("series", "cumulative_index"),
        value_key="delta_f_win_smooth_kj_mol",
        output_key="rank_cumulative_smooth_within_series",
    )

    output_dir = Path(output_dir)
    outputs = {
        "summary": output_dir / "fes_convergence_summary.csv",
        "blocks": output_dir / "fes_convergence_block_values.csv",
        "cumulative": output_dir / "fes_convergence_cumulative_values.csv",
        "curves": output_dir / "fes_convergence_curves.csv",
        "manifest": output_dir / "fes_convergence_manifest.csv",
    }
    _write_csv_with_fields(outputs["summary"], summary_rows, CONVERGENCE_SUMMARY_FIELDS)
    _write_csv_with_fields(outputs["blocks"], block_rows, CONVERGENCE_VALUE_FIELDS)
    _write_csv_with_fields(outputs["cumulative"], cumulative_rows, CONVERGENCE_VALUE_FIELDS)
    _write_csv_with_fields(outputs["curves"], curve_rows, CONVERGENCE_CURVE_FIELDS)
    _write_csv_with_fields(
        outputs["manifest"],
        manifest_rows,
        ("dataset_key", "label", "group", "series", "chemistry", "path", "n_block_paths", "n_cumulative_paths"),
    )
    return outputs


def process_curve(
    curve: FesCurve,
    reference_low: float = -math.inf,
    reference_high: float = math.inf,
    zero_low: float = -math.inf,
    zero_high: float = math.inf,
    smooth_window: int = 1,
    smooth_passes: int = 1,
) -> Tuple[List[Dict[str, object]], np.ndarray, np.ndarray, np.ndarray]:
    """Build processed rows and return shifted/smoothed/zeroed arrays."""

    shifted, reference_shift = shift_to_reference_window(
        curve.cv,
        curve.free_energy,
        reference_low=reference_low,
        reference_high=reference_high,
    )
    smoothed = moving_average_smooth(shifted, window_length=smooth_window, passes=smooth_passes)
    zeroed, zero_shift = zero_curve(curve.cv, smoothed, zero_low=zero_low, zero_high=zero_high)

    rows: List[Dict[str, object]] = []
    for index, (cv, raw_fe, shifted_fe, smooth_fe, zero_fe, unc) in enumerate(
        zip(curve.cv, curve.free_energy, shifted, smoothed, zeroed, curve.uncertainty)
    ):
        rows.append(
            {
                "dataset_key": curve.spec.dataset_key or curve.spec.path.stem,
                "label": curve.spec.label,
                "group": curve.spec.group,
                "source_path": str(curve.spec.path),
                "point_index": index,
                "cv": float(cv),
                "free_energy_raw_kj_mol": float(raw_fe),
                "free_energy_reference_shifted_kj_mol": float(shifted_fe),
                "free_energy_smooth_kj_mol": float(smooth_fe),
                "free_energy_smooth_zeroed_kj_mol": float(zero_fe),
                "reference_shift_value_kj_mol": reference_shift,
                "smooth_zero_shift_value_kj_mol": zero_shift,
                "uncertainty_kj_mol": float(unc) if np.isfinite(unc) else math.nan,
            }
        )
    return rows, shifted, smoothed, zeroed


def barrier_for_window(cv: np.ndarray, values: np.ndarray, low: float, high: float) -> Tuple[float, float, float, float, int]:
    """Return max-min barrier and extrema metadata for a CV window."""

    mask = window_mask(cv, low, high) & np.isfinite(values)
    if not np.any(mask):
        return math.nan, math.nan, math.nan, math.nan, 0
    x = np.asarray(cv, dtype=float)[mask]
    y = np.asarray(values, dtype=float)[mask]
    min_index = int(np.argmin(y))
    max_index = int(np.argmax(y))
    barrier = float(y[max_index] - y[min_index])
    return barrier, float(x[min_index]), float(y[min_index]), float(x[max_index]), int(mask.sum())


def build_barrier_rows(
    curve: FesCurve,
    shifted: np.ndarray,
    smoothed: np.ndarray,
    windows: Sequence[Tuple[str, float, float]],
) -> List[Dict[str, object]]:
    """Build barrier summary rows for one curve."""

    rows: List[Dict[str, object]] = []
    for name, low, high in windows:
        original_barrier, min_cv, min_fe, max_cv, n_points = barrier_for_window(curve.cv, shifted, low, high)
        smooth_barrier, smooth_min_cv, smooth_min_fe, smooth_max_cv, smooth_n_points = barrier_for_window(
            curve.cv,
            smoothed,
            low,
            high,
        )
        rows.append(
            {
                "dataset_key": curve.spec.dataset_key or curve.spec.path.stem,
                "label": curve.spec.label,
                "group": curve.spec.group,
                "source_path": str(curve.spec.path),
                "barrier_region": name,
                "cv_low": low,
                "cv_high": high,
                "n_points": n_points,
                "smooth_n_points": smooth_n_points,
                "barrier_original_kj_mol": original_barrier,
                "barrier_smooth_kj_mol": smooth_barrier,
                "change_smooth_minus_original_kj_mol": (
                    smooth_barrier - original_barrier
                    if math.isfinite(smooth_barrier) and math.isfinite(original_barrier)
                    else math.nan
                ),
                "original_min_cv": min_cv,
                "original_min_fe_kj_mol": min_fe,
                "original_max_cv": max_cv,
                "smooth_min_cv": smooth_min_cv,
                "smooth_min_fe_kj_mol": smooth_min_fe,
                "smooth_max_cv": smooth_max_cv,
            }
        )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    preferred = [
        "dataset_key",
        "label",
        "group",
        "barrier_region",
        "point_index",
        "cv",
        "cv_low",
        "cv_high",
        "n_points",
    ]
    ordered = [key for key in preferred if key in fieldnames]
    ordered.extend([key for key in fieldnames if key not in ordered])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def _write_csv_with_fields(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def analyze_fes_barriers(
    curve_specs: Sequence[FesCurveSpec],
    output_dir: Path,
    windows: Sequence[Tuple[str, float, float]] = DEFAULT_BARRIER_WINDOWS,
    reference_low: float = -math.inf,
    reference_high: float = math.inf,
    zero_low: float = -math.inf,
    zero_high: float = math.inf,
    smooth_window: int = 1,
    smooth_passes: int = 1,
    cv_column: int = 0,
    free_energy_column: int = 1,
    uncertainty_column: Optional[int] = 2,
) -> Dict[str, Path]:
    """Process curves and write processed-curve/barrier CSV outputs."""

    if not curve_specs:
        raise ValueError("At least one FES curve is required")
    output_dir = Path(output_dir)
    processed_rows: List[Dict[str, object]] = []
    barrier_rows: List[Dict[str, object]] = []
    manifest_rows: List[Dict[str, object]] = []

    for spec in curve_specs:
        curve = load_fes_curve(
            spec,
            cv_column=cv_column,
            free_energy_column=free_energy_column,
            uncertainty_column=uncertainty_column,
        )
        rows, shifted, smoothed, _zeroed = process_curve(
            curve,
            reference_low=reference_low,
            reference_high=reference_high,
            zero_low=zero_low,
            zero_high=zero_high,
            smooth_window=smooth_window,
            smooth_passes=smooth_passes,
        )
        processed_rows.extend(rows)
        barrier_rows.extend(build_barrier_rows(curve, shifted, smoothed, windows=windows))
        manifest_rows.append(
            {
                "dataset_key": spec.dataset_key or spec.path.stem,
                "label": spec.label,
                "group": spec.group,
                "path": str(spec.path),
                "n_points": int(len(curve.cv)),
                "cv_min": float(np.nanmin(curve.cv)),
                "cv_max": float(np.nanmax(curve.cv)),
            }
        )

    outputs = {
        "processed_curves": output_dir / "fes_processed_curves.csv",
        "barrier_summary": output_dir / "fes_barrier_summary.csv",
        "manifest": output_dir / "fes_input_manifest.csv",
    }
    _write_csv(outputs["processed_curves"], processed_rows)
    _write_csv(outputs["barrier_summary"], barrier_rows)
    _write_csv(outputs["manifest"], manifest_rows)
    return outputs


def _curve_specs_from_args(args: argparse.Namespace) -> List[FesCurveSpec]:
    specs: List[FesCurveSpec] = []
    if args.manifest is not None:
        specs.extend(load_curve_manifest(args.manifest))
    for values in args.curve or []:
        specs.append(parse_curve_spec(values))
    return specs


def get_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process 1D FES curves and compute barrier summaries")
    parser.add_argument("--manifest", type=Path, help="CSV with path,label,group,dataset_key columns")
    parser.add_argument(
        "--curve",
        nargs=3,
        action="append",
        metavar=("PATH", "LABEL", "GROUP"),
        help="Explicit curve input; may be repeated",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--barrier-window", action="append", help="Window as name:low:high; may be repeated")
    parser.add_argument("--reference-low", type=float, default=-math.inf)
    parser.add_argument("--reference-high", type=float, default=math.inf)
    parser.add_argument("--zero-low", type=float, default=-math.inf)
    parser.add_argument("--zero-high", type=float, default=math.inf)
    parser.add_argument("--smooth-window", type=int, default=1)
    parser.add_argument("--smooth-passes", type=int, default=1)
    parser.add_argument("--cv-column", type=int, default=0, help="Zero-based CV column index")
    parser.add_argument("--free-energy-column", type=int, default=1, help="Zero-based free-energy column index")
    parser.add_argument("--uncertainty-column", type=int, default=2, help="Zero-based uncertainty column index; use -1 to disable")
    return parser.parse_args(argv)


def get_fes2d_grid_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Build arguments for the standalone 2D FES grid processor."""

    parser = argparse.ArgumentParser(description="Process a regular 2D FES grid")
    parser.add_argument("--fes-file", type=Path, required=True, help="PLUMED-style 2D FES table")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for processed outputs")
    parser.add_argument("--x-range", type=float, nargs=2, metavar=("LOW", "HIGH"), help="Inclusive x range")
    parser.add_argument("--y-range", type=float, nargs=2, metavar=("LOW", "HIGH"), help="Inclusive y range")
    parser.add_argument("--max-fes", type=float, default=200.0, help="Maximum plotted FES value in kJ/mol")
    parser.add_argument("--smooth-sigma", type=float, default=0.8, help="Gaussian sigma in grid-bin units")
    parser.add_argument("--smooth-valid-threshold", type=float, default=0.25)
    parser.add_argument("--prefix", default="fes2d", help="Output filename prefix")
    parser.add_argument("--zero-scope", choices=("window", "all"), default="window")
    parser.add_argument("--missing-plot-value", choices=("max", "nan"), default="max")
    parser.add_argument("--no-grid", action="store_true", help="Skip long-form processed grid CSV")
    parser.add_argument("--write-plots", action="store_true", help="Write contour plot PNG outputs")
    parser.add_argument("--write-comparison", action="store_true", help="Also write raw-vs-smoothed comparison")
    parser.add_argument("--contour-levels", type=int, default=16)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--title", default="")
    parser.add_argument("--x-label")
    parser.add_argument("--y-label")
    parser.add_argument("--cmap", default="viridis")
    return parser.parse_args(argv)


def run_fes2d_grid(argv: Optional[Sequence[str]] = None) -> int:
    """Run 2D FES grid processing from CLI-style arguments."""

    args = get_fes2d_grid_args(argv)
    x_range = None if args.x_range is None else (float(args.x_range[0]), float(args.x_range[1]))
    y_range = None if args.y_range is None else (float(args.y_range[0]), float(args.y_range[1]))
    try:
        outputs = process_fes2d_grid(
            fes_file=args.fes_file,
            output_dir=args.output_dir,
            x_range=x_range,
            y_range=y_range,
            max_fes=float(args.max_fes),
            smooth_sigma=float(args.smooth_sigma),
            valid_threshold=float(args.smooth_valid_threshold),
            prefix=str(args.prefix),
            zero_scope=str(args.zero_scope),
            missing_plot_value=str(args.missing_plot_value),
            write_grid=not bool(args.no_grid),
            write_plots=bool(args.write_plots or args.write_comparison),
            write_comparison=bool(args.write_comparison),
            contour_levels=int(args.contour_levels),
            dpi=int(args.dpi),
            title=str(args.title),
            x_label=args.x_label,
            y_label=args.y_label,
            cmap=str(args.cmap),
        )
    except Exception as exc:
        print(f"2D FES grid processing failed: {exc}")
        return 1

    for path in outputs.values():
        print(path)
    return 0


def get_fes_reweight_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Build arguments for direct COLVAR reweighting."""

    parser = argparse.ArgumentParser(description="Reweight OPES COLVAR data into 1D and 2D FES projections")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--colvar", action="append", default=[], help="COLVAR path relative to run dir; may be repeated")
    parser.add_argument("--hills", action="append", default=[], help="HILLS path relative to run dir; may be repeated")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--cv", action="append", default=[], help="1D projected CV name; may be repeated")
    parser.add_argument("--pair", nargs=2, action="append", default=[], metavar=("X_CV", "Y_CV"))
    parser.add_argument("--bias", action="append", default=[], help="Bias column name; repeat to sum biases")
    parser.add_argument("--range", nargs=3, action="append", default=[], metavar=("CV", "LOW", "HIGH"))
    parser.add_argument("--bandwidth", nargs=2, action="append", default=[], metavar=("CV", "SIGMA"))
    parser.add_argument("--temperature", type=float, default=330.0)
    parser.add_argument("--skiprows", type=int, default=0)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--bins", type=int, default=160)
    parser.add_argument("--pair-bins", type=int, nargs=2, default=(90, 90), metavar=("NX", "NY"))
    parser.add_argument("--default-smooth-bins", type=float, default=1.5)
    parser.add_argument("--range-padding-fraction", type=float, default=0.05)
    parser.add_argument("--skip-last-data-line", action="store_true")
    parser.add_argument("--skip-last-data-line-per-file", action="store_true")
    parser.add_argument("--keep-duplicate-times", action="store_true")
    parser.add_argument("--max-fes", type=float, default=200.0)
    parser.add_argument("--write-plots", action="store_true")
    parser.add_argument("--write-comparison", action="store_true")
    parser.add_argument("--contour-levels", type=int, default=16)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--cmap", default="viridis")
    return parser.parse_args(argv)


def run_fes_reweight(argv: Optional[Sequence[str]] = None) -> int:
    """Run direct COLVAR reweighting from CLI-style arguments."""

    args = get_fes_reweight_args(argv)
    try:
        outputs = reweight_plumed_colvar(
            args.run_dir,
            colvar_names=tuple(args.colvar or ("COLVAR",)),
            hills_names=tuple(args.hills or ("HILLS",)),
            output_root=args.output_root,
            cv_names=tuple(args.cv or ()),
            pairs=tuple((str(x), str(y)) for x, y in (args.pair or ())),
            bias_names=tuple(args.bias or (".bias",)),
            ranges=parse_named_ranges(args.range),
            bandwidths=parse_named_float_map(args.bandwidth),
            temperature=float(args.temperature),
            skiprows=int(args.skiprows),
            blocks=int(args.blocks),
            bins=int(args.bins),
            pair_bins=(int(args.pair_bins[0]), int(args.pair_bins[1])),
            default_smooth_bins=float(args.default_smooth_bins),
            range_padding_fraction=float(args.range_padding_fraction),
            skip_last_data_line=bool(args.skip_last_data_line),
            skip_last_data_line_per_file=bool(args.skip_last_data_line_per_file),
            deduplicate_time=not bool(args.keep_duplicate_times),
            max_fes=float(args.max_fes),
            write_plots=bool(args.write_plots or args.write_comparison),
            write_comparison=bool(args.write_comparison),
            contour_levels=int(args.contour_levels),
            dpi=int(args.dpi),
            cmap=str(args.cmap),
        )
    except Exception as exc:
        print(f"FES reweighting failed: {exc}")
        return 1

    for path in outputs.values():
        print(path)
    return 0


def get_fes_convergence_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Build arguments for FES convergence/robustness analysis."""

    parser = argparse.ArgumentParser(description="Analyze FES Delta-F convergence from a CSV manifest")
    parser.add_argument("--manifest", type=Path, required=True, help="CSV with path and optional block/cumulative inputs")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window-low", type=float, default=20.0)
    parser.add_argument("--window-high", type=float, default=52.0)
    parser.add_argument("--smooth-window", type=int, default=1)
    parser.add_argument("--smooth-passes", type=int, default=1)
    parser.add_argument("--infer-blocks", action="store_true", help="Infer existing PATH stem_1/stem_2/... block files")
    parser.add_argument("--block-count", type=int, default=3)
    parser.add_argument("--cumulative-glob", default="fes-cum_*.dat")
    return parser.parse_args(argv)


def run_fes_convergence(argv: Optional[Sequence[str]] = None) -> int:
    """Run FES convergence analysis from CLI-style arguments."""

    args = get_fes_convergence_args(argv)
    try:
        specs = load_fes_convergence_manifest(
            args.manifest,
            infer_blocks=bool(args.infer_blocks),
            block_count=int(args.block_count),
            cumulative_glob=str(args.cumulative_glob),
        )
        outputs = analyze_fes_convergence(
            specs,
            output_dir=args.output_dir,
            window_low=float(args.window_low),
            window_high=float(args.window_high),
            smooth_window=int(args.smooth_window),
            smooth_passes=int(args.smooth_passes),
        )
    except Exception as exc:
        print(f"FES convergence analysis failed: {exc}")
        return 1

    for path in outputs.values():
        print(path)
    return 0


def get_fes2d_batch_manifest_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Build arguments for 2D FES batch manifest generation."""

    parser = argparse.ArgumentParser(description="Prepare a path-explicit 2D FES batch manifest")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--case-manifest", type=Path, help="CSV with bias_dir,case_label[,family] columns")
    input_group.add_argument(
        "--case-path-list",
        action="append",
        help="Plain path-list input as PATH or FAMILY:PATH; may be repeated",
    )
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--colvar-name", default="COLVAR_tmp")
    parser.add_argument("--run-subdir", default="fes2D/bins50")
    parser.add_argument("--fes-name", default="fes-rew.dat")
    parser.add_argument("--output-subdir", default="fes2d_plot")
    parser.add_argument("--prefix-template", default="{safe_label}_fes2d")
    parser.add_argument("--x-range", type=float, nargs=2, default=(20.0, 52.0), metavar=("LOW", "HIGH"))
    parser.add_argument("--y-range", type=float, nargs=2, default=(50.0, 380.0), metavar=("LOW", "HIGH"))
    parser.add_argument("--max-fes", type=float, default=200.0)
    parser.add_argument("--smooth-sigma", type=float, default=0.8)
    parser.add_argument("--smooth-valid-threshold", type=float, default=0.25)
    parser.add_argument("--contour-levels", type=int, default=16)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--zero-scope", choices=("window", "all"), default="window")
    parser.add_argument("--missing-plot-value", choices=("max", "nan"), default="max")
    parser.add_argument("--write-plots", action="store_true")
    parser.add_argument("--write-comparison", action="store_true")
    parser.add_argument("--command-executable", default="molsimflow")
    parser.add_argument("--create-dirs", action="store_true")
    return parser.parse_args(argv)


def run_fes2d_batch_manifest(argv: Optional[Sequence[str]] = None) -> int:
    """Run 2D FES batch manifest generation from CLI-style arguments."""

    args = get_fes2d_batch_manifest_args(argv)
    try:
        if args.case_manifest is not None:
            cases = load_fes2d_batch_case_manifest(args.case_manifest)
        else:
            cases = []
            for raw in args.case_path_list or []:
                family, path = parse_case_path_list_spec(raw)
                cases.extend(read_fes2d_case_path_list(path, family=family))
        config = Fes2DBatchConfig(
            colvar_name=args.colvar_name,
            run_subdir=args.run_subdir,
            fes_name=args.fes_name,
            output_subdir=args.output_subdir,
            prefix_template=args.prefix_template,
            x_low=float(args.x_range[0]),
            x_high=float(args.x_range[1]),
            y_low=float(args.y_range[0]),
            y_high=float(args.y_range[1]),
            max_fes=float(args.max_fes),
            smooth_sigma=float(args.smooth_sigma),
            smooth_valid_threshold=float(args.smooth_valid_threshold),
            contour_levels=int(args.contour_levels),
            dpi=int(args.dpi),
            zero_scope=args.zero_scope,
            missing_plot_value=args.missing_plot_value,
            write_plots=bool(args.write_plots or args.write_comparison),
            write_comparison=bool(args.write_comparison),
        )
        rows = prepare_fes2d_batch_manifest(
            cases,
            output_manifest=args.output_manifest,
            config=config,
            command_executable=args.command_executable,
            create_dirs=bool(args.create_dirs),
        )
    except Exception as exc:
        print(f"2D FES batch manifest generation failed: {exc}")
        return 1
    print(args.output_manifest)
    print(f"cases={len(rows)}")
    return 0


def get_fes_cumulative_reweight_manifest_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Build arguments for cumulative reweight command-table generation."""

    parser = argparse.ArgumentParser(description="Prepare a cumulative FES reweight command manifest")
    parser.add_argument("--manifest", type=Path, required=True, help="CSV with system,workdir,colvar,sample_size columns")
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--driver", type=Path, required=True, help="Path to the reweighting driver script")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fraction", type=float, action="append", help="Trajectory fraction; may be repeated")
    parser.add_argument("--output-prefix", default="fes-cum")
    parser.add_argument("--python-executable", default="python")
    parser.add_argument("--cv", default="d3d_all")
    parser.add_argument("--cv-min", type=float, default=5.0)
    parser.add_argument("--cv-max", type=float, default=52.0)
    parser.add_argument("--delta-fat", type=float, default=45.0)
    parser.add_argument("--sigma", type=float, default=0.06)
    parser.add_argument("--skiprows", type=int, default=50000)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=330.0)
    parser.add_argument("--create-dirs", action="store_true")
    return parser.parse_args(argv)


def run_fes_cumulative_reweight_manifest(argv: Optional[Sequence[str]] = None) -> int:
    """Run cumulative FES reweight command-table generation."""

    args = get_fes_cumulative_reweight_manifest_args(argv)
    try:
        specs = load_fes_cumulative_reweight_manifest(args.manifest)
        config = FesCumulativeReweightConfig(
            driver=args.driver,
            output_root=args.output_root,
            fractions=tuple(args.fraction or (0.60, 0.80, 1.00)),
            output_prefix=args.output_prefix,
            python_executable=args.python_executable,
            cv=args.cv,
            cv_min=float(args.cv_min),
            cv_max=float(args.cv_max),
            delta_f_at=float(args.delta_fat),
            sigma=float(args.sigma),
            skiprows=int(args.skiprows),
            blocks=int(args.blocks),
            temperature=float(args.temperature),
        )
        rows = prepare_fes_cumulative_reweight_manifest(
            specs,
            output_manifest=args.output_manifest,
            config=config,
            create_dirs=bool(args.create_dirs),
        )
    except Exception as exc:
        print(f"Cumulative FES reweight manifest generation failed: {exc}")
        return 1
    print(args.output_manifest)
    print(f"commands={len(rows)}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = get_args(argv)
    try:
        specs = _curve_specs_from_args(args)
        windows = [parse_window(item) for item in args.barrier_window] if args.barrier_window else list(DEFAULT_BARRIER_WINDOWS)
        outputs = analyze_fes_barriers(
            specs,
            output_dir=args.output_dir,
            windows=windows,
            reference_low=args.reference_low,
            reference_high=args.reference_high,
            zero_low=args.zero_low,
            zero_high=args.zero_high,
            smooth_window=args.smooth_window,
            smooth_passes=args.smooth_passes,
            cv_column=args.cv_column,
            free_energy_column=args.free_energy_column,
            uncertainty_column=None if int(args.uncertainty_column) < 0 else int(args.uncertainty_column),
        )
    except Exception as exc:
        print(f"FES barrier analysis failed: {exc}")
        return 1

    for path in outputs.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
