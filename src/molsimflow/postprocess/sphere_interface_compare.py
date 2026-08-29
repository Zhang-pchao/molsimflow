"""Compare existing nanodroplet or nanobubble post-processing results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PALETTE = ("#315B7D", "#B38728", "#D06B3C", "#687A3C")
LINESTYLES = ("-", "--", "-.", ":")
MARKERS = ("o", "s", "^", "D")
INK = "#20242C"
MUTED = "#68707D"
GRID = "#D9DEE7"


@dataclass(frozen=True)
class ComparisonCase:
    case_id: str
    analysis_root: Path
    ch3_sites: int
    oh_sites: int

    @property
    def ch3_fraction(self) -> float:
        return self.ch3_sites / (self.ch3_sites + self.oh_sites)

    @property
    def label(self) -> str:
        return f"CH₃:OH = {self.ch3_sites}:{self.oh_sites}"


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def load_cases(path: Path) -> list[ComparisonCase]:
    required = {"case_id", "analysis_root", "ch3_sites", "oh_sites"}
    cases: list[ComparisonCase] = []
    for row in _read_tsv(path):
        if not required.issubset(row):
            raise ValueError(f"Manifest requires columns: {sorted(required)}")
        case = ComparisonCase(
            case_id=row["case_id"].strip(),
            analysis_root=Path(row["analysis_root"]).expanduser(),
            ch3_sites=int(row["ch3_sites"]),
            oh_sites=int(row["oh_sites"]),
        )
        if not case.case_id or case.ch3_sites < 0 or case.oh_sites < 0:
            raise ValueError(f"Invalid case row: {row}")
        if case.ch3_sites + case.oh_sites <= 0:
            raise ValueError(f"Case {case.case_id} has no terminal sites")
        if not case.analysis_root.is_dir():
            raise FileNotFoundError(case.analysis_root)
        cases.append(case)
    if len(cases) < 2:
        raise ValueError("At least two cases are required")
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("Case IDs must be unique")
    return cases


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_key_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def resolve_results(analysis_root: Path, module: str) -> tuple[Path | None, str, str]:
    module_root = analysis_root / module
    values = _read_key_values(module_root / "ANALYSIS-RESULT.txt")
    latest = module_root / "latest" / "results"
    if latest.is_dir():
        status = values.get("status", "PASS_LEGACY")
        return latest.resolve(), status, "latest"
    recorded = Path(values["results"]) if values.get("results") else None
    if recorded is not None and recorded.is_dir():
        return recorded.resolve(), values.get("status", "UNKNOWN"), "recorded_failed_or_unpublished"
    return None, "MISSING", "missing"


def _float(value: object, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result


def _mean_std(values: Iterable[object]) -> tuple[float, float]:
    array = np.asarray([_float(value) for value in values], dtype=float)
    array = array[np.isfinite(array)]
    if not array.size:
        return math.nan, math.nan
    return float(np.mean(array)), float(np.std(array, ddof=1)) if array.size > 1 else 0.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], preferred: Sequence[str] = ()) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = {key for row in rows for key in row}
    fields = list(preferred) + sorted(keys - set(preferred))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _case_fields(case: ComparisonCase) -> dict[str, object]:
    return {
        "case_id": case.case_id,
        "legend_label": case.label,
        "ch3_sites": case.ch3_sites,
        "oh_sites": case.oh_sites,
        "ch3_fraction": case.ch3_fraction,
    }


def _source(
    rows: list[dict[str, object]], case: ComparisonCase, module: str, path: Path
) -> None:
    rows.append(
        {
            **_case_fields(case),
            "module": module,
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    )


def _block_core_rows(
    case: ComparisonCase,
    rows: Sequence[Mapping[str, str]],
    metrics: Sequence[str],
    timestep_fs: float,
    block_frames: int,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for start in range(0, len(rows), block_frames):
        block = rows[start : start + block_frames]
        if not block:
            continue
        record: dict[str, object] = {
            **_case_fields(case),
            "block_index": start // block_frames,
            "frame_count": len(block),
            "time_ns": np.mean([_float(row["step"]) * timestep_fs / 1.0e6 for row in block]),
        }
        for metric in metrics:
            mean, std = _mean_std(row.get(metric) for row in block)
            record[metric] = mean
            record[f"{metric}_std"] = std
        output.append(record)
    return output


def collect_comparison(
    cases: Sequence[ComparisonCase],
    kind: str,
    output_dir: Path,
    *,
    block_frames: int = 10,
    make_plots: bool = True,
    font_path: Path | None = None,
    dpi: int = 300,
) -> dict[str, object]:
    if kind not in {"nanodroplet", "nanobubble"}:
        raise ValueError("kind must be nanodroplet or nanobubble")
    if block_frames < 1:
        raise ValueError("block_frames must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    core_module = "droplet_spreading" if kind == "nanodroplet" else "attachment_kinetics"
    core_csv_name = "droplet_spreading.csv" if kind == "nanodroplet" else "attachment_kinetics.csv"
    core_metrics = (
        (
            "droplet_center_surface_dz_A",
            "droplet_lateral_radius_p90_A",
            "droplet_height_q05_q95_A",
            "footprint_convex_hull_area_A2",
        )
        if kind == "nanodroplet"
        else (
            "bubble_center_surface_dz_A",
            "bubble_lateral_radius_p90_A",
            "bubble_height_q05_q95_A",
            "bubble_contact_n2_count",
        )
    )
    modules = (
        core_module,
        "lateral_motion",
        "contact_angle_density",
        "contact_line",
        "surface_site_enrichment",
        "interfacial_water_density",
        "interfacial_water_orientation",
        "interfacial_water_hbond",
        "surface_proton_transfer",
    )
    if kind == "nanobubble":
        modules += ("precontact_n2_enrichment",)

    case_rows: list[dict[str, object]] = []
    availability: list[dict[str, object]] = []
    source_rows: list[dict[str, object]] = []
    core_rows: list[dict[str, object]] = []
    line_rows: list[dict[str, object]] = []
    density_rows: list[dict[str, object]] = []
    orientation_rows: list[dict[str, object]] = []

    for case in cases:
        resolved: dict[str, tuple[Path | None, str, str]] = {}
        for module in modules:
            result_dir, quality_status, source_mode = resolve_results(case.analysis_root, module)
            summary_status = "MISSING"
            if result_dir is not None and (result_dir / "summary.json").is_file():
                summary_status = str(_read_json(result_dir / "summary.json").get("status", "UNKNOWN"))
            availability.append(
                {
                    **_case_fields(case),
                    "module": module,
                    "quality_status": quality_status,
                    "summary_status": summary_status,
                    "source_mode": source_mode,
                    "results_dir": str(result_dir) if result_dir else "",
                }
            )
            resolved[module] = (result_dir, quality_status, source_mode)

        missing_required = [
            module
            for module in modules
            if module != "precontact_n2_enrichment" and resolved[module][0] is None
        ]
        if missing_required:
            raise FileNotFoundError(f"{case.case_id} missing required results: {missing_required}")

        row: dict[str, object] = {**_case_fields(case), "kind": kind}
        core_dir = resolved[core_module][0]
        assert core_dir is not None
        core_summary_path = core_dir / "summary.json"
        core_data_path = core_dir / core_csv_name
        core_summary = _read_json(core_summary_path)
        core_data = _read_csv(core_data_path)
        _source(source_rows, case, core_module, core_summary_path)
        _source(source_rows, case, core_module, core_data_path)
        timestep_fs = _float(core_summary.get("timestep_fs", core_summary.get("time_step_fs", 0.5)))
        core_rows.extend(
            _block_core_rows(case, core_data, core_metrics, timestep_fs, block_frames)
        )
        late = core_data[-min(201, len(core_data)) :]
        row["core_analyzed_frames"] = core_summary.get("analyzed_frames", len(core_data))
        row["core_first_step"] = core_summary.get("first_step")
        row["core_last_step"] = core_summary.get("last_step")
        row["timestep_fs"] = timestep_fs
        if kind == "nanobubble":
            row["attachment_time_ns"] = core_summary.get("attachment_time_ns", math.nan)
        for metric in core_metrics:
            mean, std = _mean_std(item.get(metric) for item in late)
            row[f"late_{metric}_mean"] = mean
            row[f"late_{metric}_std"] = std

        lateral_dir = resolved["lateral_motion"][0]
        assert lateral_dir is not None
        lateral_summary_path = lateral_dir / "summary.json"
        lateral = _read_json(lateral_summary_path)
        _source(source_rows, case, "lateral_motion", lateral_summary_path)
        for key in ("net_displacement_A", "maximum_displacement_A", "cumulative_path_length_A"):
            row[f"lateral_{key}"] = lateral.get(key, math.nan)

        angle_dir, angle_quality, _ = resolved["contact_angle_density"]
        assert angle_dir is not None
        angle_path = angle_dir / "summary.json"
        angle = _read_json(angle_path)
        _source(source_rows, case, "contact_angle_density", angle_path)
        row.update(
            {
                "contact_angle_quality_status": angle_quality,
                "contact_angle_phase_label": angle.get("phase_label", ""),
                "contact_angle_deg": angle.get("block_dense_phase_angle_mean_deg", math.nan),
                "contact_angle_std_deg": angle.get("block_dense_phase_angle_std_deg", math.nan),
                "contact_angle_primary_deg": angle.get("primary_dense_phase_contact_angle_deg", math.nan),
                "contact_angle_valid_blocks": angle.get("valid_block_count", 0),
                "contact_angle_threshold_span_deg": angle.get("threshold_angle_span_deg", math.nan),
            }
        )

        line_dir, line_quality, _ = resolved["contact_line"]
        assert line_dir is not None
        line_summary_path = line_dir / "summary.json"
        line_blocks_path = line_dir / "contact_line_blocks.csv"
        line_summary = _read_json(line_summary_path)
        blocks = _read_csv(line_blocks_path)
        _source(source_rows, case, "contact_line", line_summary_path)
        _source(source_rows, case, "contact_line", line_blocks_path)
        attachment_time = _float(row.get("attachment_time_ns"), 0.0)
        for block in blocks:
            item = {**_case_fields(case), **block}
            midpoint = 0.5 * (_float(block.get("start_time_ns")) + _float(block.get("end_time_ns")))
            item["time_mid_ns"] = midpoint
            item["time_since_attachment_ns"] = midpoint - attachment_time if kind == "nanobubble" else midpoint
            line_rows.append(item)
        valid = int(line_summary.get("valid_contact_line_frames", 0))
        analyzed = int(line_summary.get("analyzed_frames", 0))
        late_blocks = blocks[-min(10, len(blocks)) :]
        line_mean, line_std = _mean_std(block.get("mean_equivalent_radius_A") for block in late_blocks)
        row.update(
            {
                "contact_line_quality_status": line_quality,
                "contact_line_valid_frames": valid,
                "contact_line_analyzed_frames": analyzed,
                "contact_line_valid_fraction": valid / analyzed if analyzed else math.nan,
                "contact_line_jump_candidate_count": line_summary.get("jump_candidate_count", 0),
                "late_contact_line_radius_A_mean": line_mean,
                "late_contact_line_radius_A_std": line_std,
            }
        )

        site_dir = resolved["surface_site_enrichment"][0]
        assert site_dir is not None
        site_path = site_dir / "summary.json"
        site = _read_json(site_path)
        _source(source_rows, case, "surface_site_enrichment", site_path)
        for key in (
            "surface_ch3_fraction",
            "mean_footprint_ch3_enrichment",
            "mean_tpcl_ch3_enrichment",
            "mapped_frames",
        ):
            row[key] = site.get(key, math.nan)

        density_dir = resolved["interfacial_water_density"][0]
        assert density_dir is not None
        density_summary_path = density_dir / "summary.json"
        density_profile_path = density_dir / "water_density_profiles.csv"
        density = _read_json(density_summary_path)
        _source(source_rows, case, "interfacial_water_density", density_summary_path)
        _source(source_rows, case, "interfacial_water_density", density_profile_path)
        for profile in _read_csv(density_profile_path):
            density_rows.append({**_case_fields(case), **profile})
        for region, value in density.get("mean_hydration_areal_density_A-2", {}).items():
            row[f"hydration_{region}_A-2"] = value

        orientation_dir = resolved["interfacial_water_orientation"][0]
        assert orientation_dir is not None
        orientation_summary_path = orientation_dir / "summary.json"
        orientation_profile_path = orientation_dir / "orientation_profiles.csv"
        orientation = _read_json(orientation_summary_path)
        _source(source_rows, case, "interfacial_water_orientation", orientation_summary_path)
        _source(source_rows, case, "interfacial_water_orientation", orientation_profile_path)
        for profile in _read_csv(orientation_profile_path):
            orientation_rows.append({**_case_fields(case), **profile})
        for metric, value in orientation.get("mean_cos_theta", {}).items():
            row[f"orientation_{metric}"] = value
        for metric, value in orientation.get("orientation_sample_counts", {}).items():
            row[f"orientation_samples_{metric}"] = value

        hbond_dir = resolved["interfacial_water_hbond"][0]
        assert hbond_dir is not None
        hbond_path = hbond_dir / "summary.json"
        hbond = _read_json(hbond_path)
        _source(source_rows, case, "interfacial_water_hbond", hbond_path)
        for metric, value in hbond.get("mean_metrics", {}).items():
            row[f"hbond_{metric}"] = value

        proton_dir = resolved["surface_proton_transfer"][0]
        assert proton_dir is not None
        proton_path = proton_dir / "summary.json"
        proton = _read_json(proton_path)
        _source(source_rows, case, "surface_proton_transfer", proton_path)
        frames = int(proton.get("analyzed_frames", 0))
        row["proton_analyzed_frames"] = frames
        for candidate in ("h3o", "oh", "surface_site"):
            count = int(proton.get(f"frames_with_{candidate}_candidate", 0))
            row[f"proton_{candidate}_candidate_frame_fraction"] = count / frames if frames else math.nan
        row["persistent_solution_ion_candidate_event_count"] = proton.get(
            "persistent_solution_ion_candidate_event_count", 0
        )
        row["persistent_surface_site_event_count"] = proton.get(
            "persistent_surface_site_event_count", 0
        )

        if kind == "nanobubble":
            pre_dir = resolved["precontact_n2_enrichment"][0]
            if pre_dir is not None:
                pre_path = pre_dir / "summary.json"
                pre = _read_json(pre_path)
                _source(source_rows, case, "precontact_n2_enrichment", pre_path)
                for key in (
                    "analyzed_frames",
                    "frames_with_near_surface_disconnected_n2",
                    "frames_with_disconnected_n2_outside_bubble_projection",
                    "maximum_total_disconnected_n2",
                    "precontact_end_ns",
                ):
                    row[f"precontact_{key}"] = pre.get(key, math.nan)
        case_rows.append(row)

    preferred = ("case_id", "legend_label", "ch3_sites", "oh_sites", "ch3_fraction", "kind")
    _write_csv(output_dir / "case_summary.csv", case_rows, preferred)
    _write_csv(output_dir / "availability.csv", availability, preferred)
    _write_csv(output_dir / "source_manifest.csv", source_rows, preferred)
    _write_csv(output_dir / "core_dynamics.csv", core_rows, preferred)
    _write_csv(output_dir / "contact_line_blocks.csv", line_rows, preferred)
    _write_csv(output_dir / "water_density_profiles.csv", density_rows, preferred)
    _write_csv(output_dir / "orientation_profiles.csv", orientation_rows, preferred)

    quality_flags = [
        row
        for row in availability
        if row["quality_status"] in {"FAILED", "MISSING"}
    ]
    status = "PASS_WITH_QUALITY_FLAGS" if quality_flags else "PASS"
    figure_paths: list[Path] = []
    if make_plots:
        if font_path is None:
            raise ValueError("font_path is required when plots are enabled")
        figure_paths = _make_plots(
            cases,
            kind,
            case_rows,
            core_rows,
            line_rows,
            density_rows,
            orientation_rows,
            output_dir,
            Path(font_path),
            dpi,
            core_metrics,
        )
    summary = {
        "status": status,
        "kind": kind,
        "case_count": len(cases),
        "case_order": [case.case_id for case in cases],
        "legend_labels": {case.case_id: case.label for case in cases},
        "quality_flag_count": len(quality_flags),
        "quality_flags": quality_flags,
        "figures": [str(path) for path in figure_paths],
        "scientific_status": "DESCRIPTIVE_CROSS_SYSTEM_COMPARISON_NOT_EQUILIBRIUM_OR_CAUSAL_PROOF",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_report(output_dir, case_rows, summary)
    return summary


def _setup_matplotlib(font_path: Path) -> None:
    import matplotlib

    if not font_path.is_file() or font_path.stat().st_size == 0:
        raise FileNotFoundError(font_path)
    matplotlib.use("Agg", force=True)
    from matplotlib import font_manager

    font_manager.fontManager.addfont(str(font_path))
    family = font_manager.FontProperties(fname=str(font_path)).get_name()
    if family.lower() != "arial":
        raise ValueError(f"Expected Arial font, found {family!r}")
    matplotlib.rcParams.update(
        {
            "font.family": family,
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.8,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _style_axis(ax) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _add_header(fig, title: str, subtitle: str) -> None:
    fig.text(0.08, 0.98, title, ha="left", va="top", fontsize=13, fontweight="bold")
    fig.text(0.08, 0.945, subtitle, ha="left", va="top", fontsize=9, color=MUTED)


def _save_figure(fig, base: Path, dpi: int) -> list[Path]:
    png = base.with_suffix(".png")
    pdf = base.with_suffix(".pdf")
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    return [png, pdf]


def _group(rows: Sequence[Mapping[str, object]], key: str = "case_id") -> dict[str, list[Mapping[str, object]]]:
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row[key]), []).append(row)
    return grouped


def _case_style(index: int) -> dict[str, object]:
    return {
        "color": PALETTE[index % len(PALETTE)],
        "linestyle": LINESTYLES[index % len(LINESTYLES)],
        "marker": MARKERS[index % len(MARKERS)],
    }


def _make_plots(
    cases: Sequence[ComparisonCase],
    kind: str,
    case_rows: Sequence[Mapping[str, object]],
    core_rows: Sequence[Mapping[str, object]],
    line_rows: Sequence[Mapping[str, object]],
    density_rows: Sequence[Mapping[str, object]],
    orientation_rows: Sequence[Mapping[str, object]],
    output_dir: Path,
    font_path: Path,
    dpi: int,
    core_metrics: Sequence[str],
) -> list[Path]:
    _setup_matplotlib(font_path)
    import matplotlib.pyplot as plt

    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    core_grouped = _group(core_rows)
    line_grouped = _group(line_rows)
    density_grouped = _group(density_rows)
    orientation_grouped = _group(orientation_rows)
    row_by_id = {str(row["case_id"]): row for row in case_rows}

    core_labels = (
        (
            "Droplet center–surface distance (Å)",
            "Lateral radius, P₉₀ (Å)",
            "Height, q₀.₀₅–q₀.₉₅ (Å)",
            "Footprint area (Å²)",
        )
        if kind == "nanodroplet"
        else (
            "Bubble center–surface distance (Å)",
            "Lateral radius, P₉₀ (Å)",
            "Height, q₀.₀₅–q₀.₉₅ (Å)",
            "Contacting N₂ atoms",
        )
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.1), sharex=True)
    for index, case in enumerate(cases):
        values = core_grouped[case.case_id]
        style = _case_style(index)
        for ax, metric in zip(axes.flat, core_metrics):
            ax.plot(
                [_float(row["time_ns"]) for row in values],
                [_float(row[metric]) for row in values],
                label=case.label,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.35,
            )
    for ax, ylabel in zip(axes.flat, core_labels):
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Time (ns)")
        _style_axis(ax)
    axes[0, 0].legend(frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.02))
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.08, top=0.84, wspace=0.25, hspace=0.28)
    label = "Nanodroplet" if kind == "nanodroplet" else "Nanobubble"
    _add_header(fig, f"{label} core dynamics across surface terminations", "100 ps block means; all distances use the dynamic surface reference and periodic-boundary-aware centers.")
    paths.extend(_save_figure(fig, figures / "01_core_dynamics_comparison", dpi))
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.5))
    x = np.arange(len(cases))
    for index, case in enumerate(cases):
        row = row_by_id[case.case_id]
        style = _case_style(index)
        passed = str(row["contact_angle_quality_status"]).startswith("PASS")
        axes[0].errorbar(
            index,
            _float(row["contact_angle_deg"]),
            yerr=_float(row["contact_angle_std_deg"]),
            fmt=style["marker"],
            markersize=7,
            markerfacecolor=style["color"] if passed else "white",
            markeredgecolor=style["color"],
            color=style["color"],
            capsize=3,
            linewidth=1.1,
        )
        if not passed:
            axes[0].annotate("quality gate failed", (index, _float(row["contact_angle_deg"])), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=7, color=MUTED)
        values = line_grouped[case.case_id]
        axes[1].plot(
            [_float(row["time_since_attachment_ns"]) for row in values],
            [_float(row.get("mean_equivalent_radius_A")) for row in values],
            label=case.label,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markevery=max(1, len(values) // 8),
            markersize=3.5,
            linewidth=1.15,
        )
    axes[0].set_xticks(x, [case.label for case in cases], rotation=18, ha="right")
    axes[0].set_ylabel("Contact angle (°)")
    axes[0].set_xlabel("Surface termination")
    axes[1].set_ylabel("Equivalent contact-line radius (Å)")
    axes[1].set_xlabel("Time since attachment (ns)" if kind == "nanobubble" else "Time (ns)")
    axes[1].legend(frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.02))
    for ax in axes:
        _style_axis(ax)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.22, top=0.78, wspace=0.28)
    phase = "liquid-water angle" if kind == "nanodroplet" else "N₂-gas-side angle"
    _add_header(fig, f"{label} contact geometry across surface terminations", f"Left: final 2 ns block mean ± SD for the {phase}; right: contact-line block averages.")
    paths.extend(_save_figure(fig, figures / "02_contact_geometry_comparison", dpi))
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.2))
    for index, case in enumerate(cases):
        style = _case_style(index)
        density = [row for row in density_grouped[case.case_id] if row.get("region") == "tpcl"]
        orientation = [
            row
            for row in orientation_grouped[case.case_id]
            if row.get("region") == "tpcl" and row.get("observable") == "dipole"
        ]
        axes[0, 0].plot(
            [_float(row["z_A"]) for row in density],
            [_float(row["oxygen_number_density_A-3"]) for row in density],
            label=case.label,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.35,
        )
        axes[0, 1].plot(
            [_float(row["cos_theta"]) for row in orientation],
            [_float(row["probability_density"]) for row in orientation],
            label=case.label,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.35,
        )
    axes[0, 0].set_xlabel("Surface-relative z (Å)")
    axes[0, 0].set_ylabel("TPCL O number density (Å⁻³)")
    axes[0, 1].set_xlabel("cos θ (water dipole vs +z)")
    axes[0, 1].set_ylabel("TPCL probability density")
    axes[0, 0].legend(frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.02))

    regions = ("footprint", "tpcl", "far_field")
    region_labels = ("Footprint", "TPCL", "Far field")
    widths = 0.22
    for offset, (region, region_label) in enumerate(zip(regions, region_labels)):
        axes[1, 0].bar(
            x + (offset - 1) * widths,
            [_float(row_by_id[case.case_id].get(f"hydration_{region}_A-2")) for case in cases],
            width=widths,
            label=region_label,
            color=("#315B7D", "#B38728", "#D6D9DF")[offset],
            edgecolor=INK,
            linewidth=0.6,
        )
    metrics = (
        ("hbond_tpcl_water_water_hbond_degree", "Water–water degree"),
        ("hbond_tpcl_surface_water_hbond_per_h2o", "Surface–water/H₂O"),
    )
    for offset, (metric, metric_label) in enumerate(metrics):
        axes[1, 1].bar(
            x + (offset - 0.5) * 0.32,
            [_float(row_by_id[case.case_id].get(metric)) for case in cases],
            width=0.32,
            label=metric_label,
            color=("#315B7D", "#D06B3C")[offset],
            edgecolor=INK,
            linewidth=0.6,
        )
    for ax in (axes[1, 0], axes[1, 1]):
        ax.set_xticks(x, [case.label for case in cases], rotation=18, ha="right")
    axes[1, 0].set_ylabel("Hydration O areal density (Å⁻²)")
    axes[1, 1].set_ylabel("TPCL H-bond metric")
    axes[1, 0].legend(frameon=False, fontsize=8)
    axes[1, 1].legend(frameon=False, fontsize=8)
    for ax in axes.flat:
        _style_axis(ax)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.17, top=0.82, wspace=0.28, hspace=0.45)
    _add_header(fig, f"{label} interfacial-water structure across surface terminations", "Density, orientation, and geometric H-bond metrics use the same footprint/TPCL/far-field partition; H-bond lifetimes are not resolved.")
    paths.extend(_save_figure(fig, figures / "03_interfacial_water_comparison", dpi))
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.5))
    enrich = (
        ("mean_footprint_ch3_enrichment", "Footprint"),
        ("mean_tpcl_ch3_enrichment", "TPCL"),
    )
    for offset, (metric, metric_label) in enumerate(enrich):
        axes[0].bar(
            x + (offset - 0.5) * 0.32,
            [_float(row_by_id[case.case_id].get(metric)) for case in cases],
            width=0.32,
            label=metric_label,
            color=("#315B7D", "#B38728")[offset],
            edgecolor=INK,
            linewidth=0.6,
        )
    candidate_metrics = (
        ("proton_h3o_candidate_frame_fraction", "H₃O candidate"),
        ("proton_oh_candidate_frame_fraction", "OH candidate"),
        ("proton_surface_site_candidate_frame_fraction", "Surface-site candidate"),
    )
    for offset, (metric, metric_label) in enumerate(candidate_metrics):
        axes[1].plot(
            x,
            [_float(row_by_id[case.case_id].get(metric)) for case in cases],
            marker=("o", "s", "^")[offset],
            linestyle=("-", "--", "-.")[offset],
            color=("#315B7D", "#D06B3C", "#687A3C")[offset],
            label=metric_label,
            linewidth=1.2,
        )
    for ax in axes:
        ax.set_xticks(x, [case.label for case in cases], rotation=18, ha="right")
        _style_axis(ax)
    axes[0].axhline(0.0, color=INK, linewidth=0.8)
    axes[0].set_ylabel("CH₃ enrichment vs surface fraction")
    axes[1].set_ylabel("Fraction of analyzed frames")
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.22, top=0.78, wspace=0.28)
    _add_header(fig, f"{label} surface-chemistry screens across surface terminations", "Enrichment is a nearest-site boundary proxy; protonation states are sampled geometric candidates, not formal reaction assignments.")
    paths.extend(_save_figure(fig, figures / "04_surface_chemistry_comparison", dpi))
    plt.close(fig)
    return paths


def _write_report(
    output_dir: Path, case_rows: Sequence[Mapping[str, object]], summary: Mapping[str, object]
) -> None:
    lines = [
        f"# {str(summary['kind']).replace('nano', 'Nano').title()} Four-System Comparison",
        "",
        "## Stable legend labels",
        "",
        "Labels report the exact CH₃:OH termination counts in the shared 36-site source surface unit.",
        "",
        "| Case ID | Legend label | CH₃ fraction |",
        "|---|---|---:|",
    ]
    for row in case_rows:
        lines.append(
            f"| {row['case_id']} | {row['legend_label']} | {_float(row['ch3_fraction']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Contact and interface summary",
            "",
            "| Legend label | Angle / ° | Angle SD / ° | Angle gate | Valid contact line | TPCL hydration / Å⁻² | TPCL dipole cos θ | TPCL water–water degree |",
            "|---|---:|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in case_rows:
        lines.append(
            "| {label} | {angle:.2f} | {angle_sd:.2f} | {gate} | {valid:.1%} | {hydration:.4f} | {orientation:.4f} | {hbond:.4f} |".format(
                label=row["legend_label"],
                angle=_float(row.get("contact_angle_deg")),
                angle_sd=_float(row.get("contact_angle_std_deg")),
                gate=row.get("contact_angle_quality_status", ""),
                valid=_float(row.get("contact_line_valid_fraction")),
                hydration=_float(row.get("hydration_tpcl_A-2")),
                orientation=_float(row.get("orientation_tpcl_dipole")),
                hbond=_float(row.get("hbond_tpcl_water_water_hbond_degree")),
            )
        )
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- Comparisons are descriptive single-trajectory comparisons, not equilibrium or causal proofs.",
            "- Contact-angle points marked as failed remain visible only as candidates; their frozen quality gates are not relaxed.",
            "- Contact-line jumps are operational candidates, not pinning proofs.",
            "- H-bond metrics are geometric snapshots at the trajectory dump interval, not lifetimes.",
            "- Protonation and ion rows are sampled geometry candidates, not formal chemical identities.",
            "- `source_manifest.csv` records exact input paths, sizes, and SHA256 hashes.",
            "",
        ]
    )
    (output_dir / "comparison_report.md").write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--kind", choices=("nanodroplet", "nanobubble"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--font-path", type=Path)
    parser.add_argument("--block-frames", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = collect_comparison(
        load_cases(args.manifest),
        args.kind,
        args.output_dir,
        block_frames=args.block_frames,
        make_plots=not args.no_plots,
        font_path=args.font_path,
        dpi=args.dpi,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
