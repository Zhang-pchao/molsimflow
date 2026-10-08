"""Charge-selective mobile-ion microstates in a water-containing bubble film."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from molsimflow.postprocess.coupled_expulsion import MOBILE_SPECIES, in_confined_film


STATES = ("no_mobile_ion", "anion_only", "cation_only", "both_signs")


def _pandas():
    import pandas as pd

    return pd


def charge_state(n_positive: int, n_negative: int) -> str:
    """Classify one film frame by the signs of its mobile ions."""

    if n_positive and n_negative:
        return "both_signs"
    if n_positive:
        return "cation_only"
    if n_negative:
        return "anion_only"
    return "no_mobile_ion"


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys or ["status"])
        writer.writeheader()
        writer.writerows(rows)


def _manifest_row(path: Path, case_label: str) -> Mapping[str, str]:
    with path.open(newline="") as handle:
        matches = [row for row in csv.DictReader(handle) if row["case_label"] == case_label]
    if len(matches) != 1:
        raise ValueError(f"Expected one row for {case_label}, found {len(matches)}")
    return matches[0]


def _metrics(frame, weights: np.ndarray) -> dict[str, float]:
    wet = frame.n_water_film.to_numpy(float) > 0
    wet_weight = float(weights[wet].sum())
    water_weight = float(np.sum(weights[wet] * frame.loc[wet, "n_water_film"].to_numpy(float)))
    positive = frame.n_positive_film.to_numpy(float)
    negative = frame.n_negative_film.to_numpy(float)
    ion_total = float(np.sum(weights * (positive + negative)))
    values = {
        f"{state}_probability": (
            float(np.sum(weights[wet] * frame.loc[wet, "charge_state"].eq(state).to_numpy(float)) / wet_weight)
            if wet_weight else math.nan
        )
        for state in STATES
    }
    values.update(
        {
            "positive_ions_per_100_water": 100.0 * float(np.sum(weights * positive)) / water_weight if water_weight else math.nan,
            "negative_ions_per_100_water": 100.0 * float(np.sum(weights * negative)) / water_weight if water_weight else math.nan,
            "anion_fraction_of_film_ions": float(np.sum(weights * negative)) / ion_total if ion_total else math.nan,
            "charge_purity": float(np.sum(weights * np.abs(positive - negative))) / ion_total if ion_total else math.nan,
        }
    )
    return values


def summarize(frame, block_ns: float, bootstrap_samples: int, random_seed: int):
    """Return gap-conditioned point estimates and joint block-bootstrap intervals."""

    pd = _pandas()
    data = frame.copy()
    data["block_id"] = np.floor(data.time_ns.to_numpy(float) / block_ns + 1.0e-10).astype(int)
    rng = np.random.default_rng(random_seed)
    rows = []
    for gap_bin, chunk in data.groupby("gap_bin", sort=False):
        blocks = np.sort(chunk.block_id.unique())
        point = _metrics(chunk, np.ones(len(chunk)))
        draws = {metric: [] for metric in point}
        for _ in range(bootstrap_samples):
            multiplicity = Counter(rng.choice(blocks, size=len(blocks), replace=True).tolist())
            weights = chunk.block_id.map(multiplicity).to_numpy(float)
            sampled = _metrics(chunk, weights)
            for metric, value in sampled.items():
                if math.isfinite(value):
                    draws[metric].append(value)
        left = float(str(gap_bin).split("-", 1)[0])
        for metric, value in point.items():
            samples = np.asarray(draws[metric], dtype=float)
            rows.append(
                {
                    "case_label": str(chunk.case_label.iloc[0]),
                    "gap_bin": gap_bin,
                    "gap_center_A": left + 1.0,
                    "frame_count": len(chunk),
                    "wet_frame_count": int((chunk.n_water_film > 0).sum()),
                    "occupied_wet_frame_count": int(((chunk.n_water_film > 0) & ((chunk.n_positive_film + chunk.n_negative_film) > 0)).sum()),
                    "effective_block_count": len(blocks),
                    "metric": metric,
                    "mean": value,
                    "ci95_low": float(np.quantile(samples, 0.025)) if len(samples) >= 20 else math.nan,
                    "ci95_high": float(np.quantile(samples, 0.975)) if len(samples) >= 20 else math.nan,
                    "bootstrap_draw_count": len(samples),
                }
            )
    return pd.DataFrame(rows)


def analyze_case(args) -> int:
    pd = _pandas()
    case = _manifest_row(Path(args.case_manifest), args.case_label)
    if not case.get("ion_samples_csv", "").strip():
        raise ValueError(f"{args.case_label} has no classified-ion table")
    radius_a = float(case["nominal_radius_a_A"])
    radius_b = float(case["nominal_radius_b_A"])
    frames = pd.read_csv(case["coupled_frame_csv"])
    ions = pd.read_csv(case["ion_samples_csv"])
    ions = ions[ions.species.isin(MOBILE_SPECIES)].copy()
    all_mobile = ions.copy()
    ions = ions[(ions.r_A_A >= radius_a) & (ions.r_B_A >= radius_b)].copy()
    ions["strict_film"] = in_confined_film(
        ions.s_A,
        ions.rho_A,
        ions.gap_A,
        radius_a,
        radius_b,
        args.rho_core_A,
    )
    film = ions[ions.strict_film].copy()
    film["positive"] = (film.formal_charge_e > 0).astype(int)
    film["negative"] = (film.formal_charge_e < 0).astype(int)
    counts = film.groupby("global_frame")[["positive", "negative"]].sum()
    frames = frames.merge(counts, how="left", left_on="global_frame", right_index=True)
    frames[["positive", "negative"]] = frames[["positive", "negative"]].fillna(0).astype(int)
    frames = frames.rename(columns={"positive": "n_positive_film", "negative": "n_negative_film"})
    frames["film_net_formal_charge_e"] = frames.n_positive_film - frames.n_negative_film
    frames["film_abs_formal_charge_e"] = frames.n_positive_film + frames.n_negative_film
    frames["charge_state"] = [charge_state(pos, neg) for pos, neg in zip(frames.n_positive_film, frames.n_negative_film)]
    frames["case_label"] = args.case_label
    keep = [
        "case_label", "segment", "local_frame", "global_frame", "timestep", "time_ns", "gap_A", "gap_bin",
        "n_water_film", "n_positive_film", "n_negative_film", "film_net_formal_charge_e",
        "film_abs_formal_charge_e", "charge_state",
    ]
    output = Path(args.output_root) / args.case_label
    output.mkdir(parents=True, exist_ok=True)
    frame_path = output / "charge_state_frame_summary.csv"
    gap_path = output / "charge_state_gap_summary.csv"
    state_path = output / "charge_state_counts.csv"
    sensitivity_path = output / "radius_sensitivity_summary.csv"
    frames[keep].to_csv(frame_path, index=False)
    summarize(frames[keep], args.block_ns, args.bootstrap_samples, args.random_seed).to_csv(gap_path, index=False)
    state_counts = (
        frames[frames.n_water_film > 0]
        .groupby(["gap_bin", "charge_state"], observed=True)
        .size()
        .rename("frame_count")
        .reset_index()
    )
    state_counts.insert(0, "case_label", args.case_label)
    state_counts.to_csv(state_path, index=False)
    sensitivity_rows = []
    center_distance = all_mobile.gap_A + radius_a + radius_b
    for radius in args.sensitivity_radius_A:
        sensitivity_gap = center_distance - 2.0 * radius
        selected = all_mobile[in_confined_film(all_mobile.s_A, all_mobile.rho_A, sensitivity_gap, radius, radius, args.rho_core_A)]
        for gap_bin in sorted(frames.gap_bin.unique(), key=lambda value: float(str(value).split("-", 1)[0])):
            chunk = selected[selected.gap_bin == gap_bin]
            sensitivity_rows.append(
                {
                    "case_label": args.case_label,
                    "nominal_surface_radius_A": radius,
                    "gap_bin_at_19A": gap_bin,
                    "positive_observations": int((chunk.formal_charge_e > 0).sum()),
                    "negative_observations": int((chunk.formal_charge_e < 0).sum()),
                    "total_mobile_observations": len(chunk),
                }
            )
    _write_csv(sensitivity_path, sensitivity_rows)
    artifact_rows = []
    for path in (frame_path, gap_path, state_path, sensitivity_path):
        artifact_rows.append({"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    _write_csv(output / "artifact_manifest.csv", artifact_rows)
    occupied = int((frames.film_abs_formal_charge_e > 0).sum())
    print(f"case={args.case_label} frames={len(frames)} occupied={occupied} output={output}")
    return 0


def assemble(args) -> int:
    pd = _pandas()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with Path(args.case_manifest).open(newline="") as handle:
        cases = list(csv.DictReader(handle))
    analyzed = [row["case_label"] for row in cases if row.get("ion_samples_csv", "").strip()]
    summaries = pd.concat(
        [pd.read_csv(Path(args.output_root) / label / "charge_state_gap_summary.csv") for label in analyzed],
        ignore_index=True,
    )
    frames = pd.concat(
        [pd.read_csv(Path(args.output_root) / label / "charge_state_frame_summary.csv") for label in analyzed],
        ignore_index=True,
    )
    sensitivity = pd.concat(
        [pd.read_csv(Path(args.output_root) / label / "radius_sensitivity_summary.csv") for label in analyzed],
        ignore_index=True,
    )
    figure_dir = Path(args.figure_dir)
    plot_dir = figure_dir / "plot_data"
    plot_dir.mkdir(parents=True, exist_ok=True)
    summaries.to_csv(plot_dir / "multicase_charge_state_gap_summary.csv", index=False)
    frames.to_csv(plot_dir / "multicase_charge_state_frame_summary.csv", index=False)
    sensitivity.to_csv(plot_dir / "multicase_radius_sensitivity_summary.csv", index=False)

    inventory = []
    for case in cases:
        label = case["case_label"]
        q = frames[frames.case_label == label]
        inventory.append(
            {
                "case_label": label,
                "classified_ion_source": "available" if case.get("ion_samples_csv", "").strip() else "N/A",
                "canonical_frame_count": len(q) if len(q) else (pd.read_csv(case["coupled_frame_csv"]).shape[0]),
                "wet_frame_count": int((q.n_water_film > 0).sum()) if len(q) else "N/A",
                "occupied_wet_frame_count": int(((q.n_water_film > 0) & (q.film_abs_formal_charge_e > 0)).sum()) if len(q) else "N/A",
            }
        )
    _write_csv(plot_dir / "general_case_inventory.csv", inventory)

    state_colors = {
        "no_mobile_ion": "#bdbdbd",
        "anion_only": "#2c7fb8",
        "cation_only": "#d95f5f",
        "both_signs": "#756bb1",
    }
    state_labels = {
        "no_mobile_ion": "no mobile ion",
        "anion_only": "anion only",
        "cation_only": "cation only",
        "both_signs": "both signs",
    }
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.09, top=0.79, hspace=0.30, wspace=0.18)
    panel_labels = "abcd"
    for index, (ax, case_label) in enumerate(zip(axes.flat, analyzed)):
        for state in STATES:
            q = summaries[(summaries.case_label == case_label) & (summaries.metric == f"{state}_probability") & (summaries.gap_center_A < 18)].sort_values("gap_center_A")
            supported = q.effective_block_count >= 4
            ax.plot(q.gap_center_A, q["mean"], color=state_colors[state], lw=1.7, label=state_labels[state])
            ax.fill_between(q.gap_center_A, q.ci95_low, q.ci95_high, color=state_colors[state], alpha=0.10)
            ax.scatter(q.loc[supported, "gap_center_A"], q.loc[supported, "mean"], color=state_colors[state], s=25, zorder=3)
            ax.scatter(q.loc[~supported, "gap_center_A"], q.loc[~supported, "mean"], facecolor="white", edgecolor=state_colors[state], s=32, zorder=3)
        ax.set_title(f"{panel_labels[index]}  {case_label}")
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlim(0, 18)
        ax.grid(axis="y", color="#dddddd", lw=0.7)
        if index >= 2:
            ax.set_xlabel("Nominal gap h (Å)")
        if index % 2 == 0:
            ax.set_ylabel("P(charge state | water remains)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.89))
    fig.suptitle(
        "Charge-selective mobile-ion retention in the confined wet film\n"
        "S systems; strict 3-D film; 20 ps joint block bootstrap; open markers have <4 blocks; Bulk-water-S is N/A",
        fontsize=14,
        y=0.985,
    )
    png = figure_dir / "candidate_fig02_charge_selective_retention.png"
    pdf = figure_dir / "candidate_fig02_charge_selective_retention.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    support = summaries[["case_label", "gap_bin", "frame_count", "wet_frame_count", "occupied_wet_frame_count", "effective_block_count"]].drop_duplicates()
    support.to_csv(plot_dir / "frame_block_support.csv", index=False)
    study_root = figure_dir.parent.parent
    source_rows = []
    for case in cases:
        for role in ("coupled_frame_csv", "ion_samples_csv"):
            raw_path = case.get(role, "").strip()
            if not raw_path:
                continue
            path = Path(raw_path)
            source_rows.append(
                {
                    "case_label": case["case_label"],
                    "role": role,
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "mtime_ns": path.stat().st_mtime_ns,
                }
            )
    _write_csv(study_root / "manifests" / "source_manifest.csv", source_rows)
    report = study_root / "reports" / "VALIDATION.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "# Figure 2 validation\n\n"
        f"- General S cases: {', '.join(row['case_label'] for row in cases)}\n"
        f"- Cases with classified-ion tables: {', '.join(analyzed)}\n"
        f"- Canonical analyzed frames: {len(frames)}\n"
        f"- Wet frames: {int((frames.n_water_film > 0).sum())}\n"
        f"- Occupied wet-film frames: {int(((frames.n_water_film > 0) & (frames.film_abs_formal_charge_e > 0)).sum())}\n"
        "- State probabilities sum to one within numerical tolerance for every case/gap bin.\n"
        "- Bulk-water-S ion quantities are N/A, not zero.\n"
        "- Formal species charges are a classification proxy, not exact electrostatics.\n"
        "- Radius sensitivity is tabulated at 17, 18, 19, and 20 A at fixed bubble-center distance.\n"
        "- Gap bins are structural conditional ensembles, not kinetic time ordering.\n"
        "- Figure status: candidate for review; manuscript untouched.\n",
        encoding="utf-8",
    )
    artifacts = [png, pdf, report, *plot_dir.glob("*.csv")]
    _write_csv(study_root / "manifests" / "artifact_manifest.csv", [
        {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in sorted(artifacts)
    ])
    print(f"FIG02_VALIDATION_OK cases={len(cases)} analyzed={len(analyzed)} frames={len(frames)} figure={png}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    case = sub.add_parser("analyze-case")
    case.add_argument("--case-manifest", required=True)
    case.add_argument("--case-label", required=True)
    case.add_argument("--output-root", required=True)
    case.add_argument("--rho-core-A", type=float, default=6.0)
    case.add_argument("--sensitivity-radius-A", type=float, nargs="+", default=(17.0, 18.0, 19.0, 20.0))
    case.add_argument("--block-ns", type=float, default=0.020)
    case.add_argument("--bootstrap-samples", type=int, default=1000)
    case.add_argument("--random-seed", type=int, default=20260822)
    case.set_defaults(func=analyze_case)
    assembly = sub.add_parser("assemble")
    assembly.add_argument("--case-manifest", required=True)
    assembly.add_argument("--output-root", required=True)
    assembly.add_argument("--figure-dir", required=True)
    assembly.set_defaults(func=assemble)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
