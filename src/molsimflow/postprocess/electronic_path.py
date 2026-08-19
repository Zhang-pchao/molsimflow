"""Build geometry-gated charge and spin profiles along configured reaction paths."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


def _read_table(path: Path) -> List[Dict[str, str]]:
    delimiter = "\t" if Path(path).suffix.lower() in {".tsv", ".tab"} else ","
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"Table has no header: {path}")
        return [dict(row) for row in reader]


def _require_columns(rows: Sequence[Mapping[str, str]], columns: Sequence[str], name: str) -> None:
    if not rows:
        raise ValueError(f"{name} is empty")
    missing = [column for column in columns if column not in rows[0]]
    if missing:
        raise ValueError(f"{name} is missing columns: {', '.join(missing)}")


def _write_tsv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        if not rows:
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _optional_float(value: object) -> Optional[float]:
    text = str(value).strip()
    return float(text) if text else None


def _summary(values: Sequence[float]) -> Dict[str, object]:
    if not values:
        return {"mean": "", "sample_sd": "", "minimum": "", "maximum": ""}
    return {
        "mean": mean(values),
        "sample_sd": stdev(values) if len(values) > 1 else "",
        "minimum": min(values),
        "maximum": max(values),
    }


def _load_config(path: Path) -> Mapping[str, object]:
    payload = json.loads(Path(path).read_text())
    systems = payload.get("systems")
    if not isinstance(systems, dict) or not systems:
        raise ValueError("Config must define a non-empty 'systems' mapping")
    return payload


def build_electronic_path_profiles(
    atom_table_path: Path,
    frame_table_path: Path,
    config_path: Path,
    output_dir: Path,
    make_plots: bool = True,
) -> Dict[str, Path]:
    """Create frame-level and state-level electronic profiles with geometry admission gates."""

    atom_rows = _read_table(atom_table_path)
    frame_rows = _read_table(frame_table_path)
    _require_columns(
        atom_rows,
        [
            "system",
            "stratum",
            "replicate",
            "atom_index_one_based",
            "charge",
            "spin",
        ],
        "Atom table",
    )
    _require_columns(
        frame_rows,
        [
            "system",
            "stratum",
            "replicate",
            "geometry_state",
            "organic_charge",
            "organic_spin",
        ],
        "Frame table",
    )
    config = _load_config(config_path)

    atom_index: Dict[Tuple[str, str, str, int], Mapping[str, str]] = {}
    for row in atom_rows:
        key = (
            row["system"],
            row["stratum"],
            row["replicate"],
            int(row["atom_index_one_based"]),
        )
        if key in atom_index:
            raise ValueError(f"Duplicate atom-table key: {key}")
        atom_index[key] = row

    frames_by_stratum: Dict[Tuple[str, str], List[Mapping[str, str]]] = defaultdict(list)
    for row in frame_rows:
        frames_by_stratum[(row["system"], row["stratum"])].append(row)

    long_rows: List[Dict[str, object]] = []
    route_order: List[Tuple[str, str]] = []
    systems = config["systems"]
    assert isinstance(systems, dict)
    for system, system_values in systems.items():
        if not isinstance(system_values, dict):
            raise ValueError(f"System config must be a mapping: {system}")
        atom_labels = system_values.get("atom_labels")
        routes = system_values.get("routes")
        if not isinstance(atom_labels, dict) or not atom_labels:
            raise ValueError(f"System {system} must define atom_labels")
        if not isinstance(routes, dict) or not routes:
            raise ValueError(f"System {system} must define routes")
        for route, route_values in routes.items():
            if not isinstance(route_values, dict) or not isinstance(route_values.get("states"), list):
                raise ValueError(f"Route {system}/{route} must define a states list")
            route_order.append((str(system), str(route)))
            for state_order, state in enumerate(route_values["states"]):
                if not isinstance(state, dict):
                    raise ValueError(f"State entries must be mappings: {system}/{route}")
                state_label = str(state["label"])
                stratum = str(state["stratum"])
                expected_geometry = str(state.get("expected_geometry_state", "")).strip()
                selected_frames = sorted(
                    frames_by_stratum.get((str(system), stratum), []),
                    key=lambda row: row["replicate"],
                )
                if not selected_frames:
                    raise ValueError(f"No frames for configured stratum: {system}/{stratum}")
                for frame in selected_frames:
                    observed_geometry = frame["geometry_state"]
                    admitted = not expected_geometry or observed_geometry == expected_geometry
                    common = {
                        "system": system,
                        "route": route,
                        "state_order": state_order,
                        "state_label": state_label,
                        "stratum": stratum,
                        "replicate": frame["replicate"],
                        "expected_geometry_state": expected_geometry,
                        "observed_geometry_state": observed_geometry,
                        "admission_status": "admitted" if admitted else "excluded_geometry_mismatch",
                    }
                    for atom_label, atom_number in atom_labels.items():
                        key = (str(system), stratum, frame["replicate"], int(atom_number))
                        if key not in atom_index:
                            raise ValueError(f"Configured atom is absent from atom table: {key}")
                        atom = atom_index[key]
                        long_rows.append(
                            {
                                **common,
                                "descriptor_label": atom_label,
                                "atom_index_one_based": int(atom_number),
                                "element": atom.get("element", ""),
                                "charge": float(atom["charge"]),
                                "spin": float(atom["spin"]),
                            }
                        )
                    long_rows.append(
                        {
                            **common,
                            "descriptor_label": "organic_total",
                            "atom_index_one_based": "",
                            "element": "organic_fragment",
                            "charge": float(frame["organic_charge"]),
                            "spin": float(frame["organic_spin"]),
                        }
                    )

    grouped: Dict[Tuple[str, str, int, str, str], List[Mapping[str, object]]] = defaultdict(list)
    for row in long_rows:
        grouped[
            (
                str(row["system"]),
                str(row["route"]),
                int(row["state_order"]),
                str(row["state_label"]),
                str(row["descriptor_label"]),
            )
        ].append(row)

    summary_rows: List[Dict[str, object]] = []
    for key, rows in sorted(grouped.items()):
        admitted = [row for row in rows if row["admission_status"] == "admitted"]
        charge_all = [float(row["charge"]) for row in rows]
        spin_all = [float(row["spin"]) for row in rows]
        charge_admitted = [float(row["charge"]) for row in admitted]
        spin_admitted = [float(row["spin"]) for row in admitted]
        all_charge_summary = _summary(charge_all)
        all_spin_summary = _summary(spin_all)
        admitted_charge_summary = _summary(charge_admitted)
        admitted_spin_summary = _summary(spin_admitted)
        first = rows[0]
        summary_rows.append(
            {
                "system": key[0],
                "route": key[1],
                "state_order": key[2],
                "state_label": key[3],
                "stratum": first["stratum"],
                "expected_geometry_state": first["expected_geometry_state"],
                "observed_geometry_states": ";".join(
                    sorted({str(row["observed_geometry_state"]) for row in rows})
                ),
                "descriptor_label": key[4],
                "atom_index_one_based": first["atom_index_one_based"],
                "element": first["element"],
                "n_total": len(rows),
                "n_admitted": len(admitted),
                "state_status": "admitted" if admitted else "no_admitted_frames",
                "mean_charge_all": all_charge_summary["mean"],
                "sample_sd_charge_all": all_charge_summary["sample_sd"],
                "min_charge_all": all_charge_summary["minimum"],
                "max_charge_all": all_charge_summary["maximum"],
                "mean_spin_all": all_spin_summary["mean"],
                "sample_sd_spin_all": all_spin_summary["sample_sd"],
                "mean_charge_admitted": admitted_charge_summary["mean"],
                "sample_sd_charge_admitted": admitted_charge_summary["sample_sd"],
                "min_charge_admitted": admitted_charge_summary["minimum"],
                "max_charge_admitted": admitted_charge_summary["maximum"],
                "mean_spin_admitted": admitted_spin_summary["mean"],
                "sample_sd_spin_admitted": admitted_spin_summary["sample_sd"],
            }
        )

    output_dir = Path(output_dir)
    outputs = {
        "frames": output_dir / "electronic_path_frame_profiles.tsv",
        "summary": output_dir / "electronic_path_state_summary.tsv",
        "metadata": output_dir / "electronic_path_metadata.json",
    }
    _write_tsv(outputs["frames"], long_rows)
    _write_tsv(outputs["summary"], summary_rows)
    outputs["metadata"].write_text(
        json.dumps(
            {
                "analysis_scope": (
                    "Configured path-level Hirshfeld charge/spin summaries with geometry-based "
                    "frame admission; descriptors are not kinetics or committor assignments."
                ),
                "atom_table_path": str(Path(atom_table_path).resolve()),
                "frame_table_path": str(Path(frame_table_path).resolve()),
                "config_path": str(Path(config_path).resolve()),
                "frame_descriptor_rows": len(long_rows),
                "state_descriptor_rows": len(summary_rows),
                "route_order": route_order,
            },
            indent=2,
        )
        + "\n"
    )
    if make_plots:
        outputs.update(_plot_profiles(summary_rows, route_order, output_dir))
    return outputs


def _plot_profiles(
    summary_rows: Sequence[Mapping[str, object]],
    route_order: Sequence[Tuple[str, str]],
    output_dir: Path,
) -> Dict[str, Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Plotting requires matplotlib; use --no-plots for table-only output") from exc

    by_route: Dict[Tuple[str, str], List[Mapping[str, object]]] = defaultdict(list)
    for row in summary_rows:
        by_route[(str(row["system"]), str(row["route"]))].append(row)

    figure, axes = plt.subplots(1, len(route_order), figsize=(5.2 * len(route_order), 4.6), squeeze=False)
    for axis, route_key in zip(axes[0], route_order):
        rows = by_route[route_key]
        atom_labels = sorted(
            {str(row["descriptor_label"]) for row in rows if row["descriptor_label"] != "organic_total"}
        )
        states = sorted({(int(row["state_order"]), str(row["state_label"])) for row in rows})
        for atom_label in atom_labels:
            selected = {
                int(row["state_order"]): row
                for row in rows
                if row["descriptor_label"] == atom_label
            }
            x_values, means, errors = [], [], []
            for state_order, _ in states:
                row = selected[state_order]
                value = _optional_float(row["mean_charge_admitted"])
                if value is None:
                    continue
                lower = _optional_float(row["min_charge_admitted"])
                upper = _optional_float(row["max_charge_admitted"])
                x_values.append(state_order)
                means.append(value)
                errors.append((value - lower, upper - value))
            if x_values:
                axis.errorbar(
                    x_values,
                    means,
                    yerr=[[value[0] for value in errors], [value[1] for value in errors]],
                    marker="o",
                    capsize=3,
                    label=atom_label,
                )
        for state_order, state_label in states:
            state_rows = [row for row in rows if int(row["state_order"]) == state_order]
            if state_rows and all(row["state_status"] == "no_admitted_frames" for row in state_rows):
                axis.scatter(
                    [state_order],
                    [0.04],
                    marker="x",
                    color="black",
                    transform=axis.get_xaxis_transform(),
                    clip_on=False,
                )
                axis.text(
                    state_order,
                    0.08,
                    "geometry excluded",
                    rotation=90,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    transform=axis.get_xaxis_transform(),
                )
        axis.axhline(0.0, color="0.7", linewidth=0.8, linestyle="--")
        axis.set_xticks([order for order, _ in states])
        axis.set_xticklabels([label for _, label in states], rotation=35, ha="right")
        axis.set_ylabel("Hirshfeld charge (e)")
        axis.set_title(f"{route_key[0]}: {route_key[1]}")
        axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    charge_path = output_dir / "electronic_path_atomic_charge_profiles.png"
    figure.savefig(charge_path, dpi=240, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, len(route_order), figsize=(5.2 * len(route_order), 4.6), squeeze=False)
    for axis, route_key in zip(axes[0], route_order):
        rows = [
            row
            for row in by_route[route_key]
            if row["descriptor_label"] == "organic_total"
        ]
        rows.sort(key=lambda row: int(row["state_order"]))
        admitted = [row for row in rows if row["state_status"] == "admitted"]
        x_values = [int(row["state_order"]) for row in admitted]
        charges = [float(row["mean_charge_admitted"]) for row in admitted]
        spins = [float(row["mean_spin_admitted"]) for row in admitted]
        axis.plot(x_values, charges, marker="o", color="#D95F02", label="organic charge")
        axis.set_ylabel("Organic-fragment charge (e)", color="#D95F02")
        axis.tick_params(axis="y", labelcolor="#D95F02")
        twin = axis.twinx()
        twin.plot(x_values, spins, marker="s", color="#1B9E77", label="organic spin")
        twin.set_ylabel("Organic-fragment spin (e)", color="#1B9E77")
        twin.tick_params(axis="y", labelcolor="#1B9E77")
        axis.axhline(0.0, color="0.7", linewidth=0.8, linestyle="--")
        axis.set_xticks([int(row["state_order"]) for row in rows])
        axis.set_xticklabels([str(row["state_label"]) for row in rows], rotation=35, ha="right")
        axis.set_title(f"{route_key[0]}: {route_key[1]}")
    figure.tight_layout()
    fragment_path = output_dir / "electronic_path_organic_charge_spin_profiles.png"
    figure.savefig(fragment_path, dpi=240, bbox_inches="tight")
    plt.close(figure)
    return {"atomic_charge_plot": charge_path, "organic_charge_spin_plot": fragment_path}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--atom-table", type=Path, required=True, help="Atom-level CSV or TSV"
    )
    parser.add_argument(
        "--frame-table", type=Path, required=True, help="Frame-level CSV or TSV"
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Path-profile JSON configuration"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Directory for tables and plots"
    )
    parser.add_argument(
        "--no-plots", action="store_true", help="Write tables without matplotlib plots"
    )
    args = parser.parse_args(argv)
    outputs = build_electronic_path_profiles(
        args.atom_table,
        args.frame_table,
        args.config,
        args.output_dir,
        make_plots=not args.no_plots,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
