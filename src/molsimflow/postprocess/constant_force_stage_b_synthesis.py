"""Synthesize Stage-B transport mechanisms while preserving scientific gate boundaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from molsimflow.postprocess.constant_force_oxygen import (
    resolve_path,
    sha256,
    write_output_hashes,
    write_tsv,
)


def _read_table(path: Path) -> list[dict[str, str]]:
    delimiter = "\t" if path.suffix in {".tsv", ".tab"} else ","
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def _float(value: object) -> float:
    return float(str(value))


def _verify_hashes(root: Path) -> None:
    manifest = root / "OUTPUT-SHA256SUMS"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    marker = f"/{root.name}/"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, raw_path = line.split(maxsplit=1)
        raw_path = raw_path.strip().removeprefix("./")
        path = Path(raw_path)
        if not path.is_absolute():
            path = root / path
        elif not path.exists() and marker in raw_path:
            path = root / raw_path.split(marker, 1)[1]
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Source result hash mismatch: {path}")


def classify_exchange(rows: Sequence[Mapping[str, str]]) -> tuple[str, float, str]:
    selected = [row for row in rows if row["category"] == "PERSISTENT_ISLAND_TRANSFER"]
    if len(selected) != 2:
        raise ValueError("Expected X and Y persistent-island-transfer rows")
    maximum = max(abs(_float(row["signed_fraction_of_total_response"])) for row in selected)
    signs = [
        math.copysign(1.0, _float(row["response_velocity_mps"]))
        == math.copysign(1.0, _float(row["total_response_velocity_mps"]))
        for row in selected
    ]
    if maximum > 0.5 and all(signs):
        decision = "EXCHANGE_DOMINATED_CANDIDATE"
    elif maximum >= 0.2 and all(signs):
        decision = "COUPLED_EXCHANGE_ADVECTION"
    else:
        decision = "EXCHANGE_CONTRIBUTION_LIMITED"
    detail = (
        f"maximum absolute signed fraction={maximum:.6f}; same-sign axes={sum(signs)}/{len(signs)}"
    )
    return decision, maximum, detail


def _window_signs(
    rows: Sequence[Mapping[str, str]],
    *,
    branch_id: str,
    layer_index: int,
    window_ps: float = 1000.0,
) -> list[int]:
    selected = [
        row
        for row in rows
        if row["branch_id"] == branch_id
        and int(row["layer_index"]) == layer_index
        and _float(row["mean_count"]) >= 1.0
    ]
    grouped: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for row in selected:
        index = int(_float(row["start_ps"]) // window_ps)
        grouped[index].append(
            (_float(row["raw_axis_velocity_mps"]), _float(row["molecule_samples"]))
        )
    output = []
    for index in sorted(grouped):
        values = grouped[index]
        denominator = sum(weight for _, weight in values)
        mean = sum(value * weight for value, weight in values) / denominator
        output.append(1 if mean > 0 else -1 if mean < 0 else 0)
    return output


def classify_layer_response(rows: Sequence[Mapping[str, str]]) -> tuple[str, str]:
    y_layers = sorted(
        {
            int(row["layer_index"])
            for row in rows
            if row["branch_id"] == "f8e-5_y" and _float(row["mean_count"]) >= 1.0
        }
    )
    signs = {
        layer: _window_signs(rows, branch_id="f8e-5_y", layer_index=layer) for layer in y_layers
    }
    persistent_pair = None
    for left in y_layers:
        for right in y_layers:
            if left >= right or len(signs[left]) != 4 or len(signs[right]) != 4:
                continue
            if all(value > 0 for value in signs[left]) and all(value < 0 for value in signs[right]):
                persistent_pair = (left, right)
            if all(value < 0 for value in signs[left]) and all(value > 0 for value in signs[right]):
                persistent_pair = (left, right)
    if persistent_pair:
        decision = "TIME_DEPENDENT_LAYER_OPPOSED_Y_CANDIDATE"
        detail = (
            f"layers {persistent_pair[0]} and {persistent_pair[1]} retain opposite raw signs "
            "across four 1-ns windows; other layers and whole-film response remain time-dependent"
        )
    else:
        decision = "STATIONARY_COUNTERFLOW_NOT_SUPPORTED"
        detail = "no persistently opposite occupied-layer pair across four 1-ns windows"
    return decision, detail


def classify_tpcl(
    anchor_summary: Sequence[Mapping[str, str]],
    matched_rows: Sequence[Mapping[str, str]],
) -> tuple[str, str]:
    mixed = [row for row in anchor_summary if row["case_id"] == "mixed291"]
    retention = np.asarray(
        [_float(row["mean_anchor_pair_retained_fraction"]) for row in mixed],
        dtype=float,
    )
    field = "event_minus_control_delta_anchor_pair_retained_fraction"
    contrasts = [
        _float(row[field])
        for row in matched_rows
        if row["case_id"] == "mixed291" and math.isfinite(_float(row[field]))
    ]
    positive = sum(value > 0 for value in contrasts)
    negative = sum(value < 0 for value in contrasts)
    decision = "DYNAMIC_ANCHOR_ASSOCIATION_ONLY"
    detail = (
        f"mixed291 mean 10-ps retained fraction range={np.nanmin(retention):.3f}-"
        f"{np.nanmax(retention):.3f}; matched event contrasts positive/negative="
        f"{positive}/{negative}; no independent replicate or high-frequency lifetime"
    )
    return decision, detail


def _plot_gate_map(
    rows: Sequence[Mapping[str, object]], output: Path, font_path: Path | None
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager
    from matplotlib import pyplot as plt

    if font_path is not None:
        font_manager.fontManager.addfont(font_path)
        properties = font_manager.FontProperties(fname=font_path)
        font_manager.findfont(properties, fallback_to_default=False)
        matplotlib.rcParams["font.family"] = properties.get_name()
    gates = (
        "numerical_gate",
        "association_gate",
        "replicate_gate",
        "reverse_force_gate",
        "intervention_gate",
    )
    labels = {
        "PASS": 1.0,
        "PARTIAL": 0.5,
        "NOT_TESTED": 0.0,
        "FAIL": -1.0,
    }
    matrix = np.asarray([[labels[str(row[gate])] for gate in gates] for row in rows])
    figure, axis = plt.subplots(figsize=(9.0, 3.8))
    image = axis.imshow(matrix, cmap="RdYlGn", vmin=-1.0, vmax=1.0, aspect="auto")
    axis.set_xticks(
        range(len(gates)), [gate.replace("_gate", "").replace("_", " ") for gate in gates]
    )
    axis.set_yticks(range(len(rows)), [str(row["mechanism"]) for row in rows])
    for i, row in enumerate(rows):
        for j, gate in enumerate(gates):
            axis.text(
                j,
                i,
                str(row[gate]).replace("NOT_TESTED", "not tested"),
                ha="center",
                va="center",
                fontsize=8,
            )
    figure.colorbar(image, ax=axis, shrink=0.75, label="gate status")
    figure.tight_layout()
    figure.savefig(output / "stage_b_transport_mode_map.png", dpi=240)
    plt.close(figure)


def run_contract(contract_path: Path, output_path: Path) -> dict[str, object]:
    contract_path = Path(contract_path).resolve()
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"Output path already exists: {output}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if int(contract.get("schema_version", -1)) != 1:
        raise ValueError("schema_version must be 1")
    base = contract_path.parent
    roots = {
        name: resolve_path(contract[name], base)
        for name in ("b1_results", "b2_results", "b3_results", "b4_results")
    }
    for root in roots.values():
        _verify_hashes(root)
    exchange_decision, exchange_fraction, exchange_detail = classify_exchange(
        _read_table(roots["b1_results"] / "category_response_summary.tsv")
    )
    layer_decision, layer_detail = classify_layer_response(
        _read_table(roots["b2_results"] / "layer_response_blocks_50ps.tsv")
    )
    tpcl_decision, tpcl_detail = classify_tpcl(
        _read_table(roots["b3_results"] / "anchor_branch_summary.tsv"),
        _read_table(roots["b3_results"] / "event_matched_anchor_response.tsv"),
    )
    b4_summary = json.loads((roots["b4_results"] / "summary.json").read_text(encoding="utf-8"))
    response_rows = _read_table(roots["b4_results"] / "transport_response_matrix.tsv")
    maximum_lateral = max(
        abs(_float(row[field]))
        for row in response_rows
        for field in ("x_lateral_mps", "y_lateral_mps")
    )
    decisions: list[dict[str, object]] = [
        {
            "mechanism": "mixed275 directed exchange transport",
            "decision": exchange_decision,
            "numerical_gate": "PASS",
            "association_gate": "FAIL" if exchange_fraction < 0.2 else "PARTIAL",
            "replicate_gate": "NOT_TESTED",
            "reverse_force_gate": "NOT_TESTED",
            "intervention_gate": "NOT_TESTED",
            "detail": exchange_detail,
        },
        {
            "mechanism": "oh_only layer-opposed response",
            "decision": layer_decision,
            "numerical_gate": "PASS",
            "association_gate": "PARTIAL",
            "replicate_gate": "NOT_TESTED",
            "reverse_force_gate": "NOT_TESTED",
            "intervention_gate": "NOT_TESTED",
            "detail": layer_detail,
        },
        {
            "mechanism": "finite-droplet dynamic TPCL anchoring",
            "decision": tpcl_decision,
            "numerical_gate": "PASS",
            "association_gate": "PARTIAL",
            "replicate_gate": "NOT_TESTED",
            "reverse_force_gate": "NOT_TESTED",
            "intervention_gate": "NOT_TESTED",
            "detail": tpcl_detail,
        },
        {
            "mechanism": "functional-pattern anisotropy",
            "decision": "SPATIAL_ASSOCIATION_ONLY_PATTERN_CAUSALITY_UNTESTED",
            "numerical_gate": "PASS" if b4_summary["status"] == "PASS" else "FAIL",
            "association_gate": "PARTIAL",
            "replicate_gate": "NOT_TESTED",
            "reverse_force_gate": "NOT_TESTED",
            "intervention_gate": "NOT_TESTED",
            "detail": (
                f"complete 2x2 response matrices; maximum absolute lateral response="
                f"{maximum_lateral:.6f} m/s; ch3_only intrinsic X/Y control retained"
            ),
        },
    ]
    output.mkdir(parents=True)
    write_tsv(
        output / "stage_b_mechanism_decision.tsv",
        decisions,
        tuple(decisions[0]),
    )
    input_paths = {contract_path}
    for root in roots.values():
        input_paths.add(root / "OUTPUT-SHA256SUMS")
    input_rows = [
        {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(input_paths)
    ]
    write_tsv(output / "input_manifest.tsv", input_rows, tuple(input_rows[0]))
    font = resolve_path(contract["font_path"], base) if contract.get("font_path") else None
    _plot_gate_map(decisions, output, font)
    report_lines = [
        "# Stage B: Pure-water mechanism validation",
        "",
        "All numerical and artifact gates below are separate from physical interpretation and scientific acceptance.",
        "F0, X, and Y are distinct forcing branches, not independent replicates. Time blocks are single-trajectory descriptive diagnostics.",
        "",
        "## Decisions",
        "",
    ]
    for row in decisions:
        report_lines.extend(
            [
                f"### {row['mechanism']}",
                "",
                f"Decision: `{row['decision']}`",
                "",
                str(row["detail"]),
                "",
            ]
        )
    report_lines.extend(
        [
            "## Scientific boundary",
            "",
            "The existing 48 ns support closed descriptive transport decompositions and morphology-specific associations. They do not provide independent-replica uncertainty, force-reversal symmetry, composition-preserving pattern intervention, or sub-picosecond anchor lifetimes.",
            "No new MD or electrolyte calculation was submitted by Stage B postprocessing.",
            "",
        ]
    )
    (output / "STAGE-B-REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")
    summary = {
        "status": "PASS",
        "mechanism_rows": len(decisions),
        "exchange_decision": exchange_decision,
        "layer_decision": layer_decision,
        "tpcl_decision": tpcl_decision,
        "anisotropy_decision": "SPATIAL_ASSOCIATION_ONLY_PATTERN_CAUSALITY_UNTESTED",
        "scheduler_completion_is_scientific_acceptance": False,
        "independent_replicates_available": False,
        "reverse_force_available": False,
        "pattern_intervention_available": False,
        "new_md_submitted": False,
        "electrolytes_status": "HOLD",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_output_hashes(output)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_contract(args.contract, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
