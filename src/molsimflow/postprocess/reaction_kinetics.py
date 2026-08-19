"""Generic transition-state rate sensitivity and two-channel competition."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

GAS_CONSTANT_KJ_MOL_K = 0.00831446261815324
BOLTZMANN_J_K = 1.380649e-23
PLANCK_J_S = 6.62607015e-34
DEFAULT_BARRIER_SHIFTS_KJ_MOL = (-10.0, -5.0, 0.0, 5.0, 10.0)


@dataclass(frozen=True)
class PathwayBarrier:
    """A user-named pathway and its activation free-energy barrier."""

    label: str
    barrier_kj_mol: float


def eyring_rate(
    barrier_kj_mol: float,
    *,
    temperature_K: float = 298.15,
    transmission_coefficient: float = 1.0,
) -> float:
    """Return the Eyring transition-state-theory rate in inverse seconds."""

    if not math.isfinite(barrier_kj_mol) or barrier_kj_mol < 0.0:
        raise ValueError("barrier_kj_mol must be finite and non-negative")
    if not math.isfinite(temperature_K) or temperature_K <= 0.0:
        raise ValueError("temperature_K must be positive")
    if not math.isfinite(transmission_coefficient) or transmission_coefficient <= 0.0:
        raise ValueError("transmission_coefficient must be positive")
    prefactor = transmission_coefficient * BOLTZMANN_J_K * temperature_K / PLANCK_J_S
    return float(prefactor * math.exp(-barrier_kj_mol / (GAS_CONSTANT_KJ_MOL_K * temperature_K)))


def build_barrier_sensitivity(
    pathways: Sequence[PathwayBarrier],
    barrier_shifts_kj_mol: Sequence[float] = DEFAULT_BARRIER_SHIFTS_KJ_MOL,
    *,
    temperature_K: float = 298.15,
    transmission_coefficient: float = 1.0,
) -> list[dict[str, object]]:
    """Evaluate Eyring rates over user-selected barrier shifts."""

    rows: list[dict[str, object]] = []
    for pathway in pathways:
        baseline_rate = eyring_rate(
            pathway.barrier_kj_mol,
            temperature_K=temperature_K,
            transmission_coefficient=transmission_coefficient,
        )
        for shift in barrier_shifts_kj_mol:
            shifted_barrier = pathway.barrier_kj_mol + float(shift)
            rate = eyring_rate(
                shifted_barrier,
                temperature_K=temperature_K,
                transmission_coefficient=transmission_coefficient,
            )
            rows.append(
                {
                    "label": pathway.label,
                    "temperature_K": temperature_K,
                    "transmission_coefficient": transmission_coefficient,
                    "baseline_barrier_kj_mol": pathway.barrier_kj_mol,
                    "barrier_shift_kj_mol": float(shift),
                    "barrier_kj_mol": shifted_barrier,
                    "rate_s_inv": rate,
                    "lifetime_s": 1.0 / rate,
                    "rate_ratio_to_baseline": rate / baseline_rate,
                }
            )
    if not rows:
        raise ValueError("At least one pathway and one barrier shift are required")
    return rows


def build_two_channel_competition(
    pathways: Sequence[PathwayBarrier],
    competitor_rates_s_inv: Sequence[float],
    barrier_shifts_kj_mol: Sequence[float] = (0.0,),
    *,
    temperature_K: float = 298.15,
    transmission_coefficient: float = 1.0,
) -> list[dict[str, object]]:
    """Compare each pathway with user-supplied effective first-order rates."""

    if any(not math.isfinite(float(rate)) or float(rate) <= 0.0 for rate in competitor_rates_s_inv):
        raise ValueError("competitor_rates_s_inv must contain positive finite values")
    rows: list[dict[str, object]] = []
    for pathway in pathways:
        for shift in barrier_shifts_kj_mol:
            barrier = pathway.barrier_kj_mol + float(shift)
            pathway_rate = eyring_rate(
                barrier,
                temperature_K=temperature_K,
                transmission_coefficient=transmission_coefficient,
            )
            for competitor_rate_value in competitor_rates_s_inv:
                competitor_rate = float(competitor_rate_value)
                ratio = pathway_rate / competitor_rate
                rows.append(
                    {
                        "label": pathway.label,
                        "temperature_K": temperature_K,
                        "barrier_shift_kj_mol": float(shift),
                        "barrier_kj_mol": barrier,
                        "pathway_rate_s_inv": pathway_rate,
                        "competitor_rate_s_inv": competitor_rate,
                        "pathway_to_competitor_ratio": ratio,
                        "conditional_pathway_fraction": pathway_rate
                        / (pathway_rate + competitor_rate),
                        "faster_channel": "pathway" if ratio > 1.0 else "competitor",
                    }
                )
    if not rows:
        raise ValueError("At least one pathway, shift, and competitor rate are required")
    return rows


def _read_pathways(path: Path) -> list[PathwayBarrier]:
    delimiter = "\t" if Path(path).suffix.lower() in {".tsv", ".tab"} else ","
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter=delimiter))
    required = {"label", "barrier_kj_mol"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Input table must contain columns: {', '.join(sorted(required))}")
    pathways = [
        PathwayBarrier(str(row["label"]).strip(), float(row["barrier_kj_mol"])) for row in rows
    ]
    if any(not pathway.label for pathway in pathways):
        raise ValueError("Pathway labels must not be empty")
    return pathways


def _write_tsv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def run_reaction_kinetics(
    input_path: Path,
    output_dir: Path,
    *,
    temperature_K: float = 298.15,
    transmission_coefficient: float = 1.0,
    barrier_shifts_kj_mol: Sequence[float] = DEFAULT_BARRIER_SHIFTS_KJ_MOL,
    competitor_rates_s_inv: Sequence[float] | None = None,
) -> dict[str, Path]:
    """Run the table-driven workflow and return written output paths."""

    pathways = _read_pathways(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {"sensitivity": output_dir / "barrier_sensitivity.tsv"}
    sensitivity = build_barrier_sensitivity(
        pathways,
        barrier_shifts_kj_mol,
        temperature_K=temperature_K,
        transmission_coefficient=transmission_coefficient,
    )
    _write_tsv(outputs["sensitivity"], sensitivity)
    if competitor_rates_s_inv:
        outputs["competition"] = output_dir / "two_channel_competition.tsv"
        _write_tsv(
            outputs["competition"],
            build_two_channel_competition(
                pathways,
                competitor_rates_s_inv,
                barrier_shifts_kj_mol,
                temperature_K=temperature_K,
                transmission_coefficient=transmission_coefficient,
            ),
        )
    outputs["metadata"] = output_dir / "metadata.json"
    outputs["metadata"].write_text(
        json.dumps(
            {
                "workflow": "reaction_kinetics",
                "input": str(input_path),
                "temperature_K": temperature_K,
                "transmission_coefficient": transmission_coefficient,
                "barrier_shifts_kj_mol": list(barrier_shifts_kj_mol),
                "competitor_rates_s_inv": list(competitor_rates_s_inv or ()),
                "pathway_count": len(pathways),
                "interpretation": "Two-channel fractions are conditional on user-supplied competitor rates.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="CSV/TSV with label and barrier_kj_mol"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--temperature-K", type=float, default=298.15)
    parser.add_argument("--transmission-coefficient", type=float, default=1.0)
    parser.add_argument(
        "--barrier-shifts-kj-mol",
        type=float,
        nargs="+",
        default=list(DEFAULT_BARRIER_SHIFTS_KJ_MOL),
    )
    parser.add_argument("--competitor-rates-s-inv", type=float, nargs="+")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outputs = run_reaction_kinetics(
        args.input,
        args.output_dir,
        temperature_K=args.temperature_K,
        transmission_coefficient=args.transmission_coefficient,
        barrier_shifts_kj_mol=args.barrier_shifts_kj_mol,
        competitor_rates_s_inv=args.competitor_rates_s_inv,
    )
    for path in outputs.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
