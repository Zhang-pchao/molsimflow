"""Compare a PIMD thermo estimator across bead counts."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from molsimflow.postprocess.pimd_reweight import read_thermo


def _read_cases(manifest: Path, field: str, burn_in_ps: float) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            beads = int(row["beads"])
            log = Path(row["log"])
            if not log.is_absolute():
                log = manifest.parent / log
            thermo = read_thermo(log)
            selected = sorted(
                (values["Time"], values[field])
                for values in thermo.values()
                if values["Time"] >= burn_in_ps and field in values
            )
            if len(selected) < 2:
                raise ValueError(f"{log}: fewer than two {field} samples after burn-in")
            cases.append(
                {
                    "label": row.get("label") or f"P={beads}",
                    "beads": beads,
                    "log": log,
                    "time": np.asarray([item[0] for item in selected]),
                    "values": np.asarray([item[1] for item in selected]),
                }
            )
    if len(cases) < 2 or len({case["beads"] for case in cases}) != len(cases):
        raise ValueError("manifest requires at least two unique bead counts")
    return sorted(cases, key=lambda case: int(case["beads"]))


def _block_stats(values: np.ndarray, blocks: int) -> tuple[float, float, int]:
    count = min(blocks, len(values))
    if count < 2:
        raise ValueError("at least two non-empty blocks are required")
    means = np.asarray([chunk.mean() for chunk in np.array_split(values, count)])
    return float(values.mean()), float(means.std(ddof=1) / math.sqrt(count)), count


def analyze(
    manifest: Path,
    output: Path,
    *,
    field: str = "f_pi[7]",
    burn_in_ps: float = 0.0,
    blocks: int = 5,
    sigma: float = 2.0,
    write_plot: bool = False,
) -> None:
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    cases = _read_cases(manifest, field, burn_in_ps)
    output.mkdir(parents=True)

    summaries: list[dict[str, object]] = []
    for case in cases:
        mean, sem, used_blocks = _block_stats(case["values"], blocks)
        summaries.append(
            {
                "label": case["label"],
                "beads": case["beads"],
                "samples": len(case["values"]),
                "start_time_ps": float(case["time"][0]),
                "end_time_ps": float(case["time"][-1]),
                "mean_eV": mean,
                "block_sem_eV": sem,
                "blocks": used_blocks,
            }
        )

    reference = summaries[-1]
    comparisons: list[dict[str, object]] = []
    for row in summaries[:-1]:
        difference = float(row["mean_eV"]) - float(reference["mean_eV"])
        combined = math.hypot(float(row["block_sem_eV"]), float(reference["block_sem_eV"]))
        z_score = abs(difference) / combined if combined else math.inf
        comparisons.append(
            {
                "beads": row["beads"],
                "reference_beads": reference["beads"],
                "difference_eV": difference,
                "combined_block_sem_eV": combined,
                "z_score": z_score,
                "within_sigma": z_score <= sigma,
            }
        )

    _write_csv(output / "bead_summary.csv", summaries)
    _write_csv(output / "reference_comparison.csv", comparisons)
    if write_plot:
        _plot(cases, field, output / "bead_convergence.png")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(cases: list[dict[str, object]], field: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    for case in cases:
        axis.plot(case["time"], case["values"], lw=1.0, label=str(case["label"]))
    axis.set(xlabel="Time (ps)", ylabel=f"{field} (eV)")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=240)
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--field", default="f_pi[7]")
    parser.add_argument("--burn-in-ps", type=float, default=0.0)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--write-plot", action="store_true")
    args = parser.parse_args(argv)
    analyze(
        args.manifest,
        args.output,
        field=args.field,
        burn_in_ps=args.burn_in_ps,
        blocks=args.blocks,
        sigma=args.sigma,
        write_plot=args.write_plot,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
