"""Build a two-coordinate dynamic state map from accepted TPCL events.

The two coordinates are a local event-row rate and a cross-arc ordered-pair
excess relative to independent per-arc circular time shifts.  All uncertainty
is within-trajectory block resampling; it is not replicate-level inference.
"""

# Python 3.9 is supported; keep Optional rather than requiring PEP 604 syntax.
# ruff: noqa: UP045

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from molsimflow.postprocess.tpcl_propagation import DEFAULT_EVENT_STATUS, load_events

SCIENTIFIC_STATUS = (
    "RETROSPECTIVE_SINGLE_TRAJECTORY_DYNAMIC_STATE_MAP_"
    "NOT_REPLICATE_LEVEL_OR_CAUSAL_PROPAGATION_EVIDENCE"
)


@dataclass(frozen=True)
class DynamicStateSource:
    """One case routed by explicit accepted input paths."""

    case_id: str
    propagation_summary: Path
    events: Path


@dataclass(frozen=True)
class LagWindow:
    """One ordered-pair lag window in ps."""

    name: str
    start_ps: float
    end_ps: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _bh_adjust(p_values: Sequence[float]) -> list[float]:
    """Return Benjamini-Hochberg q values in original order."""

    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 1.0
    for reverse_rank, index in enumerate(order[::-1], start=1):
        rank = len(values) - reverse_rank + 1
        running = min(running, float(values[index]) * len(values) / rank)
        adjusted[index] = running
    return adjusted.tolist()


def read_sources(path: Path) -> list[DynamicStateSource]:
    """Read a path-explicit tab-separated source manifest."""

    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {"case_id", "propagation_summary", "events"}
    if not rows or required.difference(rows[0]):
        raise ValueError(f"{path}: missing source rows or columns")
    sources = [
        DynamicStateSource(
            case_id=row["case_id"],
            propagation_summary=Path(row["propagation_summary"]),
            events=Path(row["events"]),
        )
        for row in rows
    ]
    case_ids = [source.case_id for source in sources]
    if any(not case_id for case_id in case_ids) or len(set(case_ids)) != len(case_ids):
        raise ValueError(f"{path}: case_id values must be nonempty and unique")
    return sources


def parse_windows(raw: str) -> tuple[LagWindow, ...]:
    """Parse ``name:start_ps:end_ps`` comma-separated lag windows."""

    windows = []
    for item in raw.split(","):
        fields = item.split(":")
        if len(fields) != 3 or not fields[0]:
            raise ValueError(f"invalid lag window {item!r}")
        window = LagWindow(fields[0], float(fields[1]), float(fields[2]))
        if window.start_ps < 0.0 or window.end_ps <= window.start_ps:
            raise ValueError(f"invalid lag window {item!r}")
        windows.append(window)
    if not windows or len({window.name for window in windows}) != len(windows):
        raise ValueError("lag windows must be nonempty and uniquely named")
    return tuple(windows)


def _pair_counts(
    times_ps: np.ndarray,
    arcs: np.ndarray,
    n_arcs: int,
    windows: Sequence[LagWindow],
) -> np.ndarray:
    if len(times_ps) < 2:
        return np.zeros(len(windows), dtype=float)
    delta_t = times_ps[None, :] - times_ps[:, None]
    direct = np.abs(arcs[None, :] - arcs[:, None])
    cross_arc = np.minimum(direct, n_arcs - direct) > 0
    return np.asarray(
        [
            np.count_nonzero(
                cross_arc & (delta_t > window.start_ps) & (delta_t <= window.end_ps)
            )
            for window in windows
        ],
        dtype=float,
    )


def _shift_per_arc(
    times_ps: np.ndarray,
    arcs: np.ndarray,
    left_ps: float,
    duration_ps: float,
    rng: np.random.Generator,
) -> np.ndarray:
    shifted = times_ps.copy()
    for arc in np.unique(arcs):
        selected = arcs == arc
        offset = rng.uniform(0.0, duration_ps)
        shifted[selected] = left_ps + np.mod(times_ps[selected] - left_ps + offset, duration_ps)
    return shifted


def _case_blocks(
    times_ps: np.ndarray,
    arcs: np.ndarray,
    *,
    n_arcs: int,
    start_ps: float,
    stop_ps: float,
    block_ps: float,
    windows: Sequence[LagWindow],
    null_samples: int,
    rng: np.random.Generator,
) -> tuple[list[dict[str, object]], np.ndarray]:
    block_count = math.ceil((stop_ps - start_ps) / block_ps)
    block_rows = []
    null_totals = np.zeros((null_samples, len(windows)), dtype=float)
    for block_index in range(block_count):
        left = start_ps + block_index * block_ps
        right = min(stop_ps, left + block_ps)
        final = block_index + 1 == block_count
        selected = (times_ps >= left) & ((times_ps <= right) if final else (times_ps < right))
        block_times, block_arcs = times_ps[selected], arcs[selected]
        observed = _pair_counts(block_times, block_arcs, n_arcs, windows)
        block_null = np.zeros((null_samples, len(windows)), dtype=float)
        for sample in range(null_samples):
            shifted = _shift_per_arc(block_times, block_arcs, left, right - left, rng)
            block_null[sample] = _pair_counts(shifted, block_arcs, n_arcs, windows)
        null_totals += block_null
        block_rows.append(
            {
                "block_index": block_index,
                "block_start_ps": left,
                "block_end_ps": right,
                "duration_ps": right - left,
                "event_count": int(np.count_nonzero(selected)),
                "observed": observed,
                "null_mean": np.mean(block_null, axis=0),
            }
        )
    return block_rows, null_totals


def _bootstrap_intervals(
    block_rows: Sequence[Mapping[str, object]],
    *,
    n_arcs: int,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> tuple[tuple[float, float], list[tuple[float, float]]]:
    count = len(block_rows)
    rate_draws = []
    chi_draws: list[list[float]] = []
    window_count = len(np.asarray(block_rows[0]["observed"]))
    chi_draws = [[] for _ in range(window_count)]
    for _ in range(bootstrap_samples):
        indices = rng.integers(0, count, size=count)
        duration_ps = sum(float(block_rows[index]["duration_ps"]) for index in indices)
        events = sum(int(block_rows[index]["event_count"]) for index in indices)
        rate_draws.append(events / (n_arcs * duration_ps / 1000.0))
        observed = np.sum([np.asarray(block_rows[index]["observed"]) for index in indices], axis=0)
        expected = np.sum([np.asarray(block_rows[index]["null_mean"]) for index in indices], axis=0)
        for window_index, (obs, exp) in enumerate(zip(observed, expected)):
            if exp > 0.0:
                chi_draws[window_index].append(float(obs / exp - 1.0))
    rate_interval = tuple(float(value) for value in np.quantile(rate_draws, [0.025, 0.975]))
    chi_intervals = [
        tuple(float(value) for value in np.quantile(draws, [0.025, 0.975]))
        if draws
        else (math.nan, math.nan)
        for draws in chi_draws
    ]
    return rate_interval, chi_intervals


def analyze_case(
    source: DynamicStateSource,
    *,
    windows: Sequence[LagWindow],
    block_ps: float,
    null_samples: int,
    bootstrap_samples: int,
    event_status: str,
    random_seed: int,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Return yielding, cooperativity, block, and null rows for one case."""

    summary = json.loads(source.propagation_summary.read_text(encoding="utf-8"))
    if summary.get("status") != "PASS" or summary.get("case_id") != source.case_id:
        raise ValueError(f"{source.case_id}: propagation summary is not an accepted matching PASS")
    n_arcs = int(summary["arc_count"])
    start_ps = float(summary["first_time_ns"]) * 1000.0
    stop_ps = float(summary["last_time_ns"]) * 1000.0
    if n_arcs < 4 or stop_ps <= start_ps:
        raise ValueError(f"{source.case_id}: invalid arc count or trajectory support")
    if block_ps < 2.0 * max(window.end_ps for window in windows):
        raise ValueError("block_ps must be at least twice the largest lag-window endpoint")
    events = load_events(source.events, event_status)
    if len(events) != int(summary["admitted_event_rows"]):
        raise ValueError(f"{source.case_id}: event count does not match propagation summary")
    times_ps = np.asarray([float(event["transition_time_ns"]) * 1000.0 for event in events])
    arcs = np.asarray([int(event["arc_index"]) for event in events], dtype=int)
    if np.any(times_ps < start_ps) or np.any(times_ps > stop_ps) or np.any(arcs < 0) or np.any(arcs >= n_arcs):
        raise ValueError(f"{source.case_id}: events fall outside accepted support")

    rng = np.random.default_rng(random_seed)
    block_rows, null_totals = _case_blocks(
        times_ps,
        arcs,
        n_arcs=n_arcs,
        start_ps=start_ps,
        stop_ps=stop_ps,
        block_ps=block_ps,
        windows=windows,
        null_samples=null_samples,
        rng=rng,
    )
    rate_interval, chi_intervals = _bootstrap_intervals(
        block_rows,
        n_arcs=n_arcs,
        bootstrap_samples=bootstrap_samples,
        rng=rng,
    )
    duration_ns = (stop_ps - start_ps) / 1000.0
    yielding = {
        "case_id": source.case_id,
        "event_rows": len(events),
        "event_clusters": int(summary["event_cluster_count"]),
        "arc_count": n_arcs,
        "duration_ns": duration_ns,
        "yielding_rate_per_arc_ns": len(events) / (n_arcs * duration_ns),
        "yielding_rate_ci025": rate_interval[0],
        "yielding_rate_ci975": rate_interval[1],
        "cluster_rate_per_ns": int(summary["event_cluster_count"]) / duration_ns,
        "block_ps": block_ps,
        "block_count": len(block_rows),
        "scientific_status": SCIENTIFIC_STATUS,
    }
    observed_totals = np.sum([np.asarray(row["observed"]) for row in block_rows], axis=0)
    cooperativity = []
    null_rows = []
    for index, window in enumerate(windows):
        expected = float(np.mean(null_totals[:, index]))
        observed = float(observed_totals[index])
        upper_p = float(
            (1 + np.count_nonzero(null_totals[:, index] >= observed)) / (null_samples + 1)
        )
        lower_p = float(
            (1 + np.count_nonzero(null_totals[:, index] <= observed)) / (null_samples + 1)
        )
        cooperativity.append(
            {
                "case_id": source.case_id,
                "window": window.name,
                "lag_start_ps": window.start_ps,
                "lag_end_ps": window.end_ps,
                "observed_cross_arc_pairs": int(observed),
                "null_mean_cross_arc_pairs": expected,
                "cooperative_excess_fraction": observed / expected - 1.0 if expected > 0.0 else math.nan,
                "cooperative_excess_ci025": chi_intervals[index][0],
                "cooperative_excess_ci975": chi_intervals[index][1],
                "null_q025_pairs": float(np.quantile(null_totals[:, index], 0.025)),
                "null_q975_pairs": float(np.quantile(null_totals[:, index], 0.975)),
                "empirical_lower_p": lower_p,
                "empirical_upper_p": upper_p,
                "empirical_two_sided_p": min(1.0, 2.0 * min(lower_p, upper_p)),
                "informative_null_count": expected >= 5.0,
                "scientific_status": SCIENTIFIC_STATUS,
            }
        )
        null_rows.extend(
            {
                "case_id": source.case_id,
                "window": window.name,
                "null_sample": sample,
                "cross_arc_pairs": int(value),
            }
            for sample, value in enumerate(null_totals[:, index])
        )
    flat_blocks = []
    for row in block_rows:
        for index, window in enumerate(windows):
            flat_blocks.append(
                {
                    "case_id": source.case_id,
                    "block_index": row["block_index"],
                    "block_start_ps": row["block_start_ps"],
                    "block_end_ps": row["block_end_ps"],
                    "duration_ps": row["duration_ps"],
                    "event_count": row["event_count"],
                    "window": window.name,
                    "lag_start_ps": window.start_ps,
                    "lag_end_ps": window.end_ps,
                    "observed_cross_arc_pairs": int(np.asarray(row["observed"])[index]),
                    "null_mean_cross_arc_pairs": float(np.asarray(row["null_mean"])[index]),
                }
            )
    return yielding, cooperativity, flat_blocks, null_rows


def run_analysis(
    sources_path: Path,
    output_dir: Path,
    *,
    windows: Sequence[LagWindow],
    block_ps: float,
    null_samples: int,
    bootstrap_samples: int,
    event_status: str,
    random_seed: int,
) -> dict[str, object]:
    if null_samples < 20 or bootstrap_samples < 20:
        raise ValueError("null_samples and bootstrap_samples must each be at least 20")
    sources = read_sources(sources_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    yielding_rows, cooperative_rows, block_rows, null_rows = [], [], [], []
    for index, source in enumerate(sources):
        yielding, cooperative, blocks, nulls = analyze_case(
            source,
            windows=windows,
            block_ps=block_ps,
            null_samples=null_samples,
            bootstrap_samples=bootstrap_samples,
            event_status=event_status,
            random_seed=random_seed + index,
        )
        yielding_rows.append(yielding)
        cooperative_rows.extend(cooperative)
        block_rows.extend(blocks)
        null_rows.extend(nulls)
    for row, q_value in zip(
        cooperative_rows,
        _bh_adjust([float(row["empirical_two_sided_p"]) for row in cooperative_rows]),
    ):
        row["bh_q_primary_family"] = q_value
    _write_csv(output / "yielding_by_case.csv", yielding_rows)
    _write_csv(output / "cooperativity_by_case.csv", cooperative_rows)
    _write_csv(output / "block_metrics.csv", block_rows)
    _write_csv(output / "null_totals.csv", null_rows)
    summary = {
        "status": "PASS",
        "case_count": len(sources),
        "windows": [window.__dict__ for window in windows],
        "block_ps": block_ps,
        "null_samples": null_samples,
        "bootstrap_samples": bootstrap_samples,
        "scientific_status": SCIENTIFIC_STATUS,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    manifest = {
        **summary,
        "sources_manifest": {"path": str(sources_path), "sha256": _sha256(sources_path)},
        "inputs": [
            {
                "case_id": source.case_id,
                "propagation_summary": {
                    "path": str(source.propagation_summary),
                    "sha256": _sha256(source.propagation_summary),
                },
                "events": {"path": str(source.events), "sha256": _sha256(source.events)},
            }
            for source in sources
        ],
        "event_status": event_status,
        "random_seed": random_seed,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output / "REPORT.md").write_text(
        "# TPCL dynamic state map\n\n"
        "This analysis separates local event-row rate from cross-arc ordered-pair excess. "
        "Pairs crossing 200 ps block boundaries are excluded; each null independently "
        "circular-shifts every arc inside each block. Confidence intervals are whole-block "
        "resampling diagnostics from one trajectory, not replicate-level uncertainty. "
        "The outputs do not establish triggering, propagation, an intrinsic correlation "
        "length, or a physical rate constant.\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--windows", default="fast:0:5,slow:5:50")
    parser.add_argument("--block-ps", type=float, default=200.0)
    parser.add_argument("--null-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--event-status", default=DEFAULT_EVENT_STATUS)
    parser.add_argument("--random-seed", type=int, default=20260904)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    run_analysis(
        args.sources,
        args.output_dir,
        windows=parse_windows(args.windows),
        block_ps=args.block_ps,
        null_samples=args.null_samples,
        bootstrap_samples=args.bootstrap_samples,
        event_status=args.event_status,
        random_seed=args.random_seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
