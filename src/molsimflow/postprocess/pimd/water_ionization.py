"""Optional water trajectory diagnostics; no case-specific inputs or defaults."""
from __future__ import annotations
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple
import numpy as np
from .io import dump_frames
from .observables import soft_voronoi_occupancies


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


_require = require


def reconstruction_within_tolerance(
    error: float, tolerance: float | None
) -> bool:
    return tolerance is None or float(error) <= float(tolerance)



def ring_polymer_spread(
    paths: Sequence[Path], selected_steps: Sequence[int], type_labels: Mapping[int, str]
) -> List[Dict[str, float]]:
    wanted = {int(step) for step in selected_steps}
    require(bool(wanted), "no requested trajectory steps")
    first_step = min(wanted)
    last_step = max(wanted)
    generators = [dump_frames(path) for path in paths]
    output: List[Dict[str, float]] = []
    while True:
        frames = []
        for generator in generators:
            try:
                frames.append(next(generator))
            except StopIteration:
                frames = []
                break
        if not frames:
            break
        steps = {frame[0] for frame in frames}
        require(len(steps) == 1, "trajectory timestep mismatch")
        step = frames[0][0]
        require(all(np.array_equal(frame[2], frames[0][2]) for frame in frames), "atom order mismatch")
        require(all(np.array_equal(frame[3], frames[0][3]) for frame in frames), "atom type mismatch")
        if step < int(first_step):
            continue
        if step > int(last_step):
            break
        if step not in wanted:
            continue
        box = frames[0][1][:, 1] - frames[0][1][:, 0]
        positions = np.asarray([frame[4] for frame in frames])
        reference = positions[0]
        delta = positions - reference[None, :, :]
        delta -= box[None, None, :] * np.rint(delta / box[None, None, :])
        unwrapped = reference[None, :, :] + delta
        centroid = np.mean(unwrapped, axis=0)
        rg2 = np.mean(np.sum((unwrapped - centroid[None, :, :]) ** 2, axis=2), axis=0)
        row: Dict[str, float] = {
            "step": float(step),
            "rg_all_A": float(math.sqrt(float(np.mean(rg2)))),
            "rg_p95_A": float(math.sqrt(float(np.percentile(rg2, 95.0)))),
            "rg_max_A": float(math.sqrt(float(np.max(rg2)))),
        }
        types = frames[0][3]
        for atom_type, label in sorted(type_labels.items()):
            mask = types == int(atom_type)
            require(bool(np.any(mask)), f"atom type absent from trajectory: {atom_type}")
            row[f"rg_{label}_A"] = float(math.sqrt(float(np.mean(rg2[mask]))))
        output.append(row)
    observed = {int(row["step"]) for row in output}
    require(observed == wanted, "selected trajectory steps missing")
    return output



def ionization_candidate_details(
    paths: Sequence[Path],
    selected_steps: np.ndarray,
    time_ps: np.ndarray,
    centroid_values: np.ndarray,
    bead_values: np.ndarray,
    type_labels: Mapping[int, str],
    *,
    kappa: float,
    reference: float,
    minimum_score: float,
    reconstruction_tolerance: float | None,
    sampling_representation: str = "centroid",
) -> List[Dict[str, object]]:
    require(bead_values.shape == (len(selected_steps), len(paths)), "candidate bead shape mismatch")
    center_types = [key for key, value in type_labels.items() if value == "O"]
    assigned_types = [key for key, value in type_labels.items() if value == "H"]
    require(len(center_types) == 1 and len(assigned_types) == 1, "unique H/O type labels required")
    step_to_index = {int(step): index for index, step in enumerate(selected_steps)}
    wanted = {
        int(selected_steps[index])
        for index in range(len(selected_steps))
        if centroid_values[index] >= minimum_score
        or np.any(bead_values[index] >= minimum_score)
    }
    generators = [dump_frames(path) for path in paths]
    output: List[Dict[str, object]] = []
    while wanted:
        frames = []
        for generator in generators:
            try:
                frames.append(next(generator))
            except StopIteration:
                frames = []
                break
        if not frames:
            break
        steps = {frame[0] for frame in frames}
        require(len(steps) == 1, "candidate trajectory timestep mismatch")
        step = int(frames[0][0])
        if step not in wanted:
            continue
        index = step_to_index[step]
        require(all(np.array_equal(frame[2], frames[0][2]) for frame in frames), "candidate atom order mismatch")
        require(all(np.array_equal(frame[3], frames[0][3]) for frame in frames), "candidate atom type mismatch")
        ids = frames[0][2]
        types = frames[0][3]
        box = frames[0][1][:, 1] - frames[0][1][:, 0]
        positions = np.asarray([frame[4] for frame in frames])
        reference_positions = positions[0]
        delta = positions - reference_positions[None, :, :]
        delta -= box[None, None, :] * np.rint(delta / box[None, None, :])
        centroid_positions = reference_positions + np.mean(delta, axis=0)
        representations = []
        if sampling_representation == "centroid":
            representations.append(
                (sampling_representation, 0, centroid_values[index], centroid_positions)
            )
        representations.extend(
            (f"bead_{bead + 1}", bead + 1, bead_values[index, bead], positions[bead])
            for bead in range(len(paths))
        )
        center_ids = ids[types == int(center_types[0])]
        for label, bead, recorded, current_positions in representations:
            if float(recorded) < float(minimum_score):
                continue
            occupancies, hard_counts = soft_voronoi_occupancies(
                current_positions, types, box, center_types[0], assigned_types[0], kappa
            )
            score = float(np.sum((occupancies - float(reference)) ** 2))
            error = abs(score - float(recorded))
            require(
                reconstruction_within_tolerance(error, reconstruction_tolerance),
                f"ionization reconstruction mismatch at step {step} {label}",
            )
            minimum = int(np.argmin(occupancies))
            maximum = int(np.argmax(occupancies))
            output.append(
                {
                    "time_ps": float(time_ps[index]),
                    "step": step,
                    "representation": label,
                    "bead": bead,
                    "recorded_score": float(recorded),
                    "reconstructed_score": score,
                    "absolute_reconstruction_error": error,
                    "minimum_soft_occupancy": float(occupancies[minimum]),
                    "minimum_soft_occupancy_O_id": int(center_ids[minimum]),
                    "maximum_soft_occupancy": float(occupancies[maximum]),
                    "maximum_soft_occupancy_O_id": int(center_ids[maximum]),
                    "minimum_hard_H_count": int(np.min(hard_counts)),
                    "maximum_hard_H_count": int(np.max(hard_counts)),
                    "hard_undercoordinated_centers": int(np.count_nonzero(hard_counts <= 1)),
                    "hard_overcoordinated_centers": int(np.count_nonzero(hard_counts >= 3)),
                    "hard_pair_like": int(
                        np.any(hard_counts <= 1) and np.any(hard_counts >= 3)
                    ),
                }
            )
        wanted.remove(step)
    require(not wanted, "candidate trajectory frames missing")
    return output



def threshold_run_rows(
    time_ps: np.ndarray,
    series: Mapping[str, np.ndarray],
    thresholds: Sequence[float],
) -> List[Dict[str, object]]:
    stride_ps = float(np.median(np.diff(time_ps)))
    rows: List[Dict[str, object]] = []
    for label, values in series.items():
        values = np.asarray(values, dtype=float)
        for threshold in thresholds:
            mask = values >= float(threshold)
            runs: List[Tuple[int, int]] = []
            start: int | None = None
            for index, active in enumerate(mask):
                if active and start is None:
                    start = index
                if start is not None and (not active or index == len(mask) - 1):
                    stop = index if active and index == len(mask) - 1 else index - 1
                    runs.append((start, stop))
                    start = None
            longest = max((stop - start + 1 for start, stop in runs), default=0)
            rows.append(
                {
                    "representation": label,
                    "threshold": float(threshold),
                    "frames_at_or_above": int(np.count_nonzero(mask)),
                    "fraction_frames": float(np.mean(mask)),
                    "contiguous_runs": len(runs),
                    "longest_run_frames": longest,
                    "longest_run_ps": longest * stride_ps,
                }
            )
    return rows
