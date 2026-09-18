import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from molsimflow.cli import build_parser as build_cli_parser
from molsimflow.postprocess.constant_force_events import (
    _deduplicate_species_rows,
    _write_tsv,
    build_parser,
    merge_event_samples,
    run_contract,
    stitch_motion_tables,
)


def _write_motion(path: Path, rows: list[tuple[float, ...]]) -> None:
    path.write_text(
        "# synthetic motion\n"
        "# TimeStep v_dxrel v_dyrel v_topclear\n"
        + "\n".join(" ".join(map(str, row)) for row in rows)
        + "\n",
        encoding="utf-8",
    )


def _atom_rows(
    step: int,
    image_z: int = 0,
) -> list[tuple[int, int, float, float, float, int, int, int]]:
    # One framework Si/OH pair, two waters that transiently form OH/H3O,
    # and one intact high-z water used for identity and return checks.
    rows = [
        (1, 8, 1.0, 1.0, 1.0, 0, 0, 0),
        (2, 2, 1.0, 1.0, 2.0, 0, 0, 0),
        (3, 1, 1.0, 1.0, 2.9, 0, 0, 0),
        (10, 2, 3.0, 3.0, 5.0, 0, 0, 0),
        (11, 1, 3.0, 3.0, 4.1, 0, 0, 0),
        (12, 1, 3.0, 3.0, 5.9, 0, 0, 0),
        (20, 2, 6.0, 6.0, 5.0, 0, 0, 0),
        (21, 1, 6.0, 6.0, 4.1, 0, 0, 0),
        (22, 1, 6.0, 6.0, 5.9, 0, 0, 0),
        (30, 2, 8.0, 8.0, 7.0 if step != 10 else 9.0, 0, 0, image_z),
        (31, 1, 8.0, 8.0, 6.1 if step != 10 else 8.1, 0, 0, image_z),
        (32, 1, 8.0, 8.0, 7.9 if step != 10 else 9.9, 0, 0, image_z),
    ]
    if step == 10:
        # Hydrogen 12 moves from O10 to O20 and returns at step 20.
        rows[5] = (12, 1, 6.0, 6.0, 5.0, 0, 0, 0)
    return rows


def _write_dump(path: Path, *, image_z: int = 0) -> None:
    blocks = []
    for step in (0, 10, 20):
        rows = _atom_rows(step, image_z=image_z if step == 10 else 0)
        block = [
            "ITEM: TIMESTEP",
            str(step),
            "ITEM: NUMBER OF ATOMS",
            str(len(rows)),
            "ITEM: BOX BOUNDS pp pp ff",
            "0 10",
            "0 10",
            "0 12",
            "ITEM: ATOMS id type x y z ix iy iz",
        ]
        block.extend(" ".join(map(str, row)) for row in rows)
        blocks.extend(block)
    path.write_text("\n".join(blocks) + "\n", encoding="utf-8")


def _write_species(path: Path) -> None:
    fields = [
        "step",
        "time_ps",
        "O_solution",
        "OH_solution",
        "H2O_solution",
        "H3O_solution",
        "OH4plus_solution",
        "unassigned_H",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(
            [
                dict(zip(fields, (0, 0.0, 0, 0, 3, 0, 0, 0))),
                dict(zip(fields, (10, 10.0, 0, 1, 1, 1, 0, 0))),
                dict(zip(fields, (20, 20.0, 0, 0, 3, 0, 0, 0))),
            ]
        )


def _write_contract(tmp_path: Path, *, image_z: int = 0) -> Path:
    dump = tmp_path / "state.lammpstrj"
    motion_a = tmp_path / "motion_a.dat"
    motion_b = tmp_path / "motion_b.dat"
    species = tmp_path / "species.tsv"
    _write_dump(dump, image_z=image_z)
    _write_motion(motion_a, [(0, 0.0, 0.0, 2.0), (10, 1.0, 2.0, 0.5)])
    _write_motion(motion_b, [(10, 0.0, 0.0, 0.5), (20, 2.0, 3.0, 2.0)])
    _write_species(species)
    contract = {
        "schema_version": 1,
        "time_origin_step": 0,
        "timestep_fs": 1000.0,
        "window_ps": 20.0,
        "merge_gap_ps": 11.0,
        "high_z_threshold_A": 8.0,
        "wall_clearance_threshold_A": 1.0,
        "types": {"hydrogen": 1, "oxygen": 2, "silicon": 8},
        "cutoffs_A": {"oh": 1.35, "si_o": 2.25},
        "cases": [
            {
                "case_id": "synthetic",
                "branch_id": "f8e-5_x",
                "state_trajectories": [dump.name],
                "oxygen_audit_trajectories": [dump.name],
                "motion_tables": [motion_a.name, motion_b.name],
                "species_tables": [species.name],
            }
        ],
    }
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    return path


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def test_stitch_motion_tables_offsets_restart_displacements(tmp_path):
    first = tmp_path / "first.dat"
    second = tmp_path / "second.dat"
    _write_motion(first, [(0, 0.0, 0.0, 2.0), (10, 1.0, 2.0, 1.5)])
    _write_motion(second, [(10, 0.0, 0.0, 1.5), (20, 2.0, 3.0, 1.0)])

    columns, values = stitch_motion_tables([first, second])

    assert values[:, columns.index("TimeStep")].tolist() == [0.0, 10.0, 20.0]
    assert values[:, columns.index("v_dxrel")].tolist() == [0.0, 1.0, 3.0]
    assert values[:, columns.index("v_dyrel")].tolist() == [0.0, 2.0, 5.0]


def test_merge_event_samples_keeps_long_episode_bounds_and_atom_ids():
    samples = [
        {
            "time_ps": time,
            "step": int(time),
            "event_type": "wall_approach",
            "severity": 1.0,
            "oxygen_id": 30,
        }
        for time in (0.0, 5.0, 10.0, 15.0)
    ]

    episodes, sources = merge_event_samples(
        samples,
        case_id="case",
        branch_id="branch",
        merge_gap_ps=5.0,
        window_ps=20.0,
    )

    assert len(episodes) == 1
    assert episodes[0].start_time_ps == -20.0
    assert episodes[0].end_time_ps == 35.0
    assert episodes[0].tracked_oxygen_ids == (30,)
    assert {row["event_id"] for row in sources} == {episodes[0].event_id}


def test_conflicting_species_duplicates_fail_closed():
    rows = [
        {"step": "10", "time_ps": "1", "OH_solution": "0"},
        {"step": "10", "time_ps": "1", "OH_solution": "1"},
    ]

    with pytest.raises(ValueError, match="Conflicting duplicate"):
        _deduplicate_species_rows(rows, step_column="step", time_column="time_ps")


def test_species_duplicate_ignores_segment_local_frame_index():
    rows = [
        {"frame_index": "1000", "step": "10", "time_ps": "1", "OH_solution": "1"},
        {"frame_index": "0", "step": "10", "time_ps": "1.0", "OH_solution": "1.0"},
    ]

    result = _deduplicate_species_rows(rows, step_column="step", time_column="time_ps")

    assert len(result) == 1


def test_empty_table_has_stable_header(tmp_path):
    path = tmp_path / "empty.tsv"
    _write_tsv(path, [], fieldnames=("event_id", "time_ps"))
    assert path.read_text(encoding="utf-8") == "event_id\ttime_ps\n"


def test_run_contract_tracks_species_return_and_intact_high_z_water(tmp_path):
    contract = _write_contract(tmp_path)
    output = tmp_path / "result"

    summary = run_contract(contract, output)

    assert summary["status"] == "PASS"
    assert summary["events"] == 1
    assert summary["tracked_intact_water_events"] == 1
    assert summary["z_image_crossing_events"] == 0
    events = _read_tsv(output / "events.tsv")
    assert events[0]["species_returned"] == "True"
    assert events[0]["tracked_identity_complete"] == "True"
    assert events[0]["tracked_h_counts"] == "2"
    assert events[0]["tracked_intact_water"] == "True"
    assert events[0]["tracked_returned_below_high_z"] == "True"
    frames = _read_tsv(output / "frame_species.tsv")
    assert {row["OH_solution"] for row in frames} == {"0", "1"}
    assert {row["H3O_solution"] for row in frames} == {"0", "1"}
    motion = _read_tsv(output / "motion_event_summary.tsv")
    assert motion[0]["dx_window_A"] == "3.0"
    assert motion[0]["dy_window_A"] == "5.0"
    manifest = _read_tsv(output / "input_manifest.tsv")
    recorded = {Path(row["path"]).name: row["sha256"] for row in manifest}
    assert recorded[contract.name] == hashlib.sha256(contract.read_bytes()).hexdigest()


def test_run_contract_reports_z_image_crossing(tmp_path):
    contract = _write_contract(tmp_path, image_z=1)
    summary = run_contract(contract, tmp_path / "result")
    assert summary["z_image_crossing_events"] == 1
    sources = _read_tsv(tmp_path / "result" / "event_sources.tsv")
    assert "z_image" in {row["event_type"] for row in sources}


def test_run_contract_refuses_existing_output(tmp_path):
    contract = _write_contract(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        run_contract(contract, output)


def test_cli_parser_accepts_contract_and_output(tmp_path):
    args = build_parser().parse_args(
        ["--contract", str(tmp_path / "contract.json"), "--output", str(tmp_path / "out")]
    )
    assert args.contract == tmp_path / "contract.json"
    assert args.output == tmp_path / "out"


def test_main_cli_registers_constant_force_events(tmp_path):
    args = build_cli_parser().parse_args(
        [
            "postprocess",
            "constant-force-events",
            "--contract",
            str(tmp_path / "contract.json"),
            "--output",
            str(tmp_path / "out"),
        ]
    )
    assert args.func.__name__ == "_cmd_postprocess_constant_force_events"
