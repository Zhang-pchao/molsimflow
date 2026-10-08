import csv

from molsimflow.postprocess.tpcl_snapshot_contrasts import run_contrasts


def test_builds_paired_difference_in_differences(tmp_path):
    source = tmp_path / "mechanics.csv"
    rows = []
    for pair_index, response in enumerate((0.2, 0.8), start=1):
        for radius in (4.0, 6.0, 8.0):
            for sample_kind, base in (("event", 10.0), ("circular_shift_control", 3.0)):
                for phase, increment in (("pre", 0.0), ("transition", 2.0), ("post", 1.0)):
                    value = base + increment
                    if sample_kind == "event" and phase != "pre":
                        value += pair_index
                    rows.append(
                        {
                            "pair_id": f"case__{pair_index:02d}",
                            "case_id": "case",
                            "sample_kind": sample_kind,
                            "phase": phase,
                            "patch_radius_A": radius,
                            "source_time_block_200ps": pair_index,
                            "response_stratum": pair_index - 1,
                            "response_affected_arc_fraction": response,
                            "mode": "radial",
                            "accepted": 1,
                            "force": value,
                        }
                    )
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output = tmp_path / "output"
    summary = run_contrasts(
        source,
        ("force",),
        output,
        primary_patch_radius_A=6.0,
        bootstrap_draws=100,
        permutation_draws=100,
        seed=7,
        group_fields=("mode",),
        required_equal=(("accepted", "1"),),
        drop_incomplete_groups=True,
    )
    assert summary["pair_count"] == 2
    assert summary["group_fields"] == ["mode"]
    assert summary["required_equal"] == [["accepted", "1"]]
    assert summary["patch_radii_A"] == [4.0, 6.0, 8.0]
    with (output / "pair_contrasts.csv").open(newline="", encoding="utf-8") as handle:
        contrasts = list(csv.DictReader(handle))
    did = [
        float(row["value"])
        for row in contrasts
        if row["patch_radius_A"] == "6.0"
        and row["mode"] == "radial"
        and row["contrast"] == "did_transition_minus_pre"
    ]
    assert did == [1.0, 2.0]
