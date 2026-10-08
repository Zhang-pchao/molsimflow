from molsimflow.postprocess.tpcl_snapshot_selection import (
    SelectionConfig,
    _eligible_local_null,
    _select_events,
)


def test_selects_response_strata_with_unique_blocks_and_rejects_nearby_null():
    rows = [
        {
            "cluster_id": str(index),
            "time_block_200ps": str(index),
            "response_affected_arc_fraction": str(index / 20),
        }
        for index in range(15)
    ]
    config = SelectionConfig(pairs_per_surface=6, response_strata=3)
    selected = _select_events(rows, config)
    assert len(selected) == 6
    assert len({int(row["time_block_200ps"]) for row, _ in selected}) == 6
    assert {stratum for _, stratum in selected} == {0, 1, 2}

    control = {"sample_time_ns": "1.0", "primary_arc_index": "2"}
    event = {"transition_time_ns": "1.003", "primary_arc_index": "10"}
    assert not _eligible_local_null(control, [event], config, arc_count=10)
    assert _eligible_local_null(control, [event], config, arc_count=12)
