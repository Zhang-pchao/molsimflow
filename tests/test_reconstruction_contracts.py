from dataclasses import replace

import numpy as np
import pytest

from molsimflow.postprocess.reconstruction_contracts import (
    HoldoutMasks, ObservationRecord, ParentKey, ParentRecord, SiteKey, SourceKey,
    grouped_holdout_masks, validate_fold_support, validate_observation_contract,
    validate_parent_lineage,
)


def observation(step=0, source="source", epoch="epoch", cadence=0.1, **kwargs):
    return ObservationRecord(
        SourceKey(source, epoch, step, cadence, 0.0005), step * 0.0005, 0, 0.0,
        (True, True, False), "atoms", "types", "molecules", f"payload-{step}", **kwargs,
    )


def parent(sample, root="p1", seed="1", arm="sham", construction=None, **kwargs):
    return ParentRecord(sample, "slab", root, construction or f"build-{root}", arm,
                        seed, **({"lineage_verified": True} | kwargs))


def test_same_step_in_distinct_sources_and_epochs_remains_distinct():
    rows = [observation(), observation(source="other"), observation(epoch="reset")]
    result = validate_observation_contract(rows)
    assert result.canonical_indices == (0, 1, 2)
    assert result.duplicate_indices == ()


def test_time_uses_explicit_epoch_origin():
    row = replace(observation(600), origin_step=600, origin_time_ps=300.0, time_ps=300.0)
    validate_observation_contract([row])
    with pytest.raises(ValueError, match="step/time"):
        validate_observation_contract([replace(row, time_ps=0.3)])


def test_hf_core_duplicates_require_full_parity_and_opt_in():
    core = observation(200, physical_frame_id="physical-1")
    hf = replace(core, key=replace(core.key, cadence_ps=0.05), stream_id="hf")
    with pytest.raises(ValueError, match="duplicate physical"):
        validate_observation_contract([core, hf])
    result = validate_observation_contract([core, hf], allow_parity_duplicates=True)
    assert result.canonical_indices == (0,)
    assert result.duplicate_indices == (1,)
    # Distinct raw source aliases are allowed only with declared physical identity.
    hf = replace(hf, key=replace(hf.key, source_id="hf-source"))
    assert validate_observation_contract([core, hf], allow_parity_duplicates=True) == result


@pytest.mark.parametrize("change", [
    {"payload_hash": "different"}, {"atom_map_hash": "different"},
    {"type_map_hash": "different"}, {"molecule_map_hash": "different"},
    {"pbc": (True, False, False)},
    {"chemical_mask_valid": False, "missing_reason": "H ownership ambiguous"},
    {"time_ps": 0.2, "origin_time_ps": 0.1},
])
def test_physical_alias_conflicts_cannot_be_deduplicated(change):
    first = observation(200, physical_frame_id="frame")
    second = replace(first, key=replace(first.key, source_id="hf"), **change)
    with pytest.raises(ValueError):
        validate_observation_contract([first, second], allow_parity_duplicates=True)


def test_duplicate_raw_row_is_rejected_even_with_parity_opt_in():
    row = observation()
    with pytest.raises(ValueError, match="same source/epoch/stream"):
        validate_observation_contract([row, row], allow_parity_duplicates=True)


def test_declared_physical_alias_cannot_cross_velocity_reset_epoch():
    first = observation(physical_frame_id="frame")
    second = replace(first, key=replace(first.key, epoch_id="reset"))
    with pytest.raises(ValueError, match="full parity"):
        validate_observation_contract([first, second], allow_parity_duplicates=True)


def test_unknown_h_is_explicit_and_not_a_negative_observation():
    rows = [observation(), observation(200, chemical_mask_valid=False,
                                      missing_reason="H ownership unresolved")]
    result = validate_observation_contract(rows, requested_keys=[r.key for r in rows])
    assert result.canonical_indices == (0, 1)
    assert result.missing_indices == (1,)
    with pytest.raises(ValueError, match="explicit missing_reason"):
        validate_observation_contract([replace(rows[1], missing_reason=None)])


def test_requested_interior_frame_cannot_be_silently_dropped():
    rows = [observation(), observation(400)]
    with pytest.raises(ValueError, match="missing requested frames: 1"):
        validate_observation_contract(rows, requested_keys=[observation(s).key for s in (0, 200, 400)])


@pytest.mark.parametrize("change", [
    {"pbc": (False, True, False)}, {"atom_map_hash": "different-atoms"},
    {"type_map_hash": "different-types"}, {"molecule_map_hash": "different-inventory"},
])
def test_metadata_must_stay_fixed_within_source_epoch(change):
    with pytest.raises(ValueError, match="mixed PBC"):
        validate_observation_contract([observation(), replace(observation(200), **change)])


def test_variable_inventory_across_sources_is_allowed_with_explicit_maps():
    rows = [observation(), replace(observation(source="new-run"), molecule_map_hash="250-N2")]
    assert len(validate_observation_contract(rows).canonical_indices) == 2


@pytest.mark.parametrize("change", [
    {"step": 0.5}, {"dt_ps": 0}, {"cadence_ps": float("nan")},
    {"cadence_ps": 0.0006}, {"epoch_id": ""},
])
def test_invalid_frame_keys_fail_before_join(change):
    row = observation()
    with pytest.raises(ValueError):
        validate_observation_contract([replace(row, key=replace(row.key, **change))])


def test_shared_parent_seeds_and_arms_do_not_create_independent_parents():
    rows = [parent("r1"), parent("r2", seed="2"), parent("treated", arm="pattern")]
    assert set(validate_parent_lineage(rows).values()) == {ParentKey("slab", "p1")}
    with pytest.raises(ValueError, match="insufficient independent"):
        validate_parent_lineage(rows, min_independent_parents=2)


def test_shared_construction_root_cannot_be_relabelled_as_new_parent():
    rows = [parent("r1"), parent("r2", root="pretend", construction="build-p1")]
    with pytest.raises(ValueError, match="shared construction root"):
        validate_parent_lineage(rows)


def test_verified_distinct_roots_can_be_grouped_but_do_not_establish_physics():
    records = [parent("r1"), parent("r2", root="p2")]
    assert len(set(validate_parent_lineage(records, min_independent_parents=2).values())) == 2
    with pytest.raises(ValueError, match="unverified"):
        validate_parent_lineage([replace(records[0], lineage_verified=False)])


def test_conflicting_root_for_one_parent_fails():
    with pytest.raises(ValueError, match="conflicting construction"):
        validate_parent_lineage([parent("a"), parent("b", construction="different")])


def test_strict_crossed_holdout_excludes_both_cross_strips():
    parents = [ParentKey("slab", p) for p in ("p1", "p1", "p2", "p2")]
    sites = [SiteKey("slab", s) for s in ("s1", "s2", "s1", "s2")]
    masks = grouped_holdout_masks(parents, sites, held_out_parents=[parents[-1]],
                                  held_out_sites=[sites[-1]])
    assert masks.train.tolist() == [True, False, False, False]
    assert masks.test.tolist() == [False, False, False, True]
    assert masks.excluded.tolist() == [False, True, True, False]


def supported_fixture():
    labels = ["WATER_REFILL", "N2_REPLACEMENT"] * 4
    parents = ["p1"] * 4 + ["p2"] * 4
    sites = ["s1"] * 2 + ["s2"] * 2 + ["s1"] * 2 + ["s2"] * 2
    masks = grouped_holdout_masks(parents, sites, held_out_parents=["p2"], held_out_sites=["s2"])
    return labels, parents, sites, masks


def test_supported_fold_passes_necessary_gate_only():
    result = validate_fold_support(*supported_fixture())
    assert result.status == "READY"
    assert result.fit_allowed
    assert result.counts == {split: {"WATER_REFILL": 1, "N2_REPLACEMENT": 1}
                             for split in ("train", "test")}


def test_absent_competing_class_returns_not_fit():
    _, parents, sites, masks = supported_fixture()
    result = validate_fold_support(["WATER_REFILL"] * 8, parents, sites, masks)
    assert result.status == "NOT_FIT"
    assert not result.fit_allowed
    assert result.counts["test"]["N2_REPLACEMENT"] == 0


def test_unknown_outcome_and_excluded_cross_strips_do_not_supply_support():
    labels, parents, sites, masks = supported_fixture()
    valid = np.ones(8, dtype=bool)
    valid[-1] = False
    result = validate_fold_support(labels, parents, sites, masks, valid=valid)
    assert result.status == "NOT_FIT"
    assert result.counts["test"]["N2_REPLACEMENT"] == 0


def test_stratum_support_is_checked_even_when_global_class_counts_pass():
    args = supported_fixture()
    strata = ["background-A"] * 4 + ["background-B"] * 4
    result = validate_fold_support(*args, strata=strata)
    assert result.status == "NOT_FIT"
    assert any("unsupported stratum 'background-B'" in reason for reason in result.reasons)


def test_manual_split_with_parent_or_site_leakage_cannot_pass():
    labels, parents, sites, _ = supported_fixture()
    train = np.array([True] * 4 + [False] * 4)
    masks = HoldoutMasks(train, ~train, np.zeros(8, dtype=bool))
    assert "site leakage" in validate_fold_support(labels, parents, sites, masks).reasons
    masks = HoldoutMasks(np.arange(8) < 2, np.arange(8) >= 2, np.zeros(8, dtype=bool))
    assert "parent leakage" in validate_fold_support(labels, parents, sites, masks).reasons


def test_replica_and_site_floors_are_enforced_separately():
    result = validate_fold_support(*supported_fixture(), min_parents_per_class=2,
                                   min_sites_per_class=2)
    assert result.status == "NOT_FIT"
    assert any("parents" in reason for reason in result.reasons)
    assert any("sites" in reason for reason in result.reasons)


def test_empty_crossed_test_cell_is_not_fit():
    parents, sites = ["p1", "p2"], ["s1", "s2"]
    masks = grouped_holdout_masks(parents, sites, held_out_parents=["p2"], held_out_sites=["s1"])
    result = validate_fold_support(["WATER_REFILL", "N2_REPLACEMENT"], parents, sites, masks)
    assert result.status == "NOT_FIT"


def test_malformed_masks_and_unknown_holdout_keys_fail_closed():
    labels, parents, sites, masks = supported_fixture()
    with pytest.raises(ValueError, match="partition"):
        validate_fold_support(labels, parents, sites, replace(masks, excluded=np.ones(8, dtype=bool)))
    with pytest.raises(ValueError, match="observed parent"):
        grouped_holdout_masks(parents, sites, held_out_parents=["absent"], held_out_sites=["s2"])
    with pytest.raises(ValueError, match="boolean"):
        validate_fold_support(labels, parents, sites, masks, valid=np.ones(8, dtype=int))


def test_observer_reference_is_fixed_within_source_epoch():
    first = replace(observation(), observer_reference_hash="immutable-scaffold")
    second = replace(observation(200), observer_reference_hash="treated-heavy-atoms")
    with pytest.raises(ValueError, match="observer reference"):
        validate_observation_contract([first, second])


def test_physical_alias_requires_same_observer_reference():
    first = replace(observation(physical_frame_id="frame"),
                    observer_reference_hash="immutable-scaffold")
    second = replace(first, stream_id="hf", observer_reference_hash="other-reference")
    with pytest.raises(ValueError):
        validate_observation_contract([first, second], allow_parity_duplicates=True)


def test_distinct_legacy_materials_do_not_require_same_observer_reference():
    first = replace(observation(source="material-A"), observer_reference_hash="reference-A")
    second = replace(observation(source="material-B"), observer_reference_hash="reference-B")
    assert validate_observation_contract([first, second]).canonical_indices == (0, 1)
    with pytest.raises(ValueError, match="observer_reference_hash"):
        validate_observation_contract([replace(first, observer_reference_hash="")])


@pytest.mark.parametrize("physical_ids", [("alias-A", "alias-B"), (None, "alias-A"),
                                         ("alias-A", None)])
@pytest.mark.parametrize("allow_duplicates", [False, True])
def test_raw_frame_cannot_change_physical_identity_between_streams(physical_ids, allow_duplicates):
    first = observation(200, physical_frame_id=physical_ids[0])
    second = replace(first, key=replace(first.key, cadence_ps=0.05), stream_id="hf",
                     physical_frame_id=physical_ids[1])
    with pytest.raises(ValueError, match="inconsistent physical identity"):
        validate_observation_contract([first, second], allow_parity_duplicates=allow_duplicates)


def test_core_and_hf_cannot_alias_different_steps_with_incompatible_time_origins():
    first = observation(0, physical_frame_id="frame")
    second = replace(observation(200, physical_frame_id="frame"), stream_id="hf",
                     time_ps=0.0, origin_step=200, origin_time_ps=0.0)
    with pytest.raises(ValueError, match="inconsistent step/time mapping"):
        validate_observation_contract([first, second], allow_parity_duplicates=True)


def test_equivalent_time_anchors_across_streams_preserve_physical_parity():
    first = observation(200, physical_frame_id="frame")
    second = replace(first, stream_id="hf", origin_step=200, origin_time_ps=0.1)
    result = validate_observation_contract([first, second], allow_parity_duplicates=True)
    assert result.canonical_indices == (0,)
    assert result.duplicate_indices == (1,)
