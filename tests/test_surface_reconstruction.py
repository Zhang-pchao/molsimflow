"""Physical counterexamples for reconstruction and staged observation clocks."""
import numpy as np
import pytest

from molsimflow.postprocess.surface_reconstruction import (
    reconstruction_outcome, confirm_reconstruction,
    same_column_anchor_departure, transition_ledger,
)


def water(ids, normal=None, bins=None, anchored=None):
    n = len(ids)
    return {
        "ids": np.asarray(ids, int),
        "normal": np.ones(n, bool) if normal is None else np.asarray(normal),
        "bins": np.zeros(n, int) if bins is None else np.asarray(bins),
        "anchored": np.ones(n, bool) if anchored is None else np.asarray(anchored),
    }


def gas(ids, bins=None, components=None, main=None):
    n = len(ids)
    return {
        "ids": np.asarray(ids, int),
        "bins": np.zeros(n, int) if bins is None else np.asarray(bins),
        "components": np.full(n, n) if components is None else np.asarray(components),
        "main": np.ones(n, bool) if main is None else np.asarray(main),
    }


def fate(current_water=None, origin_gas=None, current_gas=None, **kwargs):
    return reconstruction_outcome(
        water([1, 2, 3, 4, 5, 6]), water([4, 5, 6]),
        water([4, 5, 6]) if current_water is None else current_water,
        gas([]) if origin_gas is None else origin_gas,
        gas([]) if current_gas is None else current_gas, [0, 1], **kwargs,
    )


def test_full_frozen_identity_flux_dictionary():
    assert fate(current_water=water([1, 4, 5, 6, 7, 8]), origin_gas=gas([10]),
                current_gas=gas([10, 11, 12])) == {
        "region_O": 6, "normal_H2O": 6, "recovery_O": 1., "recovery_H2O": 1.,
        "raw_count_fate": "WATER_AND_GAS", "molecular_fate": "WATER_REFILL",
        "chemical_recovery_discordance": False,
        "original_O_remaining": 4, "original_normal_O_remaining": 4,
        "original_O_now_normal": 4, "departing_origin_O_returned": 1,
        "departing_origin_H2O_returned": 1, "incoming_H2O_since_origin": 2,
        "incoming_H2O_after_onset": 2, "water_refill_identity_subtype": "OLD_AND_NEW",
        "region_contact_N2": 3, "contact_N2_in_lost_cells": 3, "gas_gain": 2,
        "lost_column_gas_gain": 2, "intersecting_component": 3,
        "original_contact_N2_remaining": 1, "new_contact_N2": 2,
        "new_supported_contact_N2_in_lost_cells": 2,
        "main_class_new_supported_N2_in_lost_cells_all_gas_graph": 2,
        "preexisting_contact_N2_in_lost_cells": 1, "supported_new_gas_gain": True,
        "count_gas_gain_without_new_identity_support": False,
    }


@pytest.mark.parametrize("old,new", [
    (gas([10, 11, 12]), gas([10, 11, 12])),
    (gas([10, 11, 12]), gas([10, 11, 12, 13, 14], components=[3, 3, 3, 1, 1])),
    (gas([10]), gas([10, 11, 12], bins=[0, 7, 7])),
    (gas([10, 11, 12]), gas([13, 14, 15])),
])
def test_preexisting_isolated_outside_or_turnover_gas_is_not_replacement(old, new):
    r = fate(origin_gas=old, current_gas=new)
    assert r["molecular_fate"] == "UNRESOLVED_DEFICIT"
    assert not r["supported_new_gas_gain"]


def test_positive_replacement_and_mixed_controls():
    assert fate(origin_gas=gas([10]), current_gas=gas([10, 11, 12]))[
        "molecular_fate"] == "N2_REPLACEMENT"
    assert fate(current_water=water([4, 5, 6, 7]), origin_gas=gas([10]),
                current_gas=gas([10, 11, 12]))["molecular_fate"] == "MIXED_WATER_GAS"


def test_return_and_new_arrival_fluxes_stay_separate():
    assert fate(current_water=water([1, 2, 3, 4, 5, 6]))["water_refill_identity_subtype"] == "OLD_RETURN"
    assert fate(current_water=water([4, 5, 6, 7, 8, 9]))["water_refill_identity_subtype"] == "NEW_ARRIVAL"


def test_chemical_conversion_does_not_create_normal_water():
    r = fate(current_water=water([4, 5, 6, 7, 8, 9], normal=[1, 1, 1, 0, 0, 0]))
    assert r["molecular_fate"] == "CHEMISTRY_DISCORDANT_COUNTS"


def test_initial_ion_return_is_not_original_normal_water_return():
    r = reconstruction_outcome(
        water([1, 2, 3, 4, 5, 6, 7, 8, 9], normal=[1, 1, 1, 1, 1, 1, 0, 0, 0]),
        water([4, 5, 6, 8, 9], normal=[1, 1, 1, 0, 0]),
        water([4, 5, 6, 7, 8, 9]), gas([]), gas([]), [0, 1])
    assert r["departing_origin_O_returned"] == 1
    assert r["departing_origin_H2O_returned"] == 0


def test_all_gas_membership_does_not_substitute_for_main_only_graph():
    all_graph = fate(current_gas=gas([10, 11, 12], main=[1, 1, 0]))
    main_graph = fate(current_gas=gas([10, 11], components=[1, 1]))
    assert all_graph["molecular_fate"] == "N2_REPLACEMENT"
    assert all_graph["main_class_new_supported_N2_in_lost_cells_all_gas_graph"] == 2
    assert main_graph["molecular_fate"] == "UNRESOLVED_DEFICIT"


def test_missing_chemistry_is_unknown_and_nan_flags_fail():
    assert fate(chemistry_valid=False) == {
        "molecular_fate": "UNOBSERVED", "observation_status": "CHEMISTRY_UNQUALIFIED"}
    with pytest.raises(ValueError, match="unknown"):
        fate(current_water=water([4, 5, 6], normal=[1, np.nan, 1]))


@pytest.mark.parametrize("bad", [water([4, 4, 6]), water([4, 5, 6], normal=[True])])
def test_malformed_identity_alignment_fails(bad):
    with pytest.raises(ValueError):
        fate(current_water=bad)


def confirmation_case(n=201, cadence=.1, offset=0., **overrides):
    args = dict(labels=np.full(n, "N2_REPLACEMENT", dtype=object),
                geometry_ok=np.ones(n, bool), times_ps=offset + np.arange(n) * cadence,
                epochs=np.zeros(n, int), endpoint=0, duration_ps=20.,
                expected_dt_ps=cadence)
    args.update(overrides)
    return args


def test_dwell_is_inclusive_and_cadence_explicit():
    assert confirm_reconstruction(**confirmation_case()) == "COMPLETE_STABLE"
    assert confirm_reconstruction(**confirmation_case(200)) == "CENSORED_END"
    assert confirm_reconstruction(**confirmation_case(401, .05, 32.7)) == "COMPLETE_STABLE"
    with pytest.raises(ValueError, match="whole number"):
        confirm_reconstruction(**confirmation_case(duration_ps=20.03))


def test_transient_occupancy_breaks_confirmation():
    args = confirmation_case()
    args["labels"][3] = "UNRESOLVED_DEFICIT"
    assert confirm_reconstruction(**args) == "COMPLETE_CHANGED"
    assert confirm_reconstruction(**args, stride=5) == "COMPLETE_STABLE"


@pytest.mark.parametrize("kind,expected", [
    ("epoch", "CENSORED_EPOCH"), ("source", "CENSORED_SOURCE"),
    ("gap", "CENSORED_GAP"), ("chemistry", "CHEMISTRY_GAP_CENSORED"),
    ("unobserved", "CHEMISTRY_GAP_CENSORED"), ("geometry", "GEOMETRY_UNQUALIFIED"),
])
def test_native_cadence_quality_is_never_decimated(kind, expected):
    args = confirmation_case(stride=5)
    if kind == "epoch":
        args["epochs"][3:] = 1
    elif kind == "source":
        args["source_ids"] = np.array(["a"] * 3 + ["b"] * 198)
    elif kind == "gap":
        args["times_ps"][3:] += .1
    elif kind == "chemistry":
        args["chemistry_valid"] = np.ones(201, bool)
        args["chemistry_valid"][3] = False
    elif kind == "unobserved":
        args["labels"][3] = "UNOBSERVED"
    else:
        args["geometry_ok"][3] = False
    assert confirm_reconstruction(**args) == expected


def test_spatial_gas_support_can_persist_through_complete_identity_exchange():
    labels = [fate(current_gas=gas([10 + 3 * k, 11 + 3 * k, 12 + 3 * k]))["molecular_fate"]
              for k in range(201)]
    assert confirm_reconstruction(**confirmation_case(labels=labels)) == "COMPLETE_STABLE"


def test_same_column_positive_departure_and_constant_count_turnover():
    before = water([1, 2, 3, 4, 5, 6], bins=[0, 0, 1, 5, 5, 5])
    after = water([4, 5, 6], bins=[5, 5, 5])
    result = same_column_anchor_departure(before, after, (4, 4))
    assert result["same_column_anchor_departure_pass"]
    assert result["loss_cells"] == [0, 1]
    turnover = water([4, 5, 6, 7, 8, 9], bins=[5, 5, 5, 0, 0, 1])
    assert not same_column_anchor_departure(before, turnover, (4, 4))["same_column_anchor_departure_pass"]


def test_departed_anchors_outside_connected_loss_are_insufficient():
    before = water([1, 2, 3, 4, 5, 6, 7, 8, 9], bins=[10, 10, 11, 0, 0, 1, 5, 5, 5],
                   anchored=[1, 1, 1, 0, 0, 0, 0, 0, 0])
    after = water([7, 8, 9], bins=[5, 5, 5], anchored=[0, 0, 0])
    r = same_column_anchor_departure(before, after, (5, 5))
    assert r["count_pass"] and r["connected_pass"]
    assert r["loss_cells"] == [0, 1]
    assert not r["same_column_anchor_departure_pass"]


def test_conversion_and_movement_within_region_do_not_depart():
    before = water([1, 2, 3, 4, 5, 6], bins=[0, 0, 1, 5, 5, 5])
    converted = water([1, 2, 3, 4, 5, 6], bins=[0, 0, 1, 5, 5, 5],
                      normal=[0, 0, 0, 1, 1, 1], anchored=[0, 0, 0, 1, 1, 1])
    r = same_column_anchor_departure(before, converted, (4, 4))
    assert not r["same_column_anchor_departure_pass"]
    moved = water([1, 2, 3], bins=[5, 5, 5])
    r = same_column_anchor_departure(before, moved, (4, 4))
    assert r["column_initial_anchored_IDs_departed_R6"] == 0
    assert not r["same_column_anchor_departure_pass"]


def ledger(contact, *, supply=None, arrival=None, **kwargs):
    n = len(contact)
    args = dict(supply=[1] * n if supply is None else supply,
                arrival=[0] * n if arrival is None else arrival, contact=contact,
                times_ps=np.arange(n, dtype=float), epochs=[0] * n,
                expected_dt_ps=1., lag_ps=4., dwell_ps=2., persistence_ps=2.)
    args.update(kwargs)
    return transition_ledger(**args)


def test_supply_shortage_and_no_arrival_have_distinct_denominators():
    no_supply = ledger([0] * 7, supply=[0] * 7)
    assert no_supply["denominators"]["observed_no_supply_samples"] == 7
    assert no_supply["risk_origins"] == []
    supplied = ledger([0] * 7)
    assert supplied["risk_origins"][0]["arrival_status"] == "NO_ARRIVAL_WITHIN_LAG"
    assert supplied["risk_origins"][0]["establishment"]["outcome"] == 0


def test_arrival_without_establishment_is_retained():
    result = ledger([0, 1, 0, 1, 0, 0, 0], arrival=[0, 1, 1, 1, 1, 0, 0])
    row = result["risk_origins"][0]
    assert row["arrival_status"] == "OBSERVED_ARRIVAL"
    assert row["establishment"]["status"] == "NO_CONFIRMATION_WITHIN_LAG"
    assert len(result["episodes"]["contact"]) == 2


def test_establishment_clock_is_confirmation_and_maintenance_can_fail():
    row = ledger([0, 1, 1, 1, 0, 0, 0], arrival=[0, 1, 1, 1, 1, 0, 0])["risk_origins"][0]
    assert row["establishment"]["first_index"] == 1
    assert row["confirmed_time_ps"] == 3.
    assert row["maintenance_status"] == "COMPLETE_CHANGED"


def test_seam_and_unknown_cancel_pending_confirmation():
    for kwargs in [
        {"epochs": [0, 0, 1, 1, 1, 1, 1]},
        {"observation_valid": [1, 1, 0, 1, 1, 1, 1]},
    ]:
        row = ledger([0, 1, 1, 1, 1, 1, 1], **kwargs)["risk_origins"][0]
        assert row["establishment"]["outcome"] is None
        assert row["maintenance_status"] == "NOT_ESTABLISHED"


def test_late_confirmation_after_unknown_preserves_unknown_first_passage():
    row = ledger([0, -1, 1, 1, 1, 1, 1])["risk_origins"][0]
    assert row["establishment"]["outcome"] == 1
    assert not row["establishment"]["first_passage_resolved"]
    assert row["maintenance_status"] == "COMPLETE_STABLE"


def test_local_gap_does_not_assert_whole_bubble_motion():
    result = ledger([0, 1, 1, 1, 0, 1, 1])
    assert len(result["episodes"]["contact"]) == 2
    assert "net_translation" not in result
    assert "bubble_detachment" not in result


def test_unknown_contact_at_supply_origin_is_not_an_observed_risk_exclusion():
    result = ledger([-1, 0, 0, 0, 0, 0, 0])
    assert result["risk_origins"][0]["establishment"] == {
        "status": "CENSORED_UNKNOWN_ORIGIN", "outcome": None}
    assert result["denominators"]["supplied_contact_free_origins"] == 0
    assert result["denominators"]["supplied_existing_contact_origins"] == 0
    assert result["denominators"]["supplied_unknown_contact_origins"] == 1


def test_confirmation_includes_endpoint_when_stride_does_not_divide_dwell():
    assert confirm_reconstruction(
        ["A", "A", "A", "B"], [1] * 4, [0., .1, .2, .3], [0] * 4, 0,
        duration_ps=.3, expected_dt_ps=.1, stride=2) == "COMPLETE_CHANGED"


def test_establishment_uses_exact_sample_clock_for_decimal_endpoint():
    result = transition_ledger(
        [1] * 4, [0] * 4, [0, 1, 1, 1], [0., .1, .2, .3], [0] * 4,
        expected_dt_ps=.1, lag_ps=.3, dwell_ps=.2, persistence_ps=0.)
    row = result["risk_origins"][0]
    assert row["establishment"]["confirmed_index"] == 3
    assert row["confirmed_time_ps"] == .3
    assert row["maintenance_status"] == "COMPLETE_STABLE"


def test_exact_sample_clock_never_hides_missing_physical_samples():
    result = transition_ledger(
        [1] * 4, [0] * 4, [0, 1, 1, 1], [0., .1, .3, .4], [0] * 4,
        expected_dt_ps=.1, lag_ps=.4, dwell_ps=.2, persistence_ps=0.)
    row = result["risk_origins"][0]
    assert row["establishment"]["outcome"] is None
    assert row["maintenance_status"] == "NOT_ESTABLISHED"
