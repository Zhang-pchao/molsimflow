"""Contract regressions for joining historical geometry/contact tables."""
from dataclasses import replace
import copy
import json

import pytest

from molsimflow.postprocess.contact_state_inventory import (
    ContactStateCase, GRID_STATES, MATCH_VALIDATION, SourceEpoch, inventory_contact_states,
)


def fixture_rows():
    config = ContactStateCase(
        name="late_case_demo", case="surface", tier="E", dt_ps=0.5,
        origin_step=10, origin_time_ps=100.0,
        epochs=(SourceEpoch("original", 10, 14, "/archive/original"),
                SourceEpoch("restart", 16, 18, "/archive/restart")),
        periodic=(True, True, False), geometry_dt_ps=1.0, contact_dt_ps=1.0,
    )
    geometry, contact = [], []
    for k in range(5):
        segment = "original" if k < 3 else "restart"
        wet = k != 1
        row = dict(tier="E", case="surface", step=str(10 + 2*k), time_ps=str(100+k),
                   segment=segment, source_path=f"/archive/{segment}",
                   source_frame_index=str(k if k < 3 else k-3),
                   canonical_frame_sha256=f"{k:064x}", source_frame_verified="True",
                   cell_area_A2="2.5", cells="10", main_N2="30", contact_water_O="100")
        counts = (7 if wet else 8, 1, int(wet), 1, 0)
        for state, count in zip(GRID_STATES, counts):
            row[f"{state}_cells"] = str(count)
            row[f"{state}_area_A2"] = str(count * 2.5)
        geometry.append(row)
        for distance in (3.5, 4.0, 4.5):
            present = k != 0
            contact.append(dict(name=config.name, time_ps=str(100+k), segment=segment,
                                distance_A=str(distance), main_supported_patches=str(int(present)),
                                unique_main_contact_N2="3" if present else "0",
                                some_main_supported_contact=str(present)))
    return config, geometry, contact


def test_exact_join_preserves_sources_separate_definitions_and_sample_counts():
    config, geometry, contact = fixture_rows()
    original = copy.deepcopy((geometry, contact))
    result = inventory_contact_states(geometry, contact, config=config)
    assert (geometry, contact) == original
    assert result["summary"]["geometry_samples"] == 5
    assert result["summary"]["contact_samples"] == 5
    assert result["summary"]["contact_frames"] == 4
    assert result["summary"]["water_separated_frames"] == 4
    assert result["summary"]["coexistence_frames"] == 3
    assert result["summary"]["match_validation"] == MATCH_VALIDATION
    joined = result["joined"][0]
    assert joined["canonical_frame_sha256"] == geometry[0]["canonical_frame_sha256"]
    assert joined["source_path"] == "/archive/original"
    assert joined["contact_water_O"] == "100"  # Full-surface inventory retained, not relabeled.
    assert joined["contact_shells"]["4"] == contact[1]
    assert joined["water_separated_area_A2"] == "2.5"
    json.dumps(result, allow_nan=False)


def test_source_seam_censors_contact_even_when_state_persists():
    config, geometry, contact = fixture_rows()
    result = inventory_contact_states(geometry, contact, config=config)
    episodes = [r for r in result["episodes"] if r["observed_state"] == "contact"]
    assert len(episodes) == 2
    assert episodes[0]["left_censored"] is False
    assert episodes[0]["right_censored"] is True
    assert episodes[0]["end_reason"] == "source_or_gap"
    assert episodes[1]["left_censored"] is True
    assert episodes[1]["right_censored"] is True
    absence = [r for r in result["episodes"] if r["observed_state"] == "absence"]
    assert len(absence) == 1 and absence[0]["samples"] == 1
    assert absence[0]["left_censored"] is True and absence[0]["right_censored"] is False


def test_missing_entire_contact_time_is_censored_not_imputed_absence():
    config, geometry, contact = fixture_rows()
    contact = [row for row in contact if row["time_ps"] != "101"]
    result = inventory_contact_states(geometry, contact, config=config)
    assert result["summary"]["contact_samples"] == 4
    assert result["summary"]["geometry_samples"] == 5
    absence = [r for r in result["episodes"] if r["observed_state"] == "absence"][0]
    assert absence["right_censored"] is True and absence["end_reason"] == "source_or_gap"


def test_deterministic_representatives_use_observed_categories_only():
    config, geometry, contact = fixture_rows()
    a = inventory_contact_states(geometry, contact, config=config)
    b = inventory_contact_states(list(reversed(geometry)), list(reversed(contact)), config=config)
    assert a == b
    representatives = a["representatives"]
    assert {r["category"] for r in representatives} == {
        "contact_and_water_separated", "contact_without_water_separated",
        "water_separated_without_contact",
    }
    both = next(r for r in representatives if r["category"] == "contact_and_water_separated")
    assert both["time_ps"] == 103.0  # Median of the observed 102, 103, 104 times.
    assert both["input_record"] in a["joined"]
    assert "not a qualified" in both["interpretation"]


@pytest.mark.parametrize("field,value,message", [
    ("source_frame_verified", "False", "not historically verified"),
    ("source_frame_verified", "unknown", "Boolean"),
    ("canonical_frame_sha256", "abc", "hash"),
    ("source_path", "/wrong", "source path/step"),
    ("segment", "undeclared", "undeclared"),
    ("time_ps", "100.000000001", "time/step"),
    ("step", "10.5", "integer"),
    ("cell_area_A2", "nan", "Nonfinite"),
    ("water_separated_cells", "-1", "integer"),
    ("water_separated_area_A2", "3.0", "area/count"),
    ("cells", "11", "does not close"),
    ("case", "wrong", "case/tier"),
    ("tier", "wrong", "case/tier"),
])
def test_geometry_validation_rejects_corruption(field, value, message):
    config, geometry, contact = fixture_rows()
    geometry[0][field] = value
    with pytest.raises(ValueError, match=message):
        inventory_contact_states(geometry, contact, config=config)


def test_missing_geometry_field_and_duplicate_identity_fail():
    config, geometry, contact = fixture_rows()
    with pytest.raises(ValueError, match="Duplicate geometry"):
        inventory_contact_states(geometry + [geometry[0]], contact, config=config)
    del geometry[0]["mixed_area_A2"]
    with pytest.raises(ValueError, match="missing fields"):
        inventory_contact_states(geometry, contact, config=config)


def test_no_nearest_time_or_epoch_fallback():
    config, geometry, contact = fixture_rows()
    contact[0]["time_ps"] = "100.000000001"
    with pytest.raises(ValueError, match="off the declared sample grid"):
        inventory_contact_states(geometry, contact, config=config)
    contact[0]["time_ps"] = "100"
    contact[0]["segment"] = "restart"
    with pytest.raises(ValueError, match="no exact time\\+epoch"):
        inventory_contact_states(geometry, contact, config=config)


def test_duplicate_contact_and_missing_shell_fail():
    config, geometry, contact = fixture_rows()
    with pytest.raises(ValueError, match="Duplicate contact"):
        inventory_contact_states(geometry, contact + [contact[0]], config=config)
    with pytest.raises(ValueError, match="Incomplete three-shell"):
        inventory_contact_states(geometry, contact[1:], config=config)


@pytest.mark.parametrize("field,value,message", [
    ("distance_A", "5", "undeclared"),
    ("unique_main_contact_N2", "31", "exceeds"),
    ("main_supported_patches", "1", "inconsistent"),
    ("some_main_supported_contact", "maybe", "Boolean"),
])
def test_contact_validation_rejects_corruption(field, value, message):
    config, geometry, contact = fixture_rows()
    contact[0][field] = value
    with pytest.raises(ValueError, match=message):
        inventory_contact_states(geometry, contact, config=config)


def test_other_cases_are_filtered_without_claiming_their_validation():
    config, geometry, contact = fixture_rows()
    contact.append(dict(name="not_selected", time_ps="invalid"))
    assert inventory_contact_states(geometry, contact, config=config)["summary"]["contact_samples"] == 5
    with pytest.raises(ValueError, match="No contact records"):
        inventory_contact_states(geometry, [], config=config)


@pytest.mark.parametrize("changes,message", [
    ({"periodic": (1, 1, 0)}, "Boolean"),
    ({"dt_ps": 0}, "Positive"),
    ({"distance_shells": (3.5, 4, 4)}, "Three unique"),
    ({"epochs": ()}, "source epoch"),
])
def test_explicit_configuration_is_required(changes, message):
    config, geometry, contact = fixture_rows()
    with pytest.raises(ValueError, match=message):
        inventory_contact_states(geometry, contact, config=replace(config, **changes))
