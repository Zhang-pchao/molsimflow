"""Molecular reconstruction, observed confirmation and staged contact clocks.

These are local observables. They do not establish whole-bubble motion or
chemical causality. All gas graph membership is provided by the caller.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import numpy as np

from .surface_contact_process import sampled_episodes, establishment_outcome
from .surface_water_renewal import periodic_grid_components


def _count_fate(recovery, normal_recovery, gas_gain, lost_gain, component):
    """Frozen inherited count label, subordinate to identity-supported fate."""
    if recovery >= .8:
        return "O_RECOVERY_NON_H2O" if normal_recovery < .8 else "WATER_AND_GAS" if gas_gain > 0 else "WATER_REFILL"
    supported = gas_gain >= 2 and lost_gain >= 2 and component >= 3
    if supported:
        return "N2_REPLACEMENT_CANDIDATE" if recovery <= .2 else "MIXED_WATER_GAS"
    return "PARTIAL_REFILL" if recovery > .2 else "UNRESOLVED_DEFICIT"


def recovery_band(value):
    return 'REFILLED' if value >= .8 else 'PARTIAL' if value > .2 else 'DEFICIT'


def _molecular_outcome(origin_water, onset_water, current_water, origin_gas, current_gas, lost_cells):
    """Return identity fluxes and labels without treating preexisting gas as replacement.

    Water records contain unique IDs and current normal-H2O flags. Gas records are
    the current R6 contact set and contain IDs, grid bins, per-molecule connected
    component sizes, and main-cluster flags. Gross fluxes are not net stoichiometry.
    """
    def water_sets(w):
        ids=np.asarray(w['ids']);normal=np.asarray(w['normal'],bool)
        assert len(np.unique(ids))==len(ids) and len(normal)==len(ids)
        return set(map(int,ids)),set(map(int,ids[normal]))
    base,bn=water_sets(origin_water);lost,ln=water_sets(onset_water);now,nn=water_sets(current_water)
    loss_O=len(base)-len(lost);loss_normal=len(bn)-len(ln)
    assert loss_O >= 3 and loss_normal >= 3
    recovery=(len(now)-len(lost))/loss_O
    normal_recovery=(len(nn)-len(ln))/loss_normal
    old_ids=np.asarray(origin_gas['ids']);new_ids=np.asarray(current_gas['ids'])
    assert len(np.unique(old_ids))==len(old_ids) and len(np.unique(new_ids))==len(new_ids)
    before_lost=np.isin(origin_gas['bins'],lost_cells)
    in_lost=np.isin(current_gas['bins'],lost_cells)
    fresh=~np.isin(new_ids,old_ids)
    sizes=np.asarray(current_gas['components'])
    assert len(sizes)==len(new_ids)
    supported=sizes>=3
    gas_gain=len(new_ids)-len(old_ids)
    lost_gain=int(in_lost.sum()-before_lost.sum())
    component=int(sizes[in_lost].max(initial=0))
    new_supported_lost=int((fresh&in_lost&supported).sum())
    new_support=bool(gas_gain>=2 and lost_gain>=2 and component>=3 and new_supported_lost>=2)
    raw=_count_fate(recovery,normal_recovery,gas_gain,lost_gain,component)
    discordance=recovery_band(recovery)!=recovery_band(normal_recovery)
    if discordance:
        label='CHEMISTRY_DISCORDANT_COUNTS'
    elif recovery>=.8:
        label='WATER_REFILL'
    elif new_support:
        label='N2_REPLACEMENT' if recovery<=.2 else 'MIXED_WATER_GAS'
    else:
        label='PARTIAL_REFILL' if recovery>.2 else 'UNRESOLVED_DEFICIT'
    departed=base-lost
    old_return=((bn-lost) & nn)
    new_normal=(nn-base-lost)
    subtype=('OLD_AND_NEW' if old_return and new_normal else 'OLD_RETURN' if old_return
             else 'NEW_ARRIVAL' if new_normal else 'CHEMICAL_OR_REARRANGEMENT')
    return dict(region_O=len(now),normal_H2O=len(nn),recovery_O=recovery,recovery_H2O=normal_recovery,
        raw_count_fate=raw,molecular_fate=label,chemical_recovery_discordance=discordance,
        original_O_remaining=len(base&now),original_normal_O_remaining=len(bn&nn),original_O_now_normal=len(base&nn),
        departing_origin_O_returned=len(departed&now),departing_origin_H2O_returned=len(old_return),
        incoming_H2O_since_origin=len(nn-base),incoming_H2O_after_onset=len(new_normal),
        water_refill_identity_subtype=subtype if label=='WATER_REFILL' else 'NOT_REFILLED',
        region_contact_N2=len(new_ids),contact_N2_in_lost_cells=int(in_lost.sum()),
        gas_gain=gas_gain,lost_column_gas_gain=lost_gain,intersecting_component=component,
        original_contact_N2_remaining=int(np.isin(new_ids,old_ids).sum()),
        new_contact_N2=int(fresh.sum()),new_supported_contact_N2_in_lost_cells=new_supported_lost,
        main_class_new_supported_N2_in_lost_cells_all_gas_graph=int((fresh&in_lost&supported&np.asarray(current_gas['main'],bool)).sum()),
        preexisting_contact_N2_in_lost_cells=int(before_lost.sum()),
        supported_new_gas_gain=new_support,
        count_gas_gain_without_new_identity_support=bool(gas_gain>=2 and lost_gain>=2 and component>=3 and not new_support))



def _vector(values, name, *, integer=False, boolean=False):
    a = np.asarray(values)
    if a.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if boolean:
        if not np.all(np.isin(a, [False, True])):
            raise ValueError(f"{name} contains unknown or invalid flags")
        return a.astype(bool)
    if integer:
        if a.size and (not np.issubdtype(a.dtype, np.number) or not np.isfinite(a).all()
                       or not np.equal(a, np.floor(a)).all()):
            raise ValueError(f"{name} must contain finite integer values")
        return a.astype(np.int64)
    return a


def _record(record, kind, *, columns=False):
    keys = ["ids", "normal"] if kind == "water" else ["ids", "bins", "components", "main"]
    if columns:
        keys += ["bins", "anchored"]
    out = {}
    for key in keys:
        out[key] = _vector(record[key], key, integer=key not in ("normal", "anchored", "main"),
                           boolean=key in ("normal", "anchored", "main"))
    if any(len(a) != len(out["ids"]) for a in out.values()):
        raise ValueError("Molecular arrays are not identity-aligned")
    if len(np.unique(out["ids"])) != len(out["ids"]):
        raise ValueError("Duplicate molecular identity")
    if "bins" in out and np.any(out["bins"] < 0):
        raise ValueError("Negative grid cell")
    if "components" in out and np.any(out["components"] < 1):
        raise ValueError("Present gas requires a positive component size")
    if columns and np.any(out["anchored"] & ~out["normal"]):
        raise ValueError("An anchored-water flag requires observed normal water")
    return out


def reconstruction_outcome(
    origin_water: Mapping, onset_water: Mapping, current_water: Mapping,
    origin_gas: Mapping, current_gas: Mapping, lost_cells: Sequence[int], *,
    chemistry_valid: bool = True,
) -> dict[str, object]:
    """Return frozen molecular fate and identity fluxes with explicit unknowns.

    Each water record requires unique ids and observed normal flags.
    Each gas record requires ids, bins, components and main.
    Components must come from the graph corresponding to the supplied gas set.
    A main flag only reports all-gas-graph support; it never creates a main-only
    graph. Call this reducer separately with independently computed main graphs.
    Invalid chemistry returns only an UNOBSERVED fate and observation status.
    """
    valid = _vector([chemistry_valid], "chemistry_valid", boolean=True)[0]
    if not valid:
        return {"molecular_fate": "UNOBSERVED", "observation_status": "CHEMISTRY_UNQUALIFIED"}
    waters = [_record(w, "water") for w in (origin_water, onset_water, current_water)]
    gases = [_record(g, "gas") for g in (origin_gas, current_gas)]
    cells = _vector(lost_cells, "lost_cells", integer=True)
    if np.any(cells < 0) or len(np.unique(cells)) != len(cells):
        raise ValueError("Lost cells must be distinct nonnegative cells")
    if (len(waters[0]["ids"]) - len(waters[1]["ids"]) < 3
            or waters[0]["normal"].sum() - waters[1]["normal"].sum() < 3):
        raise ValueError("Reconstruction requires at least three O and normal-water losses")
    return _molecular_outcome(*waters, *gases, cells)


def _clock(times_ps, epochs, expected_dt_ps, source_ids=None):
    t = np.asarray(times_ps, float)
    ep = np.asarray(epochs)
    if t.ndim != 1 or ep.shape != t.shape or not len(t):
        raise ValueError("A nonempty aligned time/epoch trace is required")
    if not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Observed times must be finite and strictly increasing")
    if not np.isfinite(expected_dt_ps) or expected_dt_ps <= 0:
        raise ValueError("Positive finite cadence required")
    source = np.full(len(t), "", dtype=object) if source_ids is None else np.asarray(source_ids)
    if source.shape != t.shape:
        raise ValueError("Source keys must be aligned with time")
    segments = np.asarray([repr((str(s), str(e))) for s, e in zip(source, ep)])
    return t, ep, source, segments


def confirm_reconstruction(
    labels: Sequence, geometry_ok: Sequence[bool], times_ps: Sequence[float],
    epochs: Sequence, endpoint: int, *, duration_ps: float, expected_dt_ps: float,
    stride: int = 1, source_ids: Sequence | None = None,
    chemistry_valid: Sequence[bool] | None = None,
) -> str:
    """Confirm forward occupancy over inclusive observed endpoints.

    A twenty-ps dwell at 0.1 ps requires 201 observations. Stride is only an
    occupancy-label sensitivity: every chemistry, geometry and cadence sample
    remains mandatory. No missing or source-reset observation is interpolated.
    Pass a cumulative chemistry-valid mask if eligibility includes prior history.
    """
    t, ep, source, _ = _clock(times_ps, epochs, expected_dt_ps, source_ids)
    labels = _vector(labels, "labels")
    geom = _vector(geometry_ok, "geometry_ok", boolean=True)
    valid = np.ones(len(t), bool) if chemistry_valid is None else _vector(
        chemistry_valid, "chemistry_valid", boolean=True)
    if any(len(x) != len(t) for x in (labels, geom, valid)):
        raise ValueError("Confirmation arrays must have identical lengths")
    if not isinstance(endpoint, (int, np.integer)) or not 0 <= endpoint < len(t):
        raise ValueError("Invalid endpoint")
    if not isinstance(stride, (int, np.integer)) or stride <= 0:
        raise ValueError("Stride must be a positive integer")
    if not np.isfinite(duration_ps) or duration_ps < 0:
        raise ValueError("Nonnegative finite duration required")
    n_intervals = int(round(duration_ps / expected_dt_ps))
    if not np.isclose(n_intervals * expected_dt_ps, duration_ps, atol=1e-7, rtol=0):
        raise ValueError("Confirmation duration must be a whole number of cadence intervals")
    k = int(endpoint)
    end = k + n_intervals
    if end >= len(t):
        return "CENSORED_END"
    block = slice(k, end + 1)
    if np.any(source[block] != source[k]):
        return "CENSORED_SOURCE"
    if np.any(ep[block] != ep[k]):
        return "CENSORED_EPOCH"
    if not np.allclose(t[block] - t[k], np.arange(n_intervals + 1) * expected_dt_ps,
                       atol=1e-7, rtol=0):
        return "CENSORED_GAP"
    if not valid[block].all() or np.any(np.isin(labels[block], ["UNOBSERVED", None, -1])):
        return "CHEMISTRY_GAP_CENSORED"
    if not geom[block].all():
        return "GEOMETRY_UNQUALIFIED"
    sampled = np.arange(k, end + 1, stride)
    if sampled[-1] != end:
        sampled = np.append(sampled, end)
    return "COMPLETE_STABLE" if np.all(labels[sampled] == labels[k]) else "COMPLETE_CHANGED"


def same_column_anchor_departure(
    before_water: Mapping, after_water: Mapping, shape: Sequence[int], *,
    connectivity: int = 8,
) -> dict[str, object]:
    """Qualify physical anchored departures from the connected lost columns.

    Uses the inherited largest positive-O-loss component (ties use grid order).
    Requires inherited prospective count loss, at least two connected cells,
    three net O and normal-water losses in that same component, at least three
    initially anchored IDs in it absent from the entire observed region, and
    net anchored-water loss. Conversion or within-region redistribution alone
    cannot establish departure.
    """
    before = _record(before_water, "water", columns=True)
    after = _record(after_water, "water", columns=True)
    dims = _vector(shape, "shape", integer=True)
    if len(dims) != 2 or np.any(dims <= 0):
        raise ValueError("Positive two-dimensional grid shape required")
    ncell = int(np.prod(dims))
    if any(np.any(w["bins"] >= ncell) for w in (before, after)):
        raise ValueError("Grid bin outside declared shape")
    delta = np.bincount(before["bins"], minlength=ncell) - np.bincount(after["bins"], minlength=ncell)
    components = periodic_grid_components((delta > 0).reshape(tuple(dims)), connectivity).ravel()
    selected = components >= 0
    if selected.any():
        losses = np.bincount(components[selected], weights=delta[selected])
        cells = np.flatnonzero(components == int(np.argmax(losses)))
    else:
        cells = np.array([], int)
    local = [np.isin(w["bins"], cells) for w in (before, after)]
    loss_o = len(before["ids"]) - len(after["ids"])
    loss_normal = int(before["normal"].sum() - after["normal"].sum())
    loss_anchor = int(before["anchored"].sum() - after["anchored"].sum())
    column_o = int(local[0].sum() - local[1].sum())
    column_normal = int((local[0] & before["normal"]).sum() - (local[1] & after["normal"]).sum())
    departed = np.setdiff1d(before["ids"][local[0] & before["anchored"]], after["ids"])
    retained_converted = np.intersect1d(before["ids"][local[0] & before["anchored"]],
                                       after["ids"][~after["normal"]])
    count_pass = (before["normal"].sum() >= 5
                  and loss_o >= max(3, int(np.ceil(.3 * len(before["ids"]))))
                  and loss_normal >= 3)
    connected_pass = len(cells) >= 2 and column_o >= 3 and column_normal >= 3
    return {
        "loss_O": loss_o, "loss_normal": loss_normal, "loss_anchored": loss_anchor,
        "loss_cells": cells.tolist(), "column_onset_O_loss": column_o,
        "column_onset_normal_loss": column_normal,
        "column_initial_anchored_IDs_departed_R6": int(len(departed)),
        "retained_converted_initial_anchors": int(len(retained_converted)),
        "count_pass": bool(count_pass), "connected_pass": bool(connected_pass),
        "same_column_anchor_departure_pass": bool(
            count_pass and connected_pass and len(departed) >= 3 and loss_anchor >= 1),
    }


def transition_ledger(
    supply: Sequence[int], arrival: Sequence[int], contact: Sequence[int],
    times_ps: Sequence[float], epochs: Sequence, *, expected_dt_ps: float,
    lag_ps: float, dwell_ps: float, persistence_ps: float,
    source_ids: Sequence | None = None, observation_valid: Sequence[bool] | None = None,
) -> dict[str, object]:
    """Separate observed supply, local arrival, establishment and maintenance.

    States are 1 (observed presence), 0 (observed absence), -1 (unknown).
    Supply origins are the first sample of each supply episode, selected without
    future outcomes. Arrival is an independently defined geometric observation,
    supplied by the adapter. Neither a supply episode nor a contact episode
    implies net gas gain, molecular replacement or whole-bubble motion.
    """
    t, ep, source, segments = _clock(times_ps, epochs, expected_dt_ps, source_ids)
    if (not np.isfinite([lag_ps, dwell_ps, persistence_ps]).all()
            or lag_ps < dwell_ps or dwell_ps < 0 or persistence_ps < 0):
        raise ValueError("Require finite lag >= dwell >= 0 and persistence >= 0")
    observed = np.ones(len(t), bool) if observation_valid is None else _vector(
        observation_valid, "observation_valid", boolean=True)
    if len(observed) != len(t):
        raise ValueError("Observation mask must be aligned")
    states = {}
    episodes = {}
    for name, values in [("supply", supply), ("arrival", arrival), ("contact", contact)]:
        a = _vector(values, name, integer=True)
        if len(a) != len(t) or not np.isin(a, [-1, 0, 1]).all():
            raise ValueError("Expected aligned states in {-1, 0, 1}")
        a = a.copy()
        a[~observed] = -1
        states[name] = a
        episodes[name] = sampled_episodes(a, t, segments, expected_dt_ps)
    # Use exact sample coordinates for the inherited establishment helper.
    # Floating additions such as 0.1 + 0.2 must not push confirmation past
    # the saved 0.3 endpoint. Every actual source/cadence break gets a new
    # segment before conversion, so integer coordinates never fill gaps.
    joins = ((segments[1:] == segments[:-1])
             & np.isclose(np.diff(t), expected_dt_ps, atol=1e-7, rtol=0))
    clock_segments = np.r_[0, np.cumsum(~joins)]
    sample_clock = np.arange(len(t), dtype=float)

    def sample_units(duration):
        units = duration / expected_dt_ps
        nearest = round(units)
        return float(nearest) if np.isclose(units, nearest, atol=1e-7, rtol=0) else units

    risks = []
    for supply_run in episodes["supply"]:
        i = int(supply_run["first_index"])
        end = i + 1
        deadline = t[i] + lag_ps
        while (end < len(t) and t[end] <= deadline + 1e-7
               and segments[end] == segments[i]
               and np.isclose(t[end] - t[end - 1], expected_dt_ps, atol=1e-7, rtol=0)):
            end += 1
        arrived = np.flatnonzero(states["arrival"][i:end] == 1)
        j = i + int(arrived[0]) if len(arrived) else None
        full = np.isclose(t[end - 1], deadline, atol=1e-7, rtol=0)
        arrival_status = ("OBSERVED_ARRIVAL" if j is not None else
                          "NO_ARRIVAL_WITHIN_LAG" if full and np.all(states["arrival"][i:end] >= 0)
                          else "CENSORED_UNKNOWN_OR_WINDOW")
        establishment = (
            {"status": "CENSORED_UNKNOWN_ORIGIN", "outcome": None}
            if states["contact"][i] < 0 else establishment_outcome(
                states["contact"], sample_clock, clock_segments, i,
                sample_units(lag_ps), sample_units(dwell_ps), 1.))
        row = {
            "origin_index": i, "origin_time_ps": float(t[i]),
            "source_id": str(source[i]), "epoch_id": str(ep[i]),
            "supply_left_censored": bool(supply_run["left_censored"]),
            "contact_origin_state": int(states["contact"][i]),
            "arrival_status": arrival_status,
            "arrival_index": j, "arrival_time_ps": None if j is None else float(t[j]),
            "arrival_first_passage_resolved": bool(j is not None and np.all(states["arrival"][i:j] == 0)),
            "establishment": establishment,
            "maintenance_status": "NOT_ESTABLISHED",
        }
        if establishment["outcome"] == 1:
            k = int(establishment["confirmed_index"])
            row["confirmed_time_ps"] = float(t[k])
            labels = np.where(states["contact"] < 0, "UNOBSERVED",
                              np.where(states["contact"] == 1, "CONTACT", "ABSENT"))
            row["maintenance_status"] = confirm_reconstruction(
                labels, np.ones(len(t), bool), t, ep, k,
                duration_ps=persistence_ps, expected_dt_ps=expected_dt_ps,
                source_ids=source, chemistry_valid=observed)
        risks.append(row)
    return {"episodes": episodes, "risk_origins": risks,
            "denominators": {
                "observed_supply_samples": int((states["supply"] == 1).sum()),
                "observed_no_supply_samples": int((states["supply"] == 0).sum()),
                "unknown_supply_samples": int((states["supply"] < 0).sum()),
                "supplied_origins": len(risks),
                "supplied_contact_free_origins": sum(
                    r["contact_origin_state"] == 0 for r in risks),
                "supplied_existing_contact_origins": sum(
                    r["contact_origin_state"] == 1 for r in risks),
                "supplied_unknown_contact_origins": sum(
                    r["contact_origin_state"] < 0 for r in risks),
            }}
