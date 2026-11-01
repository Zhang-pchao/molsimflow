from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from molsimflow.postprocess.scaffold_reference import (
    build_scaffold_reference, reference_to_frame_sites, register_coordinates, register_scaffold,
)


def system():
    ids = [10, 20, 30, 40, 50, 60]
    elements = ["Si", "Si", "O", "O", "C", "H"]
    xyz = np.array([[9.5, 1.0, 2.], [.5, 1.0, 2.], [0., 1.0, 3.],
                    [1., 4., 6.], [6., 6., 8.], [6., 6., 9.]])
    bounds = np.array([[0., 10.], [0., 12.], [-5., 30.]])
    return ids, elements, xyz, bounds


def reference(data=None, **kwargs):
    options = dict(silicon_ids=[10, 20], bridge_oxygen_ids=[30], top_ids=[30],
                   deformation_tolerance_A=.05)
    options.update(kwargs)
    return build_scaffold_reference(*(system() if data is None else data), **options)


def translated(shift, *, change_terminals=False):
    ids, elements, xyz, bounds = system()
    xyz = xyz + shift
    xyz[:, :2] = (xyz[:, :2] - bounds[:2, 0]) % np.diff(bounds[:2], axis=1).ravel() + bounds[:2, 0]
    if change_terminals:
        elements[3:] = ["C", "O", "H"]
        xyz[3:] += np.array([2., 3., 20.])
    return ids, elements, xyz, bounds


def test_reference_is_sorted_and_hash_invariant_under_input_permutation():
    ids, elements, xyz, bounds = system()
    permutation = [5, 1, 3, 0, 4, 2]
    other = ([ids[i] for i in permutation], [elements[i] for i in permutation], xyz[permutation], bounds)
    first = reference()
    second = reference(other, silicon_ids=[20, 10])
    assert first == second
    assert first.atom_ids == (10, 20, 30)
    assert first.sha256 == second.sha256
    assert len(first.sha256) == 64


def test_reference_has_no_input_array_or_mutable_numpy_backdoor():
    data = system()
    ref = reference(data)
    before = ref.sha256
    data[2][:] = 999.
    data[3][:] = -1.
    assert ref.sha256 == before
    assert isinstance(ref.coordinates_A, tuple)
    assert isinstance(ref.coordinates_A[0], tuple)
    assert isinstance(ref.bounds_A, tuple)
    with pytest.raises(TypeError):
        ref.coordinates_A[0][0] = 999.
    with pytest.raises(FrozenInstanceError):
        ref.top_ids = (20,)


def test_terminal_atom_changes_never_change_reference_or_registration():
    ref = reference()
    altered = translated([0., 0., 0.], change_terminals=True)
    assert reference(altered).sha256 == ref.sha256
    result = register_scaffold(ref, *altered)
    np.testing.assert_allclose(result.translation_A, [0., 0., 0.])
    assert result.max_residual_A == 0.


@pytest.mark.parametrize("shift", [[1.0, -2.0, 4.0], [-2.0, 3.0, -7.0], [11.0, -14.0, 4.0]])
def test_periodic_xy_crossing_and_nonperiodic_z_translation_register_exactly(shift):
    ref = reference()
    current = translated(shift)
    result = register_scaffold(ref, *current)
    restored = register_coordinates(current[2], ref, result)
    np.testing.assert_allclose(restored, system()[2], atol=1e-14)
    assert result.translation_A[2] == shift[2]
    assert result.max_residual_A < 1e-12


def test_fixed_sites_and_relative_heights_survive_translation():
    ref = reference()
    result = register_scaffold(ref, *translated([1., -2., 4.]))
    sites = np.array([[9.75, .5], [.25, 11.5]])
    mapped = reference_to_frame_sites(sites, ref, result)
    np.testing.assert_allclose(mapped, [[.75, 10.5], [1.25, 9.5]])
    # A physical layer offset is external and does not redefine the top datum.
    water_ref = np.column_stack((sites, np.full(2, ref.height_datum_A + 2.5)))
    water_frame = water_ref + np.asarray(result.translation_A)
    water_frame[:, :2] = mapped
    aligned = register_coordinates(water_frame, ref, result)
    np.testing.assert_allclose(aligned[:, :2], sites)
    np.testing.assert_allclose(aligned[:, 2] - ref.height_datum_A - 2.0, [.5, .5])
    assert ref.height_datum_A == 3.0


def test_explicit_top_scaffold_ids_define_median_and_hash():
    first = reference(top_ids=[10, 20, 30])
    second = reference(top_ids=[30])
    assert first.height_datum_A == 2.0
    assert second.height_datum_A == 3.0
    assert first.sha256 != second.sha256
    assert first.sha256 != reference(top_ids=[10, 20, 30], deformation_tolerance_A=.1).sha256


def test_permuted_frame_ids_register_by_identity_not_row_order():
    ref = reference()
    ids, elements, xyz, bounds = translated([1., -2., 4.])
    order = [2, 5, 0, 3, 1, 4]
    result = register_scaffold(ref, [ids[i] for i in order], [elements[i] for i in order], xyz[order], bounds)
    np.testing.assert_allclose(result.translation_A, [1., -2., 4.])


@pytest.mark.parametrize("damage", ["missing", "duplicate", "changed_element"])
def test_scaffold_identity_damage_fails(damage):
    ref = reference()
    ids, elements, xyz, bounds = system()
    if damage == "missing":
        ids, elements, xyz = ids[1:], elements[1:], xyz[1:]
    elif damage == "duplicate":
        ids[-1] = ids[0]
    else:
        elements[2] = "C"
    with pytest.raises(ValueError, match="Missing|duplicate|element"):
        register_scaffold(ref, ids, elements, xyz, bounds)


def test_nonrigid_scaffold_is_rejected_without_rotation_fit():
    ref = reference()
    ids, elements, xyz, bounds = translated([1., -2., 4.])
    xyz[2, 2] += .2
    with pytest.raises(ValueError, match="deformation"):
        register_scaffold(ref, ids, elements, xyz, bounds)


def test_small_explicitly_tolerated_residual_across_xy_boundary():
    ref = reference()
    ids, elements, xyz, bounds = translated([.49, 0., 0.])
    xyz[0, 0] += .02
    xyz[:, 0] %= 10.
    result = register_scaffold(ref, ids, elements, xyz, bounds)
    np.testing.assert_allclose(result.translation_A, [.49, 0., 0.], atol=1e-14)
    assert result.max_residual_A == pytest.approx(.02)


@pytest.mark.parametrize("shift", [[5., 0., 0.], [0., -6., 0.]])
def test_exact_half_box_translation_is_ambiguous(shift):
    with pytest.raises(ValueError, match="Ambiguous half-box"):
        register_scaffold(reference(), *translated(shift))


def test_antipodal_scaffold_displacements_are_rejected_before_consensus():
    ids, elements, xyz, bounds = system()
    xyz[2, 0] += 5.
    with pytest.raises(ValueError, match="Ambiguous half-box"):
        register_scaffold(reference(), ids, elements, xyz, bounds)


@pytest.mark.parametrize("periodic", [(True, True, True), (True, False, False), (1, 1, 0)])
def test_only_explicit_boolean_ppf_periodicity_is_supported(periodic):
    with pytest.raises(ValueError, match="p-p-f"):
        reference(periodic=periodic)


def test_triclinic_changed_lengths_and_shifted_bounds_rejected():
    ref = reference()
    ids, elements, xyz, bounds = system()
    with pytest.raises(ValueError, match="Axis-aligned"):
        register_scaffold(ref, ids, elements, xyz, np.eye(3))
    for changed in (bounds + [[0, 1], [0, 0], [0, 0]], bounds + 2.):
        with pytest.raises(ValueError, match="bounds differ"):
            register_scaffold(ref, ids, elements, xyz, changed)


@pytest.mark.parametrize("kwargs", [dict(top_ids=[40]), dict(bridge_oxygen_ids=[50]),
                                    dict(silicon_ids=[30]), dict(bridge_oxygen_ids=[10]),
                                    dict(top_ids=[]), dict(deformation_tolerance_A=float("nan")),
                                    dict(deformation_tolerance_A=3.), dict(wrapping_tolerance_A=0.)])
def test_invalid_reference_selection_or_tolerance_rejected(kwargs):
    with pytest.raises(ValueError):
        reference(**kwargs)


def test_transformation_rejects_registration_from_other_reference():
    first = reference()
    result = register_scaffold(first, *system())
    other = reference(top_ids=[10, 20, 30])
    with pytest.raises(ValueError, match="different observer"):
        register_coordinates([[0., 0., 3.]], other, result)
    with pytest.raises(ValueError, match="different observer"):
        reference_to_frame_sites([[0., 0.]], other, result)


def test_nonfinite_atom_coordinates_cannot_enter_reference():
    ids, elements, xyz, bounds = system()
    xyz[2, 2] = np.nan
    with pytest.raises(ValueError, match="finite"):
        reference((ids, elements, xyz, bounds))


def test_direct_record_replacement_cannot_retain_mutable_arrays():
    ref = reference()
    coordinates = np.asarray(ref.coordinates_A)
    bounds = np.asarray(ref.bounds_A)
    copy = replace(ref, coordinates_A=coordinates, bounds_A=bounds,
                   atom_ids=np.asarray(ref.atom_ids), silicon_ids=list(ref.silicon_ids))
    before = copy.sha256
    coordinates[:] = 100.
    bounds[:] = 100.
    assert copy.sha256 == before
    assert copy.sha256 == ref.sha256
    result = register_scaffold(ref, *system())
    shift = np.asarray(result.translation_A)
    frozen_result = replace(result, translation_A=shift)
    shift[:] = 100.
    assert frozen_result.translation_A == (0., 0., 0.)
