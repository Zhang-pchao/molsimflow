"""Synthetic checks only: no project data, atom registry, or physical radii."""
import numpy as np
import pytest

from molsimflow.postprocess.molecular_placement import (
    _ObstacleIndex, _clearance, cell_domain, complete_frame, molecular_shape,
    occupied_pose_control, pair_convention, search_placements,
)

BOUNDS = [[0., 10.], [0., 10.], [-10., 30.]]


def frame(xyz=((8., 8., 8.),), labels=("O",), ids=(7,), active=None):
    return complete_frame(ids, labels, xyz, expected_atom_ids=ids,
                          obstacle_atom_ids=ids if active is None else active,
                          bounds_A=BOUNDS)


def shape():
    return molecular_shape(["N", "N"], [1., 1.], [[-.5, 0., 0.], [.5, 0., 0.]])


def rule(mode="center_distance", value=0.):
    return pair_convention({("N", "N"): value, ("N", "O"): value}, mode=mode)


def domain():
    return cell_domain([10, 20], [[0., 0., -5.], [5., 0., -5.]],
                       [[5., 10., 15.], [10., 10., 15.]])


def search(cloud=None, centers=((2., 2., 2.), (7., 2., 2.)), turns=None,
           convention=None, cells=None, **budgets):
    options = dict(max_pose_evaluations=100, max_pair_evaluations=100)
    options.update(budgets)
    return search_placements(shape(), frame() if cloud is None else cloud,
                             rule() if convention is None else convention,
                             domain() if cells is None else cells, centers,
                             [np.eye(3)] if turns is None else turns, **options)


def test_orientation_changes_fixed_shape_clearance():
    cloud = frame([[3., 2., 2.]])
    z90 = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    result = search(cloud, centers=[[2., 2., 2.]], turns=[np.eye(3), z90])
    assert result.best_single.rotation_index == 1
    assert result.best_single.obstacle_clearance.value_A == pytest.approx(np.sqrt(1.25))
    assert result.best_single.positions_A == ((2., 1.5, 2.), (2., 2.5, 2.))


def test_species_specific_clearance_can_change_limiting_atom():
    cloud = frame([[4., 2., 2.], [6., 2., 2.]], ["O", "C"], [7, 9])
    convention = pair_convention({("N", "N"): .5, ("N", "O"): .5, ("N", "C"): 3.},
                                 mode="exclusion_clearance")
    result = search(cloud, centers=[[2., 2., 2.]], convention=convention)
    assert result.best_single.obstacle_clearance == _clearance(
        result.best_single.positions_A, shape().species, cloud.positions_A, cloud.species,
        cloud.atom_ids, cloud, convention)
    assert result.best_single.obstacle_clearance.limiting_pair == (1, 9)


def test_xy_minimum_image_but_z_is_not_periodic():
    cloud = frame([[9.8, .1, 0.], [.2, .1, 29.]], ["O", "O"], [7, 8])
    index = _ObstacleIndex(cloud.positions_A, cloud.species, cloud.atom_ids, cloud, rule())
    assert index.clearance([[.2, .1, 0.]], ["N"]).value_A == pytest.approx(.4)
    assert index.clearance([[.2, .1, -9.]], ["N"]).value_A == pytest.approx(np.sqrt(81.16))


def test_exact_nearest_species_trees_match_all_atom_mic_bruteforce():
    rng = np.random.default_rng(52)
    xyz = rng.uniform([-20., -20., -8.], [30., 30., 25.], (65, 3))
    labels = np.array(["O", "C", "H"])[np.arange(65) % 3]
    ids = np.arange(100, 165)
    cloud = frame(xyz, labels, ids)
    convention = pair_convention({("N", "O"): .7, ("N", "C"): 1.2, ("N", "H"): .4},
                                 mode="exclusion_clearance")
    index = _ObstacleIndex(cloud.positions_A, cloud.species, cloud.atom_ids, cloud, convention)
    for query in rng.uniform([-25., -25., -15.], [35., 35., 32.], (25, 2, 3)):
        expected = _clearance(query, ["N", "N"], xyz, labels, ids, cloud, convention)
        actual = index.clearance(query, ["N", "N"])
        assert actual.value_A == pytest.approx(expected.value_A, abs=1e-12)
        assert actual.limiting_pair == expected.limiting_pair


def test_two_poses_can_occupy_different_cells_in_one_frozen_union():
    result = search(convention=rule("exclusion_clearance", .2))
    assert result.best_two.first.cell_ids == (10,)
    assert result.best_two.second.cell_ids == (20,)
    assert result.two_pose_threshold_witness is True
    assert result.status == "EXHAUSTIVE_DISCRETE_SEARCH"
    assert result.best_two.joint_clearance_A == pytest.approx(3.8)


def test_individually_clear_poses_are_not_a_two_molecule_witness_if_mutually_colliding():
    result = search(centers=[[2., 2., 2.], [2.1, 2., 2.]],
                    convention=rule("exclusion_clearance", .5))
    assert result.best_two.first.obstacle_clearance.value_A > 0.
    assert result.best_two.second.obstacle_clearance.value_A > 0.
    assert result.best_two.intermolecular_clearance.value_A == pytest.approx(-.4)
    assert result.best_two.limiting_component == "intermolecular"
    assert result.two_pose_threshold_witness is False


def test_primary_atomic_center_distance_never_claims_physical_nonoverlap():
    result = search()
    assert result.two_pose_threshold_witness is None
    assert result.best_two.joint_clearance_A > 0.
    assert result.best_two.joint_clearance_A <= result.best_single.obstacle_clearance.value_A


def test_full_frame_roster_rejects_missing_and_extra_atoms():
    for expected in ([7, 8], [9]):
        with pytest.raises(ValueError, match="UNKNOWN"):
            complete_frame([7], ["O"], [[1., 1., 1.]], expected_atom_ids=expected,
                           obstacle_atom_ids=[7], bounds_A=BOUNDS)
    with pytest.raises(ValueError, match="outside"):
        complete_frame([7], ["O"], [[1., 1., 1.]], expected_atom_ids=[7],
                       obstacle_atom_ids=[8], bounds_A=BOUNDS)


def test_full_roster_can_use_explicit_background_subset_and_preserves_evidence():
    cloud = frame([[1., 1., 1.], [2., 2., 2.]], ["O", "O"], [7, 8], active=[8])
    assert cloud.atom_ids == (7, 8)
    assert cloud.obstacle_atom_ids == (8,)
    assert search(cloud).best_single.obstacle_clearance.value_A > 0.


def test_unknown_species_and_missing_intermolecular_pair_fail_closed_before_search():
    with pytest.raises(ValueError, match="UNKNOWN"):
        search(frame([[3., 2., 2.]], ["Unknown"]))
    with pytest.raises(ValueError, match="UNKNOWN"):
        search(convention=pair_convention({("N", "O"): 0.}, mode="center_distance"),
               max_pose_evaluations=1)


def test_observed_control_uses_actual_positions_and_excludes_only_named_self_ids():
    cloud = frame([[2., 2., 2.], [3.3, 2., 2.], [3.4, 2., 2.]],
                  ["N", "N", "O"], [1, 2, 7])
    convention = rule("exclusion_clearance", .8)
    actual = occupied_pose_control(cloud, [1, 2], convention)
    assert actual.positions_A == ((2., 2., 2.), (3.3, 2., 2.))
    assert actual.excluded_self_ids == (1, 2)
    assert actual.clearance.limiting_pair == (1, 7)
    assert actual.clearance.value_A == pytest.approx(-.7)
    # This is a convention overlap at an observed pose, not a trajectory failure.
    with pytest.raises(TypeError):
        search_placements(shape(), cloud, convention, domain(), [[2., 2., 2.]],
                          [np.eye(3)], max_pose_evaluations=1, max_pair_evaluations=1,
                          omitted_ids=[7])


def test_occupied_control_cannot_take_fake_coordinates_or_missing_ids():
    with pytest.raises(ValueError):
        occupied_pose_control(frame(), [8], rule())
    with pytest.raises(TypeError):
        occupied_pose_control(frame(), [7], rule(), positions_A=[[0., 0., 0.]])


def test_com_domain_is_separate_from_whole_molecule_domain():
    narrow = cell_domain([5], [[1.9, 1.9, 1.9]], [[2.1, 2.1, 2.1]])
    result = search(centers=[[2., 2., 2.]], cells=narrow)
    assert result.best_single.cell_ids == (5,)
    assert result.best_single.all_probe_nuclei_in_domain is False


def test_domain_is_xy_periodic_with_half_open_cell_edges_but_not_z_periodic():
    result = search(centers=[[12., 2., 2.]])
    assert result.best_single.cell_ids == (10,)
    with pytest.raises(ValueError, match="COM samples"):
        search(centers=[[2., 2., 42.]])
    result = search(centers=[[5., 2., 2.]])
    assert result.best_single.cell_ids == (20,)


def test_finite_negative_and_budgets_are_distinct():
    negative = search(centers=[[2., 2., 2.], [2.1, 2., 2.]],
                      convention=rule("exclusion_clearance", .8))
    assert negative.status == "EXHAUSTIVE_DISCRETE_SEARCH"
    assert negative.two_pose_threshold_witness is False
    truncated = search(max_pose_evaluations=1)
    assert truncated.status == "BUDGET_EXHAUSTED"
    assert truncated.best_two is None
    assert truncated.poses_evaluated == 1 and truncated.poses_planned == 2
    paired = search(centers=[[2., 2., 2.], [3., 2., 2.], [7., 2., 2.]],
                    max_pair_evaluations=1)
    assert paired.status == "BUDGET_EXHAUSTED"
    assert paired.pairs_evaluated == 1 and paired.pairs_planned == 3


def test_nested_background_removal_and_nested_search_domain_cannot_reduce_exact_score():
    full = frame([[3., 2., 2.], [8., 8., 8.]], ["O", "O"], [7, 8])
    reduced = frame(full.positions_A, full.species, full.atom_ids, active=[8])
    first = search(full)
    second = search(reduced)
    assert second.best_two.joint_clearance_A >= first.best_two.joint_clearance_A
    expanded = search(full, centers=[[2., 2., 2.], [7., 2., 2.], [5., 7., 4.]])
    assert expanded.best_two.joint_clearance_A >= first.best_two.joint_clearance_A


def test_input_preservation_and_repeatability():
    xyz = np.array([[8., 8., 8.]])
    cloud = frame(xyz)
    offsets = np.array([[-.5, 0., 0.], [.5, 0., 0.]])
    molecule = molecular_shape(["N", "N"], [1., 1.], offsets)
    xyz[:] = -100.
    offsets[:] = -100.
    assert cloud.positions_A == ((8., 8., 8.),)
    assert molecule.offsets_A == ((-.5, 0., 0.), (.5, 0., 0.))
    assert search(cloud) == search(cloud)


@pytest.mark.parametrize("rotation", [
    [[-1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
    [[2., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
    [[np.nan, 0., 0.], [0., 1., 0.], [0., 0., 1.]],
])
def test_improper_or_malformed_rotations_are_rejected(rotation):
    with pytest.raises(ValueError):
        search(turns=[rotation])


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_invalid_budgets_are_rejected(budget):
    with pytest.raises(ValueError):
        search(max_pose_evaluations=budget)


def test_pair_conventions_require_explicit_symmetric_and_valid_thresholds():
    with pytest.raises(ValueError, match="asymmetric"):
        pair_convention({("N", "O"): 1., ("O", "N"): 2.}, mode="exclusion_clearance")
    with pytest.raises(ValueError):
        pair_convention({("N", "N"): 0.}, mode="exclusion_clearance")
    with pytest.raises(ValueError):
        pair_convention({("N", "N"): 1.}, mode="center_distance")
    with pytest.raises(ValueError):
        pair_convention({("N", "N"): np.inf}, mode="exclusion_clearance")


def test_invalid_shape_and_triclinic_input_are_rejected():
    with pytest.raises(ValueError, match="COM"):
        molecular_shape(["N", "N"], [1., 1.], [[0., 0., 0.], [1., 0., 0.]])
    with pytest.raises(ValueError):
        molecular_shape(["N"], [-1.], [[0., 0., 0.]])
    with pytest.raises(ValueError, match="orthorhombic"):
        complete_frame([7], ["O"], [[1., 1., 1.]], expected_atom_ids=[7],
                       obstacle_atom_ids=[7], bounds_A=np.eye(3))


def test_empty_explicit_obstacle_background_still_requires_interprobe_separation():
    result = search(frame(active=[]), convention=rule("exclusion_clearance", .2))
    assert np.isinf(result.best_single.obstacle_clearance.value_A)
    assert result.best_two.joint_clearance_A == pytest.approx(3.8)
    assert result.best_two.limiting_component == "intermolecular"


def test_query_budget_can_stop_before_first_pose_without_false_negative():
    result = search(convention=rule("exclusion_clearance", .5), max_obstacle_queries=1)
    assert result.status == "BUDGET_EXHAUSTED"
    assert result.obstacle_queries == 0
    assert result.obstacle_query_budget == 1
    assert result.best_single is None and result.best_two is None
    assert result.two_pose_threshold_witness is None
    assert result.finite_grid_optimum is False


def test_query_budget_counts_each_species_and_each_probe_atom():
    cloud = frame([[8., 8., 8.], [9., 9., 9.]], ["O", "C"], [7, 8])
    convention = pair_convention({("N", "O"): 0., ("N", "C"): 0., ("N", "N"): 0.},
                                 mode="center_distance")
    result = search(cloud, convention=convention, max_obstacle_queries=7)
    assert result.poses_evaluated == 1
    assert result.obstacle_queries == 4
    assert result.obstacle_query_budget == 7


def test_budget_exhaustion_retains_positive_witness_but_not_false_negative():
    positive = search(convention=rule("exclusion_clearance", .2),
                      centers=[[2., 2., 2.], [7., 2., 2.], [5., 6., 2.]],
                      max_pair_evaluations=1)
    negative = search(convention=rule("exclusion_clearance", .5),
                      centers=[[2., 2., 2.], [2.1, 2., 2.], [7., 2., 2.]],
                      max_pair_evaluations=1)
    assert positive.status == negative.status == "BUDGET_EXHAUSTED"
    assert positive.two_pose_threshold_witness is True
    assert negative.two_pose_threshold_witness is None
    assert positive.finite_grid_optimum is False


def test_chunked_joint_scores_agree_with_independent_pair_oracle():
    rng = np.random.default_rng(771)
    centers = rng.uniform([1., 1., 0.], [9., 9., 10.], (18, 3))
    cloud = frame([[4., 4., 4.], [6., 7., 9.]], ["O", "O"], [7, 8])
    convention = rule("exclusion_clearance", .4)
    result = search(cloud, centers=centers, convention=convention,
                    max_pose_evaluations=18, max_pair_evaluations=153)
    candidates = []
    for center in centers:
        candidate = np.asarray(shape().offsets_A) + center
        single = _clearance(candidate, shape().species, cloud.positions_A,
                            cloud.species, cloud.atom_ids, cloud, convention).value_A
        candidates.append((candidate, single))
    oracle = float("-inf")
    for i, (first, single) in enumerate(candidates):
        for second, other_single in candidates[i + 1:]:
            inter = _clearance(first, shape().species, second, shape().species,
                               [0, 1], cloud, convention).value_A
            oracle = max(oracle, min(single, other_single, inter))
    assert result.best_two.joint_clearance_A == pytest.approx(oracle)
    assert result.pairs_evaluated + result.pairs_pruned == 153
    assert result.finite_grid_optimum is True


def test_pair_budget_in_middle_of_chunk_is_exact():
    centers = [[float(i) / 10., 2., 2.] for i in range(100)]
    result = search(centers=centers, max_pose_evaluations=100,
                    max_pair_evaluations=101)
    assert result.pairs_evaluated == 101
    assert result.pairs_planned == 4950
    assert result.finite_grid_optimum is False


def test_sorted_pruning_certifies_exact_optimum_without_scoring_every_pair():
    cloud = frame([[0., 0., 0.]], ["O"], [7])
    centers = [[4., 4., 8.], [6., 6., 8.], [1., 1., 1.], [1.1, 1., 1.]]
    result = search(cloud, centers=centers)
    assert result.finite_grid_optimum
    assert result.pairs_evaluated + result.pairs_pruned == 6
    assert result.pairs_pruned > 0
    assert result.status == "CERTIFIED_DISCRETE_OPTIMUM"
    assert result.best_two_gap_A == 0.


def test_budget_bounds_bracket_independent_full_grid_optimum():
    rng = np.random.default_rng(941)
    for _ in range(8):
        centers = rng.uniform([0., 0., -4.], [10., 10., 14.], (9, 3))
        cloud = frame(rng.uniform([0., 0., -4.], [10., 10., 14.], (7, 3)),
                      ["O"] * 7, list(range(7)))
        full = search(cloud, centers=centers)
        for limit in (1, 2, 4, 7, 14, 36):
            partial = search(cloud, centers=centers, max_pair_evaluations=limit)
            assert partial.pairs_evaluated <= limit
            assert partial.best_two.joint_clearance_A <= (
                full.best_two.joint_clearance_A + 1e-12)
            assert partial.best_two_upper_bound_A >= (
                full.best_two.joint_clearance_A - 1e-12)
            if partial.finite_grid_optimum:
                assert partial.best_two.joint_clearance_A == pytest.approx(
                    full.best_two.joint_clearance_A)
                assert partial.best_two_gap_A == 0.


def test_unsampled_poses_prevent_full_grid_certificate_even_if_sampled_pairs_are_exact():
    result = search(centers=[[2., 2., 2.], [7., 2., 2.], [5., 6., 2.]],
                    max_pose_evaluations=2)
    assert result.finite_grid_optimum is False
    assert result.status == "BUDGET_EXHAUSTED"
    assert np.isinf(result.best_two_upper_bound_A)
    assert np.isinf(result.best_two_gap_A)


def test_all_probe_nuclei_does_not_claim_bond_segment_or_envelope_containment():
    disconnected = cell_domain([1, 2], [[1.4, 1.9, 1.9], [2.4, 1.9, 1.9]],
                                [[1.6, 2.1, 2.1], [2.6, 2.1, 2.1]])
    # The actual occupied atoms may lie in disconnected cells while their bond
    # crosses a gap. This is intentionally only a nucleus-membership predicate.
    from molsimflow.postprocess.molecular_placement import _cell_memberships
    mask = _cell_memberships([[1.5, 2., 2.], [2.5, 2., 2.]],
                             disconnected, frame())
    assert np.all(np.any(mask, axis=1))
    assert not hasattr(search().best_single, "whole_molecule_in_domain")


def test_direct_pose_evaluation_rescores_exact_frozen_witness_without_enumeration():
    from molsimflow.postprocess.molecular_placement import (
        evaluate_pose_pair_positions, evaluate_pose_positions,
    )
    cloud = frame()
    primary = search(cloud)
    pair = primary.best_two
    secondary_rule = rule("exclusion_clearance", .6)
    result = evaluate_pose_pair_positions(
        cloud, shape().species, pair.first.positions_A,
        shape().species, pair.second.positions_A, secondary_rule)
    assert result.first_obstacle_clearance == evaluate_pose_positions(
        cloud, shape().species, pair.first.positions_A, secondary_rule)
    assert result.joint_clearance_A == pytest.approx(pair.joint_clearance_A - .6)
    assert result.intermolecular_clearance == _clearance(
        pair.first.positions_A, shape().species, pair.second.positions_A,
        shape().species, [0, 1], cloud, secondary_rule)


def test_direct_evaluation_uses_selected_obstacles_without_arbitrary_exclusions():
    from molsimflow.postprocess.molecular_placement import evaluate_pose_positions
    cloud = frame([[1., 1., 1.]], ["O"], [7])
    actual = evaluate_pose_positions(cloud, ["N"], [[1., 1., 1.]], rule())
    assert actual.value_A == 0.
    with pytest.raises(TypeError):
        evaluate_pose_positions(cloud, ["N"], [[1., 1., 1.]], rule(), omitted_ids=[7])
    with pytest.raises(ValueError):
        evaluate_pose_positions(cloud, ["N"], [[np.nan, 1., 1.]], rule())


def test_direct_pair_evaluation_supports_different_shapes_and_species():
    from molsimflow.postprocess.molecular_placement import evaluate_pose_pair_positions
    cloud = frame(active=[])
    convention = pair_convention({("A", "B"): .5}, mode="exclusion_clearance")
    result = evaluate_pose_pair_positions(
        cloud, ["A"], [[1., 1., 1.]], ["B", "B"],
        [[2., 1., 1.], [4., 1., 1.]], convention)
    assert result.joint_clearance_A == .5
    assert result.intermolecular_clearance.limiting_pair == (0, 0)
