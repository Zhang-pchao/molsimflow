"""Synthetic coordinate tests, including an independent two-anchor score oracle."""
from dataclasses import FrozenInstanceError
import numpy as np
import pytest

from molsimflow.postprocess.scaffold_motif_match import (
    bridge_basis, match_anchor_pair, rank_scaffold_motifs,
)


def anchors(lengths=(1., 1.), angle=1.8):
    half = angle / 2.
    return np.array([[lengths[0] * np.sin(half), 0., -lengths[0] * np.cos(half)],
                     [-lengths[1] * np.sin(half), 0., -lengths[1] * np.cos(half)]])


def rotation():
    a, b = .61, -.37
    z = np.array([[np.cos(a), -np.sin(a), 0.],
                  [np.sin(a), np.cos(a), 0.], [0., 0., 1.]])
    x = np.array([[1., 0., 0.], [0., np.cos(b), -np.sin(b)],
                  [0., np.sin(b), np.cos(b)]])
    return x @ z


def oracle(source, target):
    """Length/half-angle formula; never constructs the bridge basis or R."""
    a, b = np.linalg.norm(source, axis=1), np.linalg.norm(target, axis=1)
    theta_s = np.arccos(np.clip(np.dot(source[0], source[1]) / np.prod(a), -1., 1.))
    theta_t = np.arccos(np.clip(np.dot(target[0], target[1]) / np.prod(b), -1., 1.))
    dot = np.cos((theta_s - theta_t) / 2.)
    return tuple(np.sqrt(max(0., np.mean(a * a + c * c - 2. * a * c * dot)))
                 for c in (b, b[::-1]))


def test_self_has_zero_mismatch_and_both_rotations_are_proper():
    source = anchors((1.4, 1.8))
    result = match_anchor_pair(source, source)
    assert result.best_score_A < 1e-14
    assert result.best_flip_indices == (0,)
    for matrix in result.rotations:
        r = np.asarray(matrix)
        np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-14)
        assert np.linalg.det(r) == pytest.approx(1.)
    np.testing.assert_allclose(result.rotations[0], np.eye(3), atol=1e-14)


def test_equal_length_self_reports_unresolved_orientation_ambiguity():
    source = anchors()
    result = match_anchor_pair(source, source)
    assert result.best_flip_indices == (0, 1)
    assert result.ambiguous
    assert not np.allclose(result.rotations[0], result.rotations[1])


def test_unequal_lengths_require_the_correct_anchor_correspondence():
    result = match_anchor_pair(anchors((2., 1.)), anchors((1., 2.)))
    assert result.scores_A == pytest.approx((1., 0.), abs=1e-14)
    assert result.best_flip_indices == (1,)
    assert result.target_orders == ((0, 1), (1, 0))


def test_score_units_are_rms_per_anchor_not_per_coordinate():
    result = match_anchor_pair(anchors(), anchors((2., 2.)))
    assert result.scores_A == pytest.approx((1., 1.))
    doubled = match_anchor_pair(anchors() * 2., anchors((2., 2.)) * 2.)
    assert doubled.best_score_A == pytest.approx(2.)


def test_independent_half_angle_length_oracle():
    rng = np.random.default_rng(458)
    q = rotation()
    for _ in range(30):
        source = anchors(rng.uniform(.7, 2.1, 2), rng.uniform(.4, 2.7))
        target = anchors(rng.uniform(.7, 2.1, 2), rng.uniform(.4, 2.7)) @ q.T
        result = match_anchor_pair(source, target)
        assert result.scores_A == pytest.approx(oracle(source, target), abs=1e-13)


def test_global_rotation_preserves_scores_and_covariantly_rotates_maps():
    source, target = anchors((1.2, 1.6), 1.5), anchors((1.7, 1.1), 2.1)
    q = rotation()
    initial = match_anchor_pair(source, target)
    rotated = match_anchor_pair(source @ q.T, target @ q.T)
    assert rotated.scores_A == pytest.approx(initial.scores_A, abs=1e-14)
    for old, new in zip(initial.rotations, rotated.rotations):
        np.testing.assert_allclose(new, q @ np.asarray(old) @ q.T, atol=1e-14)


@pytest.mark.parametrize("reverse_source,reverse_target", [(True, False), (False, True),
                                                          (True, True)])
def test_bridge_order_permutation_preserves_score_and_rotation_set(
        reverse_source, reverse_target):
    source, target = anchors((1.2, 1.6), 1.5), anchors((1.7, 1.1), 2.1)
    initial = match_anchor_pair(source, target)
    changed = match_anchor_pair(source[::-1] if reverse_source else source,
                                target[::-1] if reverse_target else target)
    order = (1, 0) if reverse_source != reverse_target else (0, 1)
    assert changed.scores_A == pytest.approx([initial.scores_A[i] for i in order])
    for new, old_index in zip(changed.rotations, order):
        np.testing.assert_allclose(new, initial.rotations[old_index], atol=1e-14)
    assert changed.best_score_A == pytest.approx(initial.best_score_A)


def test_ranking_selects_one_source_before_any_external_filtering():
    target = anchors()
    result = rank_scaffold_motifs([8, 2, 5], [anchors((1.3, 1.3)), target, target],
                                  target, preferred_source_id=5)
    assert result.selected_source_id == 5
    assert result.first_tie_group_ids == (5, 2)
    assert [x.source_id for x in result.ranked_sources] == [5, 2, 8]
    assert result.preferred_source_eligible
    assert result.selected_score_A < 1e-14


def test_absent_preferred_id_is_ignored_and_input_order_does_not_break_ties():
    target = anchors()
    for ids in ([8, 2], [2, 8]):
        result = rank_scaffold_motifs(ids, [target, target], target, preferred_source_id=4)
        assert result.selected_source_id == 2
        assert result.first_tie_group_ids == (2, 8)
        assert not result.preferred_source_eligible


def test_minimum_anchored_tie_groups_do_not_chain_nontransitive_tolerances():
    target = anchors()
    # Scores approximately 0, 0.75*tolerance and 1.5*tolerance.
    result = rank_scaffold_motifs(
        [10, 20, 1], [target, target * (1. + .75e-10), target * (1. + 1.5e-10)],
        target, preferred_source_id=1, tie_tolerance_A=1e-10)
    assert result.first_tie_group_ids == (10, 20)
    assert result.selected_source_id == 10
    assert [x.tie_group for x in result.ranked_sources] == [0, 0, 1]


def test_preferred_self_does_not_override_a_non_tied_better_score():
    target = anchors()
    result = rank_scaffold_motifs([8, 2], [target * 1.1, target], target,
                                  preferred_source_id=8)
    assert result.selected_source_id == 2


def test_inputs_and_results_do_not_have_mutable_array_backdoors():
    source = anchors()
    original = source.copy()
    result = match_anchor_pair(source, source)
    source[:] = 99.
    np.testing.assert_allclose(np.asarray(result.rotations[0]), np.eye(3), atol=1e-14)
    with pytest.raises(FrozenInstanceError):
        result.best_score_A = 99.
    with pytest.raises(TypeError):
        result.rotations[0][0][0] = 99.
    assert match_anchor_pair(original, original) == result


@pytest.mark.parametrize("bad", [
    [[0., 0., 0.], [1., 0., 0.]],
    [[1., 0., 0.], [2., 0., 0.]],
    [[1., 0., 0.], [-2., 0., 0.]],
    [[np.nan, 0., 0.], [0., 1., 0.]],
    [[1., 2., 3.]],
])
def test_degenerate_or_malformed_anchor_pairs_reject(bad):
    with pytest.raises(ValueError):
        match_anchor_pair(bad, anchors())


def test_invalid_ids_counts_and_tolerances_reject():
    for ids, values in [([], []), ([1, 1], [anchors(), anchors()]),
                        ([1.5], [anchors()]), ([True], [anchors()]), ([1], [])]:
        with pytest.raises(ValueError):
            rank_scaffold_motifs(ids, values, anchors())
    for tolerance in (-1., np.nan, np.inf):
        with pytest.raises(ValueError):
            match_anchor_pair(anchors(), anchors(), tie_tolerance_A=tolerance)
    with pytest.raises(ValueError):
        bridge_basis(anchors(), collinearity_tolerance=1.)
