"""Focused checks for matched bridge-response helpers."""

import numpy as np
import pandas as pd

from molsimflow.postprocess.matched_bridge_response import (
    _nearest_oxygen_assignment,
    block_bootstrap_ci,
    standardized_medoid,
)


def test_standardized_medoid_is_observed_and_deterministic():
    table = pd.DataFrame(
        {
            "pair_id": [1, 2, 3],
            "delta_a": [-2.0, 0.0, 3.0],
            "delta_b": [-3.0, 0.0, 4.0],
        }
    )
    audit, selected = standardized_medoid(table, ("delta_a", "delta_b"))
    assert int(audit.loc[selected, "pair_id"]) == 2
    assert int(audit["medoid_selected"].sum()) == 1


def test_block_bootstrap_is_reproducible_and_order_independent_within_blocks():
    values = np.asarray([1.0, 2.0, 4.0, 8.0])
    blocks = np.asarray(["a", "a", "b", "b"])
    first = block_bootstrap_ci(values, blocks, samples=200, seed=17)
    second = block_bootstrap_ci(values, blocks, samples=200, seed=17)
    assert first == second
    assert first[0] <= float(np.mean(values)) <= first[1]


def test_periodic_nearest_oxygen_assignment():
    oxygen = np.asarray([[0.1, 0.0, 0.0], [5.0, 0.0, 0.0]])
    hydrogen = np.asarray([[9.9, 0.0, 0.0], [5.8, 0.0, 0.0]])
    counts, assignment = _nearest_oxygen_assignment(
        oxygen,
        hydrogen,
        np.asarray([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]]),
        1.0,
    )
    assert assignment.tolist() == [0, 1]
    assert counts.tolist() == [1, 1]


if __name__ == "__main__":
    test_standardized_medoid_is_observed_and_deterministic()
    test_block_bootstrap_is_reproducible_and_order_independent_within_blocks()
    test_periodic_nearest_oxygen_assignment()
