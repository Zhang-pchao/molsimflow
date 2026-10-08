import unittest

import numpy as np
import pandas as pd

from molsimflow.postprocess.charge_retention import STATES, charge_state, summarize


class ChargeRetentionTest(unittest.TestCase):
    def test_charge_state_is_exhaustive(self):
        self.assertEqual([charge_state(0, 0), charge_state(0, 2), charge_state(1, 0), charge_state(2, 3)], list(STATES))

    def test_state_probabilities_sum_to_one(self):
        frame = pd.DataFrame(
            {
                "case_label": ["case"] * 4,
                "time_ns": [0.00, 0.01, 0.02, 0.03],
                "gap_bin": ["4-6A"] * 4,
                "n_water_film": [10, 10, 10, 10],
                "n_positive_film": [0, 0, 1, 1],
                "n_negative_film": [0, 1, 0, 1],
                "charge_state": list(STATES),
            }
        )
        result = summarize(frame, block_ns=0.02, bootstrap_samples=20, random_seed=7)
        probabilities = result[result.metric.str.endswith("_probability")]["mean"].to_numpy(float)
        self.assertTrue(np.isclose(probabilities.sum(), 1.0))
        self.assertTrue(np.allclose(probabilities, 0.25))


if __name__ == "__main__":
    unittest.main()
