import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from molsimflow.postprocess.dual_interface_ion_access import aggregate_ion_map, aggregate_water_map, summarize_windows


class DualInterfaceIonAccessTest(unittest.TestCase):
    def test_water_map_uses_frame_volume_exposure(self):
        rows = pd.DataFrame(
            {
                "gap_window": ["2-4A", "4-6A"],
                "s_left_A": [0.0, 0.0],
                "s_right_A": [1.0, 1.0],
                "s_center_A": [0.5, 0.5],
                "rho_left_A": [0.0, 0.0],
                "rho_right_A": [1.0, 1.0],
                "rho_center_A": [0.5, 0.5],
                "count": [10, 30],
                "n_frames": [5, 10],
                "bin_volume_A3": [2.0, 2.0],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "water.csv"
            rows.to_csv(path, index=False)
            result = aggregate_water_map(path, "case", 2.0, 6.0)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(float(result.iloc[0]["water_number_density_A-3"]), 40.0 / 30.0)

    def test_window_summary_keeps_bulk_ion_access_missing(self):
        frame = pd.DataFrame(
            {
                "time_ns": np.arange(8) * 0.02,
                "gap_A": [3.0] * 4 + [11.0] * 4,
                "water_density_A-3": [0.04] * 4 + [0.02] * 4,
                "n_water_film": [10] * 8,
                "n_mobile_ions_film": [np.nan] * 8,
            }
        )
        result = summarize_windows(frame, "Bulk-water-S", (2.0, 6.0), (10.0, 14.0), 0.02, 100, 7)
        ratio = result[result.metric == "water_density_near_wide_ratio"].iloc[0]
        access = result[result.metric == "mobile_ion_access_probability"]
        self.assertAlmostEqual(float(ratio["mean"]), 2.0)
        self.assertTrue(access["mean"].isna().all())

    def test_ion_map_excludes_nominal_bubble_interior(self):
        samples = pd.DataFrame(
            {
                "gap_A": [3.0, 3.0],
                "species": ["Na_plus", "Na_plus"],
                "r_A_A": [18.0, 20.0],
                "r_B_A": [20.0, 20.0],
                "s_A": [0.5, 0.5],
                "rho_A": [0.5, 0.5],
            }
        )
        frames = pd.DataFrame({"gap_A": [3.0, 3.5]})
        with tempfile.TemporaryDirectory() as tmp:
            sample_path = Path(tmp) / "samples.csv.gz"
            frame_path = Path(tmp) / "frames.csv"
            samples.to_csv(sample_path, index=False)
            frames.to_csv(frame_path, index=False)
            result = aggregate_ion_map(sample_path, frame_path, "case", 2.0, 6.0)
        self.assertEqual(float(result["count"].sum()), 1.0)


if __name__ == "__main__":
    unittest.main()
