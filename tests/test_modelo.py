from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "codigo"))

import calibrar  # noqa: E402
import calibrar_extremos  # noqa: E402
from modelo import default_params, two_tank_run  # noqa: E402


class TwoTankModelTests(unittest.TestCase):
    def test_daily_mass_balance_and_physical_bounds(self) -> None:
        rng = np.random.default_rng(20260830)
        precip = rng.gamma(shape=0.7, scale=7.0, size=5000)
        params = default_params()

        q_sim, et_used, s1, s2, ds = two_tank_run(
            precip, params, return_states=True
        )
        residual = precip - et_used - q_sim - ds

        np.testing.assert_allclose(residual, 0.0, atol=1e-12, rtol=0.0)
        self.assertTrue(np.all(q_sim >= 0.0))
        self.assertTrue(np.all(et_used >= 0.0))
        self.assertTrue(np.all((s1 >= 0.0) & (s1 <= params.D1)))
        self.assertTrue(np.all((s2 >= 0.0) & (s2 <= params.D2)))

    def test_zero_rain_from_empty_storage_has_zero_output(self) -> None:
        q_sim, et_used, s1, s2, ds = two_tank_run(
            np.zeros(30), default_params(), return_states=True
        )
        for values in [q_sim, et_used, s1, s2, ds]:
            np.testing.assert_array_equal(values, np.zeros(30))

    def test_missing_flow_does_not_remove_a_simulation_day(self) -> None:
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2000-01-01", periods=3, freq="D"),
                "prcp_daymet_mm_day": [1.0, 2.0, 3.0],
                "q_cms": [0.1, np.nan, 0.3],
            }
        )
        standard = calibrar._dataset_for_source(frame, "daymet")
        extremes = calibrar_extremos._dataset_for_source(frame, "daymet")
        self.assertEqual(len(standard), 3)
        self.assertEqual(len(extremes), 3)
        self.assertTrue(pd.isna(standard.loc[1, "q_obs_m3s"]))


if __name__ == "__main__":
    unittest.main()
