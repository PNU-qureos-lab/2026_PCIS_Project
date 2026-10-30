"""깊은 물 → IOP → 수심의 연결과 계수 선택의 실제 계산 반영을 검증합니다."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT))
from bathymetry.config import resolve_coefficients
from bathymetry.engine import solve_pixel
from bathymetry.iop import fit_iop
from test_engine import load_fixture, SENSORS


class IOPRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = json.loads(
            (APP_ROOT / "DATA" / "iop_spectra.json").read_text(encoding="utf-8-sig")
        )
        cls.fixtures = {sensor: load_fixture(sensor) for sensor in SENSORS}

    def fit_case(self, fixture, case, **kwargs):
        return fit_iop(
            case["deep_rrs"],
            case["solar_zenith"],
            fixture["waves"],
            fixture["g0"],
            fixture["g1"],
            self.source,
            **kwargs,
        )

    def assert_iop(self, result, case):
        expected = case["expected_iop"]
        parameters = result["parameters"]
        self.assertAlmostEqual(parameters["Chl_mg_m3"], expected["Chl"], delta=1e-12)
        self.assertAlmostEqual(parameters["TSM_g_m3"], expected["TSM"], delta=1e-12)
        self.assertTrue(np.isclose(parameters["u_SSE"], expected["u_SSE"], rtol=1e-10, atol=1e-15))
        self.assertEqual(parameters["starts_completed"], 98)
        self.assertFalse(parameters["coefficient_refit"])
        self.assertFalse(any(row["improves"] for row in result["neighbors"]))
        np.testing.assert_allclose(result["context"]["a"], case["a"], rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(result["context"]["bb"], case["bb"], rtol=1e-12, atol=1e-14)
        deep = np.asarray(case["deep_rrs"], float)
        # 깊은 물 관측값을 물기둥 항으로 사용하는 원본 규칙을 별도 검사합니다.
        np.testing.assert_allclose(
            result["context"]["rrs_dp"], deep / (0.52 + 1.7 * deep), rtol=0, atol=1e-15
        )

    def test_archived_iop_for_six_conditions(self):
        for sensor, fixture in self.fixtures.items():
            for case in fixture["deep_cases"] + fixture["cases"][:1]:
                with self.subTest(sensor=sensor, water=case["water_key"]):
                    self.assert_iop(self.fit_case(fixture, case), case)

    def test_end_to_end_from_deep_rrs_to_depth(self):
        for sensor, fixture in self.fixtures.items():
            case = fixture["cases"][1]
            iop = self.fit_case(fixture, case)
            library = {
                name: np.asarray(values, float) for name, values in fixture["library"].items()
            }
            result = solve_pixel(case["observed"], iop["context"], library)
            self.assertAlmostEqual(result["depth_m"], case["expected"]["depth_m"], delta=2e-7)
            self.assertAlmostEqual(result["F"], case["expected"]["F"], delta=1e-12)
            self.assertTrue(np.isclose(result["c4"], case["expected"]["c4"], rtol=2e-8, atol=1e-13))

    def test_fixed_coefficients_reach_iop_equation_unchanged(self):
        fixture = self.fixtures["Sentinel2"]
        case = fixture["cases"][0]
        for water in ("case1", "case2", "mixed"):
            chosen = resolve_coefficients("Sentinel2", water, "fixed", APP_ROOT)
            result = fit_iop(
                case["deep_rrs"],
                case["solar_zenith"],
                chosen["waves"],
                chosen["g0"],
                chosen["g1"],
                self.source,
            )
            np.testing.assert_array_equal([row["g0"] for row in result["band_rows"]], chosen["g0"])
            np.testing.assert_array_equal([row["g1"] for row in result["band_rows"]], chosen["g1"])
            for row in result["band_rows"]:
                self.assertAlmostEqual(
                    row["g0"] * row["target_u"] + row["g1"] * row["target_u"] ** 2,
                    row["converted_rrs"],
                    delta=1e-14,
                )
            self.assertFalse(result["parameters"]["coefficient_refit"])

    def test_sensor_and_mode_switch_does_not_mutate_source(self):
        original_source = json.dumps(self.source, sort_keys=True)
        fixture = self.fixtures["Sentinel2"]
        case = fixture["deep_cases"][0]
        before = self.fit_case(fixture, case)
        sd = self.fixtures["SuperDove"]
        fixed = resolve_coefficients("SuperDove", "mixed", "fixed", APP_ROOT)
        fit_iop(
            sd["deep_cases"][0]["deep_rrs"],
            20,
            fixed["waves"],
            fixed["g0"],
            fixed["g1"],
            self.source,
        )
        after = self.fit_case(fixture, case)
        self.assertEqual(before["parameters"], after["parameters"])
        self.assertEqual(json.dumps(self.source, sort_keys=True), original_source)

    def test_progress_and_cancellation(self):
        fixture = self.fixtures["Sentinel2"]
        case = fixture["deep_cases"][0]
        events = []
        self.fit_case(fixture, case, progress=events.append)
        self.assertEqual(len(events), 98)
        self.assertEqual(events[-1]["completed"], events[-1]["total"])
        with self.assertRaises(InterruptedError):
            self.fit_case(fixture, case, cancelled=lambda: True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
