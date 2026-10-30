"""위성·수역·계수·해저면 선택과 깊은 물 입력을 검증합니다."""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT))
from bathymetry.config import (
    resolve_coefficients,
    scan_bottoms,
    deep_water_values,
)
from test_engine import load_fixture, SENSORS


class UserSelectionTests(unittest.TestCase):
    """화면 밖의 CLI/설정 파일에서도 동일한 선택 제한이 적용되어야 합니다."""

    def test_fitted_coefficients_are_exact_archived_values(self):
        for sensor in SENSORS:
            expected = load_fixture(sensor)
            got = resolve_coefficients(sensor, "case1", "fitted", APP_ROOT)
            for name in ("waves", "g0", "g1"):
                np.testing.assert_array_equal(got[name], expected[name])

    def test_three_water_presets_are_exact_for_both_sensors(self):
        # 사전 설정 계수표를 독립적인 기대값으로 명시합니다. 설정 상수를 다시 읽지 않습니다.
        expected = {"case1": (0.0949, 0.0794), "case2": (0.0840, 0.1700), "mixed": (0.0895, 0.1247)}
        for sensor in SENSORS:
            count = 4 if sensor == "Sentinel2" else 6
            for water, (g0, g1) in expected.items():
                with self.subTest(sensor=sensor, water=water):
                    got = resolve_coefficients(sensor, water, "fixed", APP_ROOT)
                    np.testing.assert_array_equal(got["g0"], [g0] * count)
                    np.testing.assert_array_equal(got["g1"], [g1] * count)

    def test_fitted_case2_and_mixed_are_rejected(self):
        for sensor in SENSORS:
            for water in ("case2", "mixed"):
                with self.subTest(sensor=sensor, water=water), self.assertRaises(ValueError):
                    resolve_coefficients(sensor, water, "fitted", APP_ROOT)

    def test_repeated_selections_are_independent(self):
        # 반환 배열을 바꾼 호출자가 있어도 다음 실행의 내장값은 오염되지 않아야 합니다.
        first = resolve_coefficients("Sentinel2", "case1", "fitted", APP_ROOT)
        first["g0"][0] = -999
        resolve_coefficients("SuperDove", "case2", "fixed", APP_ROOT)
        resolve_coefficients("Sentinel2", "mixed", "fixed", APP_ROOT)
        final = resolve_coefficients("Sentinel2", "case1", "fitted", APP_ROOT)
        np.testing.assert_array_equal(final["g0"], load_fixture("Sentinel2")["g0"])

    def test_thirteen_bottoms_and_original_three_are_available(self):
        for sensor in SENSORS:
            fixture = load_fixture(sensor)
            rows = scan_bottoms(APP_ROOT / "DATA" / "bottom_reflectances", fixture["waves"])
            self.assertEqual(len(rows), 13)
            self.assertEqual(len({row["id"] for row in rows}), 13)
            self.assertTrue(all(not row.get("error") for row in rows), rows)
            self.assertFalse(any(Path(row["path"]).name == "filelist.txt" for row in rows))
            mapping = {row["id"]: row for row in rows}
            for material, spectrum in fixture["library"].items():
                np.testing.assert_allclose(
                    mapping[material]["reflectance"], spectrum, rtol=0, atol=1e-14
                )
            self.assertTrue(all(row["title"] and row["path"] for row in rows))

    def test_invalid_bottom_files_are_visible_errors(self):
        # 목록의 잘못된 파일을 조용히 제외하면 사용자가 선택 실패 원인을 모릅니다.
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder / "filelist.txt").write_text("not a spectrum", encoding="utf-8")
            (folder / "bad_order.csv").write_text(
                "443,0.2\n490,0.3\n490,0.4\n665,0.5\n", encoding="utf-8"
            )
            (folder / "short_range.csv").write_text("490,0.2\n560,0.3\n", encoding="utf-8")
            (folder / "nonfinite.csv").write_text("443,nan\n665,0.3\n", encoding="utf-8")
            rows = scan_bottoms(folder, [443, 490, 560, 665])
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(row.get("error") for row in rows))

    def test_deep_csv_requires_one_scene_and_rejects_duplicate_waves(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "deep.csv"
            path.write_text(
                "wavelength_nm,Rrs,solar_zenith_deg\n443,0.01,30\n443,0.02,30\n490,0.01,30\n560,0.01,30\n665,0.01,30\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                deep_water_values({"deep_csv": str(path)}, [443, 490, 560, 665])
            path.write_text(
                "deep_Rrs_443_sr-1,deep_Rrs_490_sr-1,deep_Rrs_560_sr-1,deep_Rrs_665_sr-1,solar_zenith_deg\n0.01,0.01,0.01,0.01,30\n0.02,0.02,0.02,0.02,30\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                deep_water_values({"deep_csv": str(path)}, [443, 490, 560, 665])

    def test_deep_water_values_and_angles_are_checked(self):
        for values, solar in (
            ([0, 0.01, 0.01, 0.01], 30),
            ([0.01] * 4, 90),
            ([0.01] * 3, 30),
            ([0.01] * 4, float("nan")),
        ):
            with self.subTest(values=values, solar=solar), self.assertRaises(ValueError):
                deep_water_values({"deep_rrs": values, "solar_zenith": solar}, [443, 490, 560, 665])


if __name__ == "__main__":
    unittest.main(verbosity=2)


class DeepWaterWavelengthAndHashRegressionTests(unittest.TestCase):
    """다른 파장의 오인식과 대용량 해시 구간의 중지 누락을 재발 방지합니다."""

    def test_fractional_wavelength_is_not_truncated(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "deep.csv"
            path.write_text(
                "wavelength_nm,Rrs\n443.9,0.01\n490,0.02\n560,0.03\n665,0.04\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "443 nm"):
                deep_water_values(
                    {"sensor": "Sentinel2", "deep_csv": str(path), "solar_zenith": 30},
                    [443, 490, 560, 665],
                )
            # 표기만 소수인 정확한 443.0은 정상적으로 해당 밴드를 가리킵니다.
            path.write_text(
                "wavelength_nm,Rrs\n443.0,0.01\n490.0,0.02\n560.0,0.03\n665.0,0.04\n",
                encoding="utf-8",
            )
            values, solar = deep_water_values(
                {"sensor": "Sentinel2", "deep_csv": str(path), "solar_zenith": 30},
                [443, 490, 560, 665],
            )
            self.assertEqual(values, [0.01, 0.02, 0.03, 0.04])
            self.assertEqual(solar, 30)

    def test_superdove_aliases_are_not_accepted_for_sentinel2(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "deep.csv"
            contents = (
                "wavelength_nm,Rrs\n444,0.01\n492,0.02\n560,0.03\n666,0.04\n",
                "deep_Rrs_444_sr-1,deep_Rrs_492_sr-1,deep_Rrs_560_sr-1,deep_Rrs_666_sr-1\n0.01,0.02,0.03,0.04\n",
            )
            for content in contents:
                with self.subTest(format="long" if content.startswith("wavelength") else "wide"):
                    path.write_text(content, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "443 nm"):
                        deep_water_values(
                            {"sensor": "Sentinel2", "deep_csv": str(path), "solar_zenith": 30},
                            [443, 490, 560, 665],
                        )

    def test_superdove_aliases_remain_supported_in_both_csv_formats(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "deep.csv"
            actual = [444, 492, 533, 566, 612, 666]
            contents = (
                "wavelength_nm,Rrs\n" + "".join(f"{w},0.01\n" for w in actual),
                ",".join(f"deep_Rrs_{w}_sr-1" for w in actual)
                + "\n"
                + ",".join(["0.01"] * 6)
                + "\n",
            )
            for content in contents:
                with self.subTest(format="long" if content.startswith("wavelength") else "wide"):
                    path.write_text(content, encoding="utf-8")
                    values, solar = deep_water_values(
                        {"sensor": "SuperDove", "deep_csv": str(path), "solar_zenith": 30},
                        [443, 490, 531, 565, 610, 665],
                    )
                    self.assertEqual(values, [0.01] * 6)
                    self.assertEqual(solar, 30)

    def test_file_hash_is_unchanged_and_cancellable_between_chunks(self):
        import hashlib
        from bathymetry.config import file_sha256

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "raster.bin"
            payload = b"x" * (3 * 1024 * 1024 + 5)
            path.write_bytes(payload)
            self.assertEqual(file_sha256(path), hashlib.sha256(payload).hexdigest())
            self.assertEqual(
                file_sha256(path, cancelled=lambda: False), hashlib.sha256(payload).hexdigest()
            )
            calls = []

            def cancel_after_one_block():
                calls.append(True)
                return len(calls) == 2

            with self.assertRaises(InterruptedError):
                file_sha256(path, cancelled=cancel_after_one_block)
            self.assertEqual(len(calls), 2)
