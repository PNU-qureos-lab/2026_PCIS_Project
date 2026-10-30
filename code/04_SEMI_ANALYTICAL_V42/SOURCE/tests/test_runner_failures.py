"""영상 계산 도중 실패·취소가 생겨도 이미 완료한 결과를 정합하게 보존하는지 검사합니다.

수치 알고리즘은 별도 원본 회귀시험에서 검증합니다. 여기서는 의도적인 작업자 장애와
보고서 실패를 주입하고 실제 TIFF/CSV/분석기록에 남은 결과를 확인합니다.
"""
from __future__ import annotations

from concurrent.futures.process import BrokenProcessPool
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.transform import from_origin

CODE_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT))
from bathymetry import runner


def successful_result(index=0):
    """출력 계층에 전달되는 정상 수치 결과 하나입니다."""
    return dict(
        depth_m=2.0 + index,
        pair=("sand", "brown"),
        F=0.25,
        fractions={"sand": 0.25, "brown": 0.75},
        c4=0.01,
        rms_relative_error_pct=10.0,
        status="no_c4_improvement",
        any_evaluation_limit=False,
        has_alternative=False,
        bottom_reflectance_negative=False,
        bottom_reflectance_above_one=False,
    )


class RunnerFailurePreservationTests(unittest.TestCase):
    """외부에서 읽는 세 결과 형식이 처리 건수와 픽셀 위치에서 일치해야 합니다."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        image = self.folder / "rrs.tif"
        with rasterio.open(
            image,
            "w",
            driver="GTiff",
            width=3,
            height=1,
            count=4,
            dtype="float32",
            crs="EPSG:32648",
            nodata=np.nan,
            transform=from_origin(500000, 1000000, 20, 20),
        ) as raster:
            raster.write(np.full((4, 1, 3), 0.02, dtype="float32"))
        self.config = dict(
            sensor="Sentinel2",
            water_type="case1",
            coefficient_mode="fitted",
            rrs_files=[str(image)],
            deep_rrs=[0.01] * 4,
            solar_zenith=30,
            bottom_files=[
                str(APP_ROOT / "DATA" / "bottom_reflectances" / name)
                for name in ("coral_sand.txt", "brown_algae.txt")
            ],
            workers=1,
            output_root=str(self.folder / "outputs"),
        )

    def _read_saved(self):
        folders = list((self.folder / "outputs").iterdir())
        self.assertEqual(len(folders), 1)
        out = folders[0]
        summary = json.loads((out / "analysis_record.json").read_text(encoding="utf-8"))
        with (out / "pixel_results.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        with rasterio.open(out / "MAPS" / "depth_m.tif") as raster:
            depth = raster.read(1)
        with rasterio.open(out / "MAPS" / "status.tif") as raster:
            status = raster.read(1)
        with rasterio.open(out / "MAPS" / "quality_flags.tif") as raster:
            quality = raster.read(1, masked=True)
            self.assertEqual(raster.nodata, 65535)
        return out, summary, rows, depth, status, quality

    def test_broken_pool_preserves_completed_batch_pixels_and_counts(self):
        class FailingPool:
            """첫 결과 뒤 프로세스가 사라지는 실제 map 예외 경로를 재현합니다."""

            shutdown_called = False

            def __init__(self, **kwargs):
                pass

            def map(self, function, items, chunksize):
                yield items[0][0], successful_result(), None
                raise BrokenProcessPool("injected worker crash")

            def shutdown(self, **kwargs):
                FailingPool.shutdown_called = True

        self.config["workers"] = 2
        with patch.object(runner, "ProcessPoolExecutor", FailingPool), patch.object(
            runner.os, "cpu_count", return_value=4
        ):
            with self.assertRaisesRegex(BrokenProcessPool, "injected worker crash"):
                runner.run_analysis(self.config, APP_ROOT)
        self.assertTrue(FailingPool.shutdown_called)
        _, summary, rows, depth, status, quality = self._read_saved()
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(
            (summary["processed_pixels"], summary["successful_pixels"], summary["failed_pixels"]),
            (1, 1, 0),
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(float(rows[0]["depth_m"]), 2.0)
        self.assertEqual(depth[0, 0], 2.0)
        self.assertTrue(np.isnan(depth[0, 1:]).all())
        np.testing.assert_array_equal(status, [[1, 0, 0]])
        # 정상 품질값 0은 유효값이고 미처리 픽셀만 65535 NoData여야 합니다.
        np.testing.assert_array_equal(np.ma.getmaskarray(quality), [[False, True, True]])
        self.assertEqual(quality[0, 0], 0)

    def test_mid_batch_cancellation_preserves_completed_pixel(self):
        cancel_path = self.folder / "cancel.request"

        def solve_then_cancel(item):
            index, _ = item
            if index == 0:
                return index, successful_result(), None
            cancel_path.touch()
            return index, None, "CANCELLED"

        with patch.object(runner, "_solve_item", side_effect=solve_then_cancel), patch(
            "bathymetry.report.create_report"
        ):
            returned = runner.run_analysis(self.config, APP_ROOT, cancel_path=cancel_path)
        _, summary, rows, depth, status, quality = self._read_saved()
        self.assertEqual(returned["status"], "cancelled")
        self.assertEqual(summary["status"], "cancelled")
        self.assertEqual(summary["processed_pixels"], 1)
        self.assertEqual(summary["successful_pixels"], 1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(depth[0, 0], 2.0)
        self.assertTrue(np.isnan(depth[0, 1:]).all())
        np.testing.assert_array_equal(status, [[1, 0, 0]])
        np.testing.assert_array_equal(np.ma.getmaskarray(quality), [[False, True, True]])

    def test_report_failure_does_not_change_successful_inversion_status(self):
        events = []
        with patch.object(
            runner,
            "_solve_item",
            side_effect=lambda item: (item[0], successful_result(item[0]), None),
        ), patch(
            "bathymetry.report.create_report", side_effect=OSError("injected preview failure")
        ):
            returned = runner.run_analysis(self.config, APP_ROOT, event=events.append)
        out, summary, rows, depth, status, quality = self._read_saved()
        self.assertEqual(returned["status"], "completed")
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["report_error"], "injected preview failure")
        self.assertTrue((out / "report_error.txt").is_file())
        self.assertEqual((summary["processed_pixels"], summary["successful_pixels"]), (3, 3))
        self.assertEqual(len(rows), 3)
        np.testing.assert_array_equal(depth, [[2.0, 3.0, 4.0]])
        np.testing.assert_array_equal(status, [[1, 1, 1]])
        self.assertFalse(np.ma.getmaskarray(quality).any())
        self.assertEqual(events[-1]["type"], "complete")

    def test_pixel_failure_is_recorded_without_losing_neighbor_results(self):
        def solve_one_failure(item):
            index, _ = item
            return (
                (index, None, "injected pixel error")
                if index == 1
                else (index, successful_result(index), None)
            )

        with patch.object(runner, "_solve_item", side_effect=solve_one_failure), patch(
            "bathymetry.report.create_report"
        ):
            runner.run_analysis(self.config, APP_ROOT)
        _, summary, rows, depth, status, quality = self._read_saved()
        self.assertEqual(summary["status"], "completed_with_failures")
        self.assertEqual(
            (summary["processed_pixels"], summary["successful_pixels"], summary["failed_pixels"]),
            (3, 2, 1),
        )
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[1]["status"], "failed")
        self.assertEqual(depth[0, 0], 2.0)
        self.assertTrue(np.isnan(depth[0, 1]))
        self.assertEqual(depth[0, 2], 4.0)
        np.testing.assert_array_equal(status, [[1, 2, 1]])
        np.testing.assert_array_equal(quality.data, [[0, 16, 0]])
        self.assertFalse(np.ma.getmaskarray(quality).any())


if __name__ == "__main__":
    unittest.main(verbosity=2)
