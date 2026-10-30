"""작은 임시 GeoTIFF로 입력 경계와 NoData 처리를 검증합니다.

실제 위성영상을 변경하거나 전체 역산을 실행하지 않습니다. 각 시험의 3×3 자료는
종료와 함께 지워지고, validate_config가 결과 파일을 만들지 않는 것도 확인합니다.
"""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import rasterio
from rasterio.transform import from_origin

CODE_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT))
from bathymetry.runner import validate_config


class RasterInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.transform = from_origin(100, 200, 3, 3)
        self.config = {
            "sensor": "Sentinel2",
            "water_type": "case1",
            "coefficient_mode": "fitted",
            "deep_rrs": [0.01, 0.008, 0.004, 0.001],
            "solar_zenith": 30,
            "workers": 1,
            "bottom_files": [
                str(APP_ROOT / "DATA" / "bottom_reflectances" / name)
                for name in ("coral_sand.txt", "brown_algae.txt")
            ],
            "output_root": str(self.folder / "result"),
        }

    def tearDown(self):
        self.temp.cleanup()

    def raster(
        self,
        name,
        *,
        count=4,
        values=None,
        transform=None,
        crs="EPSG:32748",
        dtype="float32",
        nodata=None,
        scales=None,
    ):
        """실제 GIS 드라이버가 쓰고 읽는 3×3 파일을 만듭니다."""
        values = (
            np.full((count, 3, 3), 0.01, dtype=dtype)
            if values is None
            else np.asarray(values, dtype=dtype)
        )
        path = self.folder / name
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=values.shape[2],
            height=values.shape[1],
            count=values.shape[0],
            dtype=dtype,
            crs=crs,
            transform=transform or self.transform,
            nodata=nodata,
        ) as dataset:
            dataset.write(values)
            if scales is not None:
                dataset.scales = scales
        return str(path)

    def validate(self, files, **overrides):
        return validate_config(dict(self.config, rrs_files=files, **overrides), APP_ROOT)

    def test_multiband_and_separate_band_inputs_are_equivalent(self):
        stacked = self.raster("stack.tif")
        separate = [self.raster(f"band_{index}.tif", count=1) for index in range(4)]
        first = self.validate([stacked])
        second = self.validate(separate)
        self.assertEqual(first["유효_픽셀"], 9)
        self.assertEqual(second["유효_픽셀"], 9)
        self.assertEqual(first["격자"], second["격자"])
        self.assertEqual(first["초기_후보"], 3)
        self.assertEqual(first["탐색_후보"], 3)
        self.assertFalse((self.folder / "result").exists())

    def test_positive_nodata_nonfinite_nonpositive_and_mask_are_excluded(self):
        values = np.full((4, 3, 3), 0.01, dtype="float32")
        values[0, 0, 0] = 9999  # 양수여도 TIFF의 NoData이면 제외해야 합니다.
        values[1, 0, 1] = 0
        values[2, 0, 2] = -1
        values[3, 1, 0] = np.nan
        rrs = self.raster("nodata.tif", values=values, nodata=9999)
        mask_values = np.ones((1, 3, 3), dtype="uint8")
        mask_values[0, 1, 1] = 0
        mask_values[0, 1, 2] = 2
        mask = self.raster("mask.tif", values=mask_values, dtype="uint8", nodata=0)
        self.assertEqual(self.validate([rrs], mask=mask)["유효_픽셀"], 3)

    def test_positive_nodata_in_separate_band_is_excluded(self):
        values = np.full((1, 3, 3), 0.01, dtype="float32")
        values[0, 2, 2] = 9999
        files = [self.raster("one.tif", values=values, nodata=9999)]
        files += [self.raster(f"other_{index}.tif", count=1) for index in range(3)]
        self.assertEqual(self.validate(files)["유효_픽셀"], 8)

    def test_wrong_band_count_and_duplicate_files_are_rejected(self):
        wrong = self.raster("three_bands.tif", count=3)
        one = self.raster("one.tif", count=1)
        for files in (
            [wrong],
            [one, one, one, one],
            [one, wrong],
            [one, self.raster("second.tif", count=1), self.raster("third.tif", count=1), wrong],
        ):
            with self.subTest(files=files), self.assertRaises(ValueError):
                self.validate(files)

    def test_separate_bands_must_have_same_transform_size_and_crs(self):
        normal = [self.raster(f"normal_{index}.tif", count=1) for index in range(3)]
        wrong_files = [
            self.raster("shift.tif", count=1, transform=from_origin(100.5, 200, 3, 3)),
            self.raster("crs.tif", count=1, crs="EPSG:32749"),
            self.raster("size.tif", values=np.full((1, 2, 3), 0.01)),
        ]
        for wrong in wrong_files:
            with self.subTest(wrong=wrong), self.assertRaises(ValueError):
                self.validate(normal + [wrong])

    def test_mask_must_be_single_band_on_same_grid(self):
        rrs = self.raster("rrs.tif")
        wrong_masks = [
            self.raster("two_mask.tif", count=2),
            self.raster("shift_mask.tif", count=1, transform=from_origin(100, 201, 3, 3)),
            self.raster("crs_mask.tif", count=1, crs="EPSG:4326"),
        ]
        for mask in wrong_masks:
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                self.validate([rrs], mask=mask)

    def test_scaled_integer_or_unreferenced_rrs_is_rejected(self):
        files = [
            self.raster("scaled.tif", scales=(0.001,) * 4),
            self.raster("integer.tif", values=np.ones((4, 3, 3)), dtype="uint16"),
            self.raster("no_crs.tif", crs=None),
        ]
        for file in files:
            with self.subTest(file=file), self.assertRaises(ValueError):
                self.validate([file])

    def test_empty_analysis_area_is_rejected(self):
        rrs = self.raster("rrs.tif")
        mask = self.raster("empty_mask.tif", values=np.zeros((1, 3, 3)), dtype="uint8")
        with self.assertRaises(ValueError):
            self.validate([rrs], mask=mask)

    def test_six_band_superdove_is_checked_as_six_bands(self):
        rrs = self.raster("six.tif", count=6)
        summary = self.validate([rrs], sensor="SuperDove", deep_rrs=[0.01] * 6)
        self.assertEqual(summary["파장_nm"], [443, 490, 531, 565, 610, 665])
        self.assertEqual(summary["유효_픽셀"], 9)
        with self.assertRaises(ValueError):
            self.validate([rrs])


if __name__ == "__main__":
    unittest.main(verbosity=2)
