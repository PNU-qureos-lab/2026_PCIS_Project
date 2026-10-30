"""원본 픽셀 선택·평균/중위값·재현성·마스크 독립성을 작은 GeoTIFF로 검증한다."""
from pathlib import Path
import copy, json, sys, tempfile, unittest
import numpy as np
import rasterio
from rasterio.transform import from_origin

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from bathymetry.deepwater import (
    RrsSource,
    aggregate,
    normalize_pixels,
    create_selection,
    compute_reference,
)
from bathymetry.config import deep_water_values, load_config


class DeepWaterSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.values = np.tile(
            np.array(
                [
                    [0.01, 0.02, 0.09, 0.03, 0.04],
                    [0.05, 0.06, 0.07, 0.08, 0.10],
                    [0.11, 0.12, 0.13, 0.14, 0.15],
                ],
                dtype="float64",
            ),
            (4, 1, 1),
        )
        self.values *= np.arange(1, 5)[:, None, None]
        self.values[:, 1, 0] = 9.0
        self.values[1, 1, 1] = -1
        self.values[2, 1, 2] = np.nan
        self.pixels = [[row, col] for row in (0, 2) for col in range(5)]
        self.path = self.write("image.tif", self.values)
        self.config = {"sensor": "Sentinel2", "rrs_files": [str(self.path)], "solar_zenith": 30}

    def tearDown(self):
        self.temp.cleanup()

    def write(self, name, values):
        path = self.folder / name
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=values.shape[2],
            height=values.shape[1],
            count=len(values),
            dtype="float64",
            crs="EPSG:32748",
            transform=from_origin(100, 200, 3, 3),
            nodata=9.0,
        ) as ds:
            ds.write(values)
        return path

    def select(self, pixels=None, statistic="median"):
        return create_selection(self.config, self.pixels if pixels is None else pixels, statistic)

    def test_mean_and_median_use_raw_band_values(self):
        for mode, expected in [("mean", 0.084), ("median", 0.10)]:
            selected = self.select(statistic=mode)
            result = compute_reference(self.config, selected)
            np.testing.assert_allclose(
                result["rrs"], np.arange(1, 5) * expected, rtol=0, atol=1e-15
            )

    def test_even_count_median_and_duplicate_selection(self):
        selected = self.select(self.pixels + [[0, 0]])
        self.assertEqual(selected["pixels"], self.pixels)
        result = compute_reference(self.config, selected)
        np.testing.assert_allclose(result["rrs"], np.arange(1, 5) * 0.10)
        self.assertEqual(result["valid_count"], 10)

    def test_invalid_band_excludes_whole_pixel(self):
        with RrsSource(self.config) as source:
            _, values = source.sample([[0, 0], [1, 0], [1, 1], [1, 2]])
        result = aggregate(values, "median")
        self.assertEqual(
            (result["selected_count"], result["valid_count"], result["excluded_count"]), (4, 1, 3)
        )
        np.testing.assert_allclose(result["rrs"], np.arange(1, 5) * 0.01)

    def test_only_ten_distinct_pixels_can_be_applied(self):
        for pixels in (
            [],
            self.pixels[:9],
            self.pixels[:9] + [self.pixels[0]],
            self.pixels + [[1, 3]],
        ):
            with self.assertRaises(ValueError):
                self.select(pixels)

    def test_ten_selected_pixels_must_all_be_valid(self):
        for invalid in ([1, 0], [1, 1], [1, 2]):
            with self.assertRaisesRegex(ValueError, "유효한 10개"):
                self.select(self.pixels[:9] + [invalid])

    def test_saved_selection_cannot_bypass_count_or_validity(self):
        selected = self.select()
        for pixels in (
            self.pixels[:9],
            self.pixels + [[1, 3]],
            self.pixels[:9] + [[1, 0]],
            self.pixels[:9] + [self.pixels[0]],
        ):
            broken = copy.deepcopy(selected)
            broken["pixels"] = pixels
            with self.assertRaises(ValueError):
                compute_reference(self.config, broken)

    def test_no_valid_pixels_cannot_be_applied(self):
        for pixels in ([], [[1, 0], [1, 1], [1, 2]]):
            with self.assertRaises(ValueError):
                create_selection(self.config, pixels, "median")

    def test_coordinates_are_exact_and_in_bounds(self):
        for pixels in ([[0.2, 0]], [[True, 0]], [[-1, 0]], [[0, 3]], [[0]]):
            with self.assertRaises(ValueError):
                normalize_pixels(pixels, 3, 3)

    def test_unknown_statistic_is_rejected(self):
        with self.assertRaises(ValueError):
            self.select(statistic="minimum")

    def test_reference_extraction_can_be_cancelled(self):
        selected = self.select()
        with self.assertRaises(InterruptedError):
            compute_reference(self.config, selected, cancelled=lambda: True)

    def test_analysis_mask_does_not_filter_deep_selection(self):
        self.config["mask"] = str(self.write("exclude.tif", np.zeros((1, 3, 5))))
        self.assertEqual(compute_reference(self.config, self.select())["valid_count"], 10)

    def test_changed_sensor_grid_or_source_is_rejected(self):
        selected = self.select()
        for field, value in [
            ("sensor", "SuperDove"),
            ("waves_nm", [1, 2, 3, 4]),
            ("grid", {}),
            ("source_sha256", ["bad"]),
        ]:
            broken = copy.deepcopy(selected)
            broken[field] = value
            with self.assertRaises(ValueError):
                compute_reference(self.config, broken)

    def test_same_filename_changed_image_is_rejected(self):
        selected = self.select()
        self.write("image.tif", np.ones((4, 3, 5)) * 0.012)
        with self.assertRaisesRegex(ValueError, "내용 또는 밴드 순서"):
            compute_reference(self.config, selected)

    def test_preview_tampering_does_not_change_computed_reference(self):
        selected = self.select()
        selected["preview"]["rrs"] = [8, 8, 8, 8]
        values, solar = deep_water_values(
            dict(self.config, deep_selection=selected), [443, 490, 560, 665]
        )
        np.testing.assert_allclose(values, np.arange(1, 5) * 0.10)
        self.assertEqual(solar, 30)

    def test_ambiguous_input_and_missing_angle_are_rejected(self):
        selected = self.select()
        for extra in ({"deep_rrs": [1] * 4}, {"deep_csv": "bad.csv"}, {"solar_zenith": None}):
            with self.assertRaises(ValueError):
                deep_water_values(
                    dict(self.config, deep_selection=selected, **extra), [443, 490, 560, 665]
                )

    def test_json_round_trip_and_relative_image_paths(self):
        config = dict(self.config, rrs_files=["image.tif"], deep_selection=self.select())
        path = self.folder / "config.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        loaded = load_config(path)
        np.testing.assert_allclose(
            deep_water_values(loaded, [443, 490, 560, 665])[0], np.arange(1, 5) * 0.10
        )

    def test_separate_bands_and_six_band_source(self):
        separate = [str(self.write(f"band{i}.tif", self.values[i : i + 1])) for i in range(4)]
        config = dict(self.config, rrs_files=separate)
        selected = create_selection(config, self.pixels, "mean")
        np.testing.assert_allclose(
            compute_reference(config, selected)["rrs"], np.arange(1, 5) * 0.084
        )
        sd = dict(
            sensor="SuperDove",
            rrs_files=[str(self.write("sd.tif", np.tile(self.values[:1], (6, 1, 1))))],
        )
        selected = create_selection(sd, self.pixels, "median")
        np.testing.assert_allclose(compute_reference(sd, selected)["rrs"], [0.10] * 6)

    def test_display_stretch_does_not_change_source_statistics(self):
        with RrsSource(self.config) as source:
            indices, ranges = source.rgb_stretch()
            from rasterio.windows import Window

            rgb = source.rgb(Window(0, 0, 3, 3), (60, 60), indices, ranges)
            self.assertEqual(rgb.shape, (60, 60, 3))
            self.assertEqual(rgb.dtype, np.dtype("uint8"))
        np.testing.assert_allclose(
            compute_reference(self.config, self.select())["rrs"], np.arange(1, 5) * 0.10
        )


if __name__ == "__main__":
    unittest.main()
