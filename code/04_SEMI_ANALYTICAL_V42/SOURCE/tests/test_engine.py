"""기준 구현 결과 재현과 프로그램의 사용자 선택 규칙을 검증합니다.

실행: 코드 폴더에서 python -m unittest discover -s tests -v
검증 자료는 동봉 JSON이며 외부 기준 구현 폴더 없이도 실행됩니다. 선택 사항으로
REFERENCE_CODE_ROOT 환경변수에 기존 기준 구현 경로를 주면 센서별 원본 함수를
별도 프로세스로 재실행하여 새 엔진과 직접 비교합니다.

이 시험은 알고리즘 보존·파일 선택 기능을 확인합니다. 실제 해역의 수심 정확도나
Case 2/복합해역 계수의 과학적 타당성을 인증하는 시험이 아닙니다.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT))

from bathymetry.engine import SolverSettings, make_context, solve_pixel

FIXTURE_ROOT = Path(__file__).parent / "fixtures"
SENSORS = ("Sentinel2", "SuperDove")


def load_fixture(sensor):
    return json.loads((FIXTURE_ROOT / f"reference_{sensor}.json").read_text(encoding="utf-8"))


def make_inputs(fixture, case):
    """이미 보관된 IOP로 문맥을 만들어 수심 탐색 자체를 독립 검증합니다."""
    context = make_context(
        case["a"],
        case["bb"],
        case["solar_zenith"],
        fixture["waves"],
        fixture["g0"],
        fixture["g1"],
        case["deep_rrs"],
    )
    library = {name: np.asarray(values, float) for name, values in fixture["library"].items()}
    return context, library


class EngineRegression(unittest.TestCase):
    """기준 구현이 남긴 수치와 비교합니다. 새 함수의 출력으로 기대값을 만들지 않습니다."""

    @classmethod
    def setUpClass(cls):
        cls.fixtures = {sensor: load_fixture(sensor) for sensor in SENSORS}
        cls.results = {}
        for sensor, fixture in cls.fixtures.items():
            for case in fixture["cases"]:
                context, library = make_inputs(fixture, case)
                cls.results[(sensor, case["case_id"])] = solve_pixel(
                    case["observed"], context, library
                )

    def assert_solution_close(self, result, expected):
        """허용차는 수심 격자 0.1 m보다 훨씬 작으며 OS별 부동소수점 차이만 허용합니다."""
        self.assertAlmostEqual(result["depth_m"], expected["depth_m"], delta=2e-7)
        self.assertAlmostEqual(result["F"], expected["F"], delta=1e-12)
        self.assertTrue(np.isclose(result["c4"], expected["c4"], rtol=2e-8, atol=1e-13))
        self.assertEqual(tuple(result["pair"]), tuple(expected["pair"]))
        self.assertEqual(result["status"], expected["status"])
        for material, value in expected["fractions"].items():
            self.assertAlmostEqual(result["fractions"][material], value, delta=1e-12)

    def test_six_archived_cases_match(self):
        for sensor, fixture in self.fixtures.items():
            for case in fixture["cases"]:
                with self.subTest(sensor=sensor, case=case["case_id"]):
                    result = self.results[(sensor, case["case_id"])]
                    self.assert_solution_close(result, case["expected"])
                    self.assertEqual(result["has_alternative"], case["expected"]["has_alternative"])
                    self.assertEqual(
                        result["any_evaluation_limit"], case["expected"]["any_evaluation_limit"]
                    )

    def test_three_bottoms_use_nine_seeds_then_five_pairs(self):
        for result in self.results.values():
            self.assertEqual(result["initial_candidate_count"], 9)
            self.assertEqual(result["optimized_candidate_count"], 5)
            self.assertEqual(len(result["candidates"]), 5)
            self.assertEqual(len({tuple(row["pair"]) for row in result["initial_candidates"]}), 3)
            # 세 종류를 동시에 섞지 않습니다. 최종 쌍 밖 성분은 정확히 0입니다.
            for material, fraction in result["fractions"].items():
                if material not in result["pair"]:
                    self.assertEqual(fraction, 0)
            self.assertAlmostEqual(sum(result["fractions"].values()), 1, delta=1e-12)

    def test_two_bottoms_optimize_all_three_seeds(self):
        for sensor, fixture in self.fixtures.items():
            case = fixture["cases"][1]
            context, library = make_inputs(fixture, case)
            library.pop("green")
            result = solve_pixel(case["observed"], context, library)
            self.assertEqual(result["initial_candidate_count"], 3)
            self.assertEqual(result["optimized_candidate_count"], 3)
            self.assertEqual(len(result["candidates"]), 3)
            self.assertEqual(set(result["pair"]), {"sand", "brown"})
            self.assertEqual(set(result["fractions"]), {"sand", "brown"})

    def test_names_with_underscores_are_not_split(self):
        fixture = self.fixtures["Sentinel2"]
        case = fixture["cases"][1]
        context, library = make_inputs(fixture, case)
        aliases = {name: "selected_bottom_" + name for name in library}
        result = solve_pixel(
            case["observed"], context, {aliases[name]: value for name, value in library.items()}
        )
        self.assertEqual(set(result["fractions"]), set(aliases.values()))
        self.assertEqual(
            tuple(result["pair"]), tuple(aliases[name] for name in case["expected"]["pair"])
        )
        self.assertAlmostEqual(result["depth_m"], case["expected"]["depth_m"], delta=2e-7)

    def test_unconstrained_f_is_reported_without_clipping(self):
        # 원본의 F 무제약 규칙을 유지합니다. 품질 표시를 이유로 계산값을 바꾸지 않습니다.
        fixture = self.fixtures["Sentinel2"]
        shallow = self.results[("Sentinel2", fixture["cases"][0]["case_id"])]
        deep = self.results[("Sentinel2", fixture["cases"][2]["case_id"])]
        self.assertGreater(shallow["F"], 1)
        self.assertLess(deep["F"], 0)
        self.assertTrue(shallow["f_outside"])
        self.assertTrue(deep["f_outside"])

    def test_sensor_switch_does_not_change_previous_context(self):
        # Sentinel-2 → SuperDove → Sentinel-2의 연속 호출에서 모듈 전역값 오염을 검사합니다.
        fixture = self.fixtures["Sentinel2"]
        case = fixture["cases"][1]
        context, library = make_inputs(fixture, case)
        before = {
            name: np.array(context[name], copy=True)
            for name in ("waves", "g0", "g1", "xc", "xb", "rrs_dp")
        }
        sd = self.fixtures["SuperDove"]
        sd_case = sd["cases"][1]
        sd_context, sd_library = make_inputs(sd, sd_case)
        solve_pixel(sd_case["observed"], sd_context, sd_library)
        result = solve_pixel(case["observed"], context, library)
        self.assert_solution_close(result, case["expected"])
        for name, value in before.items():
            np.testing.assert_array_equal(context[name], value)

    def test_invalid_rrs_and_bottom_count_fail_before_solving(self):
        fixture = self.fixtures["Sentinel2"]
        case = fixture["cases"][0]
        context, library = make_inputs(fixture, case)
        for bad in (0, float("nan"), float("inf")):
            observed = list(case["observed"])
            observed[0] = bad
            with self.subTest(value=bad), self.assertRaises(ValueError):
                solve_pixel(observed, context, library)
        with self.assertRaises(ValueError):
            solve_pixel(case["observed"], context, {"sand": library["sand"]})
        with self.assertRaises(ValueError):
            solve_pixel(case["observed"][:-1], context, library)

    def test_evaluation_limit_is_not_success(self):
        fixture = self.fixtures["Sentinel2"]
        case = fixture["cases"][0]
        context, library = make_inputs(fixture, case)
        result = solve_pixel(case["observed"], context, library, SolverSettings(max_evaluations=20))
        self.assertTrue(result["any_evaluation_limit"])
        self.assertTrue(any(row["status"] == "evaluation_limit" for row in result["candidates"]))
        self.assertTrue(np.isfinite(result["depth_m"]))

    def test_cancel_is_propagated(self):
        fixture = self.fixtures["Sentinel2"]
        case = fixture["cases"][0]
        context, library = make_inputs(fixture, case)
        with self.assertRaises(InterruptedError):
            solve_pixel(case["observed"], context, library, cancelled=lambda: True)

    @unittest.skipUnless(os.environ.get("REFERENCE_CODE_ROOT"), "선택 검증: REFERENCE_CODE_ROOT 미지정")
    def test_original_source_in_separate_processes(self):
        for sensor in SENSORS:
            command = [
                sys.executable,
                str(Path(__file__).with_name("reference_probe.py")),
                "--original-code",
                str(Path(os.environ["REFERENCE_CODE_ROOT"]) / sensor / "code"),
                "--fixture",
                str(FIXTURE_ROOT / f"reference_{sensor}.json"),
            ]
            # UTF-8와 bytecode 비생성을 하위 프로세스에 명시합니다. 원본 폴더는 읽기만 합니다.
            environment = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=True,
                timeout=90,
                env=environment,
            )
            for expected in json.loads(completed.stdout):
                with self.subTest(sensor=sensor, case=expected["case_id"]):
                    self.assert_solution_close(
                        self.results[(sensor, expected["case_id"])], expected
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
