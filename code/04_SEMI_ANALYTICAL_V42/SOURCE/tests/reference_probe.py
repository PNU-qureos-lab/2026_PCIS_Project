"""기준 구현 보관 결과에서 작은 검증 자료를 만들고, 원본 코드를 별도 실행합니다.

일반 검증은 동봉된 fixtures/*.json만 읽으므로 외부 기준 구현 경로가 필요 없습니다.
이 스크립트는 개발자가 새 배포본을 확인할 때만 사용하는 보조 도구입니다.

자료 추출: python reference_probe.py --extract-from <기준 구현 폴더> --output <fixtures 폴더>
원본 비교: python reference_probe.py --original-code <센서/code> --fixture <json>

원본 모듈 이름이 두 센서 모두 model/settings이기 때문에 한 프로세스에서 섞어
가져오지 않습니다. 원본 비교를 부르는 테스트가 센서마다 새 프로세스를 만듭니다.
원본 폴더와 CSV에는 쓰지 않으며, 전체 영상이나 전체 pilot을 다시 계산하지 않습니다.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys


def read_csv(path):
    """UTF-8 BOM 유무와 CSV 안의 줄바꿈을 모두 보존해 읽습니다."""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def export_fixture(reference_root, sensor, output):
    """실제 저장된 관측·IOP·최종 결과를 추출하며 새 코드의 답은 사용하지 않습니다."""
    base = Path(reference_root) / sensor
    pilot = base / "validation" / "pilot"
    iop_dir = pilot / ("02_chl_tsm_iop" if sensor == "Sentinel2" else "02_chltsm_iop")
    parameter_name = "chl_tsm_parameters.csv" if sensor == "Sentinel2" else "chltsm_parameters.csv"
    files = {
        "observations": pilot / "01_rrs_data" / "observations.csv",
        "bottoms": pilot / "01_rrs_data" / "bottom_library.csv",
        "iops": iop_dir / "iops.csv",
        "parameters": iop_dir / parameter_name,
        "estimates": pilot / "06_final_solution" / "estimates.csv",
        "coefficients": base / "g0g1_파장함수" / "g_coefficients.json",
    }
    tables = {key: read_csv(path) for key, path in files.items() if key != "coefficients"}
    coeff = json.loads(files["coefficients"].read_text(encoding="utf-8-sig"))
    # 갈조의 0.5 m 사례, 모래/갈조 혼합의 2 m 사례, 녹조의 5 m 사례를
    # 선택합니다. 범위 밖 F가 나오는 사례도 포함해 임의 clipping을 검출합니다.
    ids = [
        "ba100_Z05_SZ30_CHL03_TSM03",
        "ba50cs50_Z20_SZ30_CHL03_TSM03",
        "ga100_Z50_SZ30_CHL03_TSM03",
    ]
    if sensor == "SuperDove":
        ids = ["dov" + value for value in ids]
    cases = []
    for case_id in ids:
        result = next(row for row in tables["estimates"] if row["case_id"] == case_id)
        observations = sorted(
            (row for row in tables["observations"] if row["case_id"] == case_id),
            key=lambda row: float(row["wavelength"]),
        )
        iops = sorted(
            (row for row in tables["iops"] if row["water_key"] == result["water_key"]),
            key=lambda row: float(row["wavelength"]),
        )
        parameter = next(
            row for row in tables["parameters"] if row["water_key"] == result["water_key"]
        )
        cases.append(
            {
                "case_id": case_id,
                "water_key": result["water_key"],
                "solar_zenith": float(result["solar_zenith"]),
                "observed": [float(row["Rrs_obs"]) for row in observations],
                "deep_rrs": [float(row["Rrs_deep_observed"]) for row in iops],
                "a": [float(row["a_hat"]) for row in iops],
                "bb": [float(row["bb_hat"]) for row in iops],
                "expected_iop": {
                    "Chl": float(parameter.get("Chl_hat", parameter.get("Chl_mg_m3"))),
                    "TSM": float(parameter.get("TSM_hat", parameter.get("TSM_g_m3"))),
                    "u_SSE": float(parameter["u_SSE"]),
                    "starts_completed": int(parameter["starts_completed"]),
                },
                "expected": {
                    "depth_m": float(result["z_hat"]),
                    "pair": result["pair"].split("_"),
                    "F": float(result["F"]),
                    "fractions": {
                        material: float(result["f_" + material])
                        for material in ("sand", "brown", "green")
                    },
                    "c4": float(result["c4"]),
                    "status": result["status"],
                    "has_alternative": result["has_alternative"].lower() == "true",
                    "any_evaluation_limit": result["any_evaluation_limit"].lower() == "true",
                    "total_evaluations": int(result["total_evaluations"]),
                },
            }
        )
    bottom_rows = sorted(tables["bottoms"], key=lambda row: float(row["wavelength"]))
    deep_cases = []
    for water_key in ("SZ20_CHL01_TSM01", "SZ40_CHL05_TSM05"):
        parameter = next(row for row in tables["parameters"] if row["water_key"] == water_key)
        iops = sorted(
            (row for row in tables["iops"] if row["water_key"] == water_key),
            key=lambda row: float(row["wavelength"]),
        )
        deep_cases.append(
            {
                "water_key": water_key,
                "solar_zenith": float(parameter["solar_zenith"]),
                "deep_rrs": [float(row["Rrs_deep_observed"]) for row in iops],
                "a": [float(row["a_hat"]) for row in iops],
                "bb": [float(row["bb_hat"]) for row in iops],
                "expected_iop": {
                    "Chl": float(parameter.get("Chl_hat", parameter.get("Chl_mg_m3"))),
                    "TSM": float(parameter.get("TSM_hat", parameter.get("TSM_g_m3"))),
                    "u_SSE": float(parameter["u_SSE"]),
                    "starts_completed": int(parameter["starts_completed"]),
                },
            }
        )
    payload = {
        "description": "기준 자료에 보관된 pilot CSV에서 추출한 3개 대표 사례. 정답 생성에 코드를 사용하지 않음.",
        "sensor": sensor,
        "source_description": f"{sensor}/validation/pilot",
        "source_sha256": {
            key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in files.items()
        },
        "waves": coeff["knots_nm"],
        "g0": coeff["knot_g0"],
        "g1": coeff["knot_g1"],
        "library": {
            material: [float(row[material]) for row in bottom_rows]
            for material in ("sand", "brown", "green")
        },
        "cases": cases,
        "deep_cases": deep_cases,
    }
    destination = Path(output) / f"reference_{sensor}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    return str(destination)


def run_original(code, fixture):
    """원본 계산 함수를 순서대로 호출하고 JSON으로 비교 가능한 열만 반환합니다."""
    sys.path.insert(0, str(Path(code).resolve()))
    # 원본 폴더를 read-only로 보존합니다. __pycache__도 생성하지 않습니다.
    sys.dont_write_bytecode = True
    import numpy as np
    from settings import Settings
    from model import make_context
    from step03_initial_seeds import build_initial_seeds
    from step04_select_seeds import select_seeds
    from step05_optimize_seeds import optimize_seeds
    from step06_final_solution import classify_candidates

    payload = json.loads(Path(fixture).read_text(encoding="utf-8-sig"))
    library = {key: np.asarray(value, float) for key, value in payload["library"].items()}
    output = []
    settings = Settings()
    for case in payload["cases"]:
        context = make_context(case["a"], case["bb"], case["solar_zenith"], case["deep_rrs"])
        seeds = build_initial_seeds(np.asarray(case["observed"], float), context, library, settings)
        selected, _ = select_seeds(seeds, settings)
        records, _ = optimize_seeds(selected)
        best, _ = classify_candidates(records, settings)
        output.append(
            {
                "case_id": case["case_id"],
                "depth_m": best["z_hat"],
                "F": best["F"],
                "pair": best["pair"].split("_"),
                "c4": best["c4"],
                "fractions": {name: best["f_" + name] for name in library},
                "status": best["status"],
                "total_evaluations": best["total_evaluations"],
            }
        )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract-from", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "fixtures")
    parser.add_argument("--original-code", type=Path)
    parser.add_argument("--fixture", type=Path)
    args = parser.parse_args()
    if args.extract_from:
        print(
            json.dumps(
                [
                    export_fixture(args.extract_from, sensor, args.output)
                    for sensor in ("Sentinel2", "SuperDove")
                ],
                ensure_ascii=False,
            )
        )
    elif args.original_code and args.fixture:
        print(
            json.dumps(
                run_original(args.original_code, args.fixture), ensure_ascii=False, allow_nan=False
            )
        )
    else:
        parser.error("--extract-from 또는 --original-code와 --fixture를 지정해야 합니다.")


if __name__ == "__main__":
    main()
