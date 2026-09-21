"""7단계: 최종 대표 결과 CSV에서 그림과 그림용 통계 표를 만듭니다.

계산을 재실행하지 않고 make_figures(estimates_csv, output_dir)를 호출합니다.
입력 CSV를 고치지 않으며 PNG·PDF·통계 CSV를 output_dir에 저장합니다.
그림 번호는 기존 Figure 1·2·3·5·6·7을 유지합니다.

수심은 z_hat 대 true_z, 바닥은 모래·갈조·녹조 세 계수를 같은 의미끼리 비교합니다.
Figure 1은 수심 1장, 실제 혼합 바닥별 F 3장, 각 성분이 포함된 모든 사례의 F 3장을 저장합니다.
F가 음수 또는 1 초과인 결과도 포함합니다. Figure 1은 예시 스타일의 원형 점을 사용합니다.
계산 한도로 끝난 결과라도 필요한 수치가 모두 유한하면 포함합니다.
값이 빠지거나 NaN·무한대인 사례는 제외 이유를 별도 audit 표에 기록합니다.

Figure 5는 기준 범위로 정규화한 수심·바닥 RMSE 및 95% bootstrap 에러바를 표시합니다.
정규화 분모는 수심 4.5 m, 바닥 계수 1로 고정하며 실제 계수 0도 포함합니다.
원래 단위의 RMSE도 함께 저장합니다.
NumPy와 Matplotlib을 사용하고 화면 창 없이 파일로 그림을 그립니다.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
from settings import FIGURE5_DEPTH_RANGE_M, FIGURE5_COEFFICIENT_RANGE


COMPONENTS = (
    ("sand", "Sand", "#D59B24"),
    ("brown", "Brown algae", "#97613C"),
    ("green", "Green algae", "#009E73"),
)
GROUPS = (
    ("sand_brown", "Brown algae + sand", "#97613C"),
    ("brown_green", "Green + brown algae", "#CC79A7"),
    ("sand_green", "Green algae + sand", "#56B4E9"),
    ("green", "Green algae", "#009E73"),
    ("brown", "Brown algae", "#D55E00"),
    ("sand", "Sand", "#D59B24"),
)
NUMERIC_COLUMNS = (
    "true_z", "z_hat", "true_f_sand", "true_f_brown", "true_f_green",
    "f_sand", "f_brown", "f_green", "c4",
)
REQUIRED_COLUMNS = (
    "case_id", "true_group", *NUMERIC_COLUMNS,
    "status", "has_alternative", "f_outside",
)


def _boolean(value: str | None) -> bool:
    """CSV의 True·1·yes 등 문자열을 실제 참/거짓 값으로 해석합니다.
    """
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "t"}


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> Path:
    """그림에 사용한 숫자 표를 지정 열 순서의 UTF-8 CSV로 저장합니다.
    """
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _load_estimates(path: Path) -> tuple[dict[str, np.ndarray], list[dict]]:
    """사례당 대표 결과가 한 행인 CSV를 읽고 그림에 포함할 행을 검사합니다.

    case_id 중복이나 모르는 바닥 그룹은 오류로 알립니다.
    필수 수치가 모두 유한한 행을 포함하며 큰 오차·음수 F라는 이유로 빼지 않습니다.
    반환 data는 그림용 배열, audit는 원래 모든 행의 포함 여부와 제외 이유입니다.
    범위 밖 여부는 저장된 표시뿐 아니라 실제 세 계수에서도 다시 검사합니다.
    """
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = set(REQUIRED_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Estimates CSV is missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("The estimates CSV contains no cases.")
    ids = [row["case_id"].strip() for row in rows]
    duplicate_ids = [case_id for case_id, n in Counter(ids).items() if n > 1]
    if duplicate_ids or any(not case_id for case_id in ids):
        raise ValueError(
            "Expected one nonempty, unique case_id per selected estimate. "
            f"Duplicate IDs: {duplicate_ids[:5]}"
        )
    known_groups = {item[0] for item in GROUPS}
    unknown = {row["true_group"] for row in rows} - known_groups
    if unknown:
        raise ValueError(f"Unrecognized truth groups: {sorted(unknown)}")

    numeric: dict[str, np.ndarray] = {}
    for name in NUMERIC_COLUMNS:
        values = []
        for row in rows:
            try:
                values.append(float(row[name]))
            except (TypeError, ValueError):
                values.append(float("nan"))
        numeric[name] = np.asarray(values, dtype=float)
    # 필요한 값이 빠진 행만 제외합니다. 오차 크기나 F 범위는 이 필터의 조건이 아닙니다.
    valid = np.ones(len(rows), dtype=bool)
    for name in NUMERIC_COLUMNS:
        valid &= np.isfinite(numeric[name])

    audit = []
    for i, row in enumerate(rows):
        invalid = [name for name in NUMERIC_COLUMNS if not np.isfinite(numeric[name][i])]
        audit.append({
            "case_id": ids[i], "status": row["status"],
            "included_in_figures": bool(valid[i]),
            "reason": "" if valid[i] else "Nonfinite or missing: " + ", ".join(invalid),
            "has_alternative": _boolean(row["has_alternative"]),
            "f_outside": _boolean(row["f_outside"]) or any(
                np.isfinite(numeric[f"f_{key}"][i])
                and (numeric[f"f_{key}"][i] < 0 or numeric[f"f_{key}"][i] > 1)
                for key, _, _ in COMPONENTS
            ),
        })
    data = {name: values[valid] for name, values in numeric.items()}
    data["_true_depths_all_cases"] = np.unique(
        numeric["true_z"][np.isfinite(numeric["true_z"])]
    )
    for name in ("case_id", "true_group", "status"):
        data[name] = np.asarray([row[name] for row in rows], dtype=str)[valid]
    data["has_alternative"] = np.asarray(
        [_boolean(row["has_alternative"]) for row in rows], dtype=bool
    )[valid]
    # 저장된 표시가 빠졌거나 잘못돼도 실제 범위 밖 계수가 숨겨지지 않도록
    # 세 계수 값에서 범위 밖 여부를 다시 확인합니다.
    outside = np.asarray([_boolean(row["f_outside"]) for row in rows], dtype=bool)[valid]
    for key, _, _ in COMPONENTS:
        outside |= (data[f"f_{key}"] < 0) | (data[f"f_{key}"] > 1)
    data["f_outside"] = outside
    return data, audit


def _metrics(errors: np.ndarray) -> dict[str, float | int]:
    """오차 배열에서 개수, MAE, RMSE, 평균 부호 오차를 계산합니다.

    오차는 추정값-실제값입니다. MAE=평균 절댓값, RMSE=제곱 평균의 제곱근입니다.
    빈 그룹의 결과를 0으로 만들지 않고 NaN으로 남겨 자료 없음과 오차 0을 구분합니다.
    """
    return {
        "n": int(errors.size),
        "mae": float(np.mean(np.abs(errors))) if errors.size else float("nan"),
        "rmse": float(np.sqrt(np.mean(errors * errors))) if errors.size else float("nan"),
        "mean_signed_error": float(np.mean(errors)) if errors.size else float("nan"),
    }


def _limits(*arrays: np.ndarray, include_zero: bool = False) -> tuple[float, float]:
    """자료 전체가 보이도록 축 범위에 여백을 더합니다. 결과 값을 자르지 않습니다.
    """
    values = np.concatenate([np.asarray(a).ravel() for a in arrays])
    values = values[np.isfinite(values)]
    if include_zero:
        values = np.append(values, 0.0)
    if not values.size:
        return -0.05, 1.05
    low, high = float(values.min()), float(values.max())
    spread = high - low
    pad = 0.06 * spread if spread > 0 else max(abs(low) * 0.06, 0.05)
    return low - pad, high + pad


def _format_stats(errors: np.ndarray, unit: str = "") -> str:
    """그래프 안에 표시할 표본 수·MAE·RMSE 문자열을 만듭니다.
    """
    stats = _metrics(errors)
    suffix = f" {unit}" if unit else ""
    return (
        f"n = {stats['n']:,}\n"
        f"MAE = {stats['mae']:.4g}{suffix}\n"
        f"RMSE = {stats['rmse']:.4g}{suffix}"
    )


def _style_axis(ax) -> None:
    """눈금·격자·테두리만 꾸밉니다. 계산값과 포함 사례를 바꾸지 않습니다.
    """
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color="#D5DADF", linewidth=0.6, alpha=0.65)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out")


def _scatter(ax, x, y, outside, color="#0072B2") -> None:
    """범위 안 계수는 점, 범위 밖 계수가 있는 사례는 붉은 x로 표시합니다.

    음수 또는 1 초과 계수를 숨기지 않고 같은 그림에서 확인하기 위한 구분입니다.
    """
    regular = ~outside
    if np.any(regular):
        ax.scatter(x[regular], y[regular], s=15, c=color, alpha=0.36,
                   linewidths=0, label="Within-range coefficients")
    if np.any(outside):
        ax.scatter(x[outside], y[outside], s=23, c="#B53633", marker="x",
                   linewidths=0.8, alpha=0.72,
                   label="Out-of-range coefficient(s)")


def _save(fig, output_dir: Path, stem: str) -> list[Path]:
    """같은 그림을 PNG와 PDF로 저장하고 두 경로를 반환합니다.
    """
    paths = [output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"]
    fig.savefig(paths[0], dpi=180, facecolor="white", bbox_inches="tight")
    fig.savefig(paths[1], facecolor="white", bbox_inches="tight")
    return paths


def _figure1_statistics(truth: np.ndarray, estimate: np.ndarray) -> dict:
    """Figure 1에 표시할 R²·RMSE·MAE·Bias를 같은 사례들로 계산합니다.

    오차는 추정-실제이며 R²=1-SSE/SST를 사용합니다. 상관계수의 제곱이 아닙니다.
    순수 바닥처럼 실제값이 모두 같으면 SST=0이어서 R²가 정의되지 않습니다.
    그 경우 R²를 NaN과 상태로 기록하고 그림에는 N/A로 표시합니다.
    음수 R²나 범위 밖 추정값을 0 또는 1로 바꾸지 않습니다.
    다른 Figure가 쓰는 기존 _metrics 함수는 변경하지 않습니다.
    """
    truth, estimate = np.asarray(truth, float), np.asarray(estimate, float)
    errors = estimate - truth
    n = int(truth.size)
    sse = float(np.sum(errors**2))
    sst = float(np.sum((truth - np.mean(truth))**2)) if n else 0.0
    r2_status = "insufficient_cases" if n < 2 else "constant_true_values" if sst == 0 else "ok"
    return dict(
        n=n, r2=1.0-sse/sst if r2_status == "ok" else float("nan"),
        r2_status=r2_status,
        rmse=float(np.sqrt(np.mean(errors**2))) if n else float("nan"),
        mae=float(np.mean(np.abs(errors))) if n else float("nan"),
        bias=float(np.mean(errors)) if n else float("nan"),
    )


def make_figure1(estimates_csv: Path, output_dir: Path) -> list[Path]:
    """Figure 1만 다시 그려 output_dir/Figure1에 일곱 개의 개별 그림을 저장합니다.

    사용자 예시의 흰 배경, 세리프 글꼴, 원형 점, 검은 1:1 점선,
    왼쪽 위 통계 상자 스타일을 사용합니다. PNG와 PDF를 각각 저장합니다.

    바닥 그룹은 원본의 실제 바닥 구성으로 구분합니다. F의 의미는 다음과 같습니다.
    Sand : brown algae → 모래 계수, Sand : green algae → 모래 계수,
    Brown algae : green algae → 갈조 계수.
    마지막 세 그림은 실제 해당 성분 계수 > 0인 모든 사례(혼합+순수)의 해당 계수입니다.
    추정된 pair가 달라도 f_sand/f_brown/f_green에서 같은 성분을 읽습니다.
    쌍마다 뜻이 달라지는 원래 F 열을 직접 비교하거나 두 성분끼리 재정규화하지 않습니다.

    예: Sand는 Sand–brown, Sand–green, 순수 Sand를 포함합니다.
    포함 여부는 추정 계수가 아니라 실제 계수로 결정하며 혼합 사례는 두 성분 그림에 포함됩니다.
    점 위치에 임의 잡음을 넣지 않으며, 음수·1 초과의 추정 계수도 전부 포함합니다.
    QAA·초기화·최적화는 실행하지 않습니다. 다른 Figure는 생성하지 않습니다.
    """
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "iteration_pipeline_matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    estimates_csv = Path(estimates_csv)
    destination = Path(output_dir) / "Figure1"
    data, audit = _load_estimates(estimates_csv)
    if not data["z_hat"].size:
        raise ValueError("No complete finite selected estimates for Figure 1")
    destination.mkdir(parents=True, exist_ok=True)

    # 이름의 앞 바닥으로 F를 고정하고, 바닥쌍을 잘못 추정한 사례도 같은 성분으로 평가합니다.
    specs = [
        ("a", "water_depth", "Water depth", "all", "depth", "#2378BD"),
        ("b", "sand_brown", "Sand : brown algae", "sand_brown", "sand", "#F28C13"),
        ("c", "sand_green", "Sand : green algae", "sand_green", "sand", "#449A51"),
        ("d", "brown_green", "Brown algae : green algae", "brown_green", "brown", "#9467BD"),
        ("e", "sand", "Sand (all containing cases)", "contains_sand", "sand", "#D79C25"),
        ("f", "brown", "Brown algae (all containing cases)", "contains_brown", "brown", "#956041"),
        ("g", "green", "Green algae (all containing cases)", "contains_green", "green", "#299E77"),
    ]
    component_labels = {"sand": "sand", "brown": "brown algae", "green": "green algae"}
    paths, metrics_rows, point_rows = [], [], []
    # Figure 1 안에서만 스타일을 바꿉니다. 다른 그림의 전역 설정은 그대로입니다.
    style = {
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 15,
        "axes.titlesize": 21, "axes.titleweight": "bold", "axes.labelsize": 17,
        "xtick.labelsize": 14, "ytick.labelsize": 14,
        "axes.linewidth": 1.2, "pdf.fonttype": 42, "ps.fonttype": 42,
    }
    with plt.rc_context(style):
        for letter, key, title, group, component, color in specs:
            if group == "all":
                mask = np.ones(data["z_hat"].size, dtype=bool)
            elif group.startswith("contains_"):
                # 실제 해당 성분이 들어간 모든 혼합·순수 사례를 포함합니다.
                # 추정 F가 음수·1 초과이거나 바닥쌍을 잘못 맞춰도 제외하지 않습니다.
                mask = data[f"true_f_{component}"] > 0
            else:
                mask = data["true_group"] == group
            truth = data["true_z"][mask] if component == "depth" else data[f"true_f_{component}"][mask]
            estimate = data["z_hat"][mask] if component == "depth" else data[f"f_{component}"][mask]
            stats = _figure1_statistics(truth, estimate)
            unit = "m" if component == "depth" else "unitless"
            stem = f"figure1{letter}_{key}"
            metrics_rows.append(dict(panel=letter, title=title, true_group=group,
                F_component=component, unit=unit, **stats, image_file=stem+".png"))
            for case_id, true_value, estimated_value in zip(data["case_id"][mask], truth, estimate):
                point_rows.append(dict(panel=letter, case_id=case_id, true_group=group,
                    F_component=component, true_value=float(true_value), estimated_value=float(estimated_value)))

            fig, ax = plt.subplots(figsize=(7.2, 7.2))
            fig.subplots_adjust(left=0.15, right=0.965, bottom=0.13, top=0.90)
            fig.patch.set_facecolor("white")
            ax.set_facecolor("white")
            # 원형 점과 실제 좌표를 유지합니다. 겹친 점을 벌리기 위한 jitter는 넣지 않습니다.
            ax.scatter(truth, estimate, s=24, facecolor=color, edgecolor=color,
                       linewidth=0.45, alpha=0.55, zorder=3)
            values = np.concatenate([truth, estimate])
            low = min(0.0, float(values.min())) if values.size else 0.0
            high = max(1.0, float(values.max())) if values.size else 1.0
            padding = 0.045 * (high-low)
            low = low-padding if low < 0 or (values.size and np.any(values == 0)) else 0.0
            high += padding
            # 같은 축 범위로 1:1 관계를 표시하고 범위 밖 추정값까지 모두 보이게 합니다.
            ax.plot([low, high], [low, high], color="black", linestyle="--", linewidth=1.35, zorder=2)
            ax.set(xlim=(low, high), ylim=(low, high))
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, linestyle="--", linewidth=0.75, color="#D0D0D0", alpha=0.8)
            ax.set_axisbelow(True)
            ax.tick_params(direction="out", length=5, width=1)
            ax.set_title(f"({letter}) {title}", loc="left", pad=13)
            if component == "depth":
                ax.set_xlabel("True depth (m)", labelpad=9)
                ax.set_ylabel("Estimated depth (m)", labelpad=10)
            else:
                ax.set_xlabel(f"True F ({component_labels[component]})", labelpad=9)
                ax.set_ylabel(f"Estimated F ({component_labels[component]})", labelpad=10)

            suffix = " m" if component == "depth" else ""
            r2_text = f"{stats['r2']:.3f}" if np.isfinite(stats["r2"]) else "N/A"
            metric_text = "\n".join([
                "$R^2$ = " + r2_text,
                f"RMSE = {stats['rmse']:.3f}{suffix}",
                f"MAE = {stats['mae']:.3f}{suffix}",
                f"Bias = {stats['bias']:+.3f}{suffix}",
            ])
            if stats["r2_status"] == "constant_true_values":
                metric_text += "\n(constant true depth)" if component == "depth" else "\n(constant true F)"
            ax.text(0.045, 0.965, metric_text, transform=ax.transAxes,
                    ha="left", va="top", fontsize=14, linespacing=1.30,
                    bbox=dict(boxstyle="round,pad=0.42", facecolor="white",
                              edgecolor="#B5B5B5", linewidth=1, alpha=0.94), zorder=4)
            ax.text(0.985, 0.025, f"n = {stats['n']:,}", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=12, color="#444444")
            for extension in ("png", "pdf"):
                path = destination / f"{stem}.{extension}"
                fig.savefig(path, dpi=300, facecolor="white")
                paths.append(path)
            plt.close(fig)

    paths.append(_write_csv(destination/"figure1_metrics.csv", list(metrics_rows[0]), metrics_rows))
    paths.append(_write_csv(destination/"figure1_points.csv", list(point_rows[0]), point_rows))
    metadata = dict(
        source_estimates_csv=str(estimates_csv.resolve()), figures=7,
        input_cases=len(audit), included_depth_cases=int(data["z_hat"].size),
        excluded_cases=len(audit)-int(data["z_hat"].size),
        exclude_out_of_range=False, included_out_of_range_cases=int(data["f_outside"].sum()),
        grouping="a: all cases; b-d: true mixed bottom type; e-g: true component coefficient > 0 (mixed and pure); misclassified pairs retained",
        component_panels_overlap=True,
        component_panel_selection="true_f_component > 0; never filter on estimated component",
        F_definition={group:component for _,_,_,group,component,_ in specs if group != "all"},
        r2_formula="1 - sum((estimate-truth)^2) / sum((truth-mean(truth))^2)",
        constant_truth_r2="undefined, displayed as N/A", bias_formula="mean(estimate-truth)",
        estimates_clipped=False, points_jittered=False, pair_coefficients_renormalized=False,
        input_estimates_modified=False, inversion_performed=False,
    )
    metadata_path = destination/"figure1_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    paths.append(metadata_path)
    return paths


def make_figure5(estimates_csv: Path, output_dir: Path, *, exclude_out_of_range: bool = False,
                 depth_reference_range_m: float = FIGURE5_DEPTH_RANGE_M,
                 coefficient_reference_range: float = FIGURE5_COEFFICIENT_RANGE) -> list[Path]:
    """Figure 5: 기준 범위로 정규화한 수심·바닥 RMSE를 같은 % 축에서 비교합니다.

    수심 NRMSE(%) = 수심 RMSE / 4.5 m × 100.
    바닥 NRMSE(%) = 세 바닥 계수의 합동 RMSE / 1 × 100.
    기본 분모는 전체 실험의 실제 수심 범위 5.0-0.5=4.5 m와 계수 범위 1-0=1입니다.
    수심별로 분모를 다시 정하거나 각 실제값으로 나누지 않습니다.
    곡선 자체의 최솟값·최댓값으로 0~1에 늘여 맞추는 방식도 아닙니다.
    분모는 함수 인자로 명시적으로 변경할 수 있으며 실제 사용값을 메타데이터에 저장합니다.

    바닥 합동 RMSE=sqrt(sum_i,sum_k((f_hat_i,k-f_true_i,k)^2)/(3*n)).
    실제 계수 0을 포함한 모래·갈조·녹조의 모든 오차를 같은 가중치로 합칩니다.
    성분별 RMSE 세 개를 단순 평균하지 않습니다. 바닥 종류가 틀린 추정도 포함합니다.
    기본적으로 최종 F 범위 밖 사례까지 포함한 전체 4,860개 결과를 사용합니다.
    exclude_out_of_range=True는 범위 안 사례만 보는 보조 분석에만 사용합니다.

    에러바는 같은 실제 수심 안의 사례를 2,000회 복원추출한 RMSE의 95% 구간입니다.
    같은 사례의 수심과 세 계수를 함께 추출한 뒤, 구간 양끝도 같은 고정 분모로 나눕니다.
    난수 seed=20260914. n<2는 구간 생략. 장비·모형 전체의 불확실성 구간은 아닙니다.
    원래 단위의 RMSE 그림·통계도 저장하며 추정 자체와 다른 Figure는 수정하지 않습니다.
    """
    depth_reference_range_m = float(depth_reference_range_m)
    coefficient_reference_range = float(coefficient_reference_range)
    if (not np.isfinite(depth_reference_range_m) or depth_reference_range_m <= 0
            or not np.isfinite(coefficient_reference_range) or coefficient_reference_range <= 0):
        raise ValueError("Normalization reference ranges must be finite and positive")
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "iteration_pipeline_matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    estimates_csv, output_dir = Path(estimates_csv), Path(output_dir)
    data, audit = _load_estimates(estimates_csv)
    if exclude_out_of_range:
        keep = ~data["f_outside"]
        data = {key: values if key.startswith("_") else values[keep] for key, values in data.items()}
        for row in audit:
            if row["included_in_figures"] and row["f_outside"]:
                row["included_in_figures"] = False
                row["reason"] = "Final F outside [0,1]"
    n = len(data["z_hat"])
    if not n:
        raise ValueError("No finite selected estimates for Figure 5")
    output_dir.mkdir(parents=True, exist_ok=True)
    depths = data["_true_depths_all_cases"]
    truth = np.column_stack([data[f"true_f_{key}"] for key, _, _ in COMPONENTS])
    error = np.column_stack([data[f"f_{key}"] for key, _, _ in COMPONENTS])-truth
    depth_error = data["z_hat"]-data["true_z"]
    if np.any(truth < 0) or np.any(data["true_z"] < 0):
        raise ValueError("Reference depth and fractions must be nonnegative")
    # 실제 계수 0을 포함한 모든 성분의 오차를 먼저 계산합니다.
    squared_sums = np.column_stack((depth_error**2, np.sum(error**2, axis=1)))
    term_counts = np.column_stack((np.ones(n), np.full(n,3)))
    names = ("depth_rmse_m", "bottom_joint_rmse")
    scales = (100/depth_reference_range_m, 100/coefficient_reference_range)
    normalized_names = ("depth_normalized_rmse_pct", "bottom_joint_normalized_rmse_pct")
    rng = np.random.default_rng(20260914)
    resamples = 2000
    rows = []
    for depth in depths:
        mask = data["true_z"] == depth
        count = int(mask.sum())
        sums, counts = squared_sums[mask], term_counts[mask]
        totals = counts.sum(axis=0)
        point = np.sqrt(np.divide(sums.sum(axis=0), totals, out=np.full(2,np.nan), where=totals>0))
        low, high = np.full(2,np.nan), np.full(2,np.nan)
        if count > 1:
            samples = []
            for start in range(0, resamples, 200):
                indices = rng.integers(0,count,size=(min(200,resamples-start),count))
                ss, nn = sums[indices].sum(axis=1), counts[indices].sum(axis=1)
                samples.append(np.sqrt(np.divide(ss,nn,out=np.full_like(ss,np.nan),where=nn>0)))
            boot = np.concatenate(samples)
            for j in range(2):
                valid = boot[:,j][np.isfinite(boot[:,j])]
                if valid.size:
                    low[j],high[j] = np.quantile(valid,[0.025,0.975])
        row = dict(true_z_m=float(depth), n_cases=count, n_coefficient_terms=int(totals[1]),
                   n_zero_true_coefficient_terms_included=int((truth[mask]==0).sum()),
                   depth_reference_range_m=depth_reference_range_m,
                   coefficient_reference_range=coefficient_reference_range)
        for j,name in enumerate(names):
            row[name]=float(point[j])
            row[name+"_ci95_low"]=float(low[j])
            row[name+"_ci95_high"]=float(high[j])
        for j,name in enumerate(normalized_names):
            row[name]=float(point[j]*scales[j])
            row[name+"_ci95_low"]=float(low[j]*scales[j])
            row[name+"_ci95_high"]=float(high[j]*scales[j])
        rows.append(row)

    paths = []
    style = {"font.family":"DejaVu Sans", "font.size":11, "axes.titlesize":13,
             "axes.labelsize":12, "legend.fontsize":10, "pdf.fonttype":42, "ps.fonttype":42}
    scope = f"Final F in [0,1]; {n:,} cases" if exclude_out_of_range else f"All finite estimates; {n:,} cases"

    def draw(ax, name, label, color, marker):
        point = np.array([row[name] for row in rows])
        low = np.array([row[name+"_ci95_low"] for row in rows])
        high = np.array([row[name+"_ci95_high"] for row in rows])
        ax.plot(depths,point,color=color,marker=marker,markersize=6,linewidth=1.8,alpha=0.65,label=label)
        valid = np.isfinite(low)&np.isfinite(high)
        ax.vlines(depths[valid],low[valid],high[valid],color=color,linewidth=1.2,alpha=0.65)
        for end in (low,high):
            ax.plot(depths[valid],end[valid],linestyle="none",marker="_",markersize=8,
                    markeredgewidth=1.2,color=color,alpha=0.65)
        ax.set_xlabel("True depth (m)")
        ax.set_xticks(depths)
        _style_axis(ax)

    with plt.rc_context(style):
        fig,ax = plt.subplots(figsize=(9.4,6),layout="constrained")
        draw(ax,"depth_normalized_rmse_pct","Depth", "#0072B2","o")
        draw(ax,"bottom_joint_normalized_rmse_pct","Bottom coefficients (pooled)","#CC79A7","s")
        # 두 선과 에러바를 모두 추가한 뒤 축을 정해 두 번째 선의 큰 값도 전부 표시합니다.
        ax.set_ylim(bottom=0)
        ax.set_title("Figure 5. Normalized RMSE of depth and bottom coefficients")
        ax.set_ylabel("Normalized RMSE (% of reference range)")
        ax.legend(loc="best")
        fig.supxlabel(scope+" | Error bars: 95% bootstrap intervals"+
                     f"\nFixed reference ranges: depth = {depth_reference_range_m:g} m; coefficient = {coefficient_reference_range:g}"+
                     "\nAll three bottom coefficients included, including true F = 0",fontsize=9,color="#555555")
        paths.extend(_save(fig,output_dir,"figure5_normalized_rmse"))
        plt.close(fig)
        # 일반 RMSE는 수심과 계수의 단위가 다르므로 두 패널로 나누어 표시합니다.
        fig,axes = plt.subplots(1,2,figsize=(12,5.5),layout="constrained")
        for ax,name,label,color,marker,ylabel in (
            (axes[0],"depth_rmse_m","Water depth","#0072B2","o","Depth RMSE (m)"),
            (axes[1],"bottom_joint_rmse","Bottom coefficients (pooled)","#CC79A7","s","Coefficient RMSE (unitless)")):
            draw(ax,name,label,color,marker)
            ax.set_ylim(bottom=0)
            ax.set_title(label)
            ax.set_ylabel(ylabel)
        fig.suptitle("Figure 5. RMSE of depth and bottom coefficients")
        fig.supxlabel(scope+" | Error bars: 95% bootstrap intervals\nTrue F = 0 included in ordinary RMSE",fontsize=9,color="#555555")
        paths.extend(_save(fig,output_dir,"figure5_rmse"))
        plt.close(fig)

    paths.append(_write_csv(output_dir/"figure5_statistics.csv",list(rows[0]),rows))
    paths.append(_write_csv(output_dir/"figure5_case_audit.csv",list(audit[0]),audit))
    component_audit = []
    for i,cid in enumerate(data["case_id"]):
        for j,(key,_,_) in enumerate(COMPONENTS):
            component_audit.append(dict(case_id=cid,component=key,true_z_m=float(data["true_z"][i]),
                true_fraction=float(truth[i,j]),estimated_fraction=float(data[f"f_{key}"][i]),
                included_in_rmse=True,included_in_normalized_rmse=True))
    paths.append(_write_csv(output_dir/"figure5_component_audit.csv",list(component_audit[0]),component_audit))
    overall = np.sqrt(squared_sums.sum(axis=0)/term_counts.sum(axis=0))
    metadata = dict(figure=5,source_estimates_csv=str(estimates_csv.resolve()),input_cases=len(audit),
        included_cases=n,excluded_cases=len(audit)-n,exclude_out_of_range=exclude_out_of_range,
        normalization="Fixed reference-range normalized RMSE, not pointwise relative RMSE",
        normalized_formula="100 * RMSE / reference_range",
        depth_reference_range_m=depth_reference_range_m,coefficient_reference_range=coefficient_reference_range,
        default_reference_basis="Full experiment true depth span 5.0-0.5=4.5 m; bottom coefficient span 1-0=1",
        denominator_is_fixed_across_depth_bins=True,curve_minmax_scaling=False,
        depth_rmse_formula="sqrt(sum_i((z_hat_i-true_z_i)**2)/n)",
        bottom_joint_rmse_formula="sqrt(sum_i,k((f_hat_i,k-f_true_i,k)**2)/(3*n))",
        coefficient_weighting="Equal weight per case and per sand/brown/green coefficient",
        zero_true_fraction="Included in both ordinary and normalized RMSE",reference_range_normalization=True,
        overall_metrics=dict(depth_rmse_m=float(overall[0]),bottom_joint_rmse=float(overall[1]),
            depth_normalized_rmse_pct=float(overall[0]*scales[0]),bottom_joint_normalized_rmse_pct=float(overall[1]*scales[1])),
        error_bar="2.5th and 97.5th percentiles of case-bootstrap RMSE divided by fixed reference range times 100",
        bootstrap_resamples=resamples,bootstrap_random_seed=20260914,
        bootstrap_unit="Whole case within each true-depth bin; depth and component rows resampled together",
        error_bar_is_individual_error_sd=False,source_modified=False,inversion_performed=False)
    path = output_dir/"figure5_metadata.json"
    path.write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
    paths.append(path)
    return paths
def make_figure2(estimates_csv: Path, output_dir: Path, *, exclude_out_of_range: bool = False) -> list[Path]:
    """Figure 2만 지정 폴더에 저장합니다. 수심·혼합비 추정을 다시 하지 않습니다.

    점과 선은 기존과 같은 수심별 MAE입니다. 전체·혼합 바닥 3종·순수 바닥 3종으로 나눕니다.
    세 패널을 묶은 그림과 각 패널의 개별 PNG·PDF를 함께 저장합니다.
    에러바는 각 그룹·실제 수심에서 절대오차 abs(z_hat-true_z)의 표본 표준편차입니다.
    즉, MAE ± 1 SD이며 표준오차나 신뢰구간이 아닙니다. ddof=1로 계산합니다.
    MAE-SD가 음수인 경우에도 에러바를 0에서 잘라내지 않습니다.
    이는 절대오차 관측값이 음수라는 뜻이 아니라 대칭 SD 범위의 아래쪽 값입니다.

    exclude_out_of_range=True이면 최근 Figure 1의 2차 버전처럼 범위 밖 F 사례를 제외합니다.
    원본 CSV와 다른 그림의 포함 기준은 바꾸지 않습니다. F=0, F=1은 포함합니다.
    표본이 1개이면 SD는 정의되지 않으므로 NaN으로 남기고 그 점의 에러바를 그리지 않습니다.
    """
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "iteration_pipeline_matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    estimates_csv, output_dir = Path(estimates_csv), Path(output_dir)
    data, audit = _load_estimates(estimates_csv)
    if exclude_out_of_range:
        keep = ~data["f_outside"]
        data = {key: values if key.startswith("_") else values[keep] for key, values in data.items()}
        for row in audit:
            if row["included_in_figures"] and row["f_outside"]:
                row["included_in_figures"] = False
                row["reason"] = "Final F outside [0,1]"
    n = len(data["z_hat"])
    if not n:
        raise ValueError("No complete finite selected estimates for Figure 2")
    output_dir.mkdir(parents=True, exist_ok=True)
    depths = data["_true_depths_all_cases"]
    error = data["z_hat"] - data["true_z"]
    rows = []
    for group in ["all"] + [item[0] for item in GROUPS]:
        group_mask = np.ones(n, dtype=bool) if group == "all" else data["true_group"] == group
        for depth in depths:
            values = error[group_mask & (data["true_z"] == depth)]
            stats = _metrics(values)
            sd = float(np.std(np.abs(values), ddof=1)) if values.size > 1 else float("nan")
            rows.append(dict(true_group=group, true_z_m=float(depth), **stats,
                             std_absolute_error=sd, mae_minus_sd=stats["mae"]-sd,
                             mae_plus_sd=stats["mae"]+sd))

    style = {"font.family": "DejaVu Sans", "font.size": 10,
             "axes.titlesize": 12, "axes.labelsize": 10,
             "legend.fontsize": 8, "figure.titlesize": 14,
             "pdf.fonttype": 42, "ps.fonttype": 42}
    note = f"One selected estimate per case; n = {n:,}; out-of-range cases = {int(data['f_outside'].sum()):,}"
    if exclude_out_of_range:
        note += f"; F outside [0,1] excluded = {sum(row['reason'] == 'Final F outside [0,1]' for row in audit):,}"
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.3), layout="constrained")
        all_rows = [row for row in rows if row["true_group"] == "all"]
        axes[0].errorbar(depths, [row["mae"] for row in all_rows],
                         yerr=[row["std_absolute_error"] for row in all_rows],
                         color="#0072B2", marker="o", linewidth=1.8,
                         elinewidth=1.0, capsize=3, capthick=1.0, alpha=0.6)
        axes[0].set_title("All cases (each case has equal weight)")
        markers = ["o", "s", "^", "D", "v", "P"]
        for (group, label, color), marker in zip(GROUPS, markers):
            group_rows = [row for row in rows if row["true_group"] == group]
            # 혼합 바닥 세 선과 순수 바닥 세 선을 별도 그래프에 배치합니다.
            group_axis = axes[2] if group in {"sand", "brown", "green"} else axes[1]
            group_axis.errorbar(depths, [row["mae"] for row in group_rows],
                             yerr=[row["std_absolute_error"] for row in group_rows],
                             color=color, marker=marker, linewidth=1.5, markersize=5,
                             elinewidth=0.8, capsize=3, capthick=0.8, alpha=0.6, label=label)
        axes[1].set_title("Mixed bottoms (3 groups)")
        axes[2].set_title("Pure bottoms (3 groups)")
        axes[1].legend(loc="best")
        axes[2].legend(loc="best")
        for ax in axes:
            ax.set_xlabel("True depth (m)")
            ax.set_ylabel("Mean absolute depth error (m)")
            ax.set_xticks(depths)
            # SD의 아래쪽 끝까지 표시합니다. 음수로 내려가는 에러바를 잘라내지 않습니다.
            ax.set_ylim(bottom=min(0.0, ax.get_ylim()[0]))
            _style_axis(ax)
        fig.suptitle("Figure 2. Depth error versus true depth\n" + note +
                     "\nError bars: +/- 1 SD of absolute depth errors", fontsize=12)
        paths = _save(fig, output_dir, "figure2_depth_mae")
        # 개별 그래프도 바로 사용할 수 있도록 제목·축·범례까지 포함해 따로 저장합니다.
        # 동일한 그래프 영역을 저장하므로 개별 그림에서도 데이터·에러바는 같습니다.
        # 개별 저장 시 이웃 패널이나 공통 제목의 일부가 같이 잘리지 않도록 숨깁니다.
        fig.set_layout_engine(None)
        fig._suptitle.set_visible(False)
        for ax, stem in zip(axes, ("figure2a_all", "figure2b_mixed", "figure2c_pure")):
            for other in axes:
                other.set_visible(other is ax)
            ax.set_title(ax.get_title() + "\nError bars: +/- 1 SD", fontsize=12)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            bounds = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted()).expanded(1.06, 1.08)
            for extension in ("png", "pdf"):
                path = output_dir / f"{stem}.{extension}"
                fig.savefig(path, dpi=180, facecolor="white", bbox_inches=bounds)
                paths.append(path)
        plt.close(fig)
    paths.append(_write_csv(output_dir / "figure2_depth_mae_counts.csv", list(rows[0]), rows))
    paths.append(_write_csv(output_dir / "figure2_case_audit.csv", list(audit[0]), audit))
    metadata = dict(source_estimates_csv=str(estimates_csv.resolve()), input_cases=len(audit),
                    included_cases=n, excluded_cases=len(audit)-n,
                    exclude_out_of_range=exclude_out_of_range,
                    point_statistic="mean(abs(z_hat - true_z))",
                    error_bar="plus/minus 1 sample SD of absolute depth errors", sd_ddof=1,
                    error_bar_is_standard_error=False, error_bar_is_confidence_interval=False,
                    negative_lower_bars_clipped=False,
                    grouping="true depth and true bottom group; equal weight per included case",
                    panels={"all": ["all"], "mixed": ["sand_brown", "brown_green", "sand_green"],
                            "pure": ["green", "brown", "sand"]},
                    source_modified=False, inversion_performed=False)
    meta_path = output_dir / "figure2_metadata.json"
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    paths.append(meta_path)
    return paths


def make_figure3(estimates_csv: Path, output_dir: Path, *, exclude_out_of_range: bool = False) -> list[Path]:
    """Figure 3을 전체·혼합 바닥·순수 바닥의 세 그림으로 저장합니다.

    그룹은 실제 바닥 종류로 정합니다. 추정 바닥이 틀린 사례도 포함 조건을 만족하면
    원래 실제 그룹에 남깁니다. 기본적으로 최종 F 범위 밖 사례도 전부 포함합니다.
    exclude_out_of_range=True는 범위 안 사례만 보는 보조 분석에만 사용합니다.
    전체는 모든 사례를 모은 한 선, 혼합은 모래–갈조·모래–녹조·갈조–녹조 세 선,
    순수는 모래·갈조·녹조 세 선입니다. 각 선은 그룹의 바닥 합동 상대 RMSE입니다.

    상대 RMSE(%)=100*sqrt(유효한 상대 제곱오차의 합 / 유효한 성분 항의 수).
    상대오차=(추정 계수-실제 계수)/실제 계수이며 실제 계수>0인 항만 유효합니다.
    혼합 사례는 두 성분, 순수 사례는 한 성분을 포함합니다. 유효 항마다 같은 가중치이며
    전체 값은 그룹별 RMSE나 세 성분별 RMSE를 단순 평균한 값이 아닙니다.
    실제 계수 0에 대한 오차까지 포함한 일반 합동 RMSE도 별도 하위 폴더에 저장합니다.
    분모 하한, 정답 기반 후보 교체, 계수 보정은 하지 않습니다.

    에러바는 각 실제 바닥 그룹·수심 내 사례를 2,000번 복원추출한 95% bootstrap 구간입니다.
    같은 사례의 세 계수를 함께 추출합니다. n<2이면 점은 유지하고 에러바만 생략합니다.
    추정 결과와 다른 Figure는 바꾸지 않습니다.
    """
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "iteration_pipeline_matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    estimates_csv, output_dir = Path(estimates_csv), Path(output_dir)
    data, audit = _load_estimates(estimates_csv)
    if exclude_out_of_range:
        keep = ~data["f_outside"]
        data = {key: values if key.startswith("_") else values[keep] for key, values in data.items()}
        for row in audit:
            if row["included_in_figures"] and row["f_outside"]:
                row["included_in_figures"] = False
                row["reason"] = "Final F outside [0,1]"
    n = len(data["z_hat"])
    if not n:
        raise ValueError("No finite selected estimates for Figure 3")
    output_dir.mkdir(parents=True, exist_ok=True)
    ordinary_dir = output_dir / "일반_RMSE"
    ordinary_dir.mkdir(exist_ok=True)
    depths = data["_true_depths_all_cases"]
    truth = np.column_stack([data[f"true_f_{key}"] for key, _, _ in COMPONENTS])
    estimate = np.column_stack([data[f"f_{key}"] for key, _, _ in COMPONENTS])
    if np.any(truth < 0):
        raise ValueError("Reference bottom fractions must be nonnegative")
    valid = truth > 0
    errors = estimate - truth
    relative = np.divide(errors, truth, out=np.zeros_like(errors), where=valid)
    # 무효 성분은 제곱오차 합과 평균 분모 모두에서 제외합니다. 오차 0으로 센 것이 아닙니다.
    squared_sums = np.column_stack((np.sum(errors**2, axis=1), np.sum((100*relative)**2, axis=1)))
    term_counts = np.column_stack((np.full(n, 3), valid.sum(axis=1)))
    specs = (
        ("all", "All bottom types", "#0072B2", "o"),
        ("sand_brown", "Sand : brown algae", "#97613C", "o"),
        ("sand_green", "Sand : green algae", "#56B4E9", "s"),
        ("brown_green", "Brown algae : green algae", "#CC79A7", "^"),
        ("sand", "Sand", "#D59B24", "o"),
        ("brown", "Brown algae", "#D55E00", "s"),
        ("green", "Green algae", "#009E73", "^"),
    )
    rng = np.random.default_rng(20260914)
    resamples = 2000
    rows = []
    for group, _, _, _ in specs:
        group_mask = np.ones(n, dtype=bool) if group == "all" else data["true_group"] == group
        for depth in depths:
            mask = group_mask & (data["true_z"] == depth)
            count = int(mask.sum())
            sums, counts = squared_sums[mask], term_counts[mask]
            totals = counts.sum(axis=0)
            point = np.sqrt(np.divide(sums.sum(axis=0), totals, out=np.full(2,np.nan), where=totals>0))
            low, high = np.full(2,np.nan), np.full(2,np.nan)
            if count > 1:
                samples = []
                for start in range(0,resamples,200):
                    indices = rng.integers(0,count,size=(min(200,resamples-start),count))
                    ss, nn = sums[indices].sum(axis=1), counts[indices].sum(axis=1)
                    samples.append(np.sqrt(np.divide(ss,nn,out=np.full_like(ss,np.nan),where=nn>0)))
                boot = np.concatenate(samples)
                for j in range(2):
                    values = boot[:,j][np.isfinite(boot[:,j])]
                    if values.size:
                        low[j], high[j] = np.quantile(values,[0.025,0.975])
            row = dict(true_group=group,true_z_m=float(depth),n_cases=count,
                       n_coefficient_terms=int(totals[0]),n_relative_coefficient_terms=int(totals[1]),
                       n_zero_true_coefficient_terms=int((~valid[mask]).sum()))
            for j,name in enumerate(("rmse","relative_rmse_pct")):
                row[name]=float(point[j])
                row[name+"_ci95_low"]=float(low[j])
                row[name+"_ci95_high"]=float(high[j])
            rows.append(row)

    panels = (("a","All bottom types",specs[:1]), ("b","Mixed bottoms",specs[1:4]),
              ("c","Pure bottoms",specs[4:]))
    stems = {"a":"figure3a_all", "b":"figure3b_mixed", "c":"figure3c_pure"}
    style = {"font.family":"DejaVu Sans","font.size":11,"axes.titlesize":14,
             "axes.labelsize":12,"legend.fontsize":10,"pdf.fonttype":42,"ps.fonttype":42}
    paths = []
    with plt.rc_context(style):
        for is_relative in (True,False):
            metric = "relative_rmse_pct" if is_relative else "rmse"
            directory = output_dir if is_relative else ordinary_dir
            for letter,title,series in panels:
                fig,ax = plt.subplots(figsize=(9.2,6),layout="constrained")
                panel_count = 0
                single_case_notes = []
                for group,label,color,marker in series:
                    selected = [row for row in rows if row["true_group"]==group]
                    group_count = sum(row["n_cases"] for row in selected)
                    panel_count += group_count
                    point = np.array([row[metric] for row in selected])
                    low = np.array([row[metric+"_ci95_low"] for row in selected])
                    high = np.array([row[metric+"_ci95_high"] for row in selected])
                    ax.plot(depths,point,color=color,marker=marker,markersize=6,linewidth=1.8,
                            alpha=0.65,label=f"{label} (n = {group_count:,})")
                    finite = np.isfinite(low)&np.isfinite(high)
                    ax.vlines(depths[finite],low[finite],high[finite],color=color,linewidth=1.2,alpha=0.65)
                    for end in (low,high):
                        ax.plot(depths[finite],end[finite],linestyle="none",marker="_",markersize=8,
                                markeredgewidth=1.2,color=color,alpha=0.65)
                    single_case_notes += [f"{label} at {row['true_z_m']:g} m: n = 1, no interval"
                                          for row in selected if row["n_cases"]==1]
                ax.set_xlabel("True depth (m)")
                ax.set_ylabel("Root mean square relative error (%)" if is_relative else "Pooled coefficient RMSE (unitless)")
                ax.set_title(f"Figure 3({letter}). {title}")
                ax.set_xticks(depths)
                # 모든 선과 에러바를 추가한 뒤 축을 정해 큰 값이 잘리지 않게 합니다.
                ax.set_ylim(bottom=0)
                if any(row[metric] == 0 for row in rows if row["true_group"] in {item[0] for item in series}):
                    # 오차 0인 선이 아래 축선에 가려지지 않도록 표시 여백만 추가합니다.
                    ax.set_ylim(bottom=-0.025*max(ax.get_ylim()[1],1e-6))
                ax.legend(loc="best")
                _style_axis(ax)
                scope = "Final F in [0,1]" if exclude_out_of_range else "All finite estimates"
                note = f"{scope}; {panel_count:,} cases | Error bars: 95% bootstrap intervals"
                note += "\nPooled bottom error; true F > 0 terms only" if is_relative else "\nAll three coefficients included, including true F = 0"
                if single_case_notes:
                    note += "\n"+"; ".join(single_case_notes)
                fig.supxlabel(note,fontsize=9,color="#555555")
                paths.extend(_save(fig,directory,stems[letter]))
                plt.close(fig)
    paths.append(_write_csv(output_dir/"figure3_group_statistics.csv",list(rows[0]),rows))
    paths.append(_write_csv(output_dir/"figure3_case_audit.csv",list(audit[0]),audit))
    component_audit = []
    for i,cid in enumerate(data["case_id"]):
        for j,(key,_,_) in enumerate(COMPONENTS):
            component_audit.append(dict(case_id=cid,true_group=str(data["true_group"][i]),component=key,
                true_z_m=float(data["true_z"][i]),true_fraction=float(truth[i,j]),estimated_fraction=float(estimate[i,j]),
                included_in_rmse=True,included_in_relative_rmse=bool(valid[i,j]),
                relative_exclusion_reason="" if valid[i,j] else "True component fraction is zero"))
    paths.append(_write_csv(output_dir/"figure3_component_audit.csv",list(component_audit[0]),component_audit))
    metadata = dict(figure=3,source_estimates_csv=str(estimates_csv.resolve()),input_cases=len(audit),included_cases=n,
        excluded_cases=len(audit)-n,exclude_out_of_range=exclude_out_of_range,
        grouping="True bottom type and true depth; misclassified selected estimates remain in their true group",
        panels={"all":["all"],"mixed":["sand_brown","sand_green","brown_green"],"pure":["sand","brown","green"]},
        relative_formula="100 * sqrt(sum_valid_terms(((estimated_fraction-true_fraction)/true_fraction)**2) / number_of_valid_terms)",
        rmse_formula="sqrt(sum_all_terms((estimated_fraction-true_fraction)**2) / (3*n_cases))",
        relative_weighting="Equal weight per positive true coefficient term; whole-case bootstrap",
        zero_true_fraction="Excluded only from relative metric; retained in ordinary RMSE in 일반_RMSE subfolder",
        denominator_floor=None,error_bar="2.5th and 97.5th percentiles of case-bootstrap RMSE",bootstrap_resamples=resamples,
        bootstrap_random_seed=20260914,bootstrap_unit="Whole case within each true-depth and true-bottom-group subset",
        single_case_rule="Keep point; interval undefined and omitted",source_modified=False,inversion_performed=False)
    path = output_dir/"figure3_metadata.json"
    path.write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
    paths.append(path)
    return paths

def make_figures(estimates_csv: Path, output_dir: Path) -> list[Path]:
    """기존 Figure 1·2·3·5·6·7을 만들고 생성한 파일 경로들을 반환합니다.

    Figure 1: 수심 하나와 실제 바닥 구성별 F 여섯 개를 개별 이미지로 저장.
    Figure 2: 실제 수심별 수심 MAE와 ±1 SD 에러바. 전체·혼합 3종·순수 3종 패널.
    Figure 3: 전체·혼합 3종·순수 3종의 세 그림. 바닥 합동 상대 RMSE와 95% bootstrap 에러바.
    실제 계수 0까지 포함한 일반 RMSE 세 그림은 일반_RMSE 하위 폴더에 저장합니다.
    Figure 3은 최종 F 범위 밖 사례도 포함하며, 실제 계수 0은 상대 지표에서만 제외합니다.
    Figure 5: 수심·바닥 합동 NRMSE(기준 범위 4.5 m·1)와 일반 RMSE, 95% bootstrap 에러바.
    Figure 5는 최종 F 범위 밖 사례와 실제 계수 0을 모두 포함합니다.
    Figure 6: 수심의 부호 있는 오차와 각 바닥 계수의 부호 있는 오차 관계.
    Figure 7: 수심·세 바닥 계수 오차의 히스토그램.

    그룹은 추정된 바닥쌍이 아니라 원본 실험의 실제 바닥 구성으로 나눕니다.
    전체 값은 그룹별 지표 여섯 개의 단순 평균이 아닙니다.
    Figure 3의 바닥 합동 상대 RMSE는 실제 계수>0인 성분 항마다 같은 가중치를 줍니다.
    Figure 5의 NRMSE는 실제 계수 0을 포함한 세 성분과 각 사례에 같은 가중치를 줍니다.
    없는 그룹·수심 조합은 누락으로 남기고 그 위치를 오차 0으로 연결하지 않습니다.
    모든 그림의 기본 결과에 최종 F 범위 밖 사례를 포함하며 추정값을 자르지 않습니다.
    범위 밖 F 제외 결과는 전용 필터 옵션을 사용한 보조 분석으로만 만듭니다.
    """
    # Matplotlib 글꼴 캐시는 호출자가 지정한 위치 또는 운영체제 임시 폴더를 씁니다.
    # 이 설정은 그림 계산값이나 결과 저장 폴더를 바꾸지 않습니다.
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "iteration_pipeline_matplotlib")
    )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    estimates_csv, output_dir = Path(estimates_csv), Path(output_dir)
    data, audit = _load_estimates(estimates_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [_write_csv(
        output_dir / "figure_case_audit.csv",
        ["case_id", "status", "included_in_figures", "reason", "has_alternative", "f_outside"],
        audit,
    )]
    if not data["z_hat"].size:
        raise ValueError(f"No complete finite estimates to plot; see {paths[0]}")

    n = data["z_hat"].size
    excluded_n = len(audit) - n
    outside = data["f_outside"]
    depth_error = data["z_hat"] - data["true_z"]
    coefficient_errors = {
        key: data[f"f_{key}"] - data[f"true_f_{key}"] for key, _, _ in COMPONENTS
    }
    variables = [("depth", "Depth", data["true_z"], data["z_hat"], "m", "#0072B2")]
    variables += [
        (key, label, data[f"true_f_{key}"], data[f"f_{key}"], "", color)
        for key, label, color in COMPONENTS
    ]
    depths = data["_true_depths_all_cases"]
    note = (
        f"One selected estimate per case; n = {n:,}; "
        f"out-of-range cases = {int(outside.sum()):,}"
    )
    if excluded_n:
        note += f"\nIncomplete/nonfinite cases excluded = {excluded_n:,} (see audit)"
    style = {
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 12, "axes.labelsize": 10,
        "legend.fontsize": 8, "figure.titlesize": 14,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }
    with plt.rc_context(style):
        # Figure 1만 전용 함수로 생성합니다. 함수 내부 스타일은 밖으로 퍼지지 않습니다.
        paths.extend(make_figure1(estimates_csv, output_dir))

        # Figure 2만 전용 함수로 생성합니다. 다른 그림의 통계·포함 기준은 바꾸지 않습니다.
        paths.extend(make_figure2(estimates_csv, output_dir / "Figure2"))

        # Figure 3도 기본적으로 최종 F 범위 밖 사례를 포함합니다.
        paths.extend(make_figure3(estimates_csv, output_dir / "Figure3"))

        # Figure 6: 추정-실제의 부호를 보존해 수심·바닥 오차가 같이 움직이는지 봅니다.
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
        for ax, (key, label, color) in zip(axes, COMPONENTS):
            error = coefficient_errors[key]
            _scatter(ax, depth_error, error, outside, color)
            ax.axhline(0, color="#555555", linestyle="--", linewidth=0.8)
            ax.axvline(0, color="#555555", linestyle="--", linewidth=0.8)
            ax.set_xlim(_limits(depth_error, include_zero=True))
            ax.set_ylim(_limits(error, include_zero=True))
            ax.set_title(label)
            ax.set_xlabel("Depth error: estimated - true (m)")
            ax.set_ylabel("Coefficient error: estimated - true (unitless)")
            _style_axis(ax)
        axes[0].legend(loc="best")
        fig.suptitle("Figure 6. Signed depth and coefficient errors\n" + note, fontsize=12)
        paths.extend(_save(fig, output_dir, "figure6_error_relationships"))
        plt.close(fig)

        # Figure 7: 모든 유효한 오차 범위를 히스토그램에 포함합니다.
        fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.8), layout="constrained")
        for ax, (key, label, truth, estimate, unit, color) in zip(axes.flat, variables):
            error = estimate - truth
            if error.max() == error.min():
                bounds = _limits(error, include_zero=True)
                bins = np.linspace(*bounds, 16)
            else:
                # 같은 간격의 구간으로 전체 오차 범위를 포함합니다.
                # 큰 오차의 꼬리나 범위 밖 바닥 계수를 잘라내지 않습니다.
                n_bins = min(70, max(15, int(np.ceil(np.sqrt(error.size)))))
                bins = np.linspace(float(error.min()), float(error.max()), n_bins + 1)
            ax.hist(error, bins=bins, color=color, alpha=0.78, edgecolor="white", linewidth=0.35)
            ax.axvline(0, color="#333333", linestyle="--", linewidth=1)
            ax.set_xlim(_limits(error, include_zero=True))
            ax.set_title(label + "\n" + _format_stats(error, unit).replace("\n", "  |  "), fontsize=10.5)
            ax.set_xlabel("Estimated - true" + (" (m)" if unit else " (unitless)"))
            ax.set_ylabel("Number of cases")
            _style_axis(ax)
        fig.suptitle("Figure 7. Signed error distributions\n" + note, fontsize=12)
        paths.extend(_save(fig, output_dir, "figure7_error_distributions"))
        plt.close(fig)
    # Figure 5는 RMSE 전용 함수에서 그립니다. 다른 그림의 MAE 설정을 일괄 변경하지 않습니다.
    paths.extend(make_figure5(estimates_csv, output_dir / "Figure5"))
    return paths


def main() -> None:
    """터미널에서 입력 CSV와 저장 폴더를 받아 그림만 다시 생성합니다.

    예: python step07_figures.py runs/pilot/06_final_solution/estimates.csv runs/pilot/07_figures
    Figure 1만 수정할 때는 명령 뒤에 --figure1-only를 붙입니다.
    Figure 2만 만들 때는 --figure2-only를 붙이고 output_dir에 최종 저장 폴더를 지정합니다.
    Figure 2에서만 범위 밖 F를 제외하려면 --figure2-filter-f를 함께 붙입니다.
    Figure 3만 만들려면 --figure3-only를 붙이고 최종 Figure3 폴더를 지정합니다.
    기본적으로 범위 밖 F 사례를 포함합니다. 보조 제외 버전은 --figure3-filter-f를 붙입니다.
    예전 --figure3-include-out-of-range 옵션도 전체 포함으로 동일하게 작동합니다.
    Figure 5만 만들려면 --figure5-only를 붙이고 최종 Figure5 폴더를 지정합니다.
    Figure 5도 기본적으로 범위 밖 F 사례를 포함하며 --figure5-filter-f를 붙이면 보조 제외 버전입니다.
    예전 --figure5-include-out-of-range 옵션도 전체 포함으로 동일하게 작동합니다.
    이 명령은 QAA나 수심·혼합비 최적화를 다시 실행하지 않습니다.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("estimates_csv", type=Path, help="CSV with one selected estimate per case")
    parser.add_argument("output_dir", type=Path, help="Directory for PNG, PDF, and summary CSV files")
    only = parser.add_mutually_exclusive_group()
    only.add_argument("--figure1-only", action="store_true", help="Regenerate only the seven separate Figure 1 plots")
    only.add_argument("--figure2-only", action="store_true", help="Regenerate only Figure 2 with MAE +/- SD error bars")
    parser.add_argument("--figure2-filter-f", action="store_true", help="With --figure2-only, exclude final F outside [0,1]")
    only.add_argument("--figure3-only", action="store_true", help="Regenerate only Figure 3 RMSE plots with bootstrap intervals")
    figure3_scope = parser.add_mutually_exclusive_group()
    figure3_scope.add_argument("--figure3-include-out-of-range", action="store_true", help="Keep all final F values (the default)")
    figure3_scope.add_argument("--figure3-filter-f", action="store_true", help="Supplementary only: exclude final F outside [0,1]")
    only.add_argument("--figure5-only", action="store_true", help="Regenerate only Figure 5 fixed-range normalized RMSE with bootstrap intervals")
    figure5_scope = parser.add_mutually_exclusive_group()
    figure5_scope.add_argument("--figure5-include-out-of-range", action="store_true", help="Keep all final F values (the default)")
    figure5_scope.add_argument("--figure5-filter-f", action="store_true", help="Supplementary only: exclude final F outside [0,1]")
    args = parser.parse_args()
    if (args.figure5_include_out_of_range or args.figure5_filter_f) and not args.figure5_only:
        parser.error("Figure 5 inclusion options require --figure5-only")
    if args.figure5_only:
        for path in make_figure5(args.estimates_csv, args.output_dir, exclude_out_of_range=args.figure5_filter_f):
            print(path.resolve())
        return
    if (args.figure3_include_out_of_range or args.figure3_filter_f) and not args.figure3_only:
        parser.error("Figure 3 inclusion options require --figure3-only")
    if args.figure3_only:
        for path in make_figure3(args.estimates_csv, args.output_dir, exclude_out_of_range=args.figure3_filter_f):
            print(path.resolve())
        return
    if args.figure2_filter_f and not args.figure2_only:
        parser.error("--figure2-filter-f requires --figure2-only")
    if args.figure2_only:
        for path in make_figure2(args.estimates_csv, args.output_dir, exclude_out_of_range=args.figure2_filter_f):
            print(path.resolve())
        return
    plotter = make_figure1 if args.figure1_only else make_figures
    for path in plotter(args.estimates_csv, args.output_dir):
        print(path.resolve())


if __name__ == "__main__":
    main()
