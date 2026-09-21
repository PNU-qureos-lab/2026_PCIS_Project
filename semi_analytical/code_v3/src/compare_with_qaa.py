"""저장된 QAA 및 HOPE-PGX의 수심·바닥 결과를 동일 사례에서 사후 비교합니다.

추정·최적화·후보 재선정은 하지 않습니다. HOPE 실행의 전체 사례 목록을 분모로
유지하고, 두 방법 모두 성공한 공통 사례에서 동일 N의 오차 지표를 계산합니다.
실패율·범위 밖 F 비율은 성공 사례만의 비율로 축소하지 않고 전체 분모로 저장합니다.
F가 범위 밖인 성공 사례도 주요 비교에 포함하므로 기존 실험의 평가와 연결됩니다.

실행: python compare_with_qaa.py RESULTS BASELINE_RESULTS
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "hope_matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

QUANTITIES = {"depth": ("true_z", "z_hat", "m"),
              "sand": ("true_f_sand", "f_sand", "fraction"),
              "brown": ("true_f_brown", "f_brown", "fraction"),
              "green": ("true_f_green", "f_green", "fraction")}
COLORS = {"QAA": "#8798A6", "HOPE-PGX": "#2878B5"}
METHOD_PREFIXES = {"QAA": "qaa", "HOPE-PGX": "hope"}


def _statistics(errors):
    """추정−실제 오차를 받아 집계하며, 빈 집합의 지표는 0이 아닌 NaN입니다."""
    errors = np.asarray(errors, float)
    if not np.isfinite(errors).all():
        raise ValueError("Metrics must use explicitly selected finite common-success rows")
    return dict(n=len(errors), RMSE=float(np.sqrt(np.mean(errors ** 2))) if len(errors) else np.nan,
                MAE=float(np.mean(np.abs(errors))) if len(errors) else np.nan,
                Bias=float(np.mean(errors)) if len(errors) else np.nan)


def _save(fig, output, stem):
    paths = []
    for extension in ("png", "pdf"):
        path = output / f"{stem}.{extension}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


def compare_results(result_dir: Path, baseline_dir: Path) -> list[Path]:
    """전체 대응표·공통 성공 집합의 지표·실패/범위 밖 집계와 그림을 저장합니다."""
    result_dir, baseline_dir = Path(result_dir), Path(baseline_dir)
    output = result_dir / "08_qaa_comparison"
    output.mkdir(parents=True, exist_ok=True)
    refs = pd.read_csv(result_dir / "01_rrs_data" / "reference_truth.csv", float_precision="round_trip")
    ref_fields = [c for c in ("case_id", "water_key", "bottom_code", "true_group", "true_z",
                             "solar_zenith", "chl", "tsm", "true_f_sand", "true_f_brown", "true_f_green") if c in refs]
    table = refs[ref_fields].copy()
    if table.case_id.duplicated().any():
        raise ValueError("Reference case_id is not unique")
    paths, counts = [], []
    sources = {"qaa": baseline_dir / "06_final_solution" / "estimates.csv",
               "hope": result_dir / "06_final_solution" / "estimates.csv"}
    true_columns = [item[0] for item in QUANTITIES.values()]
    for method, prefix in METHOD_PREFIXES.items():
        estimates = pd.read_csv(sources[prefix], float_precision="round_trip")
        if estimates.case_id.duplicated().any():
            raise ValueError(f"Duplicate case_id in {method}")
        # 같은 사례 ID가 실제로 같은 수심·바닥 정답을 가리키는지 검사합니다.
        check = table[["case_id", *true_columns]].merge(
            estimates[["case_id", *true_columns]], on="case_id", suffixes=("", "_saved"), validate="one_to_one")
        for field in true_columns:
            if not np.allclose(check[field], check[field + "_saved"], atol=1e-12, rtol=1e-10):
                raise ValueError(f"Reference mismatch in {method}: {field}")
        fields = [c for c in ("case_id", "z_hat", "F", "f_sand", "f_brown", "f_green", "c4",
                              "status", "f_outside", "has_alternative", "any_evaluation_limit",
                              "rms_relative_error_pct", "pair", "total_evaluations",
                              "solver_success") if c in estimates]
        selected = estimates[fields].copy()
        selected["row_present"] = True
        selected = selected.rename(columns={c: f"{prefix}_{c}" for c in selected if c != "case_id"})
        table = table.merge(selected, on="case_id", how="left", validate="one_to_one")
        finite_fields = [f"{prefix}_{c}" for c in ("c4", "z_hat", "f_sand", "f_brown", "f_green")]
        finite = np.isfinite(table[finite_fields].to_numpy(float)).all(axis=1)
        success = finite & table[f"{prefix}_status"].fillna("failed").ne("failed").to_numpy()
        # 14차는 유한한 미수렴 결과도 남깁니다. 비교 그림에서 이를 '성공'으로
        # 오표시하지 않도록 명시적 solver_success가 있으면 그 상태까지 확인합니다.
        # 원본 QAA에는 해당 열이 없으므로 원래의 종료 상태 정의를 유지합니다.
        table[f"{prefix}_finite_result"] = finite
        if f"{prefix}_solver_success" in table:
            success &= table[f"{prefix}_solver_success"].fillna(False).to_numpy(bool)
        elif prefix == 'hope':
            # 15차 H/F는 12차 격자법이므로 기울기 수렴 여부 열이 없습니다.
            # 여덟 이웃에서 개선이 없는 정상 종료와 단순 계산 한도 종료를 구분합니다.
            # 유한한 한도 종료 원값은 finite_result로 계속 보관합니다.
            success &= table[f"{prefix}_status"].eq('no_c4_improvement').to_numpy()
        table[f"{prefix}_success"] = success
        # F=0/1인 순수 바닥은 범위 안입니다. F를 자르거나 실패 결과로 바꾸지 않습니다.
        table[f"{prefix}_outside_recomputed"] = success & ((table[f"{prefix}_F"] < 0) | (table[f"{prefix}_F"] > 1))
        for quantity, (truth_col, estimate_col, _) in QUANTITIES.items():
            table[f"{prefix}_{quantity}_error"] = table[f"{prefix}_{estimate_col}"] - table[truth_col]
        table[f"{prefix}_bottom_3component_rmse"] = np.sqrt(np.mean(np.column_stack(
            [table[f"{prefix}_{m}_error"].to_numpy() ** 2 for m in ("sand", "brown", "green")]), axis=1))
        counts.append(dict(method=method, total_requested_cases=len(table), source_saved_cases=len(estimates),
                           missing_case_rows=int(table[f"{prefix}_row_present"].isna().sum()),
                           finite_result_cases=int(finite.sum()),
                           finite_nonconverged_cases=int((finite & ~success).sum()),
                           successful_cases=int(success.sum()), failed_cases=int((~success).sum()),
                           failure_pct_of_total=float(100 * (~success).sum() / len(table)),
                           fraction_outside_count=int(table[f"{prefix}_outside_recomputed"].sum()),
                           fraction_outside_pct_of_total=float(100 * table[f"{prefix}_outside_recomputed"].sum() / len(table))))
    table["common_success"] = table.qaa_success & table.hope_success
    paired = table.loc[table.common_success].copy()
    for quantity in QUANTITIES:
        table[f"{quantity}_absolute_error_change_hope_minus_qaa"] = (
            np.abs(table[f"hope_{quantity}_error"]) - np.abs(table[f"qaa_{quantity}_error"]))
    metrics = []
    groups = [("all", "all", paired)]
    for field in ("true_z", "true_group", "water_key", "solar_zenith", "chl", "tsm"):
        if field in paired:
            groups += [(field, str(value), frame) for value, frame in paired.groupby(field, sort=True)]
    for group_by, group_value, frame in groups:
        for quantity, (_, _, unit) in QUANTITIES.items():
            for method, prefix in METHOD_PREFIXES.items():
                metrics.append(dict(group_by=group_by, group_value=group_value, quantity=quantity,
                                    method=method, unit=unit, total_requested_cases=len(table),
                                    success_set="QAA and HOPE both successful; F outside included",
                                    **_statistics(frame[f"{prefix}_{quantity}_error"])))
    metric_table = pd.DataFrame(metrics)
    count_table = pd.DataFrame(counts)
    for frame, name in ((table, "case_comparison.csv"), (metric_table, "metrics.csv"),
                        (count_table, "failure_and_outside_counts.csv")):
        path = output / name
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        paths.append(path)
    summary = dict(total_requested_cases=len(table), common_success_cases=len(paired),
                   only_qaa_success=int((table.qaa_success & ~table.hope_success).sum()),
                   only_hope_success=int((~table.qaa_success & table.hope_success).sum()),
                   both_failed=int((~table.qaa_success & ~table.hope_success).sum()),
                   truth_used_for_reselection=False, outside_F_excluded=False,
                   success_rule="finite C4/H/sand/brown/green; HOPE grid requires no_c4_improvement, or explicit solver_success when available; QAA uses original status",
                   count_denominator="all requested reference cases, including failed cases",
                   sources={k: str(v.resolve()) for k, v in sources.items()}, counts=counts)
    path = output / "summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    paths.append(path)
    style = {"font.family": "DejaVu Sans", "font.size": 10, "axes.titleweight": "bold",
             "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42}
    with plt.rc_context(style):
        # 단위가 다른 수심(m)과 바닥 계수(무차원)는 한 세로축에 억지로 넣지 않습니다.
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), layout="constrained", gridspec_kw={"width_ratios": [1, 2.1]})
        all_metrics = metric_table[(metric_table.group_by == "all")]
        for ax, quantities, ylabel in ((axes[0], ["depth"], "RMSE (m)"),
                                         (axes[1], ["sand", "brown", "green"], "RMSE (fraction)")):
            for offset, method in zip((-.18, .18), METHOD_PREFIXES):
                values = [float(all_metrics[(all_metrics.method == method) & (all_metrics.quantity == q)].RMSE.iloc[0]) for q in quantities]
                bars = ax.bar(np.arange(len(quantities)) + offset, values, width=.34,
                              color=COLORS[method], label=method)
                ax.bar_label(bars, labels=[f"{v:.4g}" for v in values], padding=4, fontsize=9)
            ax.set_xticks(np.arange(len(quantities)), [q.capitalize() for q in quantities])
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", color="#DCE1E5", linewidth=.6)
            ax.set_axisbelow(True)
            ax.margins(y=.22)
        axes[0].set_title("Depth")
        axes[1].set_title("Bottom coefficients")
        axes[1].legend()
        fig.suptitle(f"Same {len(paired):,}/{len(table):,} successful cases | Out-of-range F included", fontsize=13)
        paths += _save(fig, output, "qaa_vs_hope_rmse")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.3), layout="constrained")
        for ax, field, xlabel in ((axes[0], "depth_error", "Absolute depth error (m)"),
                                  (axes[1], "bottom_3component_rmse", "Per-case bottom coefficient RMSE")):
            for method, prefix in METHOD_PREFIXES.items():
                values = np.sort(np.abs(paired[f"{prefix}_{field}"].to_numpy(float)))
                ax.step(values, np.arange(1, len(values) + 1) / len(values) if len(values) else [],
                        where="post", label=method, color=COLORS[method], linewidth=2)
            ax.set(xlabel=xlabel, ylabel="Cumulative share of common successful cases", ylim=(0, 1.015), xlim=(0, None))
            ax.grid(color="#DCE1E5", linewidth=.6)
            ax.legend(loc="lower right")
        fig.suptitle(f"Error distributions | {len(paired):,} common successful cases", fontsize=13)
        paths += _save(fig, output, "qaa_vs_hope_error_cdf")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.3), layout="constrained")
        for method, prefix in METHOD_PREFIXES.items():
            by_depth = []
            for depth, frame in paired.groupby("true_z", sort=True):
                by_depth.append((float(depth), float(np.sqrt(np.mean(frame[f"{prefix}_depth_error"] ** 2))),
                                 float(np.sqrt(np.mean(frame[f"{prefix}_bottom_3component_rmse"] ** 2)))))
            if by_depth:
                values = np.asarray(by_depth)
                for j, ax in enumerate(axes):
                    ax.plot(values[:, 0], values[:, j + 1], "o-", c=COLORS[method], label=method, lw=2)
        for ax, ylabel in zip(axes, ("Depth RMSE (m)", "Pooled bottom coefficient RMSE")):
            ax.set(xlabel="True depth (m)", ylabel=ylabel)
            ax.grid(color="#DCE1E5", linewidth=.6)
            ax.legend()
        fig.suptitle("Errors by true depth | Same cases for both methods at each depth", fontsize=13)
        paths += _save(fig, output, "qaa_vs_hope_by_depth")
    path = output / "설명.md"
    path.write_text(f"""# 기존 QAA와 HOPE 계열 P·G·X 결과 비교

## 비교 집합

- 요청된 전체 사례는 {len(table):,}개이고 두 방법 모두 성공한 공통 사례는 {len(paired):,}개입니다.
- 성공은 C4·수심·모래/갈조/녹조 계수가 모두 유한하고 status가 failed가 아닌 경우입니다.
- metrics.csv의 두 방법은 언제나 같은 공통 성공 사례에서 계산합니다. 방법마다 성공 사례를 따로 골라 N이 달라지는 비교를 하지 않습니다.
- 실패 수·F 범위 밖 수와 백분율의 분모는 전체 요청 사례 {len(table):,}개입니다.
- F<0 또는 F>1인 성공 결과도 주요 지표·그림에 포함합니다. F=0, F=1은 정상 범위입니다.
- case_comparison.csv는 성공 여부와 관계없이 전체 사례를 유지합니다. 저장 행 누락도 실패 집계에 포함합니다.

## 통계와 그림

- 오차=추정−실제, RMSE=√평균(오차²), MAE=평균(|오차|), Bias=평균(오차)입니다.
- 수심 오차는 m, 바닥 계수 오차는 무차원 혼합계수 단위입니다. 계수 0.1은 10%p에 해당합니다.
- qaa_vs_hope_rmse는 전체 공통 성공 사례의 수심과 세 바닥 계수 RMSE를 비교합니다.
- qaa_vs_hope_error_cdf는 사례별 수심 절대오차와 세 바닥 계수 RMSE의 누적분포입니다.
- qaa_vs_hope_by_depth는 실제 수심별 수심 RMSE와 세 바닥 성분을 모은 RMSE입니다.
- metrics.csv에는 전체·수심·바닥 그룹·수체 조건·태양각·chl·tsm별 같은 사례 집합의 지표가 있습니다.
- 참값은 저장된 대표해를 평가하는 데만 사용합니다. 추정, 최적화, 대표 후보 재선정은 하지 않습니다.
- PNG는 300 dpi이고 PDF는 벡터 형식입니다.
""", encoding="utf-8")
    paths.append(path)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("baseline_dir", type=Path)
    args = parser.parse_args()
    print(f"Saved {len(compare_results(args.result_dir, args.baseline_dir))} comparison files")
