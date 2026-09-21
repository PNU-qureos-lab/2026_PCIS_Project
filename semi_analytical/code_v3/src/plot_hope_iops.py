"""저장된 HOPE 계열 P·G·X 결과로 a·bb 산점도와 깊은 물 적합도를 그립니다.

역산을 다시 실행하지 않는 사후 평가 전용 모듈입니다. 실제 a·bb는 이 단계에서만
결합하며, 추정기나 대표해 선정으로 돌려보내지 않습니다. 동일한 깊은 물 IOP를
여러 얕은 물 사례가 재사용하므로, 점의 기본 단위는 4,860개 사례가 아니라
27개 깊은 물 조건입니다. 태양각이 다른 조건은 같은 실제 수질 IOP를 공유합니다.

실행: python plot_hope_iops.py RESULTS --baseline BASELINE_RESULTS
그림은 RESULTS/07_figures/hope_iops에 PNG(300 dpi)·벡터 PDF로 저장합니다.
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
from matplotlib.ticker import MaxNLocator

BANDS = (443, 490, 560, 665)
STYLE = {
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
    "axes.labelsize": 10, "axes.titleweight": "bold", "axes.spines.top": False,
    "axes.spines.right": False, "pdf.fonttype": 42, "savefig.facecolor": "white",
}
COLORS = ("#2878B5", "#DA8635", "#38986D", "#9564AE", "#B65272")


def _metrics(x, y):
    """유한한 쌍만 집계하며, 오차의 부호는 항상 추정값−실제값으로 계산합니다."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if not len(x):
        return dict(n=0, rmse=np.nan, mae=np.nan, bias=np.nan, r2=np.nan)
    error = y - x
    sst = float(np.sum((x - x.mean()) ** 2))
    return dict(n=len(x), rmse=float(np.sqrt(np.mean(error ** 2))),
                mae=float(np.mean(np.abs(error))), bias=float(error.mean()),
                r2=1 - float(np.sum(error ** 2)) / sst if sst > 0 else np.nan)


def _read_truth(result_dir):
    """조건별 참값의 일관성을 검사한 뒤 water_key당 참값 한 행만 남깁니다.

    pilot의 얕은 물 목록은 일부 조건만 포함할 수 있어 전체 깊은 물 참값표를
    우선 사용합니다. 기존 결과와의 호환을 위해 reference_truth.csv도 지원합니다.
    같은 water_key 안의 실제 IOP가 다르면 평균으로 숨기지 않고 중단합니다.
    """
    folder = result_dir / "01_rrs_data"
    source = folder / "deep_reference_truth.csv"
    if not source.exists():
        source = folder / "reference_truth.csv"
    raw = pd.read_csv(source, float_precision="round_trip")
    fields = [f"{v}_true_{w}" for v in ("a", "bb") for w in BANDS]
    audit = []
    for key, frame in raw.groupby("water_key", sort=True):
        for field in fields:
            values = frame[field].to_numpy(float)
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite truth: {key}, {field}")
            spread = float(values.max() - values.min())
            if not np.allclose(values, values[0], rtol=1e-10, atol=1e-12):
                raise ValueError(f"Inconsistent truth within water_key: {key}, {field}")
            audit.append(dict(water_key=key, field=field, source_rows=len(values),
                              minimum=float(values.min()), maximum=float(values.max()), spread=spread))
    metadata = [c for c in ("solar_zenith", "chl", "tsm") if c in raw]
    truth = raw[["water_key", *metadata, *fields]].drop_duplicates("water_key").copy()
    if "solar_zenith" not in truth:
        truth["solar_zenith"] = truth.water_key.str.extract(r"SZ([0-9.]+)", expand=False).astype(float)
    if truth.solar_zenith.isna().any():
        raise ValueError("Solar zenith angle missing from deep-water truth")
    return truth, pd.DataFrame(audit), source


def _save(fig, directory, stem):
    """화면 확인용 PNG와 확대·논문 편집용 벡터 PDF를 함께 저장합니다."""
    paths = []
    for extension in ("png", "pdf"):
        path = directory / f"{stem}.{extension}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def _scatter(ax, frame, variable, wavelength, estimate_field, method, colors, limits=None):
    """한 파장·한 IOP 패널: 실제값 가로축, 추정값 세로축, 1:1선·동일 축척."""
    part = frame[frame.wavelength == wavelength]
    truth_field = f"{variable}_true"
    good = np.isfinite(part[truth_field]) & np.isfinite(part[estimate_field])
    valid = part.loc[good]
    x, y = valid[truth_field].to_numpy(), valid[estimate_field].to_numpy()
    stats = _metrics(x, y)
    for angle, color in colors.items():
        group = valid[valid.solar_zenith == angle]
        ax.scatter(group[truth_field], group[estimate_field], s=34, c=color,
                   label=f"SZA {angle:g}°", alpha=.78, edgecolors="white", linewidths=.45, zorder=3)
    if limits is None:
        hi = max(float(x.max()), float(y.max()), 1e-9) if len(x) else 1.
        lo = min(float(x.min()), float(y.min()), 0.) if len(x) else 0.
        limits = (lo - .05 * (hi - lo) if lo < 0 else 0., hi + .08 * (hi - lo))
    ax.plot(limits, limits, "--", color="#555555", linewidth=1, zorder=1)
    ax.set(xlim=limits, ylim=limits)
    ax.set_aspect("equal", adjustable="box")
    # bb의 작은 소수 눈금이 2×4 요약 그림에서 겹치지 않도록 눈금 수를 제한합니다.
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.grid(color="#DCE1E5", linewidth=.6)
    ax.set_axisbelow(True)
    symbol = "a" if variable == "a" else "b_b"
    ax.set_title(f"{wavelength} nm")
    ax.set_xlabel(f"True ${symbol}$ (m$^{{-1}}$)")
    ax.set_ylabel(f"{method} ${symbol}$ (m$^{{-1}}$)")
    ax.text(.045, .96,
            f"N = {stats['n']}/{len(part)}\n$R^2$ = {stats['r2']:.3g}\nRMSE = {stats['rmse']:.3g}"
            f"\nMAE = {stats['mae']:.3g}\nBias = {stats['bias']:+.3g}",
            transform=ax.transAxes, va="top", fontsize=8.5,
            bbox=dict(facecolor="white", edgecolor="#DCE1E5", alpha=.91, pad=3), zorder=4)
    return stats


def make_iop_figures(result_dir: Path, baseline_dir: Path | None = None) -> list[Path]:
    """a·bb 및 깊은 물 Rrs 그림, 조건별 점 자료·통계·검사 기록을 생성합니다.

    이 실험의 깊은 물 조건 수는 정상적으로 27개입니다. 예외적으로 일부 조건의
    피팅이 실패하면 그 조건을 표에서 유지하고, 그림에는 유한한 점만 표시합니다.
    N=유한 점/전체 조건을 함께 적으므로 성공한 조건만으로 분모를 바꾸지 않습니다.
    QAA 비교는 저장된 기준 결과를 읽기만 하며 QAA를 다시 실행하지 않습니다.
    """
    result_dir = Path(result_dir)
    output = result_dir / "07_figures" / "hope_iops"
    output.mkdir(parents=True, exist_ok=True)
    truth, audit, truth_source = _read_truth(result_dir)
    iops = pd.read_csv(result_dir / "02_hope_pgx" / "iops.csv", float_precision="round_trip")
    if iops.duplicated(["water_key", "wavelength"]).any():
        raise ValueError("Duplicate HOPE water_key/wavelength rows")
    if len(iops) and set(iops.wavelength) != set(BANDS):
        raise ValueError("Expected exactly the four experiment wavelengths")
    if not set(iops.water_key).issubset(set(truth.water_key)):
        raise ValueError("Estimated water_key absent from deep-water reference truth")
    if not (iops.groupby("water_key").size() == len(BANDS)).all():
        raise ValueError("Each saved successful deep-water condition must have four IOP rows")
    # 파장별 참값을 긴 표로 펼쳐, 반복된 얕은 물 사례 없이 깊은 물 조건마다 한 번 결합합니다.
    truth_rows = []
    for row in truth.to_dict("records"):
        for wavelength in BANDS:
            truth_rows.append(dict(water_key=row["water_key"], wavelength=wavelength,
                                   solar_zenith=row["solar_zenith"],
                                   a_true=row[f"a_true_{wavelength}"], bb_true=row[f"bb_true_{wavelength}"]))
    # 추정 파일에 이미 있는 태양각 정보는 참값표의 평가 메타데이터로 일관되게 정리합니다.
    # 실패한 조건이 iops.csv에서 생략되어도 참값표의 전체 27조건을 왼쪽에 유지합니다.
    # 따라서 해당 조건의 추정값은 NaN이 되고, 산점도 N의 전체 분모는 줄지 않습니다.
    points = pd.DataFrame(truth_rows).merge(
        iops.drop(columns=["solar_zenith"], errors="ignore"),
        on=["water_key", "wavelength"], how="left", validate="one_to_one")
    baseline_path = None
    if baseline_dir is not None:
        baseline_path = Path(baseline_dir) / "02_qaa" / "qaa_iops.csv"
        qaa = pd.read_csv(baseline_path, float_precision="round_trip")
        points = points.merge(qaa[["water_key", "wavelength", "a_qaa", "bb_qaa"]],
                              on=["water_key", "wavelength"], how="left", validate="one_to_one")
        if points[["a_qaa", "bb_qaa"]].isna().any().any():
            raise ValueError("Missing baseline QAA IOPs for experiment conditions")
    points = points.sort_values(["water_key", "wavelength"])
    paths = []
    metrics = []
    for variable in ("a", "bb"):
        for method, column in [("HOPE-PGX", f"{variable}_hat"),
                                *([("QAA", f"{variable}_qaa")] if baseline_path else [])]:
            for wavelength, group in points.groupby("wavelength"):
                metrics.append(dict(variable=variable, method=method, wavelength_nm=int(wavelength),
                                    conditions_total=len(group), unit="m^-1",
                                    **_metrics(group[f"{variable}_true"], group[column])))
    angles = sorted(truth.solar_zenith.unique())
    colors = {a: COLORS[i % len(COLORS)] for i, a in enumerate(angles)}
    condition_count = len(truth)
    truth_fields = [f"{v}_true_{w}" for v in ("a", "bb") for w in BANDS]
    unique_water_iops = len(truth[truth_fields].drop_duplicates())
    note = (f"{condition_count} deep-water spectra; {unique_water_iops} distinct true water IOP spectra; "
            "same IOP reused across shallow cases")
    with plt.rc_context(STYLE):
        for variable in ("a", "bb"):
            fig, axes = plt.subplots(2, 2, figsize=(10.7, 10.2), layout="constrained")
            for ax, wavelength in zip(axes.flat, BANDS):
                _scatter(ax, points, variable, wavelength, f"{variable}_hat", "HOPE-PGX", colors)
            axes[1, 1].legend(loc="lower right", fontsize=8)
            fig.suptitle(f"HOPE-PGX: {'absorption a' if variable == 'a' else 'backscattering bb'}\n{note}", fontsize=12)
            paths += _save(fig, output, f"hope_{variable}_scatter_4bands")
        fig, axes = plt.subplots(2, 4, figsize=(19, 9.6), layout="constrained")
        for row_index, variable in enumerate(("a", "bb")):
            for col_index, wavelength in enumerate(BANDS):
                _scatter(axes[row_index, col_index], points, variable, wavelength,
                         f"{variable}_hat", "HOPE-PGX", colors)
        axes[-1, -1].legend(loc="lower right", fontsize=8)
        fig.suptitle(f"HOPE-PGX IOP estimates vs truth\n{note}", fontsize=14)
        paths += _save(fig, output, "hope_a_bb_scatter_summary")
        if baseline_path:
            # 각 열의 두 방법에 동일한 실제값과 축 범위를 적용하여 개선 폭을 비교합니다.
            for variable in ("a", "bb"):
                fig, axes = plt.subplots(2, 4, figsize=(19, 9.6), layout="constrained")
                for j, wavelength in enumerate(BANDS):
                    part = points[points.wavelength == wavelength]
                    values = part[[f"{variable}_true", f"{variable}_hat", f"{variable}_qaa"]].to_numpy()
                    finite = values[np.isfinite(values)]
                    hi, lo = max(float(finite.max()), 1e-9), min(float(finite.min()), 0.)
                    limits = (lo - .05 * (hi - lo) if lo < 0 else 0., hi + .08 * (hi - lo))
                    _scatter(axes[0, j], points, variable, wavelength, f"{variable}_qaa", "QAA", colors, limits)
                    _scatter(axes[1, j], points, variable, wavelength, f"{variable}_hat", "HOPE-PGX", colors, limits)
                axes[-1, -1].legend(loc="lower right", fontsize=8)
                fig.suptitle(f"{variable}: saved QAA (top) and HOPE-PGX (bottom)\n{note}", fontsize=14)
                paths += _save(fig, output, f"qaa_vs_hope_{variable}_scatter")
        # 좋은 Rrs 적합도가 좋은 a·bb를 보장하지 않으므로 IOP 그림과 별개로 저장합니다.
        fig, axes = plt.subplots(2, 2, figsize=(10.7, 9.7), layout="constrained")
        rrs_metrics = []
        for ax, wavelength in zip(axes.flat, BANDS):
            part = points[points.wavelength == wavelength]
            x = part.Rrs_deep_observed.to_numpy(float)
            y = part.Rrs_deep_model.to_numpy(float)
            good = np.isfinite(x) & np.isfinite(y)
            stats = _metrics(x, y)
            rrs_metrics.append(dict(wavelength_nm=wavelength, conditions_total=len(part), unit="sr^-1", **stats))
            hi = max(float(np.max(x[good])), float(np.max(y[good])), 1e-9) * 1.08 if good.any() else 1.
            for angle, color in colors.items():
                mask = good & (part.solar_zenith.to_numpy() == angle)
                ax.scatter(x[mask], y[mask], c=color, s=35, alpha=.8, label=f"SZA {angle:g}°")
            ax.plot([0, hi], [0, hi], "--", color="#555555", lw=1)
            ax.set(xlim=(0, hi), ylim=(0, hi), title=f"{wavelength} nm",
                   xlabel="Observed deep-water Rrs (sr$^{-1}$)", ylabel="Model deep-water Rrs (sr$^{-1}$)")
            ax.set_aspect("equal", adjustable="box")
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.grid(color="#DCE1E5", linewidth=.6)
            ax.text(.05, .95, f"N = {stats['n']}/{len(part)}\nRMSE = {stats['rmse']:.3g} sr$^{{-1}}$",
                    transform=ax.transAxes, va="top", fontsize=9)
        axes[-1, -1].legend(loc="lower right", fontsize=8)
        fig.suptitle("Deep-water spectral fit (diagnostic; not IOP accuracy)", fontsize=13)
        paths += _save(fig, output, "hope_deep_rrs_scatter")
    for frame, name in ((points, "iop_scatter_points.csv"), (audit, "truth_consistency_audit.csv"),
                        (pd.DataFrame(metrics), "iop_metrics.csv"),
                        (pd.DataFrame(rrs_metrics), "deep_rrs_metrics.csv")):
        path = output / name
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        paths.append(path)
    metadata = dict(deep_water_conditions=condition_count, distinct_true_iop_spectra=unique_water_iops,
                    wavelength_nm=list(BANDS), truth_source=str(truth_source.resolve()),
                    baseline_source=str(baseline_path.resolve()) if baseline_path else None,
                    point_unit="one deep-water condition at one wavelength",
                    shallow_cases_duplicated=False, truth_used_for_fitting=False,
                    bias="mean(estimate-truth)", r2="1-SSE/SST; NaN if SST is zero",
                    nonfinite_estimates="retained in CSV and denominator; omitted from numerical metrics and plot")
    path = output / "metadata.json"
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    paths.append(path)
    path = output / "설명.md"
    path.write_text(f"""# HOPE P·G·X의 a·bb 산점도

- 실제 a·bb를 가로축, HOPE 계열 P·G·X 추정 a·bb를 세로축으로 표시했습니다. 단위는 m⁻¹입니다.
- 각 패널은 443·490·560·665nm 중 한 파장입니다. 대각선은 1:1선이고 두 축의 축척이 같습니다.
- 점의 단위는 깊은 물 조건 {condition_count}개입니다. 서로 다른 실제 수체 IOP 스펙트럼은 {unique_water_iops}개이며, 태양각별 조건은 색으로 구분했습니다.
- 동일 a·bb를 얕은 물 수심·바닥 사례에 재사용하므로 4,860개의 독립 IOP 추정으로 세지 않습니다.
- N은 유한한 추정 수/전체 조건 수입니다. 실패 조건은 CSV와 분모에 남고 수치 통계·산점도에서만 빠집니다.
- RMSE=√평균((추정−실제)²), MAE=평균(|추정−실제|), Bias=평균(추정−실제), R²=1−SSE/SST입니다.
- QAA 비교 그림은 기존 저장 결과를 읽습니다. 위 행 QAA와 아래 행 HOPE는 같은 파장에서 같은 축 범위를 사용합니다.
- 깊은 물 Rrs 적합도 그림은 반사도 재현 정도만 표시합니다. 좋은 반사도 적합도가 정확한 IOP를 보장하지 않습니다.
- truth_consistency_audit.csv에서 같은 water_key 안의 실제 IOP가 일정한지 검사한 결과를 확인할 수 있습니다.
- 모든 그림은 PNG 300 dpi와 벡터 PDF 두 형식으로 저장했습니다.
""", encoding="utf-8")
    paths.append(path)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    print(f"Saved {len(make_iop_figures(args.result_dir, args.baseline))} IOP figure/audit files")
