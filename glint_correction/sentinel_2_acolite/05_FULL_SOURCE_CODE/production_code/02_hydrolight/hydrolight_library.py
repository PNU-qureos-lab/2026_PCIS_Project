"""Utilities for scene-specific V4 HydroLight library interpolation."""

from __future__ import annotations

# WORKFLOW ROLE — STEP 02C: SCENE-SPECIFIC HYDROLIGHT LIBRARY
# Interpolate the water-only simulation grid to the scene solar zenith angle
# and return spectra/parameters for the residual-glint validation.

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator


ROOT = Path(__file__).resolve().parents[5]  # glint_correction folder
V4 = ROOT / "outputs" / "Tidung_Glint_Removal_V4_SceneSpecific_HydroLight_20260911"
GRID = (
    V4
    / "07_HydroLight_Scene_Libraries"
    / "SZA_GRID"
    / "hydrolight_sza_water_grid_all_platforms.csv"
)
BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A"]
PARAMETERS = ["chl_mg_m3", "cdom440_m_1", "nap_g_m3"]


def scene_library(
    platform: str, sza_deg: float, *, dense: bool = True
) -> tuple[np.ndarray, pd.DataFrame]:
    """Return BOA-like pi*Rrs spectra at one scene's SZA.

    ``dense=False`` preserves the 5 x 5 x 8 HydroLight node grid and is used
    for controlled validation. ``dense=True`` interpolates that validated
    grid to 9 x 9 x 15 constituent combinations for the real-scene search.
    No additive scene bias is introduced here.
    """
    grid = pd.read_csv(GRID)
    subset = grid[grid.platform.eq(platform)].copy()
    if subset.empty:
        raise ValueError(f"No HydroLight grid records for {platform}")
    sza_nodes = np.sort(subset.sza_deg.unique())
    if sza_deg < sza_nodes.min() or sza_deg > sza_nodes.max():
        raise ValueError(
            f"Scene SZA {sza_deg:.4f} is outside HydroLight grid "
            f"{sza_nodes.min():.1f}-{sza_nodes.max():.1f}"
        )

    rows: list[dict] = []
    spectra: list[np.ndarray] = []
    for keys, group in subset.groupby(PARAMETERS, sort=True):
        ordered = group.sort_values("sza_deg")
        values = np.array(
            [np.interp(sza_deg, ordered.sza_deg, ordered[band]) for band in BANDS]
        )
        rows.append(dict(zip(PARAMETERS, keys)) | {"platform": platform, "sza_deg": sza_deg})
        spectra.append(np.pi * values)
    coarse_spectra = np.asarray(spectra)
    coarse_params = pd.DataFrame(rows)
    coarse_params["nap_tsm_g_m3"] = coarse_params["nap_g_m3"]
    if not dense:
        return coarse_spectra, coarse_params

    chl_nodes = np.sort(coarse_params.chl_mg_m3.unique())
    cdom_nodes = np.sort(coarse_params.cdom440_m_1.unique())
    nap_nodes = np.sort(coarse_params.nap_g_m3.unique())
    chl_dense = np.unique(np.r_[chl_nodes, (chl_nodes[:-1] + chl_nodes[1:]) / 2])
    cdom_dense = np.unique(np.r_[cdom_nodes, (cdom_nodes[:-1] + cdom_nodes[1:]) / 2])
    nap_dense = np.unique(np.r_[nap_nodes, (nap_nodes[:-1] + nap_nodes[1:]) / 2])
    dense_points = np.array(
        [(c, d, n) for c in chl_dense for d in cdom_dense for n in nap_dense],
        dtype=float,
    )
    cube = np.empty((len(chl_nodes), len(cdom_nodes), len(nap_nodes), len(BANDS)))
    lookup = {
        (float(row.chl_mg_m3), float(row.cdom440_m_1), float(row.nap_g_m3)): i
        for i, row in coarse_params.iterrows()
    }
    for ic, c in enumerate(chl_nodes):
        for id_, d in enumerate(cdom_nodes):
            for in_, n in enumerate(nap_nodes):
                cube[ic, id_, in_, :] = coarse_spectra[lookup[(c, d, n)]]
    dense_spectra = np.column_stack(
        [
            RegularGridInterpolator(
                (chl_nodes, cdom_nodes, nap_nodes), cube[..., ib],
                bounds_error=True,
            )(dense_points)
            for ib in range(len(BANDS))
        ]
    )
    dense_params = pd.DataFrame(dense_points, columns=PARAMETERS)
    dense_params["nap_tsm_g_m3"] = dense_params["nap_g_m3"]
    dense_params["platform"] = platform
    dense_params["sza_deg"] = sza_deg
    return dense_spectra, dense_params
