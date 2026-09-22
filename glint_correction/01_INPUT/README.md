# Input provenance for the worked example

All files belong to scene `20260808_S2A` and the Tidung subset on the common
`EPSG:32748`, 20 m, 417 × 278 pixel grid.

## 01_l2a_boa_20m

Sentinel-2 L2A bottom-of-atmosphere reflectance prepared on the common grid.
Sen2Cor performed the atmospheric correction before this workflow. B01 was
resampled from its native 60 m grid, and B02/B03/B04 were resampled from their
native 10 m grids. Resampling aligns the rasters but does not create additional
spatial information.

The folder contains every BOA band supplied by L2A: B01–B09, B8A, B11 and B12.
B10 is absent because Sen2Cor does not provide the cirrus band as an L2A BOA
surface-reflectance raster. B09 is aligned from 60 m for completeness but is
not a glint-correction target because of strong water-vapour absorption.

## 02_acolite_support

Real Glint-ACOLITE support products used by Cycle 0:

- `SWIR_selection_B11_is_1.tif`: per-pixel selection between B11 and B12;
- `rhog_ref.tif`: accepted reference glint magnitude;
- `multiband_glint_factors.csv`: factors that transfer the reference glint to
  each optical band.

## 03_hydrolight_library

`model_library_B02_B8A.csv` contains all 675 HydroLight water-only simulated
spectra used by the deep-water residual-glint validation. It is the complete
simulation lookup table used by this example, not a reduced illustration.

## 04_validation_inputs

The selected 30 deep-water ROIs, fitted residuals, acceptance gate, final QC,
shallow-water ROI samples, and the corresponding source figures used to audit
the final decision.
