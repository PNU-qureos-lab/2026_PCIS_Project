# Full Glint-ACOLITE source code

This directory contains the complete production and validation code behind the
small one-scene example. The files are grouped by scientific stage, not by the
date on which they were created.

The portable demonstration is `../../02_RUN_WORKFLOW/run_complete_example.py`. The scripts
here preserve the full research implementation, including the original input
and output paths, ACOLITE calls, HydroLight execution interfaces, validation
gates, quality checks, B01 processing, and final exports.

## Processing order

### 00_grid

`prepare_20m_grid.py` and `prepare_l2a_b09_from_safe.py`

- Input: every Sentinel-2 Sen2Cor L2A BOA band (B01–B09, B8A, B11, B12).
- Operation: aligns B01–B12 to the common B11 20 m grid.
- Output: one aligned BOA GeoTIFF per band.
- Important: resampling B01 from 60 m to 20 m does not create new spatial
  information.
- B09 is also aligned from 60 m for input completeness but is not corrected as
  a water target. B10 is unavailable as an L2A BOA band.

### 01_acolite_cycle0

`l2a_glint_acolite.py` is the complete L2A Glint-ACOLITE implementation.
`acolite_helpers.py` contains the ACOLITE LUT, geometry, masking, and archive
helpers used by the production experiments.

- Input: aligned L2A BOA, masks, geometry, AOT, B11 and B12.
- Operation: forms the two ACOLITE-transferred SWIR glint candidates, retains
  the conservative per-pixel candidate, and subtracts it from each band.
- Output: Cycle-0 BOA, B11/B12 selection, `rhog_ref`, transfer factors, masks,
  metadata, and QC.

### 02_hydrolight

- `hydrolight_forward.py`: prepares and runs forward HydroLight cases.
- `hydrolight_inversion.py`: parses/model-matches HydroLight spectra.
- `hydrolight_library.py`: interpolates the scene-specific water library.
- `controlled_hydrolight_validation.py`: known-glint closure and recovery test.
- `joint_multiband_hydrolight.py`: joins B02–B8A ACOLITE products with the
  HydroLight evidence.

The worked example includes all 675 resulting simulations in
`../../01_INPUT/03_hydrolight_library/model_library_B02_B8A.csv`. The proprietary
HydroLight executable is not redistributed.

### 03_residual_validation

`residual_cycle1_validation.py`

- Input: Cycle-0 products, all 675 HydroLight cases, masks, and 30 deep-water
  ROIs.
- Operation: applies known-answer, spatial, spectral, and safety gates.
- Output: the Cycle-1 decision and, only when all gates pass, a Cycle-1
  candidate. For `20260808_S2A`, Cycle 0 is retained.

### B01 in the active workflow

B01 is **not** a separate processing stage in the portable workflow. It enters
`00_grid` at native 60 m, is aligned to 20 m, and is corrected in the same
Cycle-0 loop as B02–B8A. Its ACOLITE transfer factor is supplied with the other
band factors.

### legacy_provenance/b01_extension

- `compute_b01_acolite_factors.py`: obtains the B01 ACOLITE transfer factors.
- `add_and_correct_b01.py`: resamples B01, applies the accepted correction,
  and records its 60 m native-resolution limitation.

These two scripts document the historical operation used when B01 was added to
an archive that had already been processed for B02–B8A. They are retained for
traceability and are **not another required B01 pass** for a new run.

### 05_conversion_export

- `boa_to_rrs_rrs.py`: converts final BOA to above-surface `Rrs` and then
  below-surface `rrs`.
- `export_four_bands.py`: packages B01–B04.
- `export_shallow_area.py`: exports valid pixels inside the supplied
  shallow-water boundary while excluding land, cloud, shadow, and invalid data.

## Equations used by the code

```text
Cycle0(λ) = original_BOA(λ) − selected_ACOLITE_glint(λ)
Rrs(λ) = final_BOA(λ) / π
rrs(λ) = Rrs(λ) / (0.52 + 1.7 × Rrs(λ))
```

## Running the code

The production scripts retain original experiment paths for provenance. Edit
their `ROOT`, `L2A`, `OUT`, ACOLITE, and HydroLight paths before using them on a
different computer. For a self-contained run that works directly with the
included scene, run `../../02_RUN_WORKFLOW/run_complete_example.py` instead.

Python dependencies are listed in `requirements-full.txt`. ACOLITE is loaded
from the workspace `acolite-source` directory. HydroLight remains an external
licensed program; its complete 675-row result library is included with this
worked example.
