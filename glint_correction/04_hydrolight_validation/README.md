# HydroLight simulation for glint validation

This folder contains the HydroLight simulation component used by the glint-correction workflow.

## Role in the workflow

ACOLITE supplies the SWIR-based glint correction applied to the Sentinel-2 image. HydroLight supplies independent, water-only spectra used to evaluate whether a residual glint signal remains after that correction. HydroLight does not replace the ACOLITE correction and does not force the image to match a simulated colour.

The sequence is:

1. Apply the ACOLITE SWIR glint correction using B11 and B12.
2. Compare corrected deep-water spectra with the HydroLight water-only spectral library.
3. Test known-glint injection and recovery.
4. Retain an additional correction cycle only when the spectral and safety checks pass.

## Included library

[`hydrolight_lut_all_platforms.json`](hydrolight_lut_all_platforms.json) contains 675 Sentinel-2 band-response spectra:

- platforms: Sentinel-2A, Sentinel-2B and Sentinel-2C;
- solar zenith angles: 20°, 25°, 30°, 35° and 40°;
- three water optical backgrounds;
- 15 TSM/NAP values from 0.1 to 80 g m⁻³.

Each row stores the platform, solar zenith angle, optical background, TSM/NAP value and the simulated values convolved to the Sentinel-2 B02, B03 and B04 spectral responses.

Run the inventory check with Node.js:

```powershell
node inspect_hydrolight_lut.cjs
```

The HydroLight executable and its licensed installation files are required to regenerate the radiative-transfer simulations. They are not part of this repository. The included JSON is the resulting satellite-response lookup table used for the analysis.
