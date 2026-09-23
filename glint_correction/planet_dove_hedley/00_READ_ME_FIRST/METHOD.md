# Method and interpretation

## Input

The source TIFFs are Planet `ortho_analytic_8b_sr` Surface Reflectance products. Planet's atmospheric correction has already been applied. ACOLITE atmospheric correction is not applied again.

## Published Hedley calculation

For each visible band, an ordinary least-squares slope against the 866 nm NIR band is fitted in the training-water mask. The correction is:

`corrected = original - slope * (NIR - minimum training-water NIR)`

The minimum is the actual finite minimum, not a percentile. The NIR difference is not truncated. The fitted slope is not scaled. Negative corrected water values are preserved and reported.

## Project-specific choices

The multi-date land mask, UDM2 screening, 300 m offshore distance and spectral deep-water screen are documented implementation choices used to select training pixels. They are not claimed as part of the original publication.

## Interpretation

Corrected surface reflectance is the primary result. `Rrs_Approx` and `rrs_Approx` are convenience transformations. HydroLight may be used separately for evaluation, but it is not part of the subtraction.

