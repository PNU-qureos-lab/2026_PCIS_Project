# Planet four-band Sentinel-comparable result

This package processes one Planet scene (2025-11-17) using 444, 492, 566 and 666 nm. Planet approximate Rrs was averaged to the Sentinel 20 m grid, and only pixels inside the Sentinel 2026-08-08 candidate shallow-water mask were inverted.

## Folder contents

- `01_INPUT_20M`: the four Planet approximate-Rrs bands actually inverted.
- `02_INTERMEDIATE`: the common candidate shallow-water mask.
- `03_FINAL_PRODUCTS`: depth, fractions, habitat, cost and spectral-error rasters.
- `04_QC`: numerical summaries and the four-panel result figure.
- `05_CODE`: the complete scripts used to build the result.

## Integrity

No depths were clipped or smoothed in the GeoTIFFs. Values above 5 m are retained and shown as black in the figure. The product remains exploratory because the Planet crop does not overlap the reviewed Sentinel offshore deep-water ROIs, and Planet SR/pi is not dedicated aquatic atmospheric correction.

## Result

- Candidate pixels: 3,806
- Median depth: 2.05 m
- 5th–95th percentiles: 1.00–14.17 m
- Pixels >5 m: 43.48%
- Depth grid boundary: 0.841%
- Deep-water relative model mismatch: 40.1%

Decision: EXPLORATORY ONLY — Sentinel-comparable geometry and scale; Planet deep-water/atmospheric reference remains unvalidated

The code files retain the absolute source paths from the audited production run
as provenance. Change `SOURCE`, `SENTINEL_MASK`, and `OUTPUT` near the top of
`run_planetscope_four_band_sentinel_comparable.py` before rerunning it on a
different computer. Treat the committed map as an exploratory satellite-model result pending further reference review.

