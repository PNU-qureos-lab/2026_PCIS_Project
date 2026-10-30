# Planet Dove Standard Hedley correction by code

## Real entry point

`SOURCE/02_RUN_WORKFLOW/process_planetscope_standard_hedley.py`

## Required input

One Planet Dove archive containing the analytic surface-reflectance product and matching UDM2 files.

## Run

```bat
python SOURCE\02_RUN_WORKFLOW\process_planetscope_standard_hedley.py --input-zip "04_DATA\...\planet.zip" --output-dir "05_RESULTS\01_INTERMEDIATE\02_PLANET_DOVE_3M\COUNTRY\ISLAND\DATE"
```

Optional setting: `--distance-m 300`.

## Output

Training-water mask/ROI evidence, visible-band Hedley parameters, slope/R-squared tables, corrected reflectance/Rrs products, masks, previews and QC reports.
