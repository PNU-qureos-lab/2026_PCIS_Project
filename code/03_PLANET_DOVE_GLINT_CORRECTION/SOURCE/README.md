# Planet Dove Standard Hedley Source Code

## Runnable entry point

`02_RUN_WORKFLOW/process_planetscope_standard_hedley.py`

The workflow accepts a Planet Dove archive containing matching analytic
surface-reflectance and UDM2 products. It creates a new result package with
the input audit, training-water ROI, regression evidence, corrected
reflectance, approximate Rrs/rrs, masks, intermediate rasters, quality-control
figures, provenance, and an HTML result report.

## Command

```bat
python 02_RUN_WORKFLOW\process_planetscope_standard_hedley.py --input-zip "path\to\Planet_Dove_SR_UDM2.zip" --output-dir "path\to\result_folder"
```

Use the package-level `RUN_CODE_WORKFLOW.bat` or the Planet Dove GUI when a
guided launch is preferred. Each new result package automatically receives a
reproducible source-code copy and validation utility from the runnable files
in `02_RUN_WORKFLOW`.
