# Semi-Analytical Processing (Version 42) — Code Mode

The runnable source is in `SOURCE`. Its processing order is configuration → raster input → deep-water Rrs → valid-pixel selection → IOP/model engine → report/output. Required coefficients, bottom spectra, IOP spectra, and examples are in the adjacent `DATA` folder.

Use corrected Rrs and masks from `05_RESULTS/01_INTERMEDIATE`. Write all semi-analytical products to `05_RESULTS/02_SEMI_ANALYTICAL` and presentation maps to `05_RESULTS/03_MAPS`.

From the package root, run:

```bat
.venv\Scripts\python.exe 03_CODE\04_SEMI_ANALYTICAL_V42\SOURCE\main.py
```

Run the supplied source only after checking the sensor coefficient file (`Sentinel2.json` or `SuperDove.json`) and configuration. The complete 58-test regression suite is in `SOURCE\tests`.

Run the tests from the package root with:

```bat
cd 03_CODE\04_SEMI_ANALYTICAL_V42\SOURCE
..\..\..\..\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
cd ..\..\..\..
```
