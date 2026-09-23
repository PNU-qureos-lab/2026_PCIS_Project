# Run these folders in numerical order

## 00 — Read first

Read `README.md` and `PORTABLE_RUN_GUIDE.md` to understand the method and the
difference between the verified result and a newly generated result.

## 01 — Input

Inspect `01_INPUT`:

1. `01_l2a_boa_20m`: every available Sentinel-2 L2A BOA band on one 20 m grid;
2. `02_acolite_support`: B11/B12 selection, `rhog_ref`, and transfer factors;
3. `03_hydrolight_library`: all 675 precomputed simulations;
4. `04_validation_inputs`: deep- and shallow-water validation evidence.

Do not edit these files when reproducing the supplied example.

## 02 — Run workflow

Open `02_RUN_WORKFLOW` and double-click `RUN_ME_WINDOWS.bat`. It installs the
three portable dependencies if necessary and runs the commented Python script.

## 03 — Example output

`03_EXAMPLE_OUTPUT` is the verified answer supplied with the package. Keep it
unchanged as the comparison reference.

## 04 — Generated output

`04_GENERATED_OUTPUT` is created or refreshed by the recipient's run. Compare
its manifest, QC tables, rasters, and figures with `03_EXAMPLE_OUTPUT`.

## 05 — Full source code

`05_FULL_SOURCE_CODE/production_code` contains the complete research code for
grid preparation, ACOLITE correction, HydroLight generation/model matching,
residual validation, conversions, and exports. It is included for audit and for
adapting the method to new scenes. The portable example does not require the
recipient to run the licensed HydroLight executable.

## Scientific processing inside outputs

The output stages are also numbered:

1. prepared and audited inputs;
2. Glint-ACOLITE Cycle 0;
3. HydroLight simulations;
4. deep-water validation;
5. Cycle-1 decision;
6. shallow-water safety check;
7. final BOA, Rrs, rrs and QC.
