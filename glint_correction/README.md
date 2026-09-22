# Complete one-scene Glint-ACOLITE worked example

This folder presents the complete Tidung workflow for one small Sentinel-2 scene:

```text
Scene: 20260808_S2A
Area: Tidung subset, 417 × 278 pixels
Grid: EPSG:32748, 20 m
```

The purpose is teaching and traceability. Every numbered step has its input,
commented Python code, and at least one real example output.

## Important distinction

- **Sen2Cor** produced the original Sentinel-2 L2A BOA reflectance.
- **Glint-ACOLITE** produced the B11/B12 selection, `rhog_ref`, and optical
  transfer factors used for Cycle 0.
- **HydroLight** produced the 675 water-only spectra used during deep-water
  residual validation.
- The Python workflow connects these products, performs the correction,
  validates the result, and exports BOA, `Rrs`, and `rrs`.

The example does not claim to contain the proprietary/external HydroLight
executable or rerun ACOLITE atmospheric correction. It includes the actual
ACOLITE-derived products, the complete 675-row HydroLight library, and the
Python processing and validation logic used after those products were created.

## Folder structure

```text
glint_correction/
├── README.md
├── code/
│   ├── run_complete_example.py       Portable one-scene runner
│   ├── RUN_ME_WINDOWS.bat            Windows launcher
│   ├── requirements.txt
│   ├── input_values/                 BOA, ACOLITE and HydroLight inputs
│   └── full_source/                  Complete production and validation code
└── results_step_by_step/
    ├── reference/                    Verified results supplied with the package
    └── generated/                    Results created on another computer
```

Both output folders use the same internal scientific order:

```text
01_prepared_input
02_glint_acolite_cycle0
03_hydrolight_simulations
04_deep_water_validation
05_cycle1_decision
06_shallow_water_check
07_final_products
```

## Processing order

1. Verify that every input band uses the same CRS, transform, width and height.
2. Read the Glint-ACOLITE per-pixel B11/B12 selection and `rhog_ref`.
3. Transfer `rhog_ref` to every optical band using its ACOLITE factor.
4. Calculate `Cycle0 = original BOA − estimated glint`.
5. Load and document all 675 HydroLight water simulations.
6. Evaluate the 30 offshore ROIs and read the accepted deep-water decision.
7. Apply Cycle 1 only if the validation authorizes it.
8. Check the same 30 shallow-water ROIs before, after Cycle 0, and final.
9. Export final BOA, above-surface `Rrs`, and below-surface `rrs`.

The final filenames use `above_surface_Rrs` and `below_surface_rrs`. This is
deliberate: Windows cannot safely store two filenames that differ only by the
capitalization of `Rrs`/`rrs`.

## Run

From `code`:

```powershell
python run_complete_example.py
```

The same runner also accepts portable command-line paths:

```powershell
python run_complete_example.py --input-dir input_values --output-dir ..\results_step_by_step\generated --scene 20260808_S2A
```

The script writes to `results_step_by_step/generated`. Compare it with the verified
`results_step_by_step/reference` reference.

`code/full_source/production_code` contains the complete implementation:
20 m preparation, ACOLITE Cycle 0, all HydroLight interfaces and validation,
the Cycle-1 gate, B01 processing, reflectance conversion, and final exports.
Each script begins with workflow-role comments, and its README documents inputs
and outputs stage by stage.

A recipient without HydroLight can still rerun every downstream validation
step because the complete 675-case precomputed library is included at
`code/input_values/03_hydrolight_library/model_library_B02_B8A.csv`. See the README beside
that file for variables, units, limitations, and its SHA-256 integrity hash.

The same folder also contains the HydroLight input-record template, the pure
seawater absorption and scattering coefficients, and the mineral-particle
absorption coefficients used to document the optical setup.

## Equations

For target band `λ`:

```text
candidate_B11(λ) = factor_B11_to_λ × rhog_ref
candidate_B12(λ) = factor_B12_to_λ × rhog_ref
selected_glint(λ) = candidate chosen by the accepted per-pixel B11/B12 map
Cycle0(λ) = original_BOA(λ) − selected_glint(λ)
Rrs(λ) = final_BOA(λ) / π
rrs(λ) = Rrs(λ) / (0.52 + 1.7 × Rrs(λ))
```

B01 is aligned from native 60 m to the 20 m grid. This does not create true
20 m B01 spatial information.

The input also includes L2A B09 aligned from 60 m. B09 is kept for completeness
but excluded from the water/glint target bands because it is centred in strong
water-vapour absorption. B10 is not supplied as an L2A BOA band by Sen2Cor.

B01 is processed in the same active correction loop as B02–B8A. The separate
scripts under `code/full_source/production_code/legacy_provenance/b01_extension` only document
how B01 was historically added to an already-completed archive.
