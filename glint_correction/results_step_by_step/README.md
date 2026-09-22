# Step-by-step results

`reference/` contains the verified output for the single example scene `20260808_S2A`.

1. `01_prepared_input` — aligned Sentinel-2 Level-2A input and grid audit.
2. `02_glint_acolite_cycle0` — ACOLITE transferred glint, corrected bands and RGB.
3. `03_hydrolight_simulations` — all 675 HydroLight spectra and library summary.
4. `04_deep_water_validation` — 30-ROI HydroLight matching and residual evidence.
5. `05_cycle1_decision` — the residual gate and retained correction stage.
6. `06_shallow_water_check` — before/after safety comparison.
7. `07_final_products` — final BOA, above-surface Rrs, below-surface rrs and QC.

Running `../code/run_complete_example.py` creates the same structure under
`generated/`. The generated directory is ignored by Git so each collaborator
can rerun the workflow without committing duplicate products.
