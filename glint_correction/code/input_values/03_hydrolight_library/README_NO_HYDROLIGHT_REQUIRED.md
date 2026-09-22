# Using the package without HydroLight

The recipient does **not** need a HydroLight licence or executable to rerun the
included glint validation. The precomputed lookup table is:

`model_library_B02_B8A.csv`

It contains all 675 water-only simulations used by the workflow. The table has
one row per simulation and includes both the input variables and Sentinel-2
band-convolved HydroLight output.

## Variables included

| Column | Meaning | Unit/range |
|---|---|---|
| `case_id` | Unique simulation identifier | text |
| `platform` | Sentinel-2 spectral response | S2A, S2B, S2C |
| `sza` | Solar zenith angle | 20, 25, 30, 35, 40 degrees |
| `background` | Water optical background identifier | OLI |
| `chl` | Chlorophyll-a | 0.05, 0.1, 0.5 mg m-3 |
| `cdom` | CDOM absorption at 440 nm | 0.01, 0.03, 0.06 m-1 |
| `tsm` | Total suspended matter | 0.1–80 g m-3, 15 values |
| `wind` | Wind speed | 5 m s-1 |
| `B02`–`B8A` | Band-convolved HydroLight above-water Rrs | sr-1 |

The complete grid is `3 platforms × 5 SZA × 3 chlorophyll × 3 CDOM ×
15 TSM = 675 simulations`.

The `source_mfile` column records the original generation path for provenance.
That absolute path does not need to exist on the recipient's computer and is
not used by the portable one-scene workflow.

## How the downstream code uses it

HydroLight outputs above-water `Rrs` in `sr-1`. During comparison with the
dimensionless corrected BOA product, the validation code uses:

```text
modeled_BOA_like = π × HydroLight_Rrs
```

The portable script reads this CSV directly, draws all 675 spectra, and copies
the full table into the numbered output. Therefore, the recipient can reproduce
the included model matching, deep-water validation, graphs, and Cycle-1
decision without rerunning HydroLight.

## What cannot be changed without HydroLight

The recipient can select, interpolate, plot, and compare these 675 existing
cases. They cannot create simulations outside the supplied ranges, change the
inherent optical-property model, bottom boundary, wavelength configuration, or
other HydroLight settings without access to HydroLight.

## Integrity check

SHA-256 for `model_library_B02_B8A.csv`:

`1204F2B04CC1D44F2E12070560A6ABDB982F9E53DC61E7EC3D0DDD9E65264700`
