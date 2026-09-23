# PlanetScope input data

The raw PlanetScope Surface Reflectance and UDM2 files are not redistributed in
this public repository because access and redistribution are governed by the
Planet data licence.

To rerun the workflow, supply a Planet delivery ZIP containing paired files:

- `*_3B_AnalyticMS_SR_8b_clip.tif`
- `*_3B_udm2_clip.tif`
- the accompanying XML/JSON metadata

Run the command documented in `../02_RUN_WORKFLOW/README.md`. The committed
generated products, figures, parameter files, and QC tables are the complete
outputs of the four-scene Tidung run.
