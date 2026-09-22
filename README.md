# 2026 PCIS Project

Satellite imagery inventory, glint correction, and semi-analytical shallow-water modelling for the Pacific Island project.

## End-to-end workflow

The processing begins with one Sentinel-2 scene. The glint-correction stage
removes estimated surface reflection and produces corrected water reflectance.
The semi-analytical stage then uses that corrected reflectance to estimate water
depth and the relative contributions of coral sand and seagrass.

### What HOPE and D2 mean

- **HOPE** means **Hyperspectral Optimization Process Exemplar**. It is a
  physics-based semi-analytical method introduced by Lee and co-authors. In this
  project, the HOPE equations fit the deep-water spectrum and estimate three
  water optical parameters: `P` for phytoplankton absorption, `G` for dissolved
  and detrital absorption, and `X` for particle backscattering.
- **D2** is the configuration label used in the supplied code for the Lee
  shallow-water forward model. It predicts how much of the measured signal
  comes from the water column and how much comes from the bottom after
  attenuation through the water. The inversion changes depth and the
  sand–seagrass mixture until the modeled four-band reflectance best matches
  Sentinel-2. `D2` is a model label in this implementation, not a separate
  acronym.

```mermaid
flowchart TD
    A[Sentinel-2 L2A scene<br/>BOA reflectance bands] --> D[Prepare all bands on<br/>one 20 m grid]
    B[ACOLITE glint products<br/>B11/B12 selection, rhog_ref<br/>and transfer factors] --> E[Glint correction<br/>Cycle 0]
    C[HydroLight water-only<br/>simulation library] --> F[Deep-water residual<br/>validation]
    D --> E
    E --> F
    F --> G{Is an additional<br/>Cycle 1 correction needed?}
    G -->|Yes| H[Apply the accepted<br/>Cycle 1 adjustment]
    G -->|No| I[Keep the Cycle 0 result]
    H --> J[Final corrected BOA]
    I --> J
    J --> K[Convert to above-water Rrs<br/>B01, B02, B03 and B04]
    K --> L[Select valid pixels with<br/>the shallow-water mask]
    M[Deep-water Rrs, solar angle<br/>and pure-water coefficients] --> N[HOPE initialization<br/>estimate P, G and X]
    O[Coral-sand and<br/>seagrass spectra] --> P[D2 candidate generation<br/>and optimization]
    L --> P
    N --> P
    P --> Q[Select the solution<br/>with minimum C4]
    Q --> R[Estimated water depth]
    Q --> S[Sand and seagrass fractions]
    Q --> T[Habitat class]
    Q --> U[Model cost and<br/>processing status]
    R --> V[GeoTIFF products and<br/>result maps with legends]
    S --> V
    T --> V
    U --> V
```

## HTML files

The links below open the files on GitHub. To use them as interactive pages, download the repository ZIP, extract it, and open `index.html` locally.

- [Satellite inventory](inventory.html) — browse Sentinel-2 and Planet Dove scenes.
- [ACOLITE glint correction](glint_correction.html) — method and interactive scene comparison.
- [Semi-analytical concept](semi_analytical_concept.html) — model workflow and output interpretation.
- [Semi-analytical results map](semi_analytical_map.html) — explore Sentinel-2 depth and habitat products geographically.

## Open the interactive pages

Download this repository as a ZIP, extract the whole folder, and open `index.html` in a browser. Keep the asset folders beside the HTML files. The results map uses an online basemap; the project overlays are stored in the repository.

The Python code is included in the `semi_analytical/` and `glint_correction/` folders.

The glint package is arranged in execution order. Start with
[`glint_correction/00_READ_ME_FIRST/01_WORKFLOW_ORDER.md`](glint_correction/00_READ_ME_FIRST/01_WORKFLOW_ORDER.md),
then inspect the numbered input, runner, reference output, generated output,
and complete source-code folders.
