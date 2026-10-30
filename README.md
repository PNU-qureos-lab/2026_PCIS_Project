# PCIS 2026 Satellite Bathymetry Workflow

This repository provides the public project website and the compact processing-code distribution for:

- Sentinel-2 glint correction using the ACOLITE-based Cycle 0 / conditional Cycle 1 workflow.
- Planet Dove glint correction using the Standard Hedley method.
- Corrected remote-sensing reflectance (Rrs) preparation.
- Semi-analytical Version 42 processing for depth and benthic fractions.

## Public website

After GitHub Pages is enabled for the `main` branch, the project website is available at:

<https://pnu-qureos-lab.github.io/2026_PCIS_Project/>

Direct sections:

- [Data inventory](https://pnu-qureos-lab.github.io/2026_PCIS_Project/site/index.html#tab-data)
- [Sentinel-2 glint correction](https://pnu-qureos-lab.github.io/2026_PCIS_Project/site/index.html#methods-sentinel)
- [Planet Dove glint correction](https://pnu-qureos-lab.github.io/2026_PCIS_Project/site/index.html#methods-planet)
- [Semi-analytical processing](https://pnu-qureos-lab.github.io/2026_PCIS_Project/site/index.html#methods-semi)
- [Result viewer](https://pnu-qureos-lab.github.io/2026_PCIS_Project/site/index.html#tab-results)

## Repository structure

```text
2026_PCIS_Project/
|-- index.html                  Public landing page
|-- site/                       Interactive inventory, methods, and results
|   |-- index.html
|   |-- assets/
|   `-- vendor/
|-- code/                       Compact code-only workflow
|   |-- 01_SENTINEL2_DOWNLOAD/
|   |-- 02_SENTINEL2_GLINT_CORRECTION/
|   |-- 03_PLANET_DOVE_GLINT_CORRECTION/
|   |-- 04_SEMI_ANALYTICAL_V42/
|   `-- 05_SHARED_UTILITIES/
`-- README.md
```

## Run the code

Start with [`code/README.md`](code/README.md). Each numbered workflow folder contains its own README with required inputs and example commands.

The code is included without raw satellite data. Large Sentinel-2 source scenes, licensed Planet Dove imagery, complete intermediate outputs, and the portable offline Python environment are distributed in the separate delivery package.

Scientific reference tables and configuration files required by the processing code remain included. Generated preview PNG files used by the website are lightweight visual products, not raw satellite scenes.
