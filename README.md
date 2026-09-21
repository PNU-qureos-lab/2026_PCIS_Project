# 2026 PCIS Project

Satellite imagery inventory, glint correction, and semi-analytical shallow-water modelling for the Pacific Island project.

## HTML files

The links below open the files on GitHub. To use them as interactive pages, download the repository ZIP, extract it, and open `index.html` locally.

- [Satellite inventory](inventory.html) — browse Sentinel-2 and Planet Dove scenes.
- [ACOLITE glint correction](glint_correction.html) — method and interactive scene comparison.
- [Semi-analytical concept](semi_analytical_concept.html) — model workflow and output interpretation.
- [Semi-analytical results map](semi_analytical_map.html) — explore Sentinel-2 depth and habitat products geographically.
- [Semi-analytical image viewer](semi_analytical.html) — inspect the result images by scene and product.

## Open the interactive pages

Download this repository as a ZIP, extract the whole folder, and open `index.html` in a browser. Keep the asset folders beside the HTML files. The results map uses an online basemap; the project overlays are stored in the repository.

## Code

- [Semi-analytical Python pipeline](semi_analytical/code_v3/src/run_all.py)
- [Glint-correction Python workflow](glint_correction/00_prepare_20m_grid.py)
