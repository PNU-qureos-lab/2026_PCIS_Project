# Final analysis

The standard Hedley equation was calculated for every delivered scene. Corrected rasters are retained for audit even when QC does not support using them as final products.

## Scene decisions

- **2025-10-18 — NO CORRECTION NEEDED**: mean visible–NIR R² 0.232; 566 nm median removal 12.10%; negative in any corrected band 0.068%; spatial preservation r 0.969. The scene shows weak evidence of a removable NIR-linked component.
- **2025-11-17 — CAUTIOUS**: mean visible–NIR R² 0.421; 566 nm median removal 20.21%; negative in any corrected band 0.039%; spatial preservation r 0.972. Some NIR-linked signal is present, but correction evidence is moderate.
- **2026-06-20 — NO CORRECTION NEEDED**: mean visible–NIR R² 0.230; 566 nm median removal 18.07%; negative in any corrected band 0.437%; spatial preservation r 0.933. The scene shows weak evidence of a removable NIR-linked component.
- **2026-06-27 — REJECT**: mean visible–NIR R² 0.671; 566 nm median removal 19.21%; negative in any corrected band 3.200%; spatial preservation r 0.550. Standard Hedley produces more than 1% negative water pixels.

## Recommended use

- The strongest currently usable candidate is **2025-11-17**, but it remains conditional on independent validation.
- For scenes labelled **NO CORRECTION NEEDED**, use the original masked Planet SR rather than forcing Hedley.
- Do not use scenes labelled **REJECT** as corrected products.
- Corrected SR is the primary numerical output. Approximate Rrs/rrs products are exploratory.
- The decision thresholds are documented project QC rules; they are not part of Hedley et al. (2005).

