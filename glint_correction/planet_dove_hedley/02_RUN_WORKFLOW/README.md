# Run the PlanetScope Hedley workflow

Install the packages in `requirements.txt`, then run:

```powershell
python process_planetscope_standard_hedley.py `
  --input-zip "PATH_TO_PLANET_DELIVERY.zip" `
  --output-dir "PATH_TO_NEW_OUTPUT_FOLDER" `
  --distance-m 300
```

The script implements the standard Hedley regression equation and reports its
QC separately. It does not clip negative corrected values and it does not turn
project QC thresholds into part of the Hedley method.

Validate a completed package with:

```powershell
python validate_planetscope_standard_hedley_package.py "PATH_TO_OUTPUT_FOLDER"
```
