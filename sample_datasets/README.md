# LabOS Scientific Validation Demo Datasets

These datasets are designed for end-to-end LabOS workflow demonstrations: Upload -> Data Quality -> Analysis -> Review -> Report.
They are not random placeholder files. Each file follows the structure needed by the implemented study engines and is tuned to produce realistic PASS outcomes with natural laboratory variation.

## Files

| File | Study Type | Purpose | Expected Outcome |
|---|---|---|---|
| `method_comparison_hba1c.csv` | Method Comparison | 50 paired HbA1c specimens spanning clinical range with strong candidate/reference agreement and mild outliers. | Correlation and regression calculable; Bland-Altman and bias plots meaningful; expected PASS. |
| `precision_hba1c.csv` | Precision | 5 days x 2 runs/day x 3 replicates/run at Low QC and High QC. | Within-run, between-run/day, and total CV calculable; expected PASS. |
| `accuracy_hba1c.csv` | Accuracy | Three HbA1c concentration levels with sample IDs and replicate measured values near target. | Recovery and bias calculable; expected PASS. |
| `linearity_hba1c.csv` | Linearity | Seven concentration levels with four replicates per level and near-linear response. | Regression, recovery, and deviation calculable; expected PASS. |
| `stability_hba1c.csv` | Stability | Eight samples over 0, 24, 72, 168, and 336 hours under refrigerated and room-temperature conditions. | Drift and percent recovery calculable; expected PASS. |
| `detection_capability_hba1c.csv` | Detection Capability | Blank, low-concentration, and quantitation-level replicates. `Blank_Flag` is intended to be mapped as the sample type field. | LoB, LoD, and LoQ calculable; expected PASS. |

## Notes

- Values are synthetic but constructed to resemble diagnostic validation studies.
- Precision `Run_ID` values repeat across QC levels and contain multiple replicates, preventing the previous issue where between-run calculations could not be performed.
- Detection Capability uses `Blank_Flag` values of `Blank`, `Low Concentration`, and `Quantitation Level` so the platform can compute LoB, LoD, and LoQ.

## Platform Schema Note

`accuracy_hba1c.csv` includes `Sample_ID` in addition to the requested target, measured, and level fields because the implemented Accuracy engine requires a sample identifier for end-to-end workflow validation.
