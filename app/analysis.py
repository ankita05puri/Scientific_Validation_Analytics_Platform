"""Statistical analysis utilities for method-comparison validation studies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AnalysisResult:
    """Container for cleaned data, calculated paired values, and summary stats."""

    analyzed_data: pd.DataFrame
    summary: dict[str, float]
    reference_column: str
    candidate_column: str
    sample_id_column: str | None = None


def _safe_float(value: float) -> float:
    """Return a plain float, preserving missing values as NaN."""

    if pd.isna(value):
        return np.nan
    return float(value)


def prepare_paired_data(
    data: pd.DataFrame,
    reference_column: str,
    candidate_column: str,
    sample_id_column: str | None = None,
) -> pd.DataFrame:
    """Clean paired reference and candidate values for method-comparison analysis.

    Missing values and non-numeric entries are removed after selected columns are
    coerced to numeric values. Percent bias is only calculated when the reference
    value is non-zero.
    """

    selected_columns = [reference_column, candidate_column]
    if sample_id_column:
        selected_columns.insert(0, sample_id_column)

    paired = data[selected_columns].copy()
    if sample_id_column:
        paired = paired.rename(columns={sample_id_column: "Sample ID"})

    paired = paired.rename(
        columns={
            reference_column: "Reference",
            candidate_column: "Candidate",
        }
    )
    paired["Reference"] = pd.to_numeric(paired["Reference"], errors="coerce")
    paired["Candidate"] = pd.to_numeric(paired["Candidate"], errors="coerce")
    paired = paired.dropna(subset=["Reference", "Candidate"]).reset_index(drop=True)

    paired["Difference"] = paired["Candidate"] - paired["Reference"]
    paired["Average"] = (paired["Reference"] + paired["Candidate"]) / 2
    paired["Percent Bias"] = np.where(
        paired["Reference"] != 0,
        (paired["Difference"] / paired["Reference"]) * 100,
        np.nan,
    )
    paired["Percent Bias"] = paired["Percent Bias"].replace([np.inf, -np.inf], np.nan)
    return paired


def calculate_bland_altman_stats(analyzed_data: pd.DataFrame) -> dict[str, float]:
    """Calculate Bland-Altman style difference statistics."""

    difference = analyzed_data["Difference"]
    mean_difference = difference.mean()
    sd_difference = difference.std(ddof=1)
    upper_loa = mean_difference + (1.96 * sd_difference)
    lower_loa = mean_difference - (1.96 * sd_difference)
    return {
        "SD Difference": _safe_float(sd_difference),
        "Upper Limit of Agreement": _safe_float(upper_loa),
        "Lower Limit of Agreement": _safe_float(lower_loa),
    }


def calculate_summary(analyzed_data: pd.DataFrame) -> dict[str, float]:
    """Calculate summary statistics for paired method-comparison data."""

    n = int(len(analyzed_data))
    if n == 0:
        raise ValueError("No valid paired numeric rows are available for analysis.")

    reference = analyzed_data["Reference"]
    candidate = analyzed_data["Candidate"]
    difference = analyzed_data["Difference"]
    percent_bias = analyzed_data["Percent Bias"].dropna()

    if n >= 2 and reference.nunique() > 1 and candidate.nunique() > 1:
        correlation_r = float(np.corrcoef(reference, candidate)[0, 1])
        slope, intercept = np.polyfit(reference, candidate, 1)
    else:
        correlation_r = np.nan
        slope = np.nan
        intercept = np.nan

    r_squared = correlation_r**2 if not pd.isna(correlation_r) else np.nan

    summary = {
        "N": n,
        "Mean Reference": _safe_float(reference.mean()),
        "Mean Candidate": _safe_float(candidate.mean()),
        "Mean Difference": _safe_float(difference.mean()),
        "Mean Bias": _safe_float(difference.mean()),
        "Mean Absolute Bias": _safe_float(difference.abs().mean()),
        "Mean Percent Bias": _safe_float(percent_bias.mean()),
        "Median Percent Bias": _safe_float(percent_bias.median()),
        "SD Percent Bias": _safe_float(percent_bias.std(ddof=1)),
        "Minimum Percent Bias": _safe_float(percent_bias.min()),
        "Maximum Percent Bias": _safe_float(percent_bias.max()),
        "Correlation r": _safe_float(correlation_r),
        "R²": _safe_float(r_squared),
        "Slope": _safe_float(slope),
        "Intercept": _safe_float(intercept),
    }
    summary.update(calculate_bland_altman_stats(analyzed_data))
    return summary


def get_top_outliers(analyzed_data: pd.DataFrame, count: int = 5) -> pd.DataFrame:
    """Return samples with the largest absolute percent bias."""

    columns = ["Reference", "Candidate", "Difference", "Percent Bias"]
    if "Sample ID" in analyzed_data.columns:
        columns.insert(0, "Sample ID")

    outliers = analyzed_data.dropna(subset=["Percent Bias"]).copy()
    outliers["Absolute Percent Bias"] = outliers["Percent Bias"].abs()
    return (
        outliers.sort_values("Absolute Percent Bias", ascending=False)
        .head(count)[columns]
        .reset_index(drop=True)
    )


def evaluate_acceptance_criteria(
    summary: dict[str, float],
    min_r_squared: float,
    max_abs_mean_percent_bias: float,
    min_correlation_r: float = 0.95,
    slope_lower_limit: float = 0.90,
    slope_upper_limit: float = 1.10,
    max_abs_intercept: float | None = None,
    max_abs_mean_bias: float | None = None,
    max_abs_mean_difference: float | None = None,
    percent_samples_meeting_agreement: float | None = None,
    min_percent_samples_meeting_agreement: float = 95.0,
) -> dict[str, object]:
    """Evaluate preliminary user-defined acceptance criteria."""

    checks = []

    r_squared = summary.get("R²", np.nan)
    checks.append(
        {
            "Criterion": "R²",
            "Observed": r_squared,
            "Acceptance Limit": f">= {min_r_squared:g}",
            "Met": not pd.isna(r_squared) and r_squared >= min_r_squared,
            "Borderline": (
                not pd.isna(r_squared)
                and r_squared < min_r_squared
                and r_squared >= min_r_squared - 0.02
            ),
        }
    )

    correlation_r = summary.get("Correlation r", np.nan)
    checks.append(
        {
            "Criterion": "Correlation coefficient (r)",
            "Observed": correlation_r,
            "Acceptance Limit": f">= {min_correlation_r:g}",
            "Met": not pd.isna(correlation_r) and correlation_r >= min_correlation_r,
            "Borderline": (
                not pd.isna(correlation_r)
                and correlation_r < min_correlation_r
                and correlation_r >= min_correlation_r - 0.02
            ),
        }
    )

    slope = summary.get("Slope", np.nan)
    checks.append(
        {
            "Criterion": "Regression slope",
            "Observed": slope,
            "Acceptance Limit": f"{slope_lower_limit:g} to {slope_upper_limit:g}",
            "Met": (
                not pd.isna(slope)
                and slope_lower_limit <= slope <= slope_upper_limit
            ),
            "Borderline": (
                not pd.isna(slope)
                and slope_lower_limit * 0.98 <= slope <= slope_upper_limit * 1.02
                and not (slope_lower_limit <= slope <= slope_upper_limit)
            ),
        }
    )

    if max_abs_intercept is not None:
        intercept = summary.get("Intercept", np.nan)
        abs_intercept = abs(intercept) if not pd.isna(intercept) else np.nan
        checks.append(
            {
                "Criterion": "Absolute intercept",
                "Observed": abs_intercept,
                "Acceptance Limit": f"<= {max_abs_intercept:g}",
                "Met": not pd.isna(abs_intercept) and abs_intercept <= max_abs_intercept,
                "Borderline": (
                    not pd.isna(abs_intercept)
                    and abs_intercept > max_abs_intercept
                    and abs_intercept <= max_abs_intercept * 1.10
                ),
            }
        )

    if max_abs_mean_bias is not None:
        mean_bias = summary.get("Mean Bias", np.nan)
        abs_mean_bias = abs(mean_bias) if not pd.isna(mean_bias) else np.nan
        checks.append(
            {
                "Criterion": "Absolute mean bias",
                "Observed": abs_mean_bias,
                "Acceptance Limit": f"<= {max_abs_mean_bias:g}",
                "Met": not pd.isna(abs_mean_bias) and abs_mean_bias <= max_abs_mean_bias,
                "Borderline": (
                    not pd.isna(abs_mean_bias)
                    and abs_mean_bias > max_abs_mean_bias
                    and abs_mean_bias <= max_abs_mean_bias * 1.10
                ),
            }
        )

    mean_percent_bias = summary.get("Mean Percent Bias", np.nan)
    abs_mean_percent_bias = abs(mean_percent_bias) if not pd.isna(mean_percent_bias) else np.nan
    checks.append(
        {
            "Criterion": "Absolute Mean Percent Bias",
            "Observed": abs_mean_percent_bias,
            "Acceptance Limit": f"<= {max_abs_mean_percent_bias:g}%",
            "Met": (
                not pd.isna(abs_mean_percent_bias)
                and abs_mean_percent_bias <= max_abs_mean_percent_bias
            ),
            "Borderline": (
                not pd.isna(abs_mean_percent_bias)
                and abs_mean_percent_bias > max_abs_mean_percent_bias
                and abs_mean_percent_bias <= max_abs_mean_percent_bias * 1.10
            ),
        }
    )

    if max_abs_mean_difference is not None:
        mean_difference = summary.get("Mean Difference", np.nan)
        abs_mean_difference = abs(mean_difference) if not pd.isna(mean_difference) else np.nan
        checks.append(
            {
                "Criterion": "Absolute Mean Difference",
                "Observed": abs_mean_difference,
                "Acceptance Limit": f"<= {max_abs_mean_difference:g}",
                "Met": (
                    not pd.isna(abs_mean_difference)
                    and abs_mean_difference <= max_abs_mean_difference
                ),
                "Borderline": (
                    not pd.isna(abs_mean_difference)
                    and abs_mean_difference > max_abs_mean_difference
                    and abs_mean_difference <= max_abs_mean_difference * 1.10
                ),
            }
        )

    if percent_samples_meeting_agreement is not None:
        checks.append(
            {
                "Criterion": "Samples meeting agreement criteria",
                "Observed": percent_samples_meeting_agreement,
                "Acceptance Limit": f">= {min_percent_samples_meeting_agreement:g}%",
                "Met": percent_samples_meeting_agreement
                >= min_percent_samples_meeting_agreement,
                "Borderline": (
                    percent_samples_meeting_agreement
                    < min_percent_samples_meeting_agreement
                    and percent_samples_meeting_agreement
                    >= min_percent_samples_meeting_agreement - 5
                ),
            }
        )

    if all(check["Met"] for check in checks):
        decision = "PASS"
    elif all(check["Met"] or check["Borderline"] for check in checks):
        decision = "BORDERLINE"
    else:
        decision = "FAIL"

    return {
        "decision": decision,
        "checks": checks,
    }


def calculate_percent_samples_meeting_agreement(
    analyzed_data: pd.DataFrame, max_abs_percent_bias: float
) -> float:
    """Calculate the percentage of samples within an absolute percent-bias limit."""

    percent_bias = analyzed_data["Percent Bias"].dropna()
    if len(percent_bias) == 0:
        return np.nan
    return float((percent_bias.abs() <= max_abs_percent_bias).mean() * 100)


def run_method_comparison(
    data: pd.DataFrame,
    reference_column: str,
    candidate_column: str,
    sample_id_column: str | None = None,
) -> AnalysisResult:
    """Run the complete paired method-comparison analysis workflow."""

    analyzed_data = prepare_paired_data(
        data, reference_column, candidate_column, sample_id_column
    )
    summary = calculate_summary(analyzed_data)
    return AnalysisResult(
        analyzed_data=analyzed_data,
        summary=summary,
        reference_column=reference_column,
        candidate_column=candidate_column,
        sample_id_column=sample_id_column,
    )
