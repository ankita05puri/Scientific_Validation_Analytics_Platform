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


@dataclass(frozen=True)
class PrecisionResult:
    """Container for cleaned precision data and precision summaries."""

    analyzed_data: pd.DataFrame
    level_summary: pd.DataFrame
    day_summary: pd.DataFrame
    run_summary: pd.DataFrame
    result_column: str
    level_column: str
    day_column: str | None = None
    run_column: str | None = None
    replicate_column: str | None = None
    sample_id_column: str | None = None


@dataclass(frozen=True)
class LinearityResult:
    """Container for cleaned linearity data, summaries, and regression stats."""

    analyzed_data: pd.DataFrame
    level_summary: pd.DataFrame
    regression_summary: dict[str, float | str]
    expected_column: str
    observed_column: str
    level_column: str
    replicate_column: str | None = None
    units_column: str | None = None
    include_column: str | None = None


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


def _precision_summary_by(data: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Summarize precision measurements by one or more grouping columns."""

    summary = (
        data.groupby(group_columns, dropna=False)["Result"]
        .agg(N="count", Mean="mean", SD="std")
        .reset_index()
    )
    summary["CV%"] = np.where(summary["Mean"] != 0, (summary["SD"] / summary["Mean"]) * 100, np.nan)
    summary["Classification"] = summary["CV%"].apply(classify_precision_cv)
    return summary


def classify_precision_cv(cv_value: float) -> str:
    """Classify precision performance from an observed CV% value."""

    if pd.isna(cv_value):
        return "Not available"
    if cv_value < 1.0:
        return "Excellent"
    if cv_value < 3.0:
        return "Good"
    if cv_value <= 5.0:
        return "Acceptable"
    return "Investigate"


def prepare_precision_data(
    data: pd.DataFrame,
    result_column: str,
    level_column: str,
    day_column: str | None = None,
    run_column: str | None = None,
    replicate_column: str | None = None,
    sample_id_column: str | None = None,
) -> pd.DataFrame:
    """Clean repeated-measurement data for precision study analysis."""

    selected_columns = [result_column, level_column]
    optional_columns = [sample_id_column, day_column, run_column, replicate_column]
    selected_columns.extend([column for column in optional_columns if column])
    selected_columns = list(dict.fromkeys(selected_columns))

    precision_data = data[selected_columns].copy()
    rename_map = {
        result_column: "Result",
        level_column: "Level",
    }
    if sample_id_column:
        rename_map[sample_id_column] = "Sample ID"
    if day_column:
        rename_map[day_column] = "Day"
    if run_column:
        rename_map[run_column] = "Run"
    if replicate_column:
        rename_map[replicate_column] = "Replicate"

    precision_data = precision_data.rename(columns=rename_map)
    precision_data["Result"] = pd.to_numeric(precision_data["Result"], errors="coerce")
    for column in ["Day", "Run", "Replicate"]:
        if column in precision_data.columns:
            precision_data[column] = pd.to_numeric(precision_data[column], errors="coerce")

    precision_data = precision_data.dropna(subset=["Result", "Level"]).reset_index(drop=True)
    precision_data["Measurement Order"] = np.arange(1, len(precision_data) + 1)
    return precision_data


def calculate_precision_summary(analyzed_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculate simple precision summaries by level, day, and run."""

    if analyzed_data.empty:
        raise ValueError("No valid repeated numeric results are available for analysis.")

    level_summary = _precision_summary_by(analyzed_data, ["Level"])

    day_summary = pd.DataFrame()
    if "Day" in analyzed_data.columns:
        day_summary = _precision_summary_by(analyzed_data, ["Level", "Day"])

    run_summary = pd.DataFrame()
    run_group_columns = [column for column in ["Level", "Day", "Run"] if column in analyzed_data.columns]
    if "Run" in run_group_columns:
        run_summary = _precision_summary_by(analyzed_data, run_group_columns)

    return level_summary, day_summary, run_summary


def evaluate_precision_criteria(
    level_summary: pd.DataFrame, max_acceptable_cv: float
) -> dict[str, object]:
    """Evaluate preliminary precision acceptance criteria by level."""

    checks = []
    for _, row in level_summary.iterrows():
        observed_cv = row["CV%"]
        checks.append(
            {
                "Criterion": f"{row['Level']} CV%",
                "Observed": observed_cv,
                "Acceptance Limit": f"<= {max_acceptable_cv:g}%",
                "Classification": row.get("Classification", classify_precision_cv(observed_cv)),
                "Met": not pd.isna(observed_cv) and observed_cv <= max_acceptable_cv,
                "Borderline": (
                    not pd.isna(observed_cv)
                    and observed_cv > max_acceptable_cv
                    and observed_cv <= max_acceptable_cv * 1.20
                ),
            }
        )

    if all(check["Met"] for check in checks):
        decision = "PASS"
    elif all(check["Met"] or check["Borderline"] for check in checks):
        decision = "REVIEW"
    else:
        decision = "INVESTIGATE"

    return {
        "decision": decision,
        "checks": checks,
    }


def run_precision_study(
    data: pd.DataFrame,
    result_column: str,
    level_column: str,
    day_column: str | None = None,
    run_column: str | None = None,
    replicate_column: str | None = None,
    sample_id_column: str | None = None,
) -> PrecisionResult:
    """Run the complete simple precision study workflow."""

    analyzed_data = prepare_precision_data(
        data=data,
        result_column=result_column,
        level_column=level_column,
        day_column=day_column,
        run_column=run_column,
        replicate_column=replicate_column,
        sample_id_column=sample_id_column,
    )
    level_summary, day_summary, run_summary = calculate_precision_summary(analyzed_data)
    return PrecisionResult(
        analyzed_data=analyzed_data,
        level_summary=level_summary,
        day_summary=day_summary,
        run_summary=run_summary,
        result_column=result_column,
        level_column=level_column,
        day_column=day_column,
        run_column=run_column,
        replicate_column=replicate_column,
        sample_id_column=sample_id_column,
    )


def prepare_linearity_data(
    data: pd.DataFrame,
    expected_column: str,
    observed_column: str,
    level_column: str,
    replicate_column: str | None = None,
    units_column: str | None = None,
    include_column: str | None = None,
) -> pd.DataFrame:
    """Clean linearity data and apply Include in Analysis filtering when present."""

    selected_columns = [level_column, expected_column, observed_column]
    optional_columns = [replicate_column, units_column, include_column]
    selected_columns.extend([column for column in optional_columns if column])
    selected_columns = list(dict.fromkeys(selected_columns))

    linearity_data = data[selected_columns].copy()
    rename_map = {
        level_column: "Level",
        expected_column: "Expected Result",
        observed_column: "Observed Result",
    }
    if replicate_column:
        rename_map[replicate_column] = "Replicate"
    if units_column:
        rename_map[units_column] = "Units"
    if include_column:
        rename_map[include_column] = "Include in Analysis"

    linearity_data = linearity_data.rename(columns=rename_map)
    if "Include in Analysis" in linearity_data.columns:
        include_values = (
            linearity_data["Include in Analysis"].astype(str).str.strip().str.lower()
        )
        linearity_data = linearity_data[include_values.isin(["yes", "y", "true", "1"])]

    linearity_data["Expected Result"] = pd.to_numeric(
        linearity_data["Expected Result"], errors="coerce"
    )
    linearity_data["Observed Result"] = pd.to_numeric(
        linearity_data["Observed Result"], errors="coerce"
    )
    if "Replicate" in linearity_data.columns:
        linearity_data["Replicate"] = pd.to_numeric(
            linearity_data["Replicate"], errors="coerce"
        )
    return linearity_data.dropna(
        subset=["Level", "Expected Result", "Observed Result"]
    ).reset_index(drop=True)


def calculate_linearity_summary(analyzed_data: pd.DataFrame) -> pd.DataFrame:
    """Calculate per-level linearity recovery and bias summaries."""

    if analyzed_data.empty:
        raise ValueError("No valid linearity rows are available for analysis.")

    summary = (
        analyzed_data.groupby(["Level", "Expected Result"], dropna=False)[
            "Observed Result"
        ]
        .agg(N="count", **{"Mean Observed Result": "mean"})
        .reset_index()
    )
    summary["Difference"] = (
        summary["Mean Observed Result"] - summary["Expected Result"]
    )
    summary["Percent Recovery"] = np.where(
        summary["Expected Result"] != 0,
        (summary["Mean Observed Result"] / summary["Expected Result"]) * 100,
        np.nan,
    )
    summary["Percent Bias"] = np.where(
        summary["Expected Result"] != 0,
        (summary["Difference"] / summary["Expected Result"]) * 100,
        np.nan,
    )
    return summary.sort_values("Expected Result").reset_index(drop=True)


def calculate_linearity_regression(level_summary: pd.DataFrame) -> dict[str, float | str]:
    """Calculate regression statistics across linearity levels."""

    expected = level_summary["Expected Result"]
    observed = level_summary["Mean Observed Result"]
    if len(level_summary) >= 2 and expected.nunique() > 1 and observed.nunique() > 1:
        slope, intercept = np.polyfit(expected, observed, 1)
        correlation_r = float(np.corrcoef(expected, observed)[0, 1])
    else:
        slope = np.nan
        intercept = np.nan
        correlation_r = np.nan

    r_squared = correlation_r**2 if not pd.isna(correlation_r) else np.nan
    min_expected = expected.min()
    max_expected = expected.max()
    return {
        "Slope": _safe_float(slope),
        "Intercept": _safe_float(intercept),
        "Correlation r": _safe_float(correlation_r),
        "R²": _safe_float(r_squared),
        "Minimum Expected Result": _safe_float(min_expected),
        "Maximum Expected Result": _safe_float(max_expected),
        "Analytical Range Tested": f"{min_expected:.2f} to {max_expected:.2f}",
    }


def evaluate_linearity_criteria(
    level_summary: pd.DataFrame,
    regression_summary: dict[str, float | str],
    min_r_squared: float,
    slope_lower_limit: float,
    slope_upper_limit: float,
    max_abs_percent_bias: float,
    recovery_lower_limit: float,
    recovery_upper_limit: float,
) -> dict[str, object]:
    """Evaluate user-defined preliminary linearity acceptance criteria."""

    checks = []
    r_squared = regression_summary.get("R²", np.nan)
    checks.append(
        {
            "Criterion": "R²",
            "Observed": r_squared,
            "Acceptance Limit": f">= {min_r_squared:g}",
            "Met": not pd.isna(r_squared) and r_squared >= min_r_squared,
            "Borderline": (
                not pd.isna(r_squared)
                and r_squared < min_r_squared
                and r_squared >= min_r_squared - 0.005
            ),
        }
    )

    slope = regression_summary.get("Slope", np.nan)
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
                and slope_lower_limit * 0.99 <= slope <= slope_upper_limit * 1.01
                and not (slope_lower_limit <= slope <= slope_upper_limit)
            ),
        }
    )

    max_observed_bias = level_summary["Percent Bias"].abs().max()
    checks.append(
        {
            "Criterion": "Maximum absolute percent bias by level",
            "Observed": max_observed_bias,
            "Acceptance Limit": f"<= {max_abs_percent_bias:g}%",
            "Met": (
                not pd.isna(max_observed_bias)
                and max_observed_bias <= max_abs_percent_bias
            ),
            "Borderline": (
                not pd.isna(max_observed_bias)
                and max_observed_bias > max_abs_percent_bias
                and max_observed_bias <= max_abs_percent_bias * 1.10
            ),
        }
    )

    min_recovery = level_summary["Percent Recovery"].min()
    max_recovery = level_summary["Percent Recovery"].max()
    checks.append(
        {
            "Criterion": "Percent recovery range",
            "Observed": f"{min_recovery:.2f}% to {max_recovery:.2f}%",
            "Acceptance Limit": f"{recovery_lower_limit:g}% to {recovery_upper_limit:g}%",
            "Met": (
                not pd.isna(min_recovery)
                and not pd.isna(max_recovery)
                and min_recovery >= recovery_lower_limit
                and max_recovery <= recovery_upper_limit
            ),
            "Borderline": (
                not pd.isna(min_recovery)
                and not pd.isna(max_recovery)
                and min_recovery >= recovery_lower_limit - 2
                and max_recovery <= recovery_upper_limit + 2
                and not (
                    min_recovery >= recovery_lower_limit
                    and max_recovery <= recovery_upper_limit
                )
            ),
        }
    )

    if all(check["Met"] for check in checks):
        decision = "PASS"
    elif all(check["Met"] or check["Borderline"] for check in checks):
        decision = "BORDERLINE"
    else:
        decision = "FAIL"
    return {"decision": decision, "checks": checks}


def run_linearity_study(
    data: pd.DataFrame,
    expected_column: str,
    observed_column: str,
    level_column: str,
    replicate_column: str | None = None,
    units_column: str | None = None,
    include_column: str | None = None,
) -> LinearityResult:
    """Run the complete linearity study workflow."""

    analyzed_data = prepare_linearity_data(
        data,
        expected_column,
        observed_column,
        level_column,
        replicate_column,
        units_column,
        include_column,
    )
    level_summary = calculate_linearity_summary(analyzed_data)
    regression_summary = calculate_linearity_regression(level_summary)
    return LinearityResult(
        analyzed_data=analyzed_data,
        level_summary=level_summary,
        regression_summary=regression_summary,
        expected_column=expected_column,
        observed_column=observed_column,
        level_column=level_column,
        replicate_column=replicate_column,
        units_column=units_column,
        include_column=include_column,
    )
