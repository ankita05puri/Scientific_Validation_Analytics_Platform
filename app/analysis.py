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


@dataclass(frozen=True)
class StabilityResult:
    """Container for cleaned stability data and timepoint summaries."""

    analyzed_data: pd.DataFrame
    stability_summary: pd.DataFrame
    timepoint_summary: pd.DataFrame
    bias_summary: pd.DataFrame
    recovery_summary: pd.DataFrame
    condition_comparison: pd.DataFrame
    outlier_table: pd.DataFrame
    overall_summary: dict[str, float | str]
    sample_id_column: str
    timepoint_column: str
    result_column: str
    storage_condition_column: str | None = None
    units_column: str | None = None
    replicate_column: str | None = None
    include_column: str | None = None


@dataclass(frozen=True)
class AccuracyResult:
    """Container for cleaned accuracy data and observed-vs-expected summaries."""

    analyzed_data: pd.DataFrame
    accuracy_summary: pd.DataFrame
    bias_summary: pd.DataFrame
    recovery_summary: pd.DataFrame
    level_decision_table: pd.DataFrame
    worst_case_summary: dict[str, float | str]
    overall_summary: dict[str, float | str]
    data_quality_warnings: list[str]
    expected_column: str
    observed_column: str
    level_column: str
    sample_id_column: str
    units_column: str | None = None
    replicate_column: str | None = None
    include_column: str | None = None


@dataclass(frozen=True)
class DetectionCapabilityResult:
    """Container for cleaned LoB/LoD/LoQ detection capability outputs."""

    analyzed_data: pd.DataFrame
    lob_summary: pd.DataFrame
    lod_summary: pd.DataFrame
    loq_summary: pd.DataFrame
    methodology_table: pd.DataFrame
    data_quality_summary: pd.DataFrame
    outlier_table: pd.DataFrame
    decision_matrix: pd.DataFrame
    overall_summary: dict[str, float | str]
    sample_id_column: str
    sample_type_column: str
    concentration_column: str
    result_column: str
    replicate_column: str | None = None
    units_column: str | None = None
    include_column: str | None = None


@dataclass(frozen=True)
class DBSValidationResult:
    """Container for cleaned DBS equivalency data and validation summaries."""

    analyzed_data: pd.DataFrame
    study_summary: pd.DataFrame
    bias_summary: pd.DataFrame
    recovery_summary: pd.DataFrame
    correlation_summary: pd.DataFrame
    agreement_summary: pd.DataFrame
    hematocrit_summary: pd.DataFrame
    delay_summary: pd.DataFrame
    instrument_summary: pd.DataFrame
    outlier_review: pd.DataFrame
    scientific_findings: list[str]
    sample_review: pd.DataFrame
    overall_summary: dict[str, float | str]
    sample_id_column: str
    reference_column: str
    dbs_column: str
    include_column: str | None = None


def _safe_float(value: float) -> float:
    """Return a plain float, preserving missing values as NaN."""

    if pd.isna(value):
        return np.nan
    return float(value)


def _max_threshold_status(observed: float, limit: float, zone: float) -> str:
    """Classify a maximum-threshold observation using a borderline zone."""

    if pd.isna(observed):
        return "FAIL"
    caution_floor = max(0.0, limit - zone)
    if observed >= limit:
        return "FAIL"
    if observed > caution_floor:
        return "PASS WITH CAUTION"
    return "PASS"


def _min_threshold_status(observed: float, limit: float, zone: float) -> str:
    """Classify a minimum-threshold observation using a borderline zone."""

    if pd.isna(observed):
        return "FAIL"
    caution_ceiling = limit + zone
    if observed <= limit:
        return "FAIL"
    if observed < caution_ceiling:
        return "PASS WITH CAUTION"
    return "PASS"


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


def _timepoint_sort_key(value: object) -> tuple[int, float, str]:
    """Return a stable sort key that keeps baseline first, then numeric timepoints."""

    label = str(value).strip()
    normalized = label.lower()
    if normalized in {"baseline", "base", "day 0", "d0", "t0", "0"}:
        return (0, 0.0, label)

    digits = "".join(character if character.isdigit() or character == "." else " " for character in normalized)
    values = [part for part in digits.split() if part]
    if values:
        return (1, float(values[0]), label)
    return (2, 0.0, label)


def _is_baseline_timepoint(value: object) -> bool:
    """Return whether a timepoint label represents baseline."""

    return str(value).strip().lower() in {"baseline", "base", "day 0", "d0", "t0", "0"}


def prepare_stability_data(
    data: pd.DataFrame,
    sample_id_column: str,
    timepoint_column: str,
    result_column: str,
    storage_condition_column: str | None = None,
    units_column: str | None = None,
    replicate_column: str | None = None,
    include_column: str | None = None,
) -> pd.DataFrame:
    """Clean stability data and calculate row-level change from sample baseline."""

    selected_columns = [sample_id_column, timepoint_column, result_column]
    optional_columns = [
        storage_condition_column,
        units_column,
        replicate_column,
        include_column,
    ]
    selected_columns.extend([column for column in optional_columns if column])
    selected_columns = list(dict.fromkeys(selected_columns))

    stability_data = data[selected_columns].copy()
    rename_map = {
        sample_id_column: "Sample ID",
        timepoint_column: "Timepoint",
        result_column: "Result",
    }
    if storage_condition_column:
        rename_map[storage_condition_column] = "Storage Condition"
    if units_column:
        rename_map[units_column] = "Units"
    if replicate_column:
        rename_map[replicate_column] = "Replicate"
    if include_column:
        rename_map[include_column] = "Include in Analysis"

    stability_data = stability_data.rename(columns=rename_map)
    if "Include in Analysis" in stability_data.columns:
        include_values = (
            stability_data["Include in Analysis"].astype(str).str.strip().str.lower()
        )
        stability_data = stability_data[include_values.isin(["yes", "y", "true", "1"])]

    stability_data["Result"] = pd.to_numeric(stability_data["Result"], errors="coerce")
    if "Replicate" in stability_data.columns:
        stability_data["Replicate"] = pd.to_numeric(
            stability_data["Replicate"], errors="coerce"
        )
    stability_data = stability_data.dropna(
        subset=["Sample ID", "Timepoint", "Result"]
    ).reset_index(drop=True)
    if stability_data.empty:
        raise ValueError("No valid stability rows are available for analysis.")

    stability_data["Is Baseline"] = stability_data["Timepoint"].apply(_is_baseline_timepoint)
    baseline_means = (
        stability_data[stability_data["Is Baseline"]]
        .groupby("Sample ID", dropna=False)["Result"]
        .mean()
        .rename("Baseline Result")
    )
    if baseline_means.empty:
        raise ValueError("At least one baseline timepoint is required for stability analysis.")

    stability_data = stability_data.merge(baseline_means, on="Sample ID", how="left")
    stability_data = stability_data.dropna(subset=["Baseline Result"]).reset_index(drop=True)
    if stability_data.empty:
        raise ValueError("No analyzed rows have a matching sample baseline.")

    stability_data["Timepoint Sort"] = stability_data["Timepoint"].map(_timepoint_sort_key)
    stability_data["Difference"] = stability_data["Result"] - stability_data["Baseline Result"]
    stability_data["Bias"] = stability_data["Difference"]
    stability_data["Percent Change"] = np.where(
        stability_data["Baseline Result"] != 0,
        (stability_data["Difference"] / stability_data["Baseline Result"]) * 100,
        np.nan,
    )
    stability_data["Percent Recovery"] = np.where(
        stability_data["Baseline Result"] != 0,
        (stability_data["Result"] / stability_data["Baseline Result"]) * 100,
        np.nan,
    )
    return stability_data.sort_values(
        ["Timepoint Sort", "Sample ID", "Replicate"] if "Replicate" in stability_data.columns else ["Timepoint Sort", "Sample ID"]
    ).reset_index(drop=True)


def calculate_stability_summary(
    analyzed_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float | str]]:
    """Calculate stability summaries by timepoint."""

    group_columns = ["Timepoint", "Timepoint Sort"]
    if "Storage Condition" in analyzed_data.columns:
        group_columns.insert(0, "Storage Condition")

    timepoint_summary = (
        analyzed_data.groupby(group_columns, dropna=False)
        .agg(
            N=("Result", "count"),
            **{
                "Baseline Mean": ("Baseline Result", "mean"),
                "Timepoint Mean": ("Result", "mean"),
                "Mean Difference": ("Difference", "mean"),
                "Mean Absolute Difference": ("Difference", lambda series: series.abs().mean()),
                "Mean Percent Change": ("Percent Change", "mean"),
                "Mean Absolute Percent Change": ("Percent Change", lambda series: series.abs().mean()),
                "Percent Recovery": ("Percent Recovery", "mean"),
                "Minimum Recovery": ("Percent Recovery", "min"),
                "Maximum Recovery": ("Percent Recovery", "max"),
                "Bias": ("Bias", "mean"),
                "Maximum Absolute Bias": ("Bias", lambda series: series.abs().max()),
            },
        )
        .reset_index()
        .sort_values("Timepoint Sort")
        .reset_index(drop=True)
    )
    timepoint_summary["Absolute Difference"] = timepoint_summary["Mean Difference"].abs()

    stability_summary = timepoint_summary[
        [
            *[column for column in ["Storage Condition"] if column in timepoint_summary.columns],
            "Timepoint",
            "N",
            "Baseline Mean",
            "Timepoint Mean",
            "Absolute Difference",
            "Mean Percent Change",
            "Percent Recovery",
            "Bias",
        ]
    ].copy()
    bias_summary = timepoint_summary[
        [
            *[column for column in ["Storage Condition"] if column in timepoint_summary.columns],
            "Timepoint",
            "Bias",
            "Maximum Absolute Bias",
            "Mean Percent Change",
            "Mean Absolute Percent Change",
        ]
    ].copy()
    recovery_summary = timepoint_summary[
        [
            *[column for column in ["Storage Condition"] if column in timepoint_summary.columns],
            "Timepoint",
            "Percent Recovery",
            "Minimum Recovery",
            "Maximum Recovery",
        ]
    ].copy()

    non_baseline = analyzed_data[~analyzed_data["Is Baseline"]]
    change_source = non_baseline if not non_baseline.empty else analyzed_data
    max_change_index = change_source["Percent Change"].abs().idxmax()
    max_change_row = change_source.loc[max_change_index]
    overall_summary = {
        "N": int(len(analyzed_data)),
        "Baseline Mean": _safe_float(analyzed_data.loc[analyzed_data["Is Baseline"], "Result"].mean()),
        "Maximum Observed Change": _safe_float(change_source["Percent Change"].abs().max()),
        "Maximum Absolute Bias": _safe_float(change_source["Bias"].abs().max()),
        "Minimum Recovery": _safe_float(change_source["Percent Recovery"].min()),
        "Maximum Recovery": _safe_float(change_source["Percent Recovery"].max()),
        "Worst Timepoint": str(max_change_row["Timepoint"]),
        "Worst Sample ID": str(max_change_row["Sample ID"]),
        "Worst Storage Condition": str(max_change_row.get("Storage Condition", "Not specified")),
        "Average Change by Timepoint": _safe_float(timepoint_summary["Mean Absolute Percent Change"].mean()),
    }

    return (
        stability_summary.reset_index(drop=True),
        timepoint_summary.drop(columns=["Timepoint Sort"]).reset_index(drop=True),
        bias_summary.reset_index(drop=True),
        recovery_summary.reset_index(drop=True),
        overall_summary,
    )


def calculate_storage_condition_comparison(timepoint_summary: pd.DataFrame) -> pd.DataFrame:
    """Compare stability metrics between storage conditions at each timepoint."""

    if "Storage Condition" not in timepoint_summary.columns:
        return pd.DataFrame()

    conditions = list(dict.fromkeys(timepoint_summary["Storage Condition"].dropna().astype(str)))
    if len(conditions) < 2:
        return pd.DataFrame()

    preferred_reference = "Refrigerated" if "Refrigerated" in conditions else conditions[0]
    preferred_candidate = (
        "Room Temperature"
        if "Room Temperature" in conditions and preferred_reference != "Room Temperature"
        else next(condition for condition in conditions if condition != preferred_reference)
    )

    rows = []
    for timepoint, group in timepoint_summary.groupby("Timepoint", sort=False):
        indexed = group.set_index("Storage Condition")
        if preferred_reference not in indexed.index or preferred_candidate not in indexed.index:
            continue
        reference = indexed.loc[preferred_reference]
        candidate = indexed.loc[preferred_candidate]
        rows.append(
            {
                "Timepoint": timepoint,
                f"{preferred_reference} Mean": reference["Timepoint Mean"],
                f"{preferred_candidate} Mean": candidate["Timepoint Mean"],
                "Difference": candidate["Timepoint Mean"] - reference["Timepoint Mean"],
                f"{preferred_reference} Recovery": reference["Percent Recovery"],
                f"{preferred_candidate} Recovery": candidate["Percent Recovery"],
                "Recovery Difference": candidate["Percent Recovery"] - reference["Percent Recovery"],
                f"{preferred_reference} Percent Change": reference["Mean Percent Change"],
                f"{preferred_candidate} Percent Change": candidate["Mean Percent Change"],
                "Percent Change Difference": candidate["Mean Percent Change"] - reference["Mean Percent Change"],
                "Comparison": f"{preferred_candidate} - {preferred_reference}",
            }
        )
    return pd.DataFrame(rows)


def identify_stability_outliers(analyzed_data: pd.DataFrame) -> pd.DataFrame:
    """Identify sample/timepoint rows contributing most to observed instability."""

    non_baseline = analyzed_data[~analyzed_data["Is Baseline"]].copy()
    if non_baseline.empty:
        return pd.DataFrame()

    study_average_change = non_baseline["Percent Change"].abs().mean()
    threshold = study_average_change * 1.5
    columns = ["Sample ID", "Timepoint", "Percent Change", "Percent Recovery", "Bias"]
    if "Storage Condition" in non_baseline.columns:
        columns.insert(1, "Storage Condition")

    outliers = non_baseline[columns].copy()
    outliers["Largest Absolute Change"] = outliers["Percent Change"].abs()
    outliers["Largest Bias"] = outliers["Bias"].abs()
    outliers["Lowest Recovery"] = outliers["Percent Recovery"]
    outliers["Potential Outlier"] = outliers["Largest Absolute Change"] > threshold
    outliers["Severity Score"] = outliers["Largest Absolute Change"]
    return (
        outliers.sort_values(["Potential Outlier", "Severity Score"], ascending=[False, False])
        .reset_index(drop=True)
    )


def evaluate_stability_criteria(
    overall_summary: dict[str, float | str],
    max_percent_change: float,
    min_recovery: float,
    max_abs_bias: float,
    borderline_zone: float,
) -> dict[str, object]:
    """Evaluate user-defined preliminary stability acceptance criteria."""

    checks = []

    def max_threshold_check(label: str, observed: float, limit: float, unit: str, zone: float) -> dict[str, object]:
        caution_floor = max(0.0, limit - zone)
        return {
            "Criterion": label,
            "Observed": observed,
            "Acceptance Limit": f"< {limit:g}{unit}",
            "Met": not pd.isna(observed) and observed <= caution_floor,
            "Borderline": (
                not pd.isna(observed)
                and observed > caution_floor
                and observed < limit
            ),
        }

    def min_threshold_check(label: str, observed: float, limit: float, unit: str, zone: float) -> dict[str, object]:
        caution_ceiling = limit + zone
        return {
            "Criterion": label,
            "Observed": observed,
            "Acceptance Limit": f"> {limit:g}{unit}",
            "Met": not pd.isna(observed) and observed >= caution_ceiling,
            "Borderline": (
                not pd.isna(observed)
                and observed > limit
                and observed < caution_ceiling
            ),
        }

    max_change = overall_summary.get("Maximum Observed Change", np.nan)
    checks.append(
        max_threshold_check(
            "Maximum absolute percent change from baseline",
            max_change,
            max_percent_change,
            "%",
            borderline_zone,
        )
    )

    observed_min_recovery = overall_summary.get("Minimum Recovery", np.nan)
    checks.append(
        min_threshold_check(
            "Minimum recovery",
            observed_min_recovery,
            min_recovery,
            "%",
            borderline_zone,
        )
    )

    observed_max_bias = overall_summary.get("Maximum Absolute Bias", np.nan)
    bias_zone = max_abs_bias * (borderline_zone / 100)
    checks.append(
        max_threshold_check(
            "Maximum absolute bias",
            observed_max_bias,
            max_abs_bias,
            "",
            bias_zone,
        )
    )

    if all(check["Met"] for check in checks):
        decision = "PASS"
    elif all(check["Met"] or check["Borderline"] for check in checks):
        decision = "PASS WITH CAUTION"
    else:
        decision = "FAIL"
    return {"decision": decision, "checks": checks}


def run_stability_study(
    data: pd.DataFrame,
    sample_id_column: str,
    timepoint_column: str,
    result_column: str,
    storage_condition_column: str | None = None,
    units_column: str | None = None,
    replicate_column: str | None = None,
    include_column: str | None = None,
) -> StabilityResult:
    """Run the complete stability study workflow."""

    analyzed_data = prepare_stability_data(
        data=data,
        sample_id_column=sample_id_column,
        timepoint_column=timepoint_column,
        result_column=result_column,
        storage_condition_column=storage_condition_column,
        units_column=units_column,
        replicate_column=replicate_column,
        include_column=include_column,
    )
    (
        stability_summary,
        timepoint_summary,
        bias_summary,
        recovery_summary,
        overall_summary,
    ) = calculate_stability_summary(analyzed_data)
    condition_comparison = calculate_storage_condition_comparison(timepoint_summary)
    outlier_table = identify_stability_outliers(analyzed_data)
    return StabilityResult(
        analyzed_data=analyzed_data,
        stability_summary=stability_summary,
        timepoint_summary=timepoint_summary,
        bias_summary=bias_summary,
        recovery_summary=recovery_summary,
        condition_comparison=condition_comparison,
        outlier_table=outlier_table,
        overall_summary=overall_summary,
        sample_id_column=sample_id_column,
        timepoint_column=timepoint_column,
        result_column=result_column,
        storage_condition_column=storage_condition_column,
        units_column=units_column,
        replicate_column=replicate_column,
        include_column=include_column,
    )


def assess_accuracy_data_quality(
    data: pd.DataFrame,
    sample_id_column: str,
    level_column: str,
    expected_column: str,
    observed_column: str,
    include_column: str | None = None,
    minimum_replicates: int = 2,
) -> list[str]:
    """Return reviewer-facing data quality warnings for accuracy input data."""

    selected_columns = [
        sample_id_column,
        level_column,
        expected_column,
        observed_column,
    ]
    if include_column:
        selected_columns.append(include_column)
    quality_data = data[list(dict.fromkeys(selected_columns))].copy()
    if include_column:
        include_values = (
            quality_data[include_column].astype(str).str.strip().str.lower()
        )
        quality_data = quality_data[include_values.isin(["yes", "y", "true", "1"])]

    expected = pd.to_numeric(quality_data[expected_column], errors="coerce")
    observed = pd.to_numeric(quality_data[observed_column], errors="coerce")
    warnings: list[str] = []
    missing_expected = int(expected.isna().sum())
    missing_observed = int(observed.isna().sum())
    duplicate_ids = int(quality_data[sample_id_column].duplicated().sum())
    non_positive_expected = int((expected <= 0).fillna(False).sum())

    if missing_expected:
        warnings.append(
            f"{missing_expected} row(s) have missing or non-numeric expected values and will be excluded from analysis."
        )
    if missing_observed:
        warnings.append(
            f"{missing_observed} row(s) have missing or non-numeric observed values and will be excluded from analysis."
        )
    if duplicate_ids:
        warnings.append(
            f"{duplicate_ids} duplicate sample ID occurrence(s) were detected. Confirm that duplicate IDs represent intended replicates."
        )
    if non_positive_expected:
        warnings.append(
            f"{non_positive_expected} row(s) have zero or negative expected values. Percent bias and recovery require positive expected values."
        )

    replicate_counts = (
        quality_data.assign(
            **{
                "_Expected Numeric": expected,
                "_Observed Numeric": observed,
            }
        )
        .dropna(subset=["_Expected Numeric", "_Observed Numeric"])
        .groupby(level_column, dropna=False)
        .size()
    )
    insufficient_levels = [
        str(level)
        for level, count in replicate_counts.items()
        if count < minimum_replicates
    ]
    if insufficient_levels:
        warnings.append(
            "Insufficient replicates detected for level(s): "
            + ", ".join(insufficient_levels)
            + f". At least {minimum_replicates} replicates are recommended for reviewer confidence."
        )
    return warnings


def prepare_accuracy_data(
    data: pd.DataFrame,
    sample_id_column: str,
    level_column: str,
    expected_column: str,
    observed_column: str,
    units_column: str | None = None,
    replicate_column: str | None = None,
    include_column: str | None = None,
) -> pd.DataFrame:
    """Clean observed-vs-expected accuracy data."""

    selected_columns = [sample_id_column, level_column, expected_column, observed_column]
    optional_columns = [units_column, replicate_column, include_column]
    selected_columns.extend([column for column in optional_columns if column])
    selected_columns = list(dict.fromkeys(selected_columns))

    accuracy_data = data[selected_columns].copy()
    rename_map = {
        sample_id_column: "Sample ID",
        level_column: "Level",
        expected_column: "Expected Result",
        observed_column: "Observed Result",
    }
    if units_column:
        rename_map[units_column] = "Units"
    if replicate_column:
        rename_map[replicate_column] = "Replicate"
    if include_column:
        rename_map[include_column] = "Include in Analysis"

    accuracy_data = accuracy_data.rename(columns=rename_map)
    if "Include in Analysis" in accuracy_data.columns:
        include_values = (
            accuracy_data["Include in Analysis"].astype(str).str.strip().str.lower()
        )
        accuracy_data = accuracy_data[include_values.isin(["yes", "y", "true", "1"])]

    accuracy_data["Expected Result"] = pd.to_numeric(
        accuracy_data["Expected Result"], errors="coerce"
    )
    accuracy_data["Observed Result"] = pd.to_numeric(
        accuracy_data["Observed Result"], errors="coerce"
    )
    if "Replicate" in accuracy_data.columns:
        accuracy_data["Replicate"] = pd.to_numeric(
            accuracy_data["Replicate"], errors="coerce"
        )

    accuracy_data = accuracy_data.dropna(
        subset=["Sample ID", "Level", "Expected Result", "Observed Result"]
    ).reset_index(drop=True)
    if accuracy_data.empty:
        raise ValueError("No valid observed and expected accuracy rows are available.")
    return accuracy_data


def calculate_accuracy_summary(
    analyzed_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float | str], dict[str, float | str]]:
    """Calculate accuracy summary, bias summary, recovery summary, and overall metrics."""

    summary = (
        analyzed_data.groupby(["Level", "Expected Result"], dropna=False)["Observed Result"]
        .agg(N="count", **{"Mean Observed Result": "mean", "SD": "std"})
        .reset_index()
    )
    summary["Difference"] = summary["Mean Observed Result"] - summary["Expected Result"]
    summary["Absolute Difference"] = summary["Difference"].abs()
    summary["Percent Bias"] = np.where(
        summary["Expected Result"] != 0,
        (summary["Difference"] / summary["Expected Result"]) * 100,
        np.nan,
    )
    summary["Absolute Percent Bias"] = summary["Percent Bias"].abs()
    summary["Percent Recovery"] = np.where(
        summary["Expected Result"] != 0,
        (summary["Mean Observed Result"] / summary["Expected Result"]) * 100,
        np.nan,
    )
    summary["Mean Observed 95% CI Lower"] = np.where(
        summary["N"] > 1,
        summary["Mean Observed Result"] - (1.96 * summary["SD"] / np.sqrt(summary["N"])),
        np.nan,
    )
    summary["Mean Observed 95% CI Upper"] = np.where(
        summary["N"] > 1,
        summary["Mean Observed Result"] + (1.96 * summary["SD"] / np.sqrt(summary["N"])),
        np.nan,
    )
    summary["Bias 95% CI Lower"] = np.where(
        summary["N"] > 1,
        summary["Difference"] - (1.96 * summary["SD"] / np.sqrt(summary["N"])),
        np.nan,
    )
    summary["Bias 95% CI Upper"] = np.where(
        summary["N"] > 1,
        summary["Difference"] + (1.96 * summary["SD"] / np.sqrt(summary["N"])),
        np.nan,
    )
    summary = summary.sort_values("Expected Result").reset_index(drop=True)

    bias_summary = summary[
        [
            "Level",
            "Expected Result",
            "Mean Observed Result",
            "Difference",
            "Absolute Difference",
            "Bias 95% CI Lower",
            "Bias 95% CI Upper",
            "Percent Bias",
            "Absolute Percent Bias",
        ]
    ].copy()
    recovery_summary = summary[
        ["Level", "Expected Result", "Mean Observed Result", "Percent Recovery"]
    ].copy()

    max_bias_row = summary.loc[summary["Absolute Percent Bias"].idxmax()]
    min_recovery_row = summary.loc[summary["Percent Recovery"].idxmin()]
    max_recovery_row = summary.loc[summary["Percent Recovery"].idxmax()]
    worst_case_summary = {
        "Worst Level": str(max_bias_row["Level"]),
        "Highest Absolute Percent Bias": _safe_float(max_bias_row["Absolute Percent Bias"]),
        "Lowest Recovery": _safe_float(min_recovery_row["Percent Recovery"]),
        "Lowest Recovery Level": str(min_recovery_row["Level"]),
        "Highest Recovery": _safe_float(max_recovery_row["Percent Recovery"]),
        "Highest Recovery Level": str(max_recovery_row["Level"]),
    }
    overall_summary = {
        "N": int(analyzed_data["Observed Result"].count()),
        "Overall Mean Bias": _safe_float(summary["Difference"].mean()),
        "Overall Mean Percent Bias": _safe_float(summary["Percent Bias"].mean()),
        "Maximum Absolute Bias": _safe_float(summary["Absolute Difference"].max()),
        "Maximum Absolute Percent Bias": _safe_float(summary["Absolute Percent Bias"].max()),
        "Minimum Recovery": _safe_float(summary["Percent Recovery"].min()),
        "Maximum Recovery": _safe_float(summary["Percent Recovery"].max()),
        "Worst Level": worst_case_summary["Worst Level"],
    }
    return summary, bias_summary, recovery_summary, worst_case_summary, overall_summary


def build_accuracy_level_decision_table(
    accuracy_summary: pd.DataFrame,
    max_abs_bias: float,
    max_abs_percent_bias: float,
    min_recovery: float,
    max_recovery: float,
    borderline_zone: float,
) -> pd.DataFrame:
    """Build level-specific accuracy decisions for validation review."""

    rows: list[dict[str, object]] = []
    bias_zone = max_abs_bias * (borderline_zone / 100)
    for _, row in accuracy_summary.iterrows():
        statuses = [
            _max_threshold_status(row["Absolute Difference"], max_abs_bias, bias_zone),
            _max_threshold_status(
                row["Absolute Percent Bias"], max_abs_percent_bias, borderline_zone
            ),
            _min_threshold_status(row["Percent Recovery"], min_recovery, borderline_zone),
            _max_threshold_status(row["Percent Recovery"], max_recovery, borderline_zone),
        ]
        if "FAIL" in statuses:
            status = "FAIL"
        elif "PASS WITH CAUTION" in statuses:
            status = "PASS WITH CAUTION"
        else:
            status = "PASS"

        rows.append(
            {
                "Level": row["Level"],
                "N": int(row["N"]),
                "Mean Observed Result": row["Mean Observed Result"],
                "Mean Observed 95% CI Lower": row["Mean Observed 95% CI Lower"],
                "Mean Observed 95% CI Upper": row["Mean Observed 95% CI Upper"],
                "Expected Result": row["Expected Result"],
                "Absolute Bias": row["Absolute Difference"],
                "Bias 95% CI Lower": row["Bias 95% CI Lower"],
                "Bias 95% CI Upper": row["Bias 95% CI Upper"],
                "Percent Bias": row["Percent Bias"],
                "Recovery %": row["Percent Recovery"],
                "Pass/Fail Status": status,
            }
        )
    return pd.DataFrame(rows)


def evaluate_accuracy_criteria(
    overall_summary: dict[str, float | str],
    max_abs_bias: float,
    max_abs_percent_bias: float,
    min_recovery: float,
    max_recovery: float,
    borderline_zone: float,
) -> dict[str, object]:
    """Evaluate user-defined preliminary accuracy acceptance criteria."""

    checks = []

    def max_threshold_check(label: str, observed: float, limit: float, unit: str, zone: float) -> dict[str, object]:
        caution_floor = max(0.0, limit - zone)
        return {
            "Criterion": label,
            "Observed": observed,
            "Acceptance Limit": f"< {limit:g}{unit}",
            "Met": not pd.isna(observed) and observed <= caution_floor,
            "Borderline": (
                not pd.isna(observed)
                and observed > caution_floor
                and observed < limit
            ),
        }

    def min_threshold_check(label: str, observed: float, limit: float, unit: str, zone: float) -> dict[str, object]:
        caution_ceiling = limit + zone
        return {
            "Criterion": label,
            "Observed": observed,
            "Acceptance Limit": f"> {limit:g}{unit}",
            "Met": not pd.isna(observed) and observed >= caution_ceiling,
            "Borderline": (
                not pd.isna(observed)
                and observed > limit
                and observed < caution_ceiling
            ),
        }

    checks.append(
        max_threshold_check(
            "Maximum absolute bias",
            overall_summary.get("Maximum Absolute Bias", np.nan),
            max_abs_bias,
            "",
            max_abs_bias * (borderline_zone / 100),
        )
    )
    checks.append(
        max_threshold_check(
            "Maximum absolute percent bias",
            overall_summary.get("Maximum Absolute Percent Bias", np.nan),
            max_abs_percent_bias,
            "%",
            borderline_zone,
        )
    )
    checks.append(
        min_threshold_check(
            "Minimum recovery",
            overall_summary.get("Minimum Recovery", np.nan),
            min_recovery,
            "%",
            borderline_zone,
        )
    )
    max_recovery_observed = overall_summary.get("Maximum Recovery", np.nan)
    caution_floor = max_recovery - borderline_zone
    checks.append(
        {
            "Criterion": "Maximum recovery",
            "Observed": max_recovery_observed,
            "Acceptance Limit": f"< {max_recovery:g}%",
            "Met": not pd.isna(max_recovery_observed) and max_recovery_observed <= caution_floor,
            "Borderline": (
                not pd.isna(max_recovery_observed)
                and max_recovery_observed > caution_floor
                and max_recovery_observed < max_recovery
            ),
        }
    )

    if all(check["Met"] for check in checks):
        decision = "PASS"
    elif all(check["Met"] or check["Borderline"] for check in checks):
        decision = "PASS WITH CAUTION"
    else:
        decision = "FAIL"
    return {"decision": decision, "checks": checks}


def run_accuracy_study(
    data: pd.DataFrame,
    sample_id_column: str,
    level_column: str,
    expected_column: str,
    observed_column: str,
    units_column: str | None = None,
    replicate_column: str | None = None,
    include_column: str | None = None,
    max_abs_bias: float = 0.50,
    max_abs_percent_bias: float = 10.0,
    min_recovery: float = 90.0,
    max_recovery: float = 110.0,
    borderline_zone: float = 2.0,
) -> AccuracyResult:
    """Run the complete accuracy study workflow."""

    data_quality_warnings = assess_accuracy_data_quality(
        data=data,
        sample_id_column=sample_id_column,
        level_column=level_column,
        expected_column=expected_column,
        observed_column=observed_column,
        include_column=include_column,
    )
    analyzed_data = prepare_accuracy_data(
        data=data,
        sample_id_column=sample_id_column,
        level_column=level_column,
        expected_column=expected_column,
        observed_column=observed_column,
        units_column=units_column,
        replicate_column=replicate_column,
        include_column=include_column,
    )
    (
        accuracy_summary,
        bias_summary,
        recovery_summary,
        worst_case_summary,
        overall_summary,
    ) = calculate_accuracy_summary(analyzed_data)
    level_decision_table = build_accuracy_level_decision_table(
        accuracy_summary,
        max_abs_bias=max_abs_bias,
        max_abs_percent_bias=max_abs_percent_bias,
        min_recovery=min_recovery,
        max_recovery=max_recovery,
        borderline_zone=borderline_zone,
    )
    return AccuracyResult(
        analyzed_data=analyzed_data,
        accuracy_summary=accuracy_summary,
        bias_summary=bias_summary,
        recovery_summary=recovery_summary,
        level_decision_table=level_decision_table,
        worst_case_summary=worst_case_summary,
        overall_summary=overall_summary,
        data_quality_warnings=data_quality_warnings,
        expected_column=expected_column,
        observed_column=observed_column,
        level_column=level_column,
        sample_id_column=sample_id_column,
        units_column=units_column,
        replicate_column=replicate_column,
        include_column=include_column,
    )


def prepare_detection_capability_data(
    data: pd.DataFrame,
    sample_id_column: str,
    sample_type_column: str,
    concentration_column: str,
    result_column: str,
    replicate_column: str | None = None,
    units_column: str | None = None,
    include_column: str | None = None,
) -> pd.DataFrame:
    """Clean LoB/LoD/LoQ input data for detection capability analysis."""

    selected_columns = [
        sample_id_column,
        sample_type_column,
        concentration_column,
        result_column,
    ]
    selected_columns.extend(
        [column for column in [replicate_column, units_column, include_column] if column]
    )
    selected_columns = list(dict.fromkeys(selected_columns))
    detection_data = data[selected_columns].copy()
    rename_map = {
        sample_id_column: "Sample ID",
        sample_type_column: "Sample Type",
        concentration_column: "Concentration Level",
        result_column: "Observed Result",
    }
    if replicate_column:
        rename_map[replicate_column] = "Replicate"
    if units_column:
        rename_map[units_column] = "Units"
    if include_column:
        rename_map[include_column] = "Include in Analysis"

    detection_data = detection_data.rename(columns=rename_map)
    if "Include in Analysis" in detection_data.columns:
        include_values = (
            detection_data["Include in Analysis"].astype(str).str.strip().str.lower()
        )
        detection_data = detection_data[include_values.isin(["yes", "y", "true", "1"])]

    detection_data["Sample Type"] = detection_data["Sample Type"].astype(str).str.strip()
    detection_data["Concentration Level"] = pd.to_numeric(
        detection_data["Concentration Level"], errors="coerce"
    )
    detection_data["Observed Result"] = pd.to_numeric(
        detection_data["Observed Result"], errors="coerce"
    )
    if "Replicate" in detection_data.columns:
        detection_data["Replicate"] = pd.to_numeric(
            detection_data["Replicate"], errors="coerce"
        )
    detection_data = detection_data.dropna(
        subset=["Sample ID", "Sample Type", "Concentration Level", "Observed Result"]
    ).reset_index(drop=True)
    if detection_data.empty:
        raise ValueError("No valid detection capability rows are available.")
    return detection_data


def identify_iqr_outliers(
    data: pd.DataFrame,
    group_column: str,
    value_column: str,
) -> pd.DataFrame:
    """Identify outliers within groups using the 1.5 x IQR rule."""

    outlier_rows: list[pd.DataFrame] = []
    for _, group in data.groupby(group_column, dropna=False):
        q1 = group[value_column].quantile(0.25)
        q3 = group[value_column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - (1.5 * iqr)
        upper = q3 + (1.5 * iqr)
        flagged = group[
            (group[value_column] < lower) | (group[value_column] > upper)
        ].copy()
        if not flagged.empty:
            flagged["IQR Lower Bound"] = lower
            flagged["IQR Upper Bound"] = upper
            outlier_rows.append(flagged)
    if not outlier_rows:
        return pd.DataFrame(
            columns=[
                "Sample ID",
                "Sample Type",
                "Concentration Level",
                "Replicate",
                "Observed Result",
                "IQR Lower Bound",
                "IQR Upper Bound",
            ]
        )
    return pd.concat(outlier_rows, ignore_index=True)


def assess_detection_data_quality(
    raw_data: pd.DataFrame,
    analyzed_data: pd.DataFrame,
    selected_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assess replicate-level data quality for detection capability studies."""

    available_columns = [column for column in selected_columns if column in raw_data.columns]
    raw_selected = raw_data[available_columns].copy()
    missing_values = int(raw_selected.isna().sum().sum())
    excluded_replicates = int(len(raw_data) - len(analyzed_data))
    outlier_table = identify_iqr_outliers(
        analyzed_data, "Sample Type", "Observed Result"
    )
    blank_count = int((analyzed_data["Sample Type"].str.lower() == "blank").sum())
    low_count = int(
        (analyzed_data["Sample Type"].str.lower() == "low concentration").sum()
    )
    loq_counts = (
        analyzed_data[
            analyzed_data["Sample Type"].str.lower() == "quantitation level"
        ]
        .groupby("Concentration Level")
        .size()
    )
    min_loq_replicates = int(loq_counts.min()) if not loq_counts.empty else 0

    rows = [
        {
            "Check": "Missing values",
            "Observed": missing_values,
            "Recommendation": "No missing required values",
            "Status": "PASS" if missing_values == 0 else "FAIL",
        },
        {
            "Check": "Excluded replicates",
            "Observed": excluded_replicates,
            "Recommendation": "Review intentionally excluded or invalid rows",
            "Status": "PASS" if excluded_replicates == 0 else "BORDERLINE",
        },
        {
            "Check": "IQR outliers",
            "Observed": int(len(outlier_table)),
            "Recommendation": "Review outliers before final approval",
            "Status": "PASS" if outlier_table.empty else "BORDERLINE",
        },
        {
            "Check": "Blank replicates",
            "Observed": blank_count,
            "Recommendation": "At least 20 blank replicates",
            "Status": "PASS" if blank_count >= 20 else "BORDERLINE",
        },
        {
            "Check": "Low-level replicates",
            "Observed": low_count,
            "Recommendation": "At least 20 low-level replicates",
            "Status": "PASS" if low_count >= 20 else "BORDERLINE",
        },
        {
            "Check": "Quantitation level replicates",
            "Observed": min_loq_replicates,
            "Recommendation": "At least 5 replicates per quantitation level",
            "Status": "PASS" if min_loq_replicates >= 5 else "BORDERLINE",
        },
    ]
    return pd.DataFrame(rows), outlier_table.reset_index(drop=True)


def build_detection_methodology_table(
    lob_summary: pd.DataFrame,
    lod_summary: pd.DataFrame,
    loq_summary: pd.DataFrame,
    overall_summary: dict[str, float | str],
) -> pd.DataFrame:
    """Build formula transparency table for detection capability outputs."""

    lob_row = lob_summary.iloc[0]
    lod_row = lod_summary.iloc[0]
    loq_row = loq_summary[
        loq_summary["Concentration Level"] == overall_summary["LoQ"]
    ]
    loq_inputs = "No concentration met the target CV% criterion"
    if not loq_row.empty:
        first_loq = loq_row.iloc[0]
        loq_inputs = (
            f"Concentration={first_loq['Concentration Level']:.3f}; "
            f"Mean={first_loq['Mean']:.3f}; SD={first_loq['SD']:.3f}"
        )
    return pd.DataFrame(
        [
            {
                "Metric": "LoB",
                "Formula": "LoB = Mean Blank + 1.645 x SD Blank",
                "Inputs Used": f"Mean Blank={lob_row['Mean Blank']:.3f}; SD Blank={lob_row['SD Blank']:.3f}",
                "Final Result": overall_summary["LoB"],
            },
            {
                "Metric": "LoD",
                "Formula": "LoD = LoB + 1.645 x SD Low Concentration Sample",
                "Inputs Used": f"LoB={overall_summary['LoB']:.3f}; SD Low={lod_row['SD Low Sample']:.3f}",
                "Final Result": overall_summary["LoD"],
            },
            {
                "Metric": "CV%",
                "Formula": "CV% = (SD / Mean) x 100",
                "Inputs Used": loq_inputs,
                "Final Result": overall_summary["Operational LoQ CV%"],
            },
            {
                "Metric": "Operational LoQ",
                "Formula": "Lowest concentration meeting target CV%",
                "Inputs Used": f"Target CV%={overall_summary['Target LoQ CV%']:.2f}%",
                "Final Result": overall_summary["LoQ"],
            },
        ]
    )


def build_detection_decision_matrix(
    criteria_result: dict[str, object],
) -> pd.DataFrame:
    """Build final Detection Capability decision matrix."""

    checks = pd.DataFrame(criteria_result["checks"])
    fail_count = int((~checks["Met"] & ~checks["Borderline"]).sum())
    borderline_count = int(checks["Borderline"].sum())
    decision = str(criteria_result["decision"])
    if decision == "PASS":
        rationale = "All criteria passed."
    elif decision == "BORDERLINE":
        rationale = "No criterion exceeded its limit, but at least one criterion is within the borderline zone."
    else:
        rationale = "At least one criterion exceeded the acceptance limit."
    return pd.DataFrame(
        [
            {"Decision Rule": "PASS", "Condition": "All criteria pass", "Applies": decision == "PASS"},
            {"Decision Rule": "BORDERLINE", "Condition": "Any criterion within borderline zone", "Applies": decision == "BORDERLINE"},
            {"Decision Rule": "FAIL", "Condition": "Any criterion exceeds acceptance limit", "Applies": decision == "FAIL"},
            {
                "Decision Rule": "Final Decision",
                "Condition": f"{decision}: {rationale} Failed criteria={fail_count}; borderline criteria={borderline_count}.",
                "Applies": True,
            },
        ]
    )


def calculate_detection_capability_summary(
    analyzed_data: pd.DataFrame,
    target_loq_cv: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float | str]]:
    """Calculate LoB, LoD, and LoQ detection capability summaries."""

    sample_type = analyzed_data["Sample Type"].str.lower()
    blank_data = analyzed_data[sample_type == "blank"]
    low_data = analyzed_data[sample_type == "low concentration"]
    loq_data = analyzed_data[sample_type == "quantitation level"]

    if blank_data.empty:
        raise ValueError("At least one Blank row is required for LoB analysis.")
    if low_data.empty:
        raise ValueError("At least one Low Concentration row is required for LoD analysis.")
    if loq_data.empty:
        raise ValueError("At least one Quantitation Level row is required for LoQ analysis.")

    mean_blank = blank_data["Observed Result"].mean()
    sd_blank = blank_data["Observed Result"].std(ddof=1)
    lob = mean_blank + (1.645 * sd_blank)
    lob_summary = pd.DataFrame(
        [
            {
                "N": int(blank_data["Observed Result"].count()),
                "Mean Blank": _safe_float(mean_blank),
                "SD Blank": _safe_float(sd_blank),
                "LoB": _safe_float(lob),
            }
        ]
    )

    mean_low = low_data["Observed Result"].mean()
    sd_low = low_data["Observed Result"].std(ddof=1)
    lod = lob + (1.645 * sd_low)
    lod_summary = pd.DataFrame(
        [
            {
                "N": int(low_data["Observed Result"].count()),
                "Mean Low Sample": _safe_float(mean_low),
                "SD Low Sample": _safe_float(sd_low),
                "LoD": _safe_float(lod),
            }
        ]
    )

    loq_summary = (
        loq_data.groupby("Concentration Level", dropna=False)["Observed Result"]
        .agg(N="count", Mean="mean", SD="std")
        .reset_index()
        .sort_values("Concentration Level")
        .reset_index(drop=True)
    )
    loq_summary["CV%"] = np.where(
        loq_summary["Mean"] != 0,
        (loq_summary["SD"] / loq_summary["Mean"]) * 100,
        np.nan,
    )
    loq_summary["Bias %"] = np.where(
        loq_summary["Concentration Level"] != 0,
        ((loq_summary["Mean"] - loq_summary["Concentration Level"]) / loq_summary["Concentration Level"]) * 100,
        np.nan,
    )
    loq_summary["Recovery %"] = np.where(
        loq_summary["Concentration Level"] != 0,
        (loq_summary["Mean"] / loq_summary["Concentration Level"]) * 100,
        np.nan,
    )
    loq_summary["Pass/Fail"] = np.where(
        loq_summary["CV%"] <= target_loq_cv,
        "PASS",
        "FAIL",
    )

    passing_loq = loq_summary[loq_summary["Pass/Fail"] == "PASS"]
    operational_loq = (
        _safe_float(passing_loq["Concentration Level"].min())
        if not passing_loq.empty
        else np.nan
    )
    operational_row = (
        passing_loq.loc[passing_loq["Concentration Level"].idxmin()]
        if not passing_loq.empty
        else pd.Series(dtype=object)
    )
    worst_row = loq_summary.loc[loq_summary["CV%"].idxmax()]
    overall_summary = {
        "LoB": _safe_float(lob),
        "LoD": _safe_float(lod),
        "LoQ": operational_loq,
        "Operational LoQ CV%": _safe_float(operational_row.get("CV%", np.nan)),
        "Worst Performing Level": _safe_float(worst_row["Concentration Level"]),
        "Worst CV%": _safe_float(worst_row["CV%"]),
        "Blank Replicates": int(blank_data["Observed Result"].count()),
        "Low-Level Replicates": int(low_data["Observed Result"].count()),
        "Quantitation Levels Tested": int(loq_summary["Concentration Level"].nunique()),
        "Target LoQ CV%": float(target_loq_cv),
    }
    return lob_summary, lod_summary, loq_summary, overall_summary


def evaluate_detection_capability_criteria(
    overall_summary: dict[str, float | str],
    max_lob: float,
    max_lod: float,
    target_loq_cv: float,
    max_loq_concentration: float,
    borderline_zone: float,
) -> dict[str, object]:
    """Evaluate preliminary detection capability acceptance criteria."""

    def max_check(label: str, observed: float, limit: float, suffix: str = "") -> dict[str, object]:
        zone = borderline_zone if suffix == "%" else limit * (borderline_zone / 100)
        raw_status = _max_threshold_status(observed, limit, zone)
        status = "BORDERLINE" if raw_status == "PASS WITH CAUTION" else raw_status
        margin_percent = ((limit - observed) / limit) * 100 if limit else np.nan
        if pd.isna(observed):
            interpretation = "Observed value could not be calculated."
            margin_text = "Not available"
        elif observed <= limit:
            margin_text = f"{margin_percent:.1f}% below limit"
            interpretation = f"Observed value is {margin_text}."
        else:
            margin_text = f"{abs(margin_percent):.1f}% above limit"
            interpretation = f"Observed value exceeds the acceptance limit by {margin_text}."
        return {
            "Criterion": label,
            "Observed": observed,
            "Acceptance Limit": f"<= {limit:g}{suffix}",
            "Margin to Limit": margin_text,
            "Scientific Interpretation": interpretation,
            "Met": status == "PASS",
            "Borderline": status == "BORDERLINE",
        }

    checks = [
        max_check("Maximum acceptable LoB", overall_summary.get("LoB", np.nan), max_lob),
        max_check("Maximum acceptable LoD", overall_summary.get("LoD", np.nan), max_lod),
        max_check("Target LoQ CV%", overall_summary.get("Operational LoQ CV%", np.nan), target_loq_cv, "%"),
        max_check(
            "Maximum acceptable LoQ concentration",
            overall_summary.get("LoQ", np.nan),
            max_loq_concentration,
        ),
    ]

    if all(check["Met"] for check in checks):
        decision = "PASS"
    elif all(check["Met"] or check["Borderline"] for check in checks):
        decision = "BORDERLINE"
    else:
        decision = "FAIL"
    return {"decision": decision, "checks": checks}


def run_detection_capability_study(
    data: pd.DataFrame,
    sample_id_column: str,
    sample_type_column: str,
    concentration_column: str,
    result_column: str,
    replicate_column: str | None = None,
    units_column: str | None = None,
    include_column: str | None = None,
    target_loq_cv: float = 20.0,
) -> DetectionCapabilityResult:
    """Run the complete detection capability workflow."""

    analyzed_data = prepare_detection_capability_data(
        data=data,
        sample_id_column=sample_id_column,
        sample_type_column=sample_type_column,
        concentration_column=concentration_column,
        result_column=result_column,
        replicate_column=replicate_column,
        units_column=units_column,
        include_column=include_column,
    )
    lob_summary, lod_summary, loq_summary, overall_summary = (
        calculate_detection_capability_summary(analyzed_data, target_loq_cv)
    )
    methodology_table = build_detection_methodology_table(
        lob_summary, lod_summary, loq_summary, overall_summary
    )
    selected_columns = [
        sample_id_column,
        sample_type_column,
        concentration_column,
        result_column,
    ]
    selected_columns.extend(
        [column for column in [replicate_column, units_column, include_column] if column]
    )
    data_quality_summary, outlier_table = assess_detection_data_quality(
        data, analyzed_data, selected_columns
    )
    return DetectionCapabilityResult(
        analyzed_data=analyzed_data,
        lob_summary=lob_summary,
        lod_summary=lod_summary,
        loq_summary=loq_summary,
        methodology_table=methodology_table,
        data_quality_summary=data_quality_summary,
        outlier_table=outlier_table,
        decision_matrix=pd.DataFrame(),
        overall_summary=overall_summary,
        sample_id_column=sample_id_column,
        sample_type_column=sample_type_column,
        concentration_column=concentration_column,
        result_column=result_column,
        replicate_column=replicate_column,
        units_column=units_column,
        include_column=include_column,
    )


def prepare_dbs_validation_data(
    data: pd.DataFrame,
    sample_id_column: str,
    reference_column: str,
    dbs_column: str,
    include_column: str | None = None,
) -> pd.DataFrame:
    """Clean paired DBS and reference specimen results."""

    selected_columns = [sample_id_column, reference_column, dbs_column]
    if include_column:
        selected_columns.append(include_column)
    optional_columns = [
        column
        for column in [
            "Collection Date",
            "Extraction Date",
            "Hematocrit",
            "Replicate",
            "Instrument",
        ]
        if column in data.columns and column not in selected_columns
    ]
    selected_columns.extend(optional_columns)
    analyzed = data[selected_columns].copy()
    if include_column:
        include_mask = analyzed[include_column].astype(str).str.strip().str.lower()
        analyzed = analyzed[include_mask.isin(["yes", "y", "true", "1", "include"])]
    analyzed = analyzed.rename(
        columns={
            sample_id_column: "Sample ID",
            reference_column: "Reference Result",
            dbs_column: "DBS Result",
        }
    )
    analyzed["Reference Result"] = pd.to_numeric(
        analyzed["Reference Result"], errors="coerce"
    )
    analyzed["DBS Result"] = pd.to_numeric(analyzed["DBS Result"], errors="coerce")
    analyzed = analyzed.dropna(subset=["Reference Result", "DBS Result"]).reset_index(
        drop=True
    )
    analyzed["Difference"] = analyzed["DBS Result"] - analyzed["Reference Result"]
    analyzed["Mean of Methods"] = (
        analyzed["DBS Result"] + analyzed["Reference Result"]
    ) / 2
    analyzed["Bias"] = analyzed["Difference"]
    analyzed["Absolute Bias"] = analyzed["Difference"].abs()
    analyzed["Percent Bias"] = np.where(
        analyzed["Reference Result"] != 0,
        (analyzed["Difference"] / analyzed["Reference Result"]) * 100,
        np.nan,
    )
    analyzed["Percent Bias"] = analyzed["Percent Bias"].replace(
        [np.inf, -np.inf], np.nan
    )
    analyzed["Absolute Percent Bias"] = analyzed["Percent Bias"].abs()
    analyzed["Recovery %"] = np.where(
        analyzed["Reference Result"] != 0,
        (analyzed["DBS Result"] / analyzed["Reference Result"]) * 100,
        np.nan,
    )
    analyzed["Recovery %"] = analyzed["Recovery %"].replace([np.inf, -np.inf], np.nan)
    if "Hematocrit" in analyzed.columns:
        analyzed["Hematocrit"] = pd.to_numeric(analyzed["Hematocrit"], errors="coerce")
    if {"Collection Date", "Extraction Date"}.issubset(analyzed.columns):
        collection_date = pd.to_datetime(analyzed["Collection Date"], errors="coerce")
        extraction_date = pd.to_datetime(analyzed["Extraction Date"], errors="coerce")
        analyzed["Extraction Delay (days)"] = (
            extraction_date - collection_date
        ).dt.days
        analyzed["Delay Category"] = pd.cut(
            analyzed["Extraction Delay (days)"],
            bins=[-1, 0, 1, 3, np.inf],
            labels=["Same Day", "1 Day", "2-3 Days", "4+ Days"],
        ).astype("object")
    return analyzed


def calculate_dbs_validation_summary(
    analyzed_data: pd.DataFrame,
    max_percent_bias: float,
    min_recovery: float,
    max_recovery: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float | str]]:
    """Calculate DBS validation summary, bias, recovery, regression, and agreement."""

    if analyzed_data.empty:
        raise ValueError("No valid paired DBS/reference rows are available for analysis.")

    reference = analyzed_data["Reference Result"]
    dbs = analyzed_data["DBS Result"]
    difference = analyzed_data["Difference"]
    percent_bias = analyzed_data["Percent Bias"].dropna()
    recovery = analyzed_data["Recovery %"].dropna()
    n = int(len(analyzed_data))

    if n >= 2 and reference.nunique() > 1 and dbs.nunique() > 1:
        correlation_r = float(np.corrcoef(reference, dbs)[0, 1])
        slope, intercept = np.polyfit(reference, dbs, 1)
    else:
        correlation_r = np.nan
        slope = np.nan
        intercept = np.nan
    r_squared = correlation_r**2 if not pd.isna(correlation_r) else np.nan

    sd_difference = difference.std(ddof=1)
    mean_difference = difference.mean()
    upper_loa = mean_difference + (1.96 * sd_difference)
    lower_loa = mean_difference - (1.96 * sd_difference)

    worst_bias_row = analyzed_data.sort_values(
        "Absolute Percent Bias", ascending=False
    ).iloc[0]
    lowest_recovery_row = analyzed_data.sort_values("Recovery %", ascending=True).iloc[0]
    highest_recovery_row = analyzed_data.sort_values(
        "Recovery %", ascending=False
    ).iloc[0]
    sample_review = analyzed_data.copy()
    sample_review["Potential Outlier"] = np.where(
        (sample_review["Absolute Percent Bias"] > max_percent_bias)
        | (sample_review["Recovery %"] < min_recovery)
        | (sample_review["Recovery %"] > max_recovery),
        "YES",
        "NO",
    )
    sample_review["Outlier Reason"] = np.select(
        [
            sample_review["Absolute Percent Bias"] > max_percent_bias,
            sample_review["Recovery %"] < min_recovery,
            sample_review["Recovery %"] > max_recovery,
        ],
        ["High percent bias", "Low recovery", "High recovery"],
        default="Within criteria",
    )

    overall = {
        "N": n,
        "Mean Reference": _safe_float(reference.mean()),
        "Mean DBS": _safe_float(dbs.mean()),
        "Mean Bias": _safe_float(difference.mean()),
        "Median Bias": _safe_float(difference.median()),
        "Mean Absolute Bias": _safe_float(difference.abs().mean()),
        "Mean Percent Bias": _safe_float(percent_bias.mean()),
        "Median Percent Bias": _safe_float(percent_bias.median()),
        "Mean Absolute Percent Bias": _safe_float(percent_bias.abs().mean()),
        "Maximum Absolute Percent Bias": _safe_float(percent_bias.abs().max()),
        "Mean Recovery": _safe_float(recovery.mean()),
        "Median Recovery": _safe_float(recovery.median()),
        "Minimum Recovery": _safe_float(recovery.min()),
        "Maximum Recovery": _safe_float(recovery.max()),
        "Correlation r": _safe_float(correlation_r),
        "R²": _safe_float(r_squared),
        "Slope": _safe_float(slope),
        "Intercept": _safe_float(intercept),
        "Mean Difference": _safe_float(mean_difference),
        "SD Difference": _safe_float(sd_difference),
        "Lower Limit of Agreement": _safe_float(lower_loa),
        "Upper Limit of Agreement": _safe_float(upper_loa),
        "Highest Bias Sample": str(worst_bias_row["Sample ID"]),
        "Lowest Recovery Sample": str(lowest_recovery_row["Sample ID"]),
        "Highest Recovery Sample": str(highest_recovery_row["Sample ID"]),
        "Worst Sample": str(worst_bias_row["Sample ID"]),
    }

    study_summary = pd.DataFrame(
        [
            {"Metric": "N", "Value": overall["N"]},
            {"Metric": "Mean Reference", "Value": overall["Mean Reference"]},
            {"Metric": "Mean DBS", "Value": overall["Mean DBS"]},
            {"Metric": "Mean Bias", "Value": overall["Mean Bias"]},
            {"Metric": "Mean Recovery", "Value": overall["Mean Recovery"]},
            {"Metric": "R²", "Value": overall["R²"]},
        ]
    )
    bias_summary = pd.DataFrame(
        [
            {"Metric": "Mean Bias", "Value": overall["Mean Bias"]},
            {"Metric": "Median Bias", "Value": overall["Median Bias"]},
            {"Metric": "Mean Absolute Bias", "Value": overall["Mean Absolute Bias"]},
            {"Metric": "Mean Percent Bias", "Value": overall["Mean Percent Bias"]},
            {"Metric": "Median Percent Bias", "Value": overall["Median Percent Bias"]},
            {"Metric": "Maximum Absolute Percent Bias", "Value": overall["Maximum Absolute Percent Bias"]},
        ]
    )
    recovery_summary = pd.DataFrame(
        [
            {"Metric": "Mean Recovery", "Value": overall["Mean Recovery"]},
            {"Metric": "Median Recovery", "Value": overall["Median Recovery"]},
            {"Metric": "Minimum Recovery", "Value": overall["Minimum Recovery"]},
            {"Metric": "Maximum Recovery", "Value": overall["Maximum Recovery"]},
        ]
    )
    correlation_summary = pd.DataFrame(
        [
            {"Metric": "Pearson Correlation r", "Value": overall["Correlation r"]},
            {"Metric": "R²", "Value": overall["R²"]},
            {"Metric": "Slope", "Value": overall["Slope"]},
            {"Metric": "Intercept", "Value": overall["Intercept"]},
        ]
    )
    agreement_summary = pd.DataFrame(
        [
            {"Metric": "Mean Difference", "Value": overall["Mean Difference"]},
            {"Metric": "SD Difference", "Value": overall["SD Difference"]},
            {"Metric": "Lower 95% Limit of Agreement", "Value": overall["Lower Limit of Agreement"]},
            {"Metric": "Upper 95% Limit of Agreement", "Value": overall["Upper Limit of Agreement"]},
        ]
    )
    return (
        study_summary,
        bias_summary,
        recovery_summary,
        correlation_summary,
        agreement_summary,
        sample_review.sort_values("Absolute Percent Bias", ascending=False),
        overall,
    )


def evaluate_dbs_criteria(
    overall_summary: dict[str, float | str],
    max_percent_bias: float,
    min_recovery: float,
    max_recovery: float,
    min_r_squared: float,
    max_mean_difference: float,
    borderline_zone: float,
) -> dict[str, object]:
    """Evaluate preliminary DBS validation criteria."""

    checks = []
    max_bias = float(overall_summary.get("Maximum Absolute Percent Bias", np.nan))
    bias_status = _max_threshold_status(max_bias, max_percent_bias, borderline_zone)
    checks.append(
        {
            "Criterion": "Maximum absolute percent bias",
            "Observed": max_bias,
            "Acceptance Limit": f"<= {max_percent_bias:g}%",
            "Met": bias_status == "PASS",
            "Borderline": bias_status == "PASS WITH CAUTION",
        }
    )

    min_observed_recovery = float(overall_summary.get("Minimum Recovery", np.nan))
    min_recovery_status = _min_threshold_status(
        min_observed_recovery, min_recovery, borderline_zone
    )
    checks.append(
        {
            "Criterion": "Minimum recovery",
            "Observed": min_observed_recovery,
            "Acceptance Limit": f">= {min_recovery:g}%",
            "Met": min_recovery_status == "PASS",
            "Borderline": min_recovery_status == "PASS WITH CAUTION",
        }
    )

    max_observed_recovery = float(overall_summary.get("Maximum Recovery", np.nan))
    max_recovery_status = _max_threshold_status(
        max_observed_recovery, max_recovery, borderline_zone
    )
    checks.append(
        {
            "Criterion": "Maximum recovery",
            "Observed": max_observed_recovery,
            "Acceptance Limit": f"<= {max_recovery:g}%",
            "Met": max_recovery_status == "PASS",
            "Borderline": max_recovery_status == "PASS WITH CAUTION",
        }
    )

    r_squared = float(overall_summary.get("R²", np.nan))
    r_status = "PASS" if not pd.isna(r_squared) and r_squared >= min_r_squared else (
        "PASS WITH CAUTION"
        if not pd.isna(r_squared) and r_squared >= min_r_squared - 0.02
        else "FAIL"
    )
    checks.append(
        {
            "Criterion": "R²",
            "Observed": r_squared,
            "Acceptance Limit": f">= {min_r_squared:g}",
            "Met": r_status == "PASS",
            "Borderline": r_status == "PASS WITH CAUTION",
        }
    )

    mean_difference = abs(float(overall_summary.get("Mean Difference", np.nan)))
    difference_zone = max_mean_difference * (borderline_zone / 100)
    diff_status = _max_threshold_status(
        mean_difference, max_mean_difference, difference_zone
    )
    checks.append(
        {
            "Criterion": "Absolute mean difference",
            "Observed": mean_difference,
            "Acceptance Limit": f"<= {max_mean_difference:g}",
            "Met": diff_status == "PASS",
            "Borderline": diff_status == "PASS WITH CAUTION",
        }
    )

    if any(not check["Met"] and not check["Borderline"] for check in checks):
        decision = "FAIL"
    elif any(check["Borderline"] for check in checks):
        decision = "PASS WITH CAUTION"
    else:
        decision = "PASS"
    return {"checks": checks, "decision": decision}


def _correlation_regression_summary(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    label: str,
) -> dict[str, float | str]:
    """Calculate simple correlation/regression summary for optional DBS factors."""

    subset = data[[x_column, y_column]].dropna()
    if len(subset) >= 2 and subset[x_column].nunique() > 1 and subset[y_column].nunique() > 1:
        correlation = float(np.corrcoef(subset[x_column], subset[y_column])[0, 1])
        slope, intercept = np.polyfit(subset[x_column], subset[y_column], 1)
    else:
        correlation = np.nan
        slope = np.nan
        intercept = np.nan
    r_squared = correlation**2 if not pd.isna(correlation) else np.nan
    return {
        "Assessment": label,
        "N": int(len(subset)),
        "Correlation r": _safe_float(correlation),
        "R²": _safe_float(r_squared),
        "Slope": _safe_float(slope),
        "Intercept": _safe_float(intercept),
    }


def _effect_status(correlation: float) -> str:
    """Classify optional-factor effects using correlation magnitude."""

    if pd.isna(correlation):
        return "NOT AVAILABLE"
    magnitude = abs(float(correlation))
    if magnitude < 0.30:
        return "PASS"
    if magnitude < 0.50:
        return "REVIEW"
    return "INVESTIGATE"


def calculate_dbs_enhancement_assessments(
    analyzed_data: pd.DataFrame,
    sample_review: pd.DataFrame,
    overall_summary: dict[str, float | str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Calculate optional DBS investigation tables and reviewer-style findings."""

    findings: list[str] = []

    if "Hematocrit" in analyzed_data.columns:
        hematocrit_rows = [
            _correlation_regression_summary(
                analyzed_data, "Hematocrit", "Bias", "Hematocrit vs Bias"
            ),
            _correlation_regression_summary(
                analyzed_data, "Hematocrit", "Percent Bias", "Hematocrit vs Percent Bias"
            ),
        ]
        hematocrit_summary = pd.DataFrame(hematocrit_rows)
        hematocrit_summary["Status"] = hematocrit_summary["Correlation r"].map(_effect_status)
        strongest_hct = hematocrit_summary["Correlation r"].abs().max()
        if pd.isna(strongest_hct):
            findings.append("Hematocrit impact could not be assessed because insufficient numeric hematocrit data were available.")
        elif strongest_hct < 0.30:
            findings.append("No meaningful hematocrit-associated bias trend was observed.")
        elif strongest_hct < 0.50:
            findings.append("Mild hematocrit-associated bias was detected and should be reviewed in context of specimen equivalency.")
        else:
            findings.append("A significant hematocrit effect may impact DBS specimen equivalency and should be investigated.")
    else:
        hematocrit_summary = pd.DataFrame()
        findings.append("Hematocrit impact assessment was not performed because no Hematocrit column was available.")

    if "Extraction Delay (days)" in analyzed_data.columns:
        delay_rows = [
            _correlation_regression_summary(
                analyzed_data, "Extraction Delay (days)", "Bias", "Delay vs Bias"
            ),
            _correlation_regression_summary(
                analyzed_data, "Extraction Delay (days)", "Recovery %", "Delay vs Recovery"
            ),
            _correlation_regression_summary(
                analyzed_data, "Extraction Delay (days)", "Percent Bias", "Delay vs Percent Bias"
            ),
        ]
        delay_summary = pd.DataFrame(delay_rows)
        delay_summary["Status"] = delay_summary["Correlation r"].map(_effect_status)
        if "Delay Category" in analyzed_data.columns:
            delay_category_summary = (
                analyzed_data.groupby("Delay Category", dropna=False)
                .agg(
                    N=("Sample ID", "count"),
                    **{
                        "Mean Bias": ("Bias", "mean"),
                        "Mean Recovery": ("Recovery %", "mean"),
                        "Mean Percent Bias": ("Percent Bias", "mean"),
                    },
                )
                .reset_index()
            )
            delay_summary = pd.concat(
                [
                    delay_summary,
                    delay_category_summary.assign(
                        Assessment=lambda frame: "Average Bias by Delay Category: "
                        + frame["Delay Category"].astype(str),
                        **{"Correlation r": np.nan, "R²": np.nan, "Slope": np.nan, "Intercept": np.nan, "Status": "SUMMARY"},
                    )[
                        [
                            "Assessment",
                            "N",
                            "Correlation r",
                            "R²",
                            "Slope",
                            "Intercept",
                            "Status",
                            "Mean Bias",
                            "Mean Recovery",
                            "Mean Percent Bias",
                        ]
                    ],
                ],
                ignore_index=True,
            )
        strongest_delay = delay_summary.loc[
            delay_summary["Status"].isin(["PASS", "REVIEW", "INVESTIGATE"]),
            "Correlation r",
        ].abs().max()
        if pd.isna(strongest_delay):
            findings.append("Extraction delay assessment could not be completed because insufficient paired date data were available.")
        elif strongest_delay < 0.30:
            findings.append("No delay-associated degradation or bias trend was observed.")
        elif strongest_delay < 0.50:
            findings.append("A mild extraction-delay trend was observed and should be monitored in future DBS studies.")
        else:
            findings.append("Increased bias or recovery shift was observed with extraction delay; prolonged delays should be investigated.")
    else:
        delay_summary = pd.DataFrame()
        findings.append("Extraction delay assessment was not performed because Collection Date and Extraction Date were not both available.")

    if "Instrument" in analyzed_data.columns:
        instrument_rows = []
        for instrument, group in analyzed_data.groupby("Instrument", dropna=False):
            if len(group) >= 2 and group["Reference Result"].nunique() > 1 and group["DBS Result"].nunique() > 1:
                r_value = float(np.corrcoef(group["Reference Result"], group["DBS Result"])[0, 1])
            else:
                r_value = np.nan
            instrument_rows.append(
                {
                    "Instrument": instrument if pd.notna(instrument) else "Not specified",
                    "Sample Count": int(len(group)),
                    "Mean Bias": _safe_float(group["Bias"].mean()),
                    "Mean Recovery": _safe_float(group["Recovery %"].mean()),
                    "R²": _safe_float(r_value**2 if not pd.isna(r_value) else np.nan),
                }
            )
        instrument_summary = pd.DataFrame(instrument_rows)
        if len(instrument_summary) <= 1:
            findings.append("Instrument comparison was limited because only one instrument was represented.")
        else:
            recovery_range = instrument_summary["Mean Recovery"].max() - instrument_summary["Mean Recovery"].min()
            bias_range = instrument_summary["Mean Bias"].max() - instrument_summary["Mean Bias"].min()
            if abs(recovery_range) < 3 and abs(bias_range) < 0.3:
                findings.append("No meaningful instrument-associated performance difference was observed.")
            else:
                findings.append("Instrument-level differences were observed and should be reviewed for potential analytical or workflow effects.")
    else:
        instrument_summary = pd.DataFrame()
        findings.append("Instrument comparison was not performed because no Instrument column was available.")

    lower_loa = overall_summary.get("Lower Limit of Agreement", np.nan)
    upper_loa = overall_summary.get("Upper Limit of Agreement", np.nan)
    outlier_review = sample_review.copy()
    if not pd.isna(lower_loa) and not pd.isna(upper_loa):
        outlier_review["Outside Bland-Altman Limits"] = np.where(
            (outlier_review["Difference"] < float(lower_loa))
            | (outlier_review["Difference"] > float(upper_loa)),
            "YES",
            "NO",
        )
    else:
        outlier_review["Outside Bland-Altman Limits"] = "NO"
    outlier_review["Borderline Sample"] = np.where(
        outlier_review["Potential Outlier"].eq("NO")
        & (
            outlier_review["Absolute Percent Bias"]
            >= outlier_review["Absolute Percent Bias"].quantile(0.85)
        ),
        "YES",
        "NO",
    )
    outlier_review["Acceptance Status"] = np.select(
        [
            outlier_review["Potential Outlier"].eq("YES"),
            outlier_review["Outside Bland-Altman Limits"].eq("YES"),
            outlier_review["Borderline Sample"].eq("YES"),
        ],
        ["FAIL", "FAIL", "REVIEW"],
        default="PASS",
    )
    outlier_columns = [
        column
        for column in [
            "Sample ID",
            "Reference Result",
            "DBS Result",
            "Difference",
            "Percent Bias",
            "Hematocrit",
            "Extraction Delay (days)",
            "Instrument",
            "Potential Outlier",
            "Outside Bland-Altman Limits",
            "Borderline Sample",
            "Acceptance Status",
            "Outlier Reason",
        ]
        if column in outlier_review.columns
    ]
    outlier_review = (
        outlier_review.sort_values("Absolute Percent Bias", ascending=False)
        .head(10)[outlier_columns]
        .reset_index(drop=True)
    )
    if outlier_review.empty:
        findings.append("No sample-level outlier patterns were identified.")
    else:
        worst = outlier_review.iloc[0]
        context = []
        if "Hematocrit" in worst and pd.notna(worst["Hematocrit"]):
            context.append(f"hematocrit {worst['Hematocrit']:.1f}")
        if "Extraction Delay (days)" in worst and pd.notna(worst["Extraction Delay (days)"]):
            context.append(f"extraction delay {int(worst['Extraction Delay (days)'])} day(s)")
        context_text = "; ".join(context) if context else "no unusual optional specimen factor was available"
        findings.append(
            f"Sample {worst['Sample ID']} exhibited the largest observed bias; {context_text}."
        )

    if abs(float(overall_summary.get("Mean Percent Bias", 0))) < 5 and float(overall_summary.get("R²", 0)) >= 0.95:
        findings.append("Results suggest acceptable DBS equivalency under the evaluated study conditions.")
    else:
        findings.append("Further investigation is recommended before concluding DBS equivalency.")

    return hematocrit_summary, delay_summary, instrument_summary, outlier_review, findings


def run_dbs_validation_study(
    data: pd.DataFrame,
    sample_id_column: str,
    reference_column: str,
    dbs_column: str,
    include_column: str | None = None,
    max_percent_bias: float = 10.0,
    min_recovery: float = 90.0,
    max_recovery: float = 110.0,
) -> DBSValidationResult:
    """Run a DBS-versus-reference specimen validation analysis."""

    analyzed_data = prepare_dbs_validation_data(
        data, sample_id_column, reference_column, dbs_column, include_column
    )
    (
        study_summary,
        bias_summary,
        recovery_summary,
        correlation_summary,
        agreement_summary,
        sample_review,
        overall_summary,
    ) = calculate_dbs_validation_summary(
        analyzed_data, max_percent_bias, min_recovery, max_recovery
    )
    (
        hematocrit_summary,
        delay_summary,
        instrument_summary,
        outlier_review,
        scientific_findings,
    ) = calculate_dbs_enhancement_assessments(
        analyzed_data, sample_review, overall_summary
    )
    return DBSValidationResult(
        analyzed_data=analyzed_data,
        study_summary=study_summary,
        bias_summary=bias_summary,
        recovery_summary=recovery_summary,
        correlation_summary=correlation_summary,
        agreement_summary=agreement_summary,
        hematocrit_summary=hematocrit_summary,
        delay_summary=delay_summary,
        instrument_summary=instrument_summary,
        outlier_review=outlier_review,
        scientific_findings=scientific_findings,
        sample_review=sample_review,
        overall_summary=overall_summary,
        sample_id_column=sample_id_column,
        reference_column=reference_column,
        dbs_column=dbs_column,
        include_column=include_column,
    )
