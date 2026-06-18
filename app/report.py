"""Report and interpretation helpers for validation analytics outputs."""

from __future__ import annotations

from datetime import datetime
from html import escape
from io import BytesIO

import pandas as pd
from fpdf import FPDF


SUMMARY_COLUMNS = [
    "N",
    "Mean Reference",
    "Mean Candidate",
    "Mean Difference",
    "Mean Bias",
    "SD Difference",
    "Lower Limit of Agreement",
    "Upper Limit of Agreement",
    "Mean Percent Bias",
    "SD Percent Bias",
    "Correlation r",
    "R²",
    "Slope",
    "Intercept",
]


def build_summary_table(summary: dict[str, float]) -> pd.DataFrame:
    """Create a one-row summary table for display and export."""

    return pd.DataFrame([{column: summary.get(column) for column in SUMMARY_COLUMNS}])


def build_criteria_table(criteria_result: dict[str, object]) -> pd.DataFrame:
    """Create a table showing user-defined preliminary criteria results."""

    criteria_table = pd.DataFrame(criteria_result["checks"])
    criteria_table["Status"] = criteria_table.apply(
        lambda row: "Pass"
        if row["Met"]
        else "Borderline"
        if row["Borderline"]
        else "Fail",
        axis=1,
    )
    criteria_table = criteria_table.rename(
        columns={
            "Observed": "Observed Value",
            "Status": "Pass/Fail Status",
            "Borderline": "Borderline Status",
        }
    )
    columns = [
        "Criterion",
        "Observed Value",
        "Acceptance Limit",
        "Pass/Fail Status",
        "Borderline Status",
    ]
    if "Classification" in criteria_table.columns:
        columns.insert(3, "Classification")
    return criteria_table[columns]


def format_precision_summary_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Format precision summary values for display and export."""

    formatted = summary.copy()
    if "N" in formatted.columns:
        formatted["N"] = formatted["N"].map(lambda value: f"{int(value)}")
    if "Mean" in formatted.columns:
        formatted["Mean"] = formatted["Mean"].map(
            lambda value: "" if pd.isna(value) else f"{value:.2f}"
        )
    if "SD" in formatted.columns:
        formatted["SD"] = formatted["SD"].map(
            lambda value: "" if pd.isna(value) else f"{value:.3f}"
        )
    if "CV%" in formatted.columns:
        formatted["CV%"] = formatted["CV%"].map(
            lambda value: "" if pd.isna(value) else f"{value:.2f}%"
        )
    return formatted


def format_precision_criteria_table(criteria_table: pd.DataFrame) -> pd.DataFrame:
    """Format precision acceptance criteria results for display and export."""

    formatted = criteria_table.copy()
    if "Observed Value" in formatted.columns:
        formatted["Observed Value"] = formatted["Observed Value"].map(
            lambda value: "" if pd.isna(value) else f"{value:.2f}%"
        )
    if "Acceptance Limit" in formatted.columns:
        formatted["Acceptance Limit"] = formatted["Acceptance Limit"].str.replace(
            r"<= ([0-9.]+)%",
            lambda match: f"<= {float(match.group(1)):.2f}%",
            regex=True,
        )
    if "Pass/Fail Status" in formatted.columns:
        formatted["Pass/Fail Status"] = formatted["Pass/Fail Status"].str.upper()
    if "Borderline Status" in formatted.columns:
        formatted["Borderline Status"] = formatted["Borderline Status"].map(
            lambda value: "YES" if bool(value) else "NO"
        )
    return formatted


def format_linearity_summary_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Format linearity level summary values for display and export."""

    formatted = summary.copy()
    if "N" in formatted.columns:
        formatted["N"] = formatted["N"].map(lambda value: f"{int(value)}")
    for column in ["Expected Result", "Mean Observed Result", "Difference"]:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.2f}"
            )
    for column in ["Percent Recovery", "Percent Bias"]:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.2f}%"
            )
    return formatted


def format_linearity_regression_summary(summary: dict[str, float | str]) -> pd.DataFrame:
    """Format linearity regression summary as a two-column table."""

    rows = [
        ("Slope", f"{summary['Slope']:.4f}"),
        ("Intercept", f"{summary['Intercept']:.4f}"),
        ("Correlation r", f"{summary['Correlation r']:.4f}"),
        ("R²", f"{summary['R²']:.4f}"),
        ("Minimum Expected Result", f"{summary['Minimum Expected Result']:.2f}"),
        ("Maximum Expected Result", f"{summary['Maximum Expected Result']:.2f}"),
        ("Analytical Range Tested", str(summary["Analytical Range Tested"])),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def format_linearity_criteria_table(criteria_table: pd.DataFrame) -> pd.DataFrame:
    """Format linearity acceptance criteria result values."""

    formatted = criteria_table.copy()

    def format_observed(row: pd.Series) -> str:
        value = row["Observed Value"]
        if isinstance(value, str):
            return value
        if pd.isna(value):
            return ""
        if "bias" in str(row["Criterion"]).lower():
            return f"{value:.2f}%"
        return f"{value:.4f}"

    formatted["Observed Value"] = formatted.apply(format_observed, axis=1)
    if "Pass/Fail Status" in formatted.columns:
        formatted["Pass/Fail Status"] = formatted["Pass/Fail Status"].str.upper()
        formatted["Pass/Fail Status"] = formatted["Pass/Fail Status"].replace(
            {"BORDERLINE": "PASS WITH CAUTION"}
        )
    if "Borderline Status" in formatted.columns:
        formatted["Borderline Status"] = formatted["Borderline Status"].map(
            lambda value: "YES" if bool(value) else "NO"
        )
    return formatted


def format_stability_table(table: pd.DataFrame) -> pd.DataFrame:
    """Format stability summary tables for display and export."""

    formatted = table.copy()
    if "N" in formatted.columns:
        formatted["N"] = formatted["N"].map(lambda value: f"{int(value)}")
    two_decimal_columns = [
        "Baseline Mean",
        "Timepoint Mean",
        "Absolute Difference",
        "Mean Difference",
        "Mean Absolute Difference",
        "Bias",
        "Maximum Absolute Bias",
    ]
    for column in two_decimal_columns:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.2f}"
            )
    percent_columns = [
        "Mean Percent Change",
        "Mean Absolute Percent Change",
        "Percent Recovery",
        "Minimum Recovery",
        "Maximum Recovery",
    ]
    for column in percent_columns:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.2f}%"
            )
    return formatted


def format_stability_overall_summary(summary: dict[str, float | str]) -> pd.DataFrame:
    """Format overall stability summary as a two-column table."""

    rows = [
        ("N", f"{int(summary['N'])}"),
        ("Baseline Mean", f"{summary['Baseline Mean']:.2f}"),
        ("Maximum Observed Change", f"{summary['Maximum Observed Change']:.2f}%"),
        ("Maximum Absolute Bias", f"{summary['Maximum Absolute Bias']:.2f}"),
        ("Minimum Recovery", f"{summary['Minimum Recovery']:.2f}%"),
        ("Maximum Recovery", f"{summary['Maximum Recovery']:.2f}%"),
        ("Worst Timepoint", str(summary["Worst Timepoint"])),
        ("Worst Sample ID", str(summary["Worst Sample ID"])),
        ("Average Change by Timepoint", f"{summary['Average Change by Timepoint']:.2f}%"),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def format_stability_criteria_table(criteria_table: pd.DataFrame) -> pd.DataFrame:
    """Format stability acceptance criteria result values."""

    formatted = criteria_table.copy()

    def format_observed(row: pd.Series) -> str:
        value = row["Observed Value"]
        if pd.isna(value):
            return ""
        criterion = str(row["Criterion"]).lower()
        if "recovery" in criterion or "percent change" in criterion:
            return f"{value:.2f}%"
        return f"{value:.2f}"

    formatted["Observed Value"] = formatted.apply(format_observed, axis=1)
    if "Pass/Fail Status" in formatted.columns:
        formatted["Pass/Fail Status"] = formatted["Pass/Fail Status"].str.upper()
        formatted["Pass/Fail Status"] = formatted["Pass/Fail Status"].replace(
            {"BORDERLINE": "PASS WITH CAUTION"}
        )
    if "Borderline Status" in formatted.columns:
        formatted["Borderline Status"] = formatted["Borderline Status"].map(
            lambda value: "YES" if bool(value) else "NO"
        )
    return formatted


def format_storage_condition_comparison_table(table: pd.DataFrame) -> pd.DataFrame:
    """Format storage condition comparison values."""

    formatted = table.copy()
    for column in formatted.columns:
        if column == "Timepoint" or column == "Comparison":
            continue
        if pd.api.types.is_numeric_dtype(formatted[column]):
            suffix = "%" if "Recovery" in column or "Percent Change" in column else ""
            formatted[column] = formatted[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.2f}{suffix}"
            )
    return formatted


def format_stability_outlier_table(table: pd.DataFrame) -> pd.DataFrame:
    """Format potential stability outlier rows."""

    formatted = table.copy()
    for column in ["Percent Change", "Percent Recovery"]:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.2f}%"
            )
    for column in ["Bias", "Largest Absolute Change", "Largest Bias", "Lowest Recovery", "Severity Score"]:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.2f}"
            )
    if "Potential Outlier" in formatted.columns:
        formatted["Potential Outlier"] = formatted["Potential Outlier"].map(
            lambda value: "YES" if bool(value) else "NO"
        )
    return formatted


def get_status_class(status: str) -> str:
    """Return a CSS class suffix for a validation status value."""

    normalized = str(status).strip().lower()
    if normalized in {"pass", "passed"}:
        return "pass"
    if normalized in {"borderline", "review", "pass with caution"}:
        return "borderline"
    return "fail"


def status_badge_html(status: str) -> str:
    """Render a colored status badge for HTML reports and UI fragments."""

    safe_status = escape(str(status).upper())
    return f'<span class="status-badge status-{get_status_class(status)}">{safe_status}</span>'


def criteria_table_to_badged_html(criteria_table: pd.DataFrame) -> str:
    """Render an acceptance criteria table with colored status badges."""

    rows = []
    for _, row in criteria_table.iterrows():
        cells = []
        for column in criteria_table.columns:
            value = row[column]
            if column == "Pass/Fail Status":
                cell_value = status_badge_html(str(value))
            else:
                cell_value = escape(str(value))
            cells.append(f"<td>{cell_value}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")

    headers = "".join(f"<th>{escape(str(column))}</th>" for column in criteria_table.columns)
    return "<table>" + f"<thead><tr>{headers}</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def get_linearity_worst_case(level_summary: pd.DataFrame) -> pd.Series:
    """Return the level with the largest absolute percent bias."""

    if level_summary.empty:
        return pd.Series(dtype=object)
    return level_summary.loc[level_summary["Percent Bias"].abs().idxmax()]


def build_linearity_executive_summary(
    level_summary: pd.DataFrame,
    regression_summary: dict[str, float | str],
    decision: str,
) -> dict[str, str]:
    """Build display values for the linearity executive summary cards."""

    min_recovery = level_summary["Percent Recovery"].min()
    max_recovery = level_summary["Percent Recovery"].max()
    max_abs_bias = level_summary["Percent Bias"].abs().max()
    return {
        "Overall Decision": decision,
        "R²": f"{regression_summary['R²']:.4f}",
        "Regression Slope": f"{regression_summary['Slope']:.4f}",
        "Maximum Absolute Bias": f"{max_abs_bias:.2f}%",
        "Percent Recovery Range": f"{min_recovery:.1f}% to {max_recovery:.1f}%",
        "Analytical Range Tested": str(regression_summary["Analytical Range Tested"]),
    }


def format_linearity_equation(regression_summary: dict[str, float | str]) -> str:
    """Format the linearity regression equation for display."""

    return (
        f"Observed = {regression_summary['Slope']:.4f} x Expected + "
        f"{regression_summary['Intercept']:.4f}"
    )


def linearity_executive_summary_html(
    level_summary: pd.DataFrame,
    regression_summary: dict[str, float | str],
    decision: str,
) -> str:
    """Render linearity executive summary cards for the HTML report."""

    summary_values = build_linearity_executive_summary(
        level_summary, regression_summary, decision
    )
    cards = []
    for label, value in summary_values.items():
        value_html = status_badge_html(value) if label == "Overall Decision" else escape(value)
        cards.append(
            f"""
            <div class="summary-card">
              <div class="summary-label">{escape(label)}</div>
              <div class="summary-value">{value_html}</div>
            </div>
            """
        )
    return '<div class="summary-grid">' + "".join(cards) + "</div>"


def linearity_equation_html(regression_summary: dict[str, float | str]) -> str:
    """Render a regression equation card for the HTML report."""

    return f"""
    <div class="equation-card">
      <h3>Regression Equation</h3>
      <p class="equation">{escape(format_linearity_equation(regression_summary))}</p>
      <p>R² = {regression_summary['R²']:.4f}</p>
    </div>
    """


def linearity_worst_case_html(level_summary: pd.DataFrame) -> str:
    """Render the worst-performing level summary for the HTML report."""

    worst_case = get_linearity_worst_case(level_summary)
    if worst_case.empty:
        return "<p>No worst-performing level available.</p>"
    return f"""
    <div class="equation-card">
      <h3>Worst Performing Level</h3>
      <p class="equation">{escape(str(worst_case['Level']))}</p>
      <p>
        Absolute Bias: {abs(worst_case['Percent Bias']):.2f}%<br>
        Recovery: {worst_case['Percent Recovery']:.1f}%<br>
        Expected Result: {worst_case['Expected Result']:.2f}<br>
        Mean Observed Result: {worst_case['Mean Observed Result']:.2f}
      </p>
    </div>
    """


def build_stability_executive_summary(
    overall_summary: dict[str, float | str],
    decision: str,
    criteria_table: pd.DataFrame | None = None,
) -> dict[str, str]:
    """Build display values for stability executive summary cards."""

    borderline_count = 0
    failed_count = 0
    if criteria_table is not None and not criteria_table.empty:
        status_values = criteria_table["Pass/Fail Status"].astype(str).str.upper()
        borderline_count = int((status_values == "PASS WITH CAUTION").sum())
        failed_count = int((status_values == "FAIL").sum())

    return {
        "Overall Decision": decision,
        "Maximum Observed Change": f"{overall_summary['Maximum Observed Change']:.2f}%",
        "Minimum Recovery": f"{overall_summary['Minimum Recovery']:.2f}%",
        "Maximum Absolute Bias": f"{overall_summary['Maximum Absolute Bias']:.2f}",
        "Worst Timepoint": str(overall_summary["Worst Timepoint"]),
        "Worst Storage Condition": str(overall_summary.get("Worst Storage Condition", "Not specified")),
        "Borderline Criteria": str(borderline_count),
        "Failed Criteria": str(failed_count),
    }


def stability_executive_summary_html(
    overall_summary: dict[str, float | str],
    decision: str,
    criteria_table: pd.DataFrame | None = None,
) -> str:
    """Render stability executive summary cards for the HTML report."""

    summary_values = build_stability_executive_summary(
        overall_summary, decision, criteria_table
    )
    cards = []
    for label, value in summary_values.items():
        value_html = status_badge_html(value) if label == "Overall Decision" else escape(value)
        cards.append(
            f"""
            <div class="summary-card">
              <div class="summary-label">{escape(label)}</div>
              <div class="summary-value">{value_html}</div>
            </div>
            """
        )
    return '<div class="summary-grid">' + "".join(cards) + "</div>"


def generate_interpretation(
    summary: dict[str, float],
    metadata: dict[str, object],
    criteria: dict[str, float | None],
    decision: str,
) -> str:
    """Generate an informational scientific interpretation paragraph."""

    r_squared = summary.get("R²")
    mean_percent_bias = summary.get("Mean Percent Bias")
    mean_difference = summary.get("Mean Difference")
    study_name = metadata.get("Study Name") or "this validation study"
    biomarker = metadata.get("Assay / Biomarker") or "the selected assay"
    candidate_method = metadata.get("Candidate Method") or "candidate results"
    reference_method = metadata.get("Reference Method") or "the reference method"

    if pd.isna(r_squared):
        agreement_text = "correlation could not be calculated from the selected data"
    elif r_squared >= 0.95:
        agreement_text = "candidate results showed strong agreement with the reference method"
    elif r_squared >= 0.85:
        agreement_text = "candidate results showed moderate agreement with the reference method"
    else:
        agreement_text = "candidate results showed limited agreement with the reference method"

    r2_text = "not available" if pd.isna(r_squared) else f"{r_squared:.3f}"
    bias_text = (
        "not available"
        if pd.isna(mean_percent_bias)
        else f"{mean_percent_bias:.2f}%"
    )
    difference_text = (
        "not available" if pd.isna(mean_difference) else f"{mean_difference:.3f}"
    )
    difference_limit = criteria.get("Maximum Absolute Mean Difference")
    difference_criteria = (
        ""
        if difference_limit is None
        else f" and absolute mean difference within {difference_limit:g}"
    )

    return (
        f"For {study_name}, {agreement_text} for {biomarker}. "
        f"The comparison of {candidate_method} against {reference_method} produced "
        f"R² = {r2_text}, mean percent bias = {bias_text}, and mean difference = "
        f"{difference_text}. User-defined preliminary acceptance criteria were "
        f"R² >= {criteria['Minimum R²']:g}, absolute mean percent bias within "
        f"{criteria['Maximum Absolute Mean Percent Bias']:g}%{difference_criteria}. "
        f"The overall preliminary decision is {decision}. This interpretation is "
        "informational only and does not replace formal laboratory approval."
    )


def generate_precision_interpretation(
    level_summary: pd.DataFrame,
    metadata: dict[str, object],
    max_acceptable_cv: float,
    decision: str,
) -> str:
    """Generate an informational interpretation for precision study results."""

    study_name = metadata.get("Study Name") or "this precision study"
    biomarker = metadata.get("Assay / Biomarker") or "the selected assay"
    cv_parts = []
    classification_groups: dict[str, list[str]] = {}
    for _, row in level_summary.iterrows():
        cv_value = row["CV%"]
        cv_text = "not available" if pd.isna(cv_value) else f"{cv_value:.2f}%"
        level = str(row["Level"])
        classification = str(row.get("Classification", "Not available"))
        cv_parts.append(f"{cv_text} for the {level} QC level")
        classification_groups.setdefault(classification, []).append(level)

    cv_summary = " and ".join(cv_parts)
    classification_parts = []
    for classification, levels in classification_groups.items():
        if len(levels) == len(level_summary) and len(levels) > 1:
            classification_parts.append(f"all levels classified as {classification}")
        elif len(levels) == 2:
            classification_parts.append(
                f"the {levels[0]} and {levels[1]} QC levels classified as {classification}"
            )
        elif len(levels) == 1:
            classification_parts.append(
                f"the {levels[0]} QC level classified as {classification}"
            )
        else:
            classification_parts.append(
                f"{', '.join(levels)} QC levels classified as {classification}"
            )
    classification_text = "; ".join(classification_parts)
    if decision == "PASS":
        decision_text = "the study meets precision acceptance requirements"
    elif decision == "REVIEW":
        decision_text = "the study requires review against the precision acceptance requirements"
    else:
        decision_text = "the study requires investigation before precision acceptance"

    return (
        f"For {study_name}, the precision study included repeated measurements across multiple days, "
        f"runs, and replicate observations for {biomarker}. Observed CV values were "
        f"{cv_summary}, with {classification_text} precision performance. All observed "
        f"CV values were evaluated against the user-defined acceptance threshold of "
        f"{max_acceptable_cv:.2f}%. Based on the preliminary screening criteria, "
        f"{decision_text}. This interpretation is informational only and does not "
        "replace formal laboratory approval."
    )


def generate_linearity_interpretation(
    level_summary: pd.DataFrame,
    regression_summary: dict[str, float | str],
    criteria: dict[str, float],
    decision: str,
    metadata: dict[str, object],
) -> str:
    """Generate a professional interpretation for linearity study results."""

    biomarker = metadata.get("Assay / Biomarker") or "the selected assay"
    level_count = len(level_summary)
    min_expected = regression_summary["Minimum Expected Result"]
    max_expected = regression_summary["Maximum Expected Result"]
    min_recovery = level_summary["Percent Recovery"].min()
    max_recovery = level_summary["Percent Recovery"].max()
    max_abs_bias = level_summary["Percent Bias"].abs().max()
    r_squared = regression_summary["R²"]
    slope = regression_summary["Slope"]
    r2_assessment = "excellent" if r_squared >= criteria["Minimum R²"] else "limited"
    slope_assessment = (
        "within"
        if criteria["Slope Lower Limit"] <= slope <= criteria["Slope Upper Limit"]
        else "outside"
    )
    bias_assessment = (
        "within"
        if max_abs_bias <= criteria["Maximum Absolute Percent Bias"]
        else "outside"
    )
    recovery_assessment = (
        "within"
        if min_recovery >= criteria["Recovery Lower Limit"]
        and max_recovery <= criteria["Recovery Upper Limit"]
        else "outside"
    )
    decision_text = (
        "meets preliminary linearity acceptance requirements"
        if decision == "PASS"
        else "requires review against preliminary linearity acceptance requirements"
        if decision == "BORDERLINE"
        else "does not meet preliminary linearity acceptance requirements"
    )
    return (
        f"The linearity study evaluated {biomarker} performance across {level_count} "
        f"expected concentration levels spanning {min_expected:.2f}% to {max_expected:.2f}%. "
        f"Linear regression demonstrated {r2_assessment} agreement between expected and "
        f"observed values (R² = {r_squared:.4f}, slope = {slope:.4f}); the slope was "
        f"{slope_assessment} the user-defined range of {criteria['Slope Lower Limit']:.2f} "
        f"to {criteria['Slope Upper Limit']:.2f}. Maximum absolute bias was "
        f"{max_abs_bias:.2f}% and was {bias_assessment} the limit of "
        f"{criteria['Maximum Absolute Percent Bias']:.2f}%. Percent recovery ranged "
        f"from {min_recovery:.1f}% to {max_recovery:.1f}% and was {recovery_assessment} "
        f"the user-defined acceptance range of {criteria['Recovery Lower Limit']:.1f}% "
        f"to {criteria['Recovery Upper Limit']:.1f}%. Based on the evaluated criteria, "
        f"the study {decision_text}; the overall preliminary decision is {decision}. "
        "This interpretation is informational only and does not replace formal laboratory approval."
    )


def generate_stability_interpretation(
    overall_summary: dict[str, float | str],
    timepoint_summary: pd.DataFrame,
    criteria: dict[str, float],
    decision: str,
    metadata: dict[str, object],
) -> str:
    """Generate a professional interpretation for stability study results."""

    biomarker = metadata.get("Assay / Biomarker") or "the selected assay"
    study_name = metadata.get("Study Name") or "this stability study"
    timepoints = list(dict.fromkeys(str(value) for value in timepoint_summary["Timepoint"].tolist()))
    timepoint_text = ", ".join(timepoints)
    max_change = overall_summary["Maximum Observed Change"]
    min_recovery = overall_summary["Minimum Recovery"]
    max_bias = overall_summary["Maximum Absolute Bias"]
    worst_timepoint = overall_summary["Worst Timepoint"]
    worst_sample = overall_summary["Worst Sample ID"]
    if decision == "PASS":
        conclusion = "meets preliminary stability acceptance requirements"
    elif decision == "PASS WITH CAUTION":
        conclusion = "requires review against preliminary stability acceptance requirements"
    else:
        conclusion = "does not meet preliminary stability acceptance requirements"

    return (
        f"For {study_name}, {biomarker} stability was evaluated across the analyzed "
        f"timepoints ({timepoint_text}) relative to baseline measurements. The largest "
        f"observed absolute percent change was {max_change:.2f}% at {worst_timepoint} "
        f"for sample {worst_sample}, compared with the user-defined limit of "
        f"{criteria['Maximum Percent Change']:.2f}%. Minimum observed recovery was "
        f"{min_recovery:.2f}% against the minimum recovery criterion of "
        f"{criteria['Minimum Recovery']:.2f}%, and maximum absolute bias was "
        f"{max_bias:.2f} against the limit of {criteria['Maximum Absolute Bias']:.2f}. "
        f"Timepoint trends were evaluated for percent change, recovery, and bias. "
        f"Based on the preliminary screening criteria, the study {conclusion}; the "
        f"overall preliminary decision is {decision}. This interpretation is "
        "informational only and does not replace formal laboratory approval."
    )


def generate_storage_condition_comparison_interpretation(
    comparison_table: pd.DataFrame,
) -> str:
    """Generate interpretation for direct storage condition comparison."""

    if comparison_table.empty:
        return "Storage condition comparison was not available because fewer than two storage conditions were present."

    final_row = comparison_table.iloc[-1]
    comparison = str(final_row.get("Comparison", "condition difference"))
    mean_difference = final_row["Difference"]
    recovery_difference = final_row["Recovery Difference"]
    percent_change_difference = final_row["Percent Change Difference"]
    if abs(mean_difference) < 0.05 and abs(percent_change_difference) < 1:
        direction = "similar stability performance"
    elif percent_change_difference < 0:
        direction = "lower recovery and greater negative change for the candidate condition"
    else:
        direction = "higher recovery or less negative change for the candidate condition"

    return (
        f"Direct storage-condition comparison was calculated as {comparison}. At the "
        f"latest evaluated timepoint, the mean result difference was {mean_difference:.2f}, "
        f"recovery difference was {recovery_difference:.2f}%, and percent-change "
        f"difference was {percent_change_difference:.2f}%. The comparison suggests "
        f"{direction}."
    )


def generate_stability_risk_assessment(
    overall_summary: dict[str, float | str],
    criteria_table: pd.DataFrame,
    criteria: dict[str, float],
) -> str:
    """Generate a stability risk assessment narrative."""

    max_change = overall_summary["Maximum Observed Change"]
    minimum_recovery = overall_summary["Minimum Recovery"]
    max_bias = overall_summary["Maximum Absolute Bias"]
    status_values = criteria_table["Pass/Fail Status"].astype(str).str.upper()
    caution_count = int((status_values == "PASS WITH CAUTION").sum())
    fail_count = int((status_values == "FAIL").sum())

    if fail_count > 0:
        risk = "High Risk"
        summary = "One or more predefined criteria failed, indicating potential instability under the evaluated conditions."
    elif caution_count > 0:
        risk = "Moderate Risk"
        summary = "All criteria remained within allowable limits, but at least one result approached a predefined threshold."
    elif max_change <= criteria["Maximum Percent Change"] * 0.5 and minimum_recovery >= 98:
        risk = "Low Risk"
        summary = "The assay demonstrated minimal degradation and strong recovery throughout the evaluated period."
    else:
        risk = "Low Risk"
        summary = "The assay demonstrated acceptable stability throughout the evaluated period."

    return (
        f"{risk}: {summary} Maximum observed change was {max_change:.2f}%, "
        f"minimum recovery was {minimum_recovery:.2f}%, and maximum absolute bias was "
        f"{max_bias:.2f}. Borderline findings: {caution_count}; failed criteria: {fail_count}."
    )


def _metadata_to_html(metadata: dict[str, object]) -> str:
    """Render study metadata as HTML rows."""

    rows = []
    for key, value in metadata.items():
        rows.append(
            f"<tr><th>{escape(str(key))}</th><td>{escape(str(value or 'Not specified'))}</td></tr>"
        )
    return "<table>" + "".join(rows) + "</table>"


def _laboratory_documentation_to_html(metadata: dict[str, object]) -> str:
    """Render optional laboratory documentation fields as HTML rows."""

    laboratory_fields = [
        "Instrument Name",
        "Instrument ID",
        "Reagent Lot Number",
        "Calibrator Lot Number",
        "Quality Control Lot Number",
        "Operator Name",
        "Laboratory Site",
    ]
    rows = []
    for key in laboratory_fields:
        rows.append(
            f"<tr><th>{escape(key)}</th><td>{escape(str(metadata.get(key) or 'Not specified'))}</td></tr>"
        )
    return "<table>" + "".join(rows) + "</table>"


def _criteria_to_html(criteria: dict[str, float | None], decision: str) -> str:
    """Render acceptance criteria and decision as HTML."""

    rows = [
        ("Minimum acceptable R²", f">= {criteria['Minimum R²']:g}"),
        (
            "Minimum acceptable correlation coefficient (r)",
            f">= {criteria['Minimum Correlation r']:g}",
        ),
        (
            "Slope lower and upper limits",
            f"{criteria['Slope Lower Limit']:g} to {criteria['Slope Upper Limit']:g}",
        ),
        (
            "Maximum acceptable intercept",
            f"<= {criteria['Maximum Absolute Intercept']:g}",
        ),
        (
            "Maximum acceptable absolute mean bias",
            f"<= {criteria['Maximum Absolute Mean Bias']:g}",
        ),
        (
            "Maximum acceptable absolute mean percent bias",
            f"<= {criteria['Maximum Absolute Mean Percent Bias']:g}%",
        ),
        (
            "Sample agreement percent bias limit",
            f"<= {criteria['Sample Agreement Percent Bias Limit']:g}%",
        ),
        (
            "Minimum samples meeting agreement criteria",
            f">= {criteria['Minimum Percent Samples Meeting Agreement']:g}%",
        ),
    ]
    if criteria.get("Maximum Absolute Mean Difference") is not None:
        rows.append(
            (
                "Maximum acceptable absolute mean difference",
                f"<= {criteria['Maximum Absolute Mean Difference']:g}",
            )
        )

    rows.append(("Overall preliminary decision", decision))
    html_rows = [
        f"<tr><th>{escape(label)}</th><td>{escape(value)}</td></tr>"
        for label, value in rows
    ]
    return "<table>" + "".join(html_rows) + "</table>"


def build_html_report(
    summary_table: pd.DataFrame,
    criteria_table: pd.DataFrame,
    outlier_table: pd.DataFrame,
    interpretation: str,
    metadata: dict[str, object],
    criteria: dict[str, float | None],
    decision: str,
    reference_column: str,
    candidate_column: str,
    visualization_html: dict[str, str] | None = None,
) -> str:
    """Build a standalone validation summary report for scientific review."""

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    summary_html = summary_table.to_html(
        index=False, float_format=lambda value: f"{value:.4g}"
    )
    criteria_results_html = criteria_table_to_badged_html(criteria_table)
    outlier_html = outlier_table.to_html(
        index=False, float_format=lambda value: f"{value:.4g}"
    )
    metadata_html = _metadata_to_html(metadata)
    criteria_html = _criteria_to_html(criteria, decision)
    figures_html = ""
    for title, figure_html in (visualization_html or {}).items():
        figures_html += f"<h3>{escape(title)}</h3>{figure_html}"

    notes = escape(str(metadata.get("Notes") or "None documented."))
    deviations = escape(str(metadata.get("Deviations") or "None documented."))
    conclusions = escape(str(metadata.get("Conclusions") or "Pending final review."))

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Scientific Validation Analytics Report</title>
  <style>
    body {{
      color: #1f2933;
      font-family: Arial, sans-serif;
      line-height: 1.5;
      margin: 40px;
    }}
    h1, h2 {{
      color: #102a43;
    }}
    table {{
      border-collapse: collapse;
      margin-top: 12px;
      width: 100%;
    }}
    th, td {{
      border: 1px solid #d9e2ec;
      padding: 8px;
      text-align: right;
    }}
    th {{
      background: #f0f4f8;
    }}
    td:first-child, th:first-child {{
      text-align: left;
    }}
    .decision {{
      display: inline-block;
      font-size: 1.15rem;
      font-weight: bold;
      margin: 8px 0 16px;
      padding: 8px 12px;
      border: 1px solid #bcccdc;
      background: #f7f9fb;
    }}
    .note {{
      background: #f7f9fb;
      border-left: 4px solid #2a6f97;
      padding: 12px 16px;
    }}
    .status-badge {{
      border-radius: 999px;
      display: inline-block;
      font-size: 0.78rem;
      font-weight: bold;
      line-height: 1;
      padding: 6px 10px;
    }}
    .status-pass {{
      background: #e3f9e5;
      color: #1f7a1f;
    }}
    .status-borderline {{
      background: #fff8c5;
      color: #946200;
    }}
    .status-fail {{
      background: #ffe3e3;
      color: #c92a2a;
    }}
  </style>
</head>
<body>
  <h1>Scientific Validation Analytics Report</h1>
  <p><strong>Generated:</strong> {generated_at}</p>
  <p><strong>Reference column:</strong> {reference_column}</p>
  <p><strong>Candidate column:</strong> {candidate_column}</p>

  <h2>Study Metadata</h2>
  {metadata_html}

  <h2>User-Defined Preliminary Acceptance Criteria</h2>
  {criteria_html}
  <h3>Acceptance Criteria Results</h3>
  {criteria_results_html}

  <h2>Overall Decision</h2>
  <div class="decision">{decision}</div>

  <h2>Summary Statistics</h2>
  {summary_html}

  <h2>Method Comparison Results</h2>
  <p class="note">{interpretation}</p>

  <h2>Bland-Altman Results</h2>
  <p>Mean difference, SD of difference, and limits of agreement are included in the summary statistics table above.</p>

  <h2>Visualizations</h2>
  {figures_html}

  <h2>Outlier Review</h2>
  {outlier_html}

  <h2>Analyst Notes</h2>
  <p><strong>Notes:</strong> {notes}</p>
  <p><strong>Deviations:</strong> {deviations}</p>

  <h2>Preliminary Conclusion</h2>
  <p>{conclusions}</p>

  <h2>Interpretation</h2>
  <p class="note">{interpretation}</p>

  <h2>Signature Section</h2>
  <table>
    <tr><th>Prepared by</th><td></td><th>Date</th><td></td></tr>
    <tr><th>Reviewed by</th><td></td><th>Date</th><td></td></tr>
    <tr><th>Approved by</th><td></td><th>Date</th><td></td></tr>
  </table>
</body>
</html>
"""


def build_precision_html_report(
    level_summary: pd.DataFrame,
    day_summary: pd.DataFrame,
    run_summary: pd.DataFrame,
    analyzed_data: pd.DataFrame,
    criteria_table: pd.DataFrame,
    interpretation: str,
    metadata: dict[str, object],
    max_acceptable_cv: float,
    decision: str,
    visualization_html: dict[str, str] | None = None,
) -> str:
    """Build a precision study validation summary report."""

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    metadata_html = _metadata_to_html(metadata)
    laboratory_html = _laboratory_documentation_to_html(metadata)
    level_html = format_precision_summary_table(level_summary).to_html(index=False)
    day_html = (
        "<p>No day-level summary available.</p>"
        if day_summary.empty
        else format_precision_summary_table(day_summary).to_html(index=False)
    )
    run_html = (
        "<p>No run-level summary available.</p>"
        if run_summary.empty
        else format_precision_summary_table(run_summary).to_html(index=False)
    )
    criteria_html = criteria_table_to_badged_html(
        format_precision_criteria_table(criteria_table)
    )
    analyzed_html = analyzed_data.to_html(
        index=False,
        float_format=lambda value: f"{value:.2f}" if not pd.isna(value) else "",
    )
    figures_html = ""
    for title, figure_html in (visualization_html or {}).items():
        figures_html += f"<h3>{escape(title)}</h3>{figure_html}"

    notes = escape(str(metadata.get("Notes") or "None documented."))
    deviations = escape(str(metadata.get("Deviations") or "None documented."))
    conclusions = escape(str(metadata.get("Conclusions") or "Pending final review."))

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Precision Study Validation Report</title>
  <style>
    body {{
      color: #1f2933;
      font-family: Arial, sans-serif;
      line-height: 1.5;
      margin: 40px;
    }}
    h1, h2 {{
      color: #102a43;
    }}
    table {{
      border-collapse: collapse;
      margin-top: 12px;
      width: 100%;
    }}
    th, td {{
      border: 1px solid #d9e2ec;
      padding: 8px;
      text-align: right;
    }}
    th {{
      background: #f0f4f8;
    }}
    td:first-child, th:first-child {{
      text-align: left;
    }}
    td:nth-child(n+2) {{
      text-align: right;
    }}
    td:nth-last-child(-n+2) {{
      text-align: center;
    }}
    .decision {{
      display: inline-block;
      font-size: 1.15rem;
      font-weight: bold;
      margin: 8px 0 16px;
      padding: 8px 12px;
      border: 1px solid #bcccdc;
      background: #f7f9fb;
    }}
    .note {{
      background: #f7f9fb;
      border-left: 4px solid #2a6f97;
      padding: 12px 16px;
    }}
    .status-badge {{
      border-radius: 999px;
      display: inline-block;
      font-size: 0.78rem;
      font-weight: bold;
      line-height: 1;
      padding: 6px 10px;
    }}
    .status-pass {{
      background: #e3f9e5;
      color: #1f7a1f;
    }}
    .status-borderline {{
      background: #fff8c5;
      color: #946200;
    }}
    .status-fail {{
      background: #ffe3e3;
      color: #c92a2a;
    }}
  </style>
</head>
<body>
  <h1>Precision Study Validation Report</h1>
  <p><strong>Generated:</strong> {generated_at}</p>

  <h2>Study Metadata</h2>
  {metadata_html}

  <h2>Acceptance Criteria</h2>
  <p>User-defined preliminary criterion: CV% <= {max_acceptable_cv:.2f}%.</p>

  <h2>Precision Summary</h2>
  <div class="decision">{decision}</div>
  {level_html}

  <h2>Acceptance Criteria Results</h2>
  {criteria_html}

  <h2>Day-Level Summary</h2>
  {day_html}

  <h2>Run-Level Summary</h2>
  {run_html}

  <h2>Interpretation</h2>
  <p class="note">{interpretation}</p>

  <h2>Visualizations</h2>
  {figures_html}

  <h2>Analyzed Data</h2>
  {analyzed_html}

  <h2>Analyst Notes</h2>
  <p><strong>Notes:</strong> {notes}</p>

  <h2>Deviations</h2>
  <p><strong>Deviations:</strong> {deviations}</p>

  <h2>Preliminary Conclusion</h2>
  <p>{conclusions}</p>

  <h2>Signature Section</h2>
  <table>
    <tr><th>Prepared by</th><td></td><th>Date</th><td></td></tr>
    <tr><th>Reviewed by</th><td></td><th>Date</th><td></td></tr>
    <tr><th>Approved by</th><td></td><th>Date</th><td></td></tr>
  </table>
</body>
</html>
"""


def build_linearity_html_report(
    level_summary: pd.DataFrame,
    regression_summary: dict[str, float | str],
    criteria_table: pd.DataFrame,
    interpretation: str,
    metadata: dict[str, object],
    criteria: dict[str, float],
    decision: str,
    visualization_html: dict[str, str] | None = None,
) -> str:
    """Build a linearity study validation summary report."""

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    metadata_html = _metadata_to_html(metadata)
    laboratory_html = _laboratory_documentation_to_html(metadata)
    executive_summary_html = linearity_executive_summary_html(
        level_summary, regression_summary, decision
    )
    equation_html = linearity_equation_html(regression_summary)
    worst_case_html = linearity_worst_case_html(level_summary)
    level_html = format_linearity_summary_table(level_summary).to_html(index=False)
    criteria_html = criteria_table_to_badged_html(
        format_linearity_criteria_table(criteria_table)
    )
    regression_html = format_linearity_regression_summary(regression_summary).to_html(
        index=False
    )
    figures_html = ""
    for title, figure_html in (visualization_html or {}).items():
        figures_html += f"<h3>{escape(title)}</h3>{figure_html}"
    notes = escape(str(metadata.get("Notes") or "None documented."))
    deviations = escape(str(metadata.get("Deviations") or "None documented."))
    conclusions = escape(str(metadata.get("Conclusions") or "Pending final review."))

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Linearity Study Validation Report</title>
  <style>
    body {{
      color: #1f2933;
      font-family: Arial, sans-serif;
      line-height: 1.5;
      margin: 40px;
    }}
    h1, h2, h3 {{ color: #102a43; }}
    table {{
      border-collapse: collapse;
      margin: 12px 0 28px;
      width: 100%;
    }}
    th, td {{
      border: 1px solid #d9e2ec;
      padding: 8px;
      text-align: right;
    }}
    th {{ background: #f0f4f8; }}
    td:first-child, th:first-child {{ text-align: left; }}
    .summary-grid {{
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      margin: 14px 0 28px;
    }}
    .summary-card, .equation-card {{
      background: #ffffff;
      border: 1px solid #d9e2ec;
      border-radius: 8px;
      box-shadow: 0 1px 2px rgba(16, 42, 67, 0.04);
      padding: 14px 16px;
    }}
    .summary-label {{
      color: #52606d;
      font-size: 0.78rem;
      font-weight: bold;
      margin-bottom: 8px;
      text-transform: uppercase;
    }}
    .summary-value {{
      color: #102a43;
      font-size: 1.25rem;
      font-weight: bold;
    }}
    .equation-card {{
      border-left: 4px solid #2a6f97;
      background: #f7f9fb;
      margin: 12px 0 28px;
    }}
    .equation {{
      color: #102a43;
      font-size: 1.2rem;
      font-weight: bold;
    }}
    .decision {{
      display: inline-block;
      font-size: 1.15rem;
      font-weight: bold;
      margin: 8px 0 16px;
      padding: 8px 12px;
      border: 1px solid #bcccdc;
      background: #f7f9fb;
    }}
    .note {{
      background: #f7f9fb;
      border-left: 4px solid #2a6f97;
      padding: 12px 16px;
    }}
    .status-badge {{
      border-radius: 999px;
      display: inline-block;
      font-size: 0.78rem;
      font-weight: bold;
      line-height: 1;
      padding: 6px 10px;
    }}
    .status-pass {{
      background: #e3f9e5;
      color: #1f7a1f;
    }}
    .status-borderline {{
      background: #fff8c5;
      color: #946200;
    }}
    .status-fail {{
      background: #ffe3e3;
      color: #c92a2a;
    }}
    @media (max-width: 800px) {{
      .summary-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <h1>Linearity Study Validation Report</h1>
  <p><strong>Generated:</strong> {generated_at}</p>

  <h2>Executive Summary</h2>
  {executive_summary_html}

  <h2>Study Metadata</h2>
  {metadata_html}

  <h2>Laboratory Documentation</h2>
  {laboratory_html}

  <h2>Study Objective</h2>
  <p>{escape(str(metadata.get("Study Objective") or "Not specified."))}</p>

  <h2>Study Design</h2>
  <p>{escape(str(metadata.get("Study Design") or "Not specified."))}</p>

  <h2>Acceptance Criteria</h2>
  <p>User-defined preliminary screening criteria: minimum R² {criteria['Minimum R²']:.4f}; slope {criteria['Slope Lower Limit']:.2f} to {criteria['Slope Upper Limit']:.2f}; maximum absolute percent bias {criteria['Maximum Absolute Percent Bias']:.2f}%; recovery {criteria['Recovery Lower Limit']:.2f}% to {criteria['Recovery Upper Limit']:.2f}%.</p>

  <h2>Linearity Summary Table</h2>
  {level_html}
  {worst_case_html}

  <h2>Acceptance Criteria Results</h2>
  {criteria_html}

  {equation_html}

  <h2>Regression Summary</h2>
  {regression_html}

  <h2>Visualizations</h2>
  {figures_html}

  <h2>Interpretation</h2>
  <p class="note">{interpretation}</p>

  <h2>Notes / Deviations</h2>
  <p><strong>Notes:</strong> {notes}</p>
  <p><strong>Deviations:</strong> {deviations}</p>

  <h2>Preliminary Conclusion</h2>
  <p>{conclusions}</p>

  <h2>Signature Section</h2>
  <table>
    <tr><th>Prepared by</th><td></td><th>Date</th><td></td></tr>
    <tr><th>Reviewed by</th><td></td><th>Date</th><td></td></tr>
    <tr><th>Approved by</th><td></td><th>Date</th><td></td></tr>
  </table>
</body>
</html>
"""


def build_stability_html_report(
    stability_summary: pd.DataFrame,
    timepoint_summary: pd.DataFrame,
    recovery_summary: pd.DataFrame,
    bias_summary: pd.DataFrame,
    condition_comparison: pd.DataFrame,
    outlier_table: pd.DataFrame,
    overall_summary: dict[str, float | str],
    criteria_table: pd.DataFrame,
    risk_assessment: str,
    condition_interpretation: str,
    interpretation: str,
    metadata: dict[str, object],
    criteria: dict[str, float],
    decision: str,
    visualization_html: dict[str, str] | None = None,
) -> str:
    """Build a stability study validation summary report."""

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    metadata_html = _metadata_to_html(metadata)
    laboratory_html = _laboratory_documentation_to_html(metadata)
    formatted_criteria_table = format_stability_criteria_table(criteria_table)
    executive_summary_html = stability_executive_summary_html(
        overall_summary, decision, formatted_criteria_table
    )
    stability_html = format_stability_table(stability_summary).to_html(index=False)
    timepoint_html = format_stability_table(timepoint_summary).to_html(index=False)
    recovery_html = format_stability_table(recovery_summary).to_html(index=False)
    bias_html = format_stability_table(bias_summary).to_html(index=False)
    comparison_html = (
        "<p>Storage condition comparison was not available.</p>"
        if condition_comparison.empty
        else format_storage_condition_comparison_table(condition_comparison).to_html(index=False)
    )
    outlier_html = (
        "<p>No potential stability outliers were available.</p>"
        if outlier_table.empty
        else format_stability_outlier_table(outlier_table).to_html(index=False)
    )
    overall_html = format_stability_overall_summary(overall_summary).to_html(index=False)
    criteria_html = criteria_table_to_badged_html(formatted_criteria_table)
    figures_html = ""
    for title, figure_html in (visualization_html or {}).items():
        figures_html += f"<h3>{escape(title)}</h3>{figure_html}"

    notes = escape(str(metadata.get("Notes") or "None documented."))
    deviations = escape(str(metadata.get("Deviations") or "None documented."))
    conclusions = escape(str(metadata.get("Conclusions") or "Pending final review."))

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Stability Study Validation Report</title>
  <style>
    body {{
      color: #1f2933;
      font-family: Arial, sans-serif;
      line-height: 1.5;
      margin: 40px;
    }}
    h1, h2, h3 {{ color: #102a43; }}
    table {{
      border-collapse: collapse;
      margin: 12px 0 28px;
      width: 100%;
    }}
    th, td {{
      border: 1px solid #d9e2ec;
      padding: 8px;
      text-align: right;
    }}
    th {{ background: #f0f4f8; }}
    td:first-child, th:first-child {{ text-align: left; }}
    .summary-grid {{
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      margin: 14px 0 28px;
    }}
    .summary-card {{
      background: #ffffff;
      border: 1px solid #d9e2ec;
      border-radius: 8px;
      box-shadow: 0 1px 2px rgba(16, 42, 67, 0.04);
      padding: 14px 16px;
    }}
    .summary-label {{
      color: #52606d;
      font-size: 0.78rem;
      font-weight: bold;
      margin-bottom: 8px;
      text-transform: uppercase;
    }}
    .summary-value {{
      color: #102a43;
      font-size: 1.25rem;
      font-weight: bold;
    }}
    .note {{
      background: #f7f9fb;
      border-left: 4px solid #2a6f97;
      padding: 12px 16px;
    }}
    .status-badge {{
      border-radius: 999px;
      display: inline-block;
      font-size: 0.78rem;
      font-weight: bold;
      line-height: 1;
      padding: 6px 10px;
    }}
    .status-pass {{
      background: #e3f9e5;
      color: #1f7a1f;
    }}
    .status-borderline {{
      background: #fff8c5;
      color: #946200;
    }}
    .status-fail {{
      background: #ffe3e3;
      color: #c92a2a;
    }}
    @media (max-width: 800px) {{
      .summary-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <h1>Stability Study Validation Report</h1>
  <p><strong>Generated:</strong> {generated_at}</p>

  <h2>Executive Summary</h2>
  {executive_summary_html}

  <h2>Study Metadata</h2>
  {metadata_html}

  <h2>Laboratory Documentation</h2>
  {laboratory_html}

  <h2>Acceptance Criteria</h2>
  <p>User-defined preliminary screening criteria: maximum percent change {criteria['Maximum Percent Change']:.2f}%; minimum recovery {criteria['Minimum Recovery']:.2f}%; maximum absolute bias {criteria['Maximum Absolute Bias']:.2f}; borderline zone {criteria['Borderline Zone']:.2f}%.</p>

  <h2>Stability Summary</h2>
  {overall_html}
  {stability_html}

  <h2>Acceptance Criteria Results</h2>
  {criteria_html}

  <h2>Timepoint Analysis</h2>
  {timepoint_html}

  <h2>Recovery Analysis</h2>
  {recovery_html}

  <h2>Bias Analysis</h2>
  {bias_html}

  <h2>Storage Condition Comparison</h2>
  <p class="note">{escape(condition_interpretation)}</p>
  {comparison_html}

  <h2>Potential Stability Outliers</h2>
  {outlier_html}

  <h2>Risk Assessment</h2>
  <p class="note">{escape(risk_assessment)}</p>

  <h2>Visualizations</h2>
  {figures_html}

  <h2>Interpretation</h2>
  <p class="note">{interpretation}</p>

  <h2>Analyst Notes</h2>
  <p><strong>Notes:</strong> {notes}</p>
  <p><strong>Deviations:</strong> {deviations}</p>

  <h2>Preliminary Conclusion</h2>
  <p>{conclusions}</p>

  <h2>Signature Section</h2>
  <table>
    <tr><th>Prepared by</th><td></td><th>Date</th><td></td></tr>
    <tr><th>Reviewed by</th><td></td><th>Date</th><td></td></tr>
    <tr><th>Approved by</th><td></td><th>Date</th><td></td></tr>
  </table>
</body>
</html>
"""


def _pdf_clean(value: object) -> str:
    """Return PDF-safe text."""

    return str(value).replace("²", "2")


def _pdf_add_table(pdf: FPDF, title: str, table: pd.DataFrame, max_rows: int = 18) -> None:
    """Add a compact table to a PDF report."""

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, _pdf_clean(title), ln=True)
    pdf.set_font("Arial", "B", 8)
    columns = list(table.columns)
    usable_width = pdf.w - pdf.l_margin - pdf.r_margin
    column_width = usable_width / max(1, len(columns))
    for column in columns:
        pdf.cell(column_width, 6, _pdf_clean(column)[:22], border=1)
    pdf.ln()
    pdf.set_font("Arial", "", 8)
    for _, row in table.head(max_rows).iterrows():
        for column in columns:
            pdf.cell(column_width, 6, _pdf_clean(row[column])[:22], border=1)
        pdf.ln()
    pdf.ln(4)


def _pdf_add_line_chart(
    pdf: FPDF,
    title: str,
    labels: list[str],
    values: list[float],
    y_label: str,
) -> None:
    """Draw a simple line chart directly in the PDF."""

    if not values:
        return
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, _pdf_clean(title), ln=True)
    x0, y0 = pdf.l_margin + 8, pdf.get_y() + 6
    width, height = 165, 55
    pdf.rect(x0, y0, width, height)
    y_min = min(values)
    y_max = max(values)
    if y_min == y_max:
        y_min -= 1
        y_max += 1
    points = []
    for index, value in enumerate(values):
        x = x0 + (index / max(1, len(values) - 1)) * width
        y = y0 + height - ((value - y_min) / (y_max - y_min)) * height
        points.append((x, y))
    pdf.set_draw_color(42, 111, 151)
    for start, end in zip(points, points[1:]):
        pdf.line(start[0], start[1], end[0], end[1])
    pdf.set_fill_color(42, 111, 151)
    for x, y in points:
        pdf.ellipse(x - 1.5, y - 1.5, 3, 3, style="F")
    pdf.set_draw_color(0, 0, 0)
    pdf.set_font("Arial", "", 7)
    for label, (x, _) in zip(labels, points):
        pdf.text(x - 6, y0 + height + 5, _pdf_clean(label)[:10])
    pdf.text(x0 - 6, y0 + 4, f"{y_max:.2f}")
    pdf.text(x0 - 6, y0 + height, f"{y_min:.2f}")
    pdf.text(x0, y0 + height + 12, _pdf_clean(y_label))
    pdf.ln(height + 20)


def build_stability_pdf_report(
    stability_summary: pd.DataFrame,
    timepoint_summary: pd.DataFrame,
    recovery_summary: pd.DataFrame,
    bias_summary: pd.DataFrame,
    condition_comparison: pd.DataFrame,
    outlier_table: pd.DataFrame,
    overall_summary: dict[str, float | str],
    criteria_table: pd.DataFrame,
    risk_assessment: str,
    interpretation: str,
    metadata: dict[str, object],
    criteria: dict[str, float],
    decision: str,
) -> bytes:
    """Build a multi-page PDF stability study report."""

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    pdf.add_page()
    pdf.set_font("Arial", "B", 20)
    pdf.cell(0, 14, "Stability Study Validation Report", ln=True, align="C")
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, f"Generated: {generated_at}", ln=True, align="C")
    pdf.ln(8)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, f"Overall Decision: {decision}", ln=True)
    pdf.set_font("Arial", "", 11)
    for key, value in metadata.items():
        pdf.multi_cell(0, 6, f"{_pdf_clean(key)}: {_pdf_clean(value or 'Not specified')}")

    pdf.add_page()
    _pdf_add_table(pdf, "Acceptance Criteria", pd.DataFrame([
        {"Criterion": "Maximum Percent Change", "Limit": f"{criteria['Maximum Percent Change']:.2f}%"},
        {"Criterion": "Minimum Recovery", "Limit": f"{criteria['Minimum Recovery']:.2f}%"},
        {"Criterion": "Maximum Absolute Bias", "Limit": f"{criteria['Maximum Absolute Bias']:.2f}"},
        {"Criterion": "Borderline Zone", "Limit": f"{criteria['Borderline Zone']:.2f}%"},
    ]))
    _pdf_add_table(pdf, "Executive Summary", format_stability_overall_summary(overall_summary))
    _pdf_add_table(pdf, "Acceptance Results", format_stability_criteria_table(criteria_table))

    pdf.add_page()
    _pdf_add_table(pdf, "Stability Summary Table", format_stability_table(stability_summary))
    _pdf_add_table(pdf, "Timepoint Summary Table", format_stability_table(timepoint_summary))

    pdf.add_page()
    _pdf_add_table(pdf, "Recovery Summary", format_stability_table(recovery_summary))
    _pdf_add_table(pdf, "Bias Summary", format_stability_table(bias_summary))
    if not condition_comparison.empty:
        _pdf_add_table(
            pdf,
            "Storage Condition Comparison",
            format_storage_condition_comparison_table(condition_comparison),
        )
    if not outlier_table.empty:
        _pdf_add_table(
            pdf,
            "Potential Stability Outliers",
            format_stability_outlier_table(outlier_table),
            max_rows=12,
        )

    pdf.add_page()
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Risk Assessment", ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 6, _pdf_clean(risk_assessment))
    pdf.ln(3)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Interpretation", ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 6, _pdf_clean(interpretation))

    pdf.add_page()
    chart_summary = timepoint_summary.groupby("Timepoint", sort=False).agg(
        **{
            "Timepoint Mean": ("Timepoint Mean", "mean"),
            "Mean Percent Change": ("Mean Percent Change", "mean"),
            "Percent Recovery": ("Percent Recovery", "mean"),
            "Bias": ("Bias", "mean"),
        }
    ).reset_index()
    labels = chart_summary["Timepoint"].astype(str).tolist()
    _pdf_add_line_chart(
        pdf,
        "Mean Result vs Timepoint",
        labels,
        chart_summary["Timepoint Mean"].astype(float).tolist(),
        "Mean Result",
    )
    _pdf_add_line_chart(
        pdf,
        "Percent Change vs Timepoint",
        labels,
        chart_summary["Mean Percent Change"].astype(float).tolist(),
        "Percent Change (%)",
    )
    _pdf_add_line_chart(
        pdf,
        "Recovery vs Timepoint",
        labels,
        chart_summary["Percent Recovery"].astype(float).tolist(),
        "Recovery (%)",
    )
    _pdf_add_line_chart(
        pdf,
        "Bias vs Timepoint",
        labels,
        chart_summary["Bias"].astype(float).tolist(),
        "Bias",
    )

    output = pdf.output(dest="S")
    if isinstance(output, bytes):
        return output
    return output.encode("latin-1")
