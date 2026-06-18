"""Report and interpretation helpers for validation analytics outputs."""

from __future__ import annotations

from datetime import datetime
from html import escape
from io import BytesIO

import pandas as pd
from fpdf import FPDF

from validation.acceptance import count_statuses, normalize_status
from validation.interpretation import build_sectioned_interpretation


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


def _format_measurement(value: object, digits: int = 2, suffix: str = "") -> str:
    """Format numeric values for scientific review tables."""

    if pd.isna(value):
        return ""
    numeric_value = float(value)
    if numeric_value != 0 and (abs(numeric_value) >= 10000 or abs(numeric_value) < 0.001):
        return f"{numeric_value:.{digits}e}{suffix}"
    return f"{numeric_value:.{digits}f}{suffix}"


def _format_interval(lower: object, upper: object, digits: int = 2, suffix: str = "") -> str:
    """Format a lower-to-upper confidence interval."""

    if pd.isna(lower) or pd.isna(upper):
        return "Not available"
    return f"{_format_measurement(lower, digits, suffix)} to {_format_measurement(upper, digits, suffix)}"


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
    for optional_column in ["Margin to Limit", "Scientific Interpretation"]:
        if optional_column in criteria_table.columns:
            insert_at = 3 if optional_column == "Margin to Limit" else len(columns)
            columns.insert(insert_at, optional_column)
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
    formatted = formatted.rename(
        columns={
            "Observed Value": "Observed",
            "Pass/Fail Status": "Status",
        }
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


def format_accuracy_table(table: pd.DataFrame) -> pd.DataFrame:
    """Format accuracy summary tables for display and export."""

    formatted = table.copy()
    if "N" in formatted.columns:
        formatted["N"] = formatted["N"].map(lambda value: f"{int(value)}")
    if {
        "Mean Observed 95% CI Lower",
        "Mean Observed 95% CI Upper",
    }.issubset(formatted.columns):
        formatted["Mean Observed Result ± 95% CI"] = formatted.apply(
            lambda row: _format_interval(
                row["Mean Observed 95% CI Lower"],
                row["Mean Observed 95% CI Upper"],
                2,
            ),
            axis=1,
        )
    if {"Bias 95% CI Lower", "Bias 95% CI Upper"}.issubset(formatted.columns):
        formatted["Bias ± 95% CI"] = formatted.apply(
            lambda row: _format_interval(
                row["Bias 95% CI Lower"],
                row["Bias 95% CI Upper"],
                2,
            ),
            axis=1,
        )
    numeric_columns = [
        "Expected Result",
        "Mean Observed Result",
        "SD",
        "Difference",
        "Absolute Difference",
        "Absolute Bias",
        "Mean Observed 95% CI Lower",
        "Mean Observed 95% CI Upper",
        "Bias 95% CI Lower",
        "Bias 95% CI Upper",
    ]
    for column in numeric_columns:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(
                lambda value: _format_measurement(value, 2)
            )
    percent_columns = ["Percent Bias", "Absolute Percent Bias", "Percent Recovery"]
    for column in percent_columns:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(
                lambda value: _format_measurement(value, 2, "%")
            )
    if "Recovery %" in formatted.columns:
        formatted["Recovery %"] = formatted["Recovery %"].map(
            lambda value: _format_measurement(value, 2, "%")
        )
    return formatted


def format_accuracy_level_decision_table(table: pd.DataFrame) -> pd.DataFrame:
    """Format level-specific accuracy decisions for display and export."""

    formatted = format_accuracy_table(table)
    preferred_columns = [
        "Level",
        "N",
        "Mean Observed Result",
        "Mean Observed Result ± 95% CI",
        "Expected Result",
        "Absolute Bias",
        "Bias ± 95% CI",
        "Percent Bias",
        "Recovery %",
        "Pass/Fail Status",
    ]
    return formatted[[column for column in preferred_columns if column in formatted.columns]]


def format_accuracy_overall_summary(summary: dict[str, float | str]) -> pd.DataFrame:
    """Format overall accuracy summary as a two-column table."""

    rows = [
        ("N", f"{int(summary['N'])}"),
        ("Overall Mean Bias", f"{summary['Overall Mean Bias']:.2f}"),
        ("Overall Mean Percent Bias", f"{summary['Overall Mean Percent Bias']:.2f}%"),
        ("Maximum Absolute Bias", f"{summary['Maximum Absolute Bias']:.2f}"),
        ("Maximum Absolute Percent Bias", f"{summary['Maximum Absolute Percent Bias']:.2f}%"),
        ("Minimum Recovery", f"{summary['Minimum Recovery']:.2f}%"),
        ("Maximum Recovery", f"{summary['Maximum Recovery']:.2f}%"),
        ("Worst Level", str(summary["Worst Level"])),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def format_accuracy_worst_case_summary(summary: dict[str, float | str]) -> pd.DataFrame:
    """Format worst-case accuracy summary as a two-column table."""

    rows = [
        ("Worst Level", str(summary["Worst Level"])),
        ("Highest Absolute Percent Bias", f"{summary['Highest Absolute Percent Bias']:.2f}%"),
        ("Lowest Recovery", f"{summary['Lowest Recovery']:.2f}%"),
        ("Lowest Recovery Level", str(summary["Lowest Recovery Level"])),
        ("Highest Recovery", f"{summary['Highest Recovery']:.2f}%"),
        ("Highest Recovery Level", str(summary["Highest Recovery Level"])),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def format_accuracy_criteria_table(criteria_table: pd.DataFrame) -> pd.DataFrame:
    """Format accuracy acceptance criteria result values."""

    formatted = criteria_table.copy()

    def format_observed(row: pd.Series) -> str:
        value = row["Observed Value"]
        if pd.isna(value):
            return ""
        criterion = str(row["Criterion"]).lower()
        if "percent" in criterion or "recovery" in criterion:
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
            if column in {"Pass/Fail Status", "Status"}:
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


def build_accuracy_executive_summary(
    overall_summary: dict[str, float | str],
    worst_case_summary: dict[str, float | str],
    decision: str,
    criteria_table: pd.DataFrame | None = None,
    level_decision_table: pd.DataFrame | None = None,
) -> dict[str, str]:
    """Build display values for accuracy executive summary cards."""

    caution_count = 0
    failed_count = 0
    if criteria_table is not None and not criteria_table.empty:
        criteria_counts = count_statuses(criteria_table)
        caution_count = criteria_counts["BORDERLINE"]
        failed_count = criteria_counts["FAIL"]
    levels_passing = "Not available"
    levels_failing = "Not available"
    if level_decision_table is not None and not level_decision_table.empty:
        level_counts = count_statuses(level_decision_table)
        levels_passing = str(level_counts["PASS"])
        levels_failing = str(level_counts["FAIL"])

    return {
        "Overall Decision": decision,
        "Overall Mean Percent Bias": f"{overall_summary['Overall Mean Percent Bias']:.2f}%",
        "Maximum Absolute Percent Bias": f"{overall_summary['Maximum Absolute Percent Bias']:.2f}%",
        "Lowest Recovery": f"{overall_summary['Minimum Recovery']:.2f}%",
        "Highest Recovery": f"{overall_summary['Maximum Recovery']:.2f}%",
        "Worst Performing Level": str(worst_case_summary["Worst Level"]),
        "Levels Passing": levels_passing,
        "Levels Failing": levels_failing,
        "Borderline Criteria": str(caution_count),
        "Failed Criteria": str(failed_count),
    }


def accuracy_executive_summary_html(
    overall_summary: dict[str, float | str],
    worst_case_summary: dict[str, float | str],
    decision: str,
    criteria_table: pd.DataFrame | None = None,
    level_decision_table: pd.DataFrame | None = None,
) -> str:
    """Render accuracy executive summary cards for the HTML report."""

    summary_values = build_accuracy_executive_summary(
        overall_summary, worst_case_summary, decision, criteria_table, level_decision_table
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


def accuracy_criteria_dashboard_html(criteria_table: pd.DataFrame) -> str:
    """Render accuracy criteria statuses as dashboard cards."""

    cards = []
    for _, row in criteria_table.iterrows():
        display_status = normalize_status(row["Pass/Fail Status"])
        cards.append(
            f"""
            <div class="summary-card">
              <div class="summary-label">{escape(str(row['Criterion']))}</div>
              <div class="summary-value">{status_badge_html(display_status)}</div>
              <div style="color:#52606d;font-size:0.85rem;margin-top:8px;">
                Observed: {escape(str(row['Observed Value']))}<br>
                Limit: {escape(str(row['Acceptance Limit']))}
              </div>
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


def generate_accuracy_interpretation(
    overall_summary: dict[str, float | str],
    worst_case_summary: dict[str, float | str],
    criteria: dict[str, float],
    decision: str,
    metadata: dict[str, object],
) -> str:
    """Generate a professional interpretation for accuracy study results."""

    biomarker = metadata.get("Assay / Biomarker") or "the selected assay"
    study_name = metadata.get("Study Name") or "this accuracy study"
    design = metadata.get("Study Design") or "Replicate observations were compared with assigned expected values."
    max_abs_percent_bias = overall_summary["Maximum Absolute Percent Bias"]
    max_abs_bias = overall_summary["Maximum Absolute Bias"]
    mean_percent_bias = overall_summary["Overall Mean Percent Bias"]
    min_recovery = overall_summary["Minimum Recovery"]
    max_recovery = overall_summary["Maximum Recovery"]
    worst_level = worst_case_summary["Worst Level"]
    if decision == "PASS":
        conclusion = "The observed agreement supports preliminary assay accuracy across the evaluated levels."
    elif decision == "PASS WITH CAUTION":
        conclusion = "The observed agreement supports preliminary accuracy with caution; borderline findings should be reviewed before final approval."
    else:
        conclusion = "The observed agreement does not support preliminary assay accuracy under the selected criteria."

    return build_sectioned_interpretation(
        {
            "Objective": (
                f"{study_name} evaluated whether observed {biomarker} results agree "
                "with expected or assigned target values."
            ),
            "What was evaluated": str(design),
            "Results Summary": (
                f"Overall mean percent bias was {mean_percent_bias:.2f}%. Maximum "
                f"absolute bias was {max_abs_bias:.2f}, maximum absolute percent bias "
                f"was {max_abs_percent_bias:.2f}%, and recovery ranged from "
                f"{min_recovery:.2f}% to {max_recovery:.2f}%. The worst-performing "
                f"level was {worst_level}."
            ),
            "Acceptance Criteria Assessment": (
                f"User-defined preliminary criteria were maximum absolute bias "
                f"{criteria['Maximum Absolute Bias']:.2f}, maximum absolute percent "
                f"bias {criteria['Maximum Absolute Percent Bias']:.2f}%, and recovery "
                f"from {criteria['Minimum Recovery']:.2f}% to "
                f"{criteria['Maximum Recovery']:.2f}% with a "
                f"{criteria['Borderline Zone']:.2f}% borderline zone. The overall "
                f"criteria-based decision is {decision}."
            ),
            "Scientific Interpretation": conclusion,
            "Preliminary Conclusion": (
                f"The overall preliminary decision is {decision}. This interpretation "
                "is informational only and does not replace formal laboratory approval."
            ),
        }
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


def build_accuracy_html_report(
    accuracy_summary: pd.DataFrame,
    bias_summary: pd.DataFrame,
    recovery_summary: pd.DataFrame,
    level_decision_table: pd.DataFrame,
    worst_case_summary: dict[str, float | str],
    overall_summary: dict[str, float | str],
    criteria_table: pd.DataFrame,
    interpretation: str,
    metadata: dict[str, object],
    criteria: dict[str, float],
    decision: str,
    visualization_html: dict[str, str] | None = None,
) -> str:
    """Build an accuracy study validation summary report."""

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    metadata_html = _metadata_to_html(metadata)
    laboratory_html = _laboratory_documentation_to_html(metadata)
    formatted_criteria = format_accuracy_criteria_table(criteria_table)
    formatted_level_decisions = format_accuracy_level_decision_table(
        level_decision_table
    )
    executive_summary_html = accuracy_executive_summary_html(
        overall_summary,
        worst_case_summary,
        decision,
        formatted_criteria,
        level_decision_table,
    )
    criteria_dashboard_html = accuracy_criteria_dashboard_html(formatted_criteria)
    overall_html = format_accuracy_overall_summary(overall_summary).to_html(index=False)
    accuracy_html = format_accuracy_table(accuracy_summary).to_html(index=False)
    level_decisions_html = criteria_table_to_badged_html(formatted_level_decisions)
    bias_html = format_accuracy_table(bias_summary).to_html(index=False)
    recovery_html = format_accuracy_table(recovery_summary).to_html(index=False)
    worst_case_html = format_accuracy_worst_case_summary(worst_case_summary).to_html(index=False)
    criteria_html = criteria_table_to_badged_html(formatted_criteria)
    figures_html = ""
    for title, figure_html in (visualization_html or {}).items():
        figures_html += f"<h3>{escape(title)}</h3>{figure_html}"

    notes = escape(str(metadata.get("Notes") or "None documented."))
    deviations = escape(str(metadata.get("Deviations") or "None documented."))
    conclusions = escape(str(metadata.get("Conclusions") or "Pending final review."))
    interpretation_html = "<br><br>".join(
        escape(paragraph) for paragraph in interpretation.split("\n\n")
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Accuracy Study Validation Report</title>
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
  <h1>Accuracy Study Validation Report</h1>
  <p><strong>Generated:</strong> {generated_at}</p>

  <h2>Study Metadata</h2>
  {metadata_html}

  <h2>Laboratory Documentation</h2>
  {laboratory_html}

  <h2>Acceptance Criteria</h2>
  <p>User-defined preliminary screening criteria: maximum absolute bias {criteria['Maximum Absolute Bias']:.2f}; maximum absolute percent bias {criteria['Maximum Absolute Percent Bias']:.2f}%; recovery {criteria['Minimum Recovery']:.2f}% to {criteria['Maximum Recovery']:.2f}%; borderline zone {criteria['Borderline Zone']:.2f}%.</p>
  {criteria_dashboard_html}

  <h2>Executive Summary</h2>
  {executive_summary_html}
  {overall_html}

  <h2>Accuracy Results</h2>
  <h3>Level-Specific Decision Table</h3>
  {level_decisions_html}
  <h3>Accuracy Summary</h3>
  {accuracy_html}

  <h2>Acceptance Criteria Results</h2>
  {criteria_html}

  <h2>Bias Analysis</h2>
  {bias_html}

  <h2>Recovery Analysis</h2>
  {recovery_html}

  <h2>Worst-Case Performance</h2>
  {worst_case_html}

  <h2>Visualizations</h2>
  {figures_html}

  <h2>Interpretation</h2>
  <p class="note">{interpretation_html}</p>

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


def format_detection_table(table: pd.DataFrame) -> pd.DataFrame:
    """Format detection capability summary values for display and export."""

    formatted = table.copy()
    if "N" in formatted.columns:
        formatted["N"] = formatted["N"].map(lambda value: f"{int(value)}")
    for column in [
        "Mean Blank",
        "SD Blank",
        "LoB",
        "Mean Low Sample",
        "SD Low Sample",
        "LoD",
        "Concentration Level",
        "Mean",
        "SD",
        "Observed Result",
        "IQR Lower Bound",
        "IQR Upper Bound",
        "Final Result",
    ]:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(
                lambda value: _format_measurement(value, 3)
            )
    for column in ["CV%", "Bias %", "Recovery %"]:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(
                lambda value: _format_measurement(value, 2, "%")
            )
    return formatted


def format_detection_overall_summary(summary: dict[str, float | str]) -> pd.DataFrame:
    """Format overall detection capability results."""

    rows = [
        ("LoB", _format_measurement(summary["LoB"], 3)),
        ("LoD", _format_measurement(summary["LoD"], 3)),
        ("Operational LoQ", _format_measurement(summary["LoQ"], 3)),
        ("Operational LoQ CV%", _format_measurement(summary["Operational LoQ CV%"], 2, "%")),
        ("Blank Replicates", str(int(summary.get("Blank Replicates", 0)))),
        ("Low-Level Replicates", str(int(summary.get("Low-Level Replicates", 0)))),
        ("Quantitation Levels Tested", str(int(summary.get("Quantitation Levels Tested", 0)))),
        ("Worst Performing Level", _format_measurement(summary["Worst Performing Level"], 3)),
        ("Worst CV%", _format_measurement(summary["Worst CV%"], 2, "%")),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def format_detection_criteria_table(criteria_table: pd.DataFrame) -> pd.DataFrame:
    """Format detection capability criteria table."""

    formatted = criteria_table.copy()

    def format_observed(row: pd.Series) -> str:
        value = row["Observed Value"]
        if pd.isna(value):
            return ""
        criterion = str(row["Criterion"]).lower()
        suffix = "%" if "cv" in criterion else ""
        digits = 2 if suffix else 3
        return _format_measurement(value, digits, suffix)

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


def build_detection_executive_summary(
    overall_summary: dict[str, float | str],
    decision: str,
    criteria_table: pd.DataFrame | None = None,
) -> dict[str, str]:
    """Build display values for detection capability executive summary cards."""

    if criteria_table is not None:
        status_column = "Status" if "Status" in criteria_table.columns else "Pass/Fail Status"
        counts = count_statuses(criteria_table, status_column)
    else:
        counts = {"BORDERLINE": 0, "FAIL": 0}
    return {
        "Overall Decision": decision,
        "LoB": _format_measurement(overall_summary["LoB"], 3),
        "LoD": _format_measurement(overall_summary["LoD"], 3),
        "Operational LoQ": _format_measurement(overall_summary["LoQ"], 3),
        "LoQ CV%": _format_measurement(overall_summary["Operational LoQ CV%"], 2, "%"),
        "Blank Replicates": str(int(overall_summary.get("Blank Replicates", 0))),
        "Low-Level Replicates": str(int(overall_summary.get("Low-Level Replicates", 0))),
        "Quantitation Levels Tested": str(int(overall_summary.get("Quantitation Levels Tested", 0))),
    }


def detection_executive_summary_html(
    overall_summary: dict[str, float | str],
    decision: str,
    criteria_table: pd.DataFrame | None = None,
) -> str:
    """Render detection capability executive summary cards for HTML reports."""

    summary_values = build_detection_executive_summary(
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


def generate_detection_interpretation(
    overall_summary: dict[str, float | str],
    data_quality_summary: pd.DataFrame,
    criteria: dict[str, float],
    decision: str,
    metadata: dict[str, object],
) -> str:
    """Generate a scientific interpretation for LoB/LoD/LoQ studies."""

    study_name = metadata.get("Study Name") or "this detection capability study"
    biomarker = metadata.get("Assay / Biomarker") or "the selected assay"
    quality_status = data_quality_summary["Status"].astype(str).str.upper()
    quality_findings = (
        f"{int((quality_status == 'PASS').sum())} data quality checks passed; "
        f"{int((quality_status == 'BORDERLINE').sum())} require reviewer attention; "
        f"{int((quality_status == 'FAIL').sum())} failed."
    )
    if decision == "PASS":
        conclusion = "The observed LoB, LoD, and LoQ findings meet the selected preliminary detection capability criteria."
    elif decision == "BORDERLINE":
        conclusion = "The study meets the selected criteria with caution; borderline findings should be reviewed before formal approval."
    else:
        conclusion = "One or more detection capability criteria were not met and should be investigated before formal approval."

    return build_sectioned_interpretation(
        {
            "Objective": (
                f"{study_name} evaluated {biomarker} detection capability through "
                "replicate blank, low-concentration, and quantitation-level measurements."
            ),
            "LoB Assessment": (
                f"The calculated LoB was {_format_measurement(overall_summary['LoB'], 3)}, "
                f"compared with the user-defined maximum acceptable LoB of "
                f"{criteria['Maximum LoB']:.3f}."
            ),
            "LoD Assessment": (
                f"The calculated LoD was {_format_measurement(overall_summary['LoD'], 3)}, "
                f"compared with the user-defined maximum acceptable LoD of "
                f"{criteria['Maximum LoD']:.3f}."
            ),
            "LoQ Assessment": (
                f"The operational LoQ was {_format_measurement(overall_summary['LoQ'], 3)} "
                f"with CV% {_format_measurement(overall_summary['Operational LoQ CV%'], 2, '%')}. "
                f"The target LoQ CV% was {criteria['Target LoQ CV%']:.2f}% and the maximum "
                f"acceptable LoQ concentration was {criteria['Maximum LoQ Concentration']:.3f}."
            ),
            "Data Quality Assessment": quality_findings,
            "Acceptance Criteria Assessment": (
                f"The overall preliminary criteria-based decision is {decision}."
            ),
            "Overall Conclusion": (
                f"{conclusion} This interpretation is informational only and does not "
                "replace formal laboratory review or regulatory approval."
            ),
        }
    )


def build_detection_html_report(
    lob_summary: pd.DataFrame,
    lod_summary: pd.DataFrame,
    loq_summary: pd.DataFrame,
    methodology_table: pd.DataFrame,
    data_quality_summary: pd.DataFrame,
    outlier_table: pd.DataFrame,
    decision_matrix: pd.DataFrame,
    analyzed_data: pd.DataFrame,
    overall_summary: dict[str, float | str],
    criteria_table: pd.DataFrame,
    interpretation: str,
    metadata: dict[str, object],
    criteria: dict[str, float],
    decision: str,
    visualization_html: dict[str, str] | None = None,
) -> str:
    """Build a complete Detection Capability HTML report."""

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    metadata_html = _metadata_to_html(metadata)
    laboratory_html = _laboratory_documentation_to_html(metadata)
    formatted_criteria = format_detection_criteria_table(criteria_table)
    executive_summary_html = detection_executive_summary_html(
        overall_summary, decision, formatted_criteria
    )
    criteria_html = criteria_table_to_badged_html(formatted_criteria)
    data_quality_html = format_detection_table(data_quality_summary).to_html(index=False)
    outlier_html = (
        "<p>No IQR outliers were detected.</p>"
        if outlier_table.empty
        else format_detection_table(outlier_table).to_html(index=False)
    )
    methodology_html = format_detection_table(methodology_table).to_html(index=False)
    decision_matrix_html = format_detection_table(decision_matrix).to_html(index=False)
    figures_html = ""
    for title, figure_html in (visualization_html or {}).items():
        figures_html += f"<h3>{escape(title)}</h3>{figure_html}"
    interpretation_html = "<br><br>".join(
        escape(paragraph) for paragraph in interpretation.split("\n\n")
    )
    notes = escape(str(metadata.get("Notes") or "None documented."))
    conclusions = escape(str(metadata.get("Conclusions") or "Pending final review."))

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Detection Capability Validation Report</title>
  <style>
    body {{ color: #1f2933; font-family: Arial, sans-serif; line-height: 1.5; margin: 40px; }}
    h1, h2, h3 {{ color: #102a43; }}
    table {{ border-collapse: collapse; margin: 12px 0 28px; width: 100%; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 8px; text-align: right; }}
    th {{ background: #f0f4f8; }}
    td:first-child, th:first-child {{ text-align: left; }}
    .summary-grid {{ display: grid; gap: 12px; grid-template-columns: repeat(3, minmax(0, 1fr)); margin: 14px 0 28px; }}
    .summary-card {{ background: #ffffff; border: 1px solid #d9e2ec; border-radius: 8px; box-shadow: 0 1px 2px rgba(16, 42, 67, 0.04); padding: 14px 16px; }}
    .summary-label {{ color: #52606d; font-size: 0.78rem; font-weight: bold; margin-bottom: 8px; text-transform: uppercase; }}
    .summary-value {{ color: #102a43; font-size: 1.25rem; font-weight: bold; }}
    .note {{ background: #f7f9fb; border-left: 4px solid #2a6f97; padding: 12px 16px; }}
    .status-badge {{ border-radius: 999px; display: inline-block; font-size: 0.78rem; font-weight: bold; line-height: 1; padding: 6px 10px; }}
    .status-pass {{ background: #e3f9e5; color: #1f7a1f; }}
    .status-borderline {{ background: #fff8c5; color: #946200; }}
    .status-fail {{ background: #ffe3e3; color: #c92a2a; }}
  </style>
</head>
<body>
  <h1>Detection Capability Validation Report</h1>
  <p><strong>Generated:</strong> {generated_at}</p>

  <h2>Study Metadata</h2>
  {metadata_html}

  <h2>Laboratory Documentation</h2>
  {laboratory_html}

  <h2>Executive Summary</h2>
  {executive_summary_html}
  {format_detection_overall_summary(overall_summary).to_html(index=False)}

  <h2>Acceptance Criteria</h2>
  <p>User-defined preliminary screening criteria: maximum LoB {criteria['Maximum LoB']:.3f}; maximum LoD {criteria['Maximum LoD']:.3f}; target LoQ CV% {criteria['Target LoQ CV%']:.2f}%; maximum LoQ concentration {criteria['Maximum LoQ Concentration']:.3f}; borderline zone {criteria['Borderline Zone']:.2f}%.</p>

  <h2>Calculation Methodology</h2>
  {methodology_html}

  <h2>Data Quality Assessment</h2>
  {data_quality_html}
  <h3>IQR Outlier Review</h3>
  {outlier_html}

  <h2>LoB Analysis</h2>
  {format_detection_table(lob_summary).to_html(index=False)}

  <h2>LoD Analysis</h2>
  {format_detection_table(lod_summary).to_html(index=False)}

  <h2>LoQ Analysis</h2>
  {format_detection_table(loq_summary).to_html(index=False)}

  <h2>Acceptance Criteria Results</h2>
  {criteria_html}

  <h2>Detection Capability Decision Matrix</h2>
  {decision_matrix_html}

  <h2>Interpretation</h2>
  <p class="note">{interpretation_html}</p>

  <h2>Visualizations</h2>
  {figures_html}

  <h2>Analyzed Dataset</h2>
  {format_detection_table(analyzed_data).to_html(index=False)}

  <h2>Analyst Notes</h2>
  <p>{notes}</p>

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


def build_detection_pdf_report(
    lob_summary: pd.DataFrame,
    lod_summary: pd.DataFrame,
    loq_summary: pd.DataFrame,
    methodology_table: pd.DataFrame,
    data_quality_summary: pd.DataFrame,
    outlier_table: pd.DataFrame,
    decision_matrix: pd.DataFrame,
    overall_summary: dict[str, float | str],
    criteria_table: pd.DataFrame,
    interpretation: str,
    metadata: dict[str, object],
    criteria: dict[str, float],
    decision: str,
) -> bytes:
    """Build a professional PDF Detection Capability report."""

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    pdf.add_page()
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 12, "Detection Capability Validation Report", ln=True, align="C")
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 8, f"Generated: {generated_at}", ln=True, align="C")
    pdf.ln(6)
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, f"Overall Decision: {decision}", ln=True)
    pdf.set_font("Arial", "", 9)
    for key, value in metadata.items():
        pdf.multi_cell(0, 5, f"{_pdf_clean(key)}: {_pdf_clean(value or 'Not specified')}")

    pdf.add_page()
    _pdf_add_table(pdf, "Acceptance Criteria Results", format_detection_criteria_table(criteria_table))
    _pdf_add_table(pdf, "Executive Summary", format_detection_overall_summary(overall_summary))
    _pdf_add_table(pdf, "Decision Matrix", format_detection_table(decision_matrix))

    pdf.add_page()
    _pdf_add_table(pdf, "Calculation Methodology", format_detection_table(methodology_table))
    _pdf_add_table(pdf, "Data Quality Assessment", format_detection_table(data_quality_summary))
    if not outlier_table.empty:
        _pdf_add_table(pdf, "IQR Outlier Review", format_detection_table(outlier_table), max_rows=12)

    pdf.add_page()
    _pdf_add_table(pdf, "LoB Calculations", format_detection_table(lob_summary))
    _pdf_add_table(pdf, "LoD Calculations", format_detection_table(lod_summary))
    _pdf_add_table(pdf, "LoQ Calculations", format_detection_table(loq_summary), max_rows=12)

    pdf.add_page()
    chart_data = loq_summary.sort_values("Concentration Level")
    labels = chart_data["Concentration Level"].map(lambda value: f"{value:.3f}").tolist()
    _pdf_add_line_chart(
        pdf,
        "LoQ Precision Curve",
        labels,
        chart_data["CV%"].astype(float).tolist(),
        f"CV% (target {criteria['Target LoQ CV%']:.2f}%)",
    )
    _pdf_add_line_chart(
        pdf,
        "Recovery vs Concentration",
        labels,
        chart_data["Recovery %"].astype(float).tolist(),
        "Recovery (%)",
    )

    pdf.add_page()
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Interpretation", ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 6, _pdf_clean(interpretation))

    output = pdf.output(dest="S")
    if isinstance(output, bytes):
        return output
    return output.encode("latin-1")


def format_dbs_table(table: pd.DataFrame) -> pd.DataFrame:
    """Format DBS validation tables for scientific review."""

    formatted = table.copy()
    for column in formatted.columns:
        lower = str(column).lower()
        if lower in {"n", "replicate"}:
            formatted[column] = formatted[column].map(
                lambda value: "" if pd.isna(value) else f"{int(value)}"
            )
        elif "r²" in str(column) or "correlation" in lower or "slope" in lower:
            formatted[column] = formatted[column].map(
                lambda value: _format_measurement(value, 4)
                if pd.api.types.is_number(value)
                else value
            )
        elif "percent" in lower or "recovery" in lower or "bias" in lower:
            formatted[column] = formatted[column].map(
                lambda value: _format_measurement(value, 2, "%" if "percent" in lower or "recovery" in lower else "")
                if pd.api.types.is_number(value)
                else value
            )
        elif pd.api.types.is_numeric_dtype(formatted[column]):
            formatted[column] = formatted[column].map(
                lambda value: _format_measurement(value, 3)
            )
    return formatted


def format_dbs_criteria_table(criteria_table: pd.DataFrame) -> pd.DataFrame:
    """Format DBS acceptance criteria results."""

    formatted = criteria_table.copy()

    def format_observed(row: pd.Series) -> str:
        value = row["Observed Value"]
        if pd.isna(value):
            return ""
        criterion = str(row["Criterion"]).lower()
        suffix = "%" if "bias" in criterion or "recovery" in criterion else ""
        digits = 4 if "r²" in criterion else 2
        return _format_measurement(value, digits, suffix)

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


def format_dbs_overall_summary(summary: dict[str, float | str]) -> pd.DataFrame:
    """Format DBS overall summary metrics."""

    rows = [
        ("N", str(int(summary.get("N", 0)))),
        ("Mean Bias", _format_measurement(summary.get("Mean Bias"), 3)),
        ("Mean Percent Bias", _format_measurement(summary.get("Mean Percent Bias"), 2, "%")),
        ("Mean Recovery", _format_measurement(summary.get("Mean Recovery"), 2, "%")),
        ("R²", _format_measurement(summary.get("R²"), 4)),
        ("Slope", _format_measurement(summary.get("Slope"), 4)),
        ("Mean Difference", _format_measurement(summary.get("Mean Difference"), 3)),
        ("Lower 95% LoA", _format_measurement(summary.get("Lower Limit of Agreement"), 3)),
        ("Upper 95% LoA", _format_measurement(summary.get("Upper Limit of Agreement"), 3)),
        ("Worst Sample", str(summary.get("Worst Sample", ""))),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def build_dbs_executive_summary(
    overall_summary: dict[str, float | str],
    decision: str,
    criteria_table: pd.DataFrame | None = None,
) -> dict[str, str]:
    """Build DBS executive summary card values."""

    counts = count_statuses(criteria_table, "Pass/Fail Status") if criteria_table is not None else {"BORDERLINE": 0, "FAIL": 0}
    return {
        "Overall Decision": decision,
        "Mean Bias": _format_measurement(overall_summary["Mean Bias"], 3),
        "Mean Recovery": _format_measurement(overall_summary["Mean Recovery"], 2, "%"),
        "R²": _format_measurement(overall_summary["R²"], 4),
        "Worst Sample": str(overall_summary["Worst Sample"]),
        "Borderline Findings": str(counts.get("BORDERLINE", 0) + counts.get("PASS WITH CAUTION", 0)),
        "Failed Criteria": str(counts.get("FAIL", 0)),
    }


def dbs_executive_summary_html(
    overall_summary: dict[str, float | str],
    decision: str,
    criteria_table: pd.DataFrame | None = None,
) -> str:
    """Render DBS executive summary cards."""

    cards = []
    for label, value in build_dbs_executive_summary(
        overall_summary, decision, criteria_table
    ).items():
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


def generate_dbs_interpretation(
    overall_summary: dict[str, float | str],
    criteria: dict[str, float],
    decision: str,
    metadata: dict[str, object],
) -> str:
    """Generate a professional DBS validation interpretation."""

    study_name = metadata.get("Study Name") or "this DBS validation study"
    assay = metadata.get("Assay / Biomarker") or metadata.get("Assay") or "the assay"
    if decision == "PASS":
        conclusion = "All predefined preliminary acceptance criteria were met."
    elif decision in {"PASS WITH CAUTION", "BORDERLINE"}:
        conclusion = "One or more findings were within the user-defined borderline zone and should be reviewed before formal approval."
    else:
        conclusion = "One or more predefined preliminary acceptance criteria were not met and require investigation."
    return build_sectioned_interpretation(
        {
            "Study Objective": (
                f"{study_name} evaluated whether DBS-derived {assay} results demonstrate "
                "acceptable analytical agreement with reference venous specimens."
            ),
            "Bias Assessment": (
                f"Mean bias was {_format_measurement(overall_summary['Mean Bias'], 3)} and "
                f"mean percent bias was {_format_measurement(overall_summary['Mean Percent Bias'], 2, '%')}. "
                f"The maximum absolute percent bias was {_format_measurement(overall_summary['Maximum Absolute Percent Bias'], 2, '%')} "
                f"against a user-defined limit of {criteria['Maximum Percent Bias']:.2f}%."
            ),
            "Recovery Assessment": (
                f"Mean recovery was {_format_measurement(overall_summary['Mean Recovery'], 2, '%')}, "
                f"with observed recovery ranging from {_format_measurement(overall_summary['Minimum Recovery'], 2, '%')} "
                f"to {_format_measurement(overall_summary['Maximum Recovery'], 2, '%')}."
            ),
            "Correlation Assessment": (
                f"Correlation analysis produced r = {_format_measurement(overall_summary['Correlation r'], 4)} "
                f"and R² = {_format_measurement(overall_summary['R²'], 4)}, with slope "
                f"{_format_measurement(overall_summary['Slope'], 4)} and intercept "
                f"{_format_measurement(overall_summary['Intercept'], 4)}."
            ),
            "Agreement Assessment": (
                f"The mean difference was {_format_measurement(overall_summary['Mean Difference'], 3)}, "
                f"with 95% limits of agreement from {_format_measurement(overall_summary['Lower Limit of Agreement'], 3)} "
                f"to {_format_measurement(overall_summary['Upper Limit of Agreement'], 3)}."
            ),
            "Overall Conclusion": (
                f"{conclusion} The overall preliminary decision is {decision}. "
                "This interpretation is informational only and does not replace formal laboratory review."
            ),
        }
    )


def build_dbs_html_report(
    study_summary: pd.DataFrame,
    bias_summary: pd.DataFrame,
    recovery_summary: pd.DataFrame,
    correlation_summary: pd.DataFrame,
    agreement_summary: pd.DataFrame,
    hematocrit_summary: pd.DataFrame,
    delay_summary: pd.DataFrame,
    instrument_summary: pd.DataFrame,
    outlier_review: pd.DataFrame,
    scientific_findings: list[str],
    sample_review: pd.DataFrame,
    overall_summary: dict[str, float | str],
    criteria_table: pd.DataFrame,
    interpretation: str,
    metadata: dict[str, object],
    criteria: dict[str, float],
    decision: str,
    visualization_html: dict[str, str] | None = None,
) -> str:
    """Build a complete DBS Validation HTML report."""

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    formatted_criteria = format_dbs_criteria_table(criteria_table)
    figures_html = "".join(
        f"<h3>{escape(title)}</h3>{figure_html}"
        for title, figure_html in (visualization_html or {}).items()
    )
    interpretation_html = "<br><br>".join(
        escape(paragraph) for paragraph in interpretation.split("\n\n")
    )
    findings_html = "<ul>" + "".join(
        f"<li>{escape(finding)}</li>" for finding in scientific_findings
    ) + "</ul>"
    hematocrit_html = (
        "<p>Hematocrit impact assessment was not available.</p>"
        if hematocrit_summary.empty
        else format_dbs_table(hematocrit_summary).to_html(index=False)
    )
    delay_html = (
        "<p>Extraction delay assessment was not available.</p>"
        if delay_summary.empty
        else format_dbs_table(delay_summary).to_html(index=False)
    )
    instrument_html = (
        "<p>Instrument comparison was not available.</p>"
        if instrument_summary.empty
        else format_dbs_table(instrument_summary).to_html(index=False)
    )
    outlier_html = (
        "<p>No outliers were identified for investigation.</p>"
        if outlier_review.empty
        else format_dbs_table(outlier_review).to_html(index=False)
    )
    notes = escape(str(metadata.get("Notes") or "None documented."))
    deviations = escape(str(metadata.get("Deviations") or "None documented."))
    conclusions = escape(str(metadata.get("Conclusions") or "Pending final review."))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>DBS Validation Report</title>
  <style>
    body {{ color:#1f2933; font-family:Arial,sans-serif; line-height:1.5; margin:40px; }}
    h1,h2,h3 {{ color:#102a43; }}
    table {{ border-collapse:collapse; margin:12px 0 28px; width:100%; }}
    th,td {{ border:1px solid #d9e2ec; padding:8px; text-align:right; }}
    th {{ background:#f0f4f8; }}
    td:first-child,th:first-child {{ text-align:left; }}
    .summary-grid {{ display:grid; gap:12px; grid-template-columns:repeat(3,minmax(0,1fr)); margin:14px 0 28px; }}
    .summary-card {{ border:1px solid #d9e2ec; border-radius:8px; padding:14px 16px; background:#fff; }}
    .summary-label {{ color:#52606d; font-size:.78rem; font-weight:bold; text-transform:uppercase; }}
    .summary-value {{ color:#102a43; font-size:1.25rem; font-weight:bold; }}
    .note {{ background:#f7f9fb; border-left:4px solid #2a6f97; padding:12px 16px; }}
    .status-badge {{ border-radius:999px; display:inline-block; font-size:.78rem; font-weight:bold; padding:6px 10px; }}
    .status-pass {{ background:#e3f9e5; color:#1f7a1f; }}
    .status-borderline {{ background:#fff8c5; color:#946200; }}
    .status-fail {{ background:#ffe3e3; color:#c92a2a; }}
  </style>
</head>
<body>
  <h1>DBS Validation Report</h1>
  <p><strong>Generated:</strong> {generated_at}</p>
  <h2>Study Metadata</h2>
  {_metadata_to_html(metadata)}
  <h2>Executive Summary</h2>
  {dbs_executive_summary_html(overall_summary, decision, formatted_criteria)}
  {format_dbs_overall_summary(overall_summary).to_html(index=False)}
  <h2>Acceptance Criteria</h2>
  <p>User-defined preliminary screening criteria: maximum percent bias {criteria['Maximum Percent Bias']:.2f}%; recovery {criteria['Minimum Recovery']:.2f}% to {criteria['Maximum Recovery']:.2f}%; minimum R² {criteria['Minimum R²']:.3f}; maximum mean difference {criteria['Maximum Mean Difference']:.3f}; borderline zone {criteria['Borderline Zone']:.2f}%.</p>
  <h2>Bias Analysis</h2>
  {format_dbs_table(bias_summary).to_html(index=False)}
  <h2>Recovery Analysis</h2>
  {format_dbs_table(recovery_summary).to_html(index=False)}
  <h2>Correlation Analysis</h2>
  {format_dbs_table(correlation_summary).to_html(index=False)}
  <h2>Agreement Analysis</h2>
  {format_dbs_table(agreement_summary).to_html(index=False)}
  <h2>Hematocrit Assessment</h2>
  {hematocrit_html}
  <h2>Extraction Delay Assessment</h2>
  {delay_html}
  <h2>Instrument Assessment</h2>
  {instrument_html}
  <h2>Acceptance Criteria Results</h2>
  {criteria_table_to_badged_html(formatted_criteria)}
  <h2>Sample-Level Review</h2>
  {format_dbs_table(sample_review.head(10)).to_html(index=False)}
  <h2>Outlier Investigation</h2>
  {outlier_html}
  <h2>Scientific Review Findings</h2>
  {findings_html}
  <h2>Visualizations</h2>
  {figures_html}
  <h2>Scientific Interpretation</h2>
  <p class="note">{interpretation_html}</p>
  <h2>Analyst Notes</h2>
  <p>{notes}</p>
  <h2>Deviations</h2>
  <p>{deviations}</p>
  <h2>Preliminary Conclusion</h2>
  <p>{conclusions}</p>
  <h2>Signature Section</h2>
  <p>Prepared By: ________________________________</p>
  <p>Reviewed By: ________________________________</p>
  <p>Approved By: ________________________________</p>
</body>
</html>
"""


def build_dbs_pdf_report(
    study_summary: pd.DataFrame,
    bias_summary: pd.DataFrame,
    recovery_summary: pd.DataFrame,
    correlation_summary: pd.DataFrame,
    agreement_summary: pd.DataFrame,
    hematocrit_summary: pd.DataFrame,
    delay_summary: pd.DataFrame,
    instrument_summary: pd.DataFrame,
    outlier_review: pd.DataFrame,
    scientific_findings: list[str],
    sample_review: pd.DataFrame,
    overall_summary: dict[str, float | str],
    criteria_table: pd.DataFrame,
    interpretation: str,
    metadata: dict[str, object],
    criteria: dict[str, float],
    decision: str,
) -> bytes:
    """Build a compact DBS Validation PDF report."""

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 12, "DBS Validation Report", ln=True, align="C")
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align="C")
    _pdf_add_table(pdf, "Study Metadata", pd.DataFrame([{"Field": k, "Value": v or "Not specified"} for k, v in metadata.items()]))
    _pdf_add_table(pdf, "Executive Summary", format_dbs_overall_summary(overall_summary))
    _pdf_add_table(pdf, "Acceptance Criteria Results", format_dbs_criteria_table(criteria_table))
    _pdf_add_table(pdf, "Bias Analysis", format_dbs_table(bias_summary))
    _pdf_add_table(pdf, "Recovery Analysis", format_dbs_table(recovery_summary))
    _pdf_add_table(pdf, "Correlation Analysis", format_dbs_table(correlation_summary))
    _pdf_add_table(pdf, "Agreement Analysis", format_dbs_table(agreement_summary))
    if not hematocrit_summary.empty:
        _pdf_add_table(pdf, "Hematocrit Assessment", format_dbs_table(hematocrit_summary))
    if not delay_summary.empty:
        _pdf_add_table(pdf, "Extraction Delay Assessment", format_dbs_table(delay_summary))
    if not instrument_summary.empty:
        _pdf_add_table(pdf, "Instrument Assessment", format_dbs_table(instrument_summary))
    _pdf_add_table(pdf, "Sample-Level Review", format_dbs_table(sample_review.head(8)))
    if not outlier_review.empty:
        _pdf_add_table(pdf, "Outlier Investigation", format_dbs_table(outlier_review.head(8)))
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Scientific Review Findings", ln=True)
    pdf.set_font("Arial", "", 9)
    for finding in scientific_findings:
        pdf.multi_cell(0, 5, _pdf_clean(f"- {finding}"))
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Scientific Interpretation", ln=True)
    pdf.set_font("Arial", "", 9)
    pdf.multi_cell(0, 5, _pdf_clean(interpretation))
    pdf.ln(4)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Signature Section", ln=True)
    pdf.set_font("Arial", "", 10)
    for label in ["Prepared By:", "Reviewed By:", "Approved By:", "Approval Date:"]:
        pdf.cell(0, 8, f"{label} ________________________________", ln=True)
    output = pdf.output(dest="S")
    if isinstance(output, bytes):
        return output
    return output.encode("latin-1")


def format_microtainer_table(table: pd.DataFrame) -> pd.DataFrame:
    """Format Microtainer validation tables."""

    return format_dbs_table(table)


def format_microtainer_criteria_table(criteria_table: pd.DataFrame) -> pd.DataFrame:
    """Format Microtainer acceptance criteria results."""

    formatted = criteria_table.copy()

    def format_observed(row: pd.Series) -> str:
        value = row["Observed Value"]
        if pd.isna(value):
            return ""
        criterion = str(row["Criterion"]).lower()
        suffix = "%" if "bias" in criterion or "recovery" in criterion else ""
        digits = 4 if "r²" in criterion else 2
        return _format_measurement(value, digits, suffix)

    formatted["Observed Value"] = formatted.apply(format_observed, axis=1)
    if "Pass/Fail Status" in formatted.columns:
        formatted["Pass/Fail Status"] = formatted["Pass/Fail Status"].map(
            lambda value: "PASS" if str(value).upper() == "PASS" else "FAIL"
        )
    if "Borderline Status" in formatted.columns:
        formatted["Borderline Status"] = formatted["Borderline Status"].map(
            lambda value: "YES" if bool(value) else "NO"
        )
    return formatted


def format_microtainer_overall_summary(summary: dict[str, float | str]) -> pd.DataFrame:
    """Format Microtainer overall summary metrics."""

    rows = [
        ("N", str(int(summary.get("N", 0)))),
        ("Mean Bias", _format_measurement(summary.get("Mean Bias"), 3)),
        ("Mean Percent Bias", _format_measurement(summary.get("Mean Percent Bias"), 2, "%")),
        ("Mean Recovery", _format_measurement(summary.get("Mean Recovery"), 2, "%")),
        ("R²", _format_measurement(summary.get("R²"), 4)),
        ("Slope", _format_measurement(summary.get("Slope"), 4)),
        ("Mean Difference", _format_measurement(summary.get("Mean Difference"), 3)),
        ("Lower 95% LoA", _format_measurement(summary.get("Lower Limit of Agreement"), 3)),
        ("Upper 95% LoA", _format_measurement(summary.get("Upper Limit of Agreement"), 3)),
        ("Worst Sample", str(summary.get("Worst Sample", ""))),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def build_microtainer_executive_summary(
    overall_summary: dict[str, float | str],
    decision: str,
    criteria_table: pd.DataFrame | None = None,
) -> dict[str, str]:
    """Build Microtainer executive summary card values."""

    failed_count = 0
    borderline_count = 0
    if criteria_table is not None and not criteria_table.empty:
        if "Pass/Fail Status" in criteria_table.columns:
            failed_count = int((criteria_table["Pass/Fail Status"].astype(str).str.upper() == "FAIL").sum())
        if "Borderline Status" in criteria_table.columns:
            borderline_count = int((criteria_table["Borderline Status"].astype(str).str.upper().isin(["YES", "TRUE"])).sum())
    return {
        "Overall Decision": decision,
        "Mean Bias": _format_measurement(overall_summary["Mean Bias"], 3),
        "Mean Recovery": _format_measurement(overall_summary["Mean Recovery"], 2, "%"),
        "R²": _format_measurement(overall_summary["R²"], 4),
        "Worst Sample": str(overall_summary["Worst Sample"]),
        "Borderline Findings": str(borderline_count),
        "Failed Criteria": str(failed_count),
    }


def generate_microtainer_interpretation(
    overall_summary: dict[str, float | str],
    criteria: dict[str, float],
    decision: str,
    metadata: dict[str, object],
) -> str:
    """Generate professional Microtainer validation interpretation."""

    study_name = metadata.get("Study Name") or "this Microtainer validation study"
    assay = metadata.get("Assay / Biomarker") or "the assay"
    if decision == "PASS":
        criteria_text = "The predefined acceptance criteria were met."
    else:
        criteria_text = "One or more predefined acceptance criteria were not met."
    return build_sectioned_interpretation(
        {
            "Study Objective": (
                f"{study_name} evaluated whether microtainer-derived {assay} results demonstrate "
                "acceptable analytical agreement with reference venous specimens."
            ),
            "Analytical Findings": (
                f"Mean percent bias was "
                f"{_format_measurement(overall_summary['Mean Percent Bias'], 2, '%')}, "
                f"recovery ranged from {_format_measurement(overall_summary['Minimum Recovery'], 2, '%')} "
                f"to {_format_measurement(overall_summary['Maximum Recovery'], 2, '%')}, "
                f"R² was {_format_measurement(overall_summary['R²'], 4)}, and the mean difference was "
                f"{_format_measurement(overall_summary['Mean Difference'], 3)}. Bland-Altman limits were "
                f"{_format_measurement(overall_summary['Lower Limit of Agreement'], 3)} to "
                f"{_format_measurement(overall_summary['Upper Limit of Agreement'], 3)}."
            ),
            "Acceptance Criteria Assessment": criteria_text,
            "Validation Conclusion": (
                f"The final validation decision is {decision}. "
                "This interpretation is informational only and does not replace formal laboratory review."
            ),
        }
    )


def microtainer_executive_summary_html(
    overall_summary: dict[str, float | str],
    decision: str,
    criteria_table: pd.DataFrame | None = None,
) -> str:
    """Render Microtainer executive summary cards."""

    cards = []
    for label, value in build_microtainer_executive_summary(overall_summary, decision, criteria_table).items():
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


def build_microtainer_html_report(
    study_summary: pd.DataFrame,
    bias_summary: pd.DataFrame,
    recovery_summary: pd.DataFrame,
    correlation_summary: pd.DataFrame,
    agreement_summary: pd.DataFrame,
    volume_summary: pd.DataFrame,
    delay_summary: pd.DataFrame,
    instrument_summary: pd.DataFrame,
    collection_site_summary: pd.DataFrame,
    outlier_review: pd.DataFrame,
    scientific_findings: list[str],
    sample_review: pd.DataFrame,
    overall_summary: dict[str, float | str],
    criteria_table: pd.DataFrame,
    interpretation: str,
    metadata: dict[str, object],
    criteria: dict[str, float],
    decision: str,
    visualization_html: dict[str, str] | None = None,
) -> str:
    """Build a complete Microtainer Validation HTML report."""

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    formatted_criteria = format_microtainer_criteria_table(criteria_table)
    figures_html = "".join(f"<h3>{escape(title)}</h3>{figure_html}" for title, figure_html in (visualization_html or {}).items())
    interpretation_html = "<br><br>".join(escape(paragraph) for paragraph in interpretation.split("\n\n"))
    review_html = (
        "<p>No samples required reviewer attention.</p>"
        if outlier_review.empty
        else format_microtainer_table(outlier_review).to_html(index=False)
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Microtainer Validation Report</title>
  <style>
    body {{ color:#1f2933; font-family:Arial,sans-serif; line-height:1.5; margin:40px; }}
    h1,h2,h3 {{ color:#102a43; }}
    table {{ border-collapse:collapse; margin:12px 0 28px; width:100%; }}
    th,td {{ border:1px solid #d9e2ec; padding:8px; text-align:right; }}
    th {{ background:#f0f4f8; }}
    td:first-child,th:first-child {{ text-align:left; }}
    .summary-grid {{ display:grid; gap:12px; grid-template-columns:repeat(3,minmax(0,1fr)); margin:14px 0 28px; }}
    .summary-card {{ border:1px solid #d9e2ec; border-radius:8px; padding:14px 16px; background:#fff; }}
    .summary-label {{ color:#52606d; font-size:.78rem; font-weight:bold; text-transform:uppercase; }}
    .summary-value {{ color:#102a43; font-size:1.25rem; font-weight:bold; }}
    .note {{ background:#f7f9fb; border-left:4px solid #2a6f97; padding:12px 16px; }}
    .status-badge {{ border-radius:999px; display:inline-block; font-size:.78rem; font-weight:bold; padding:6px 10px; }}
    .status-pass {{ background:#e3f9e5; color:#1f7a1f; }}
    .status-borderline {{ background:#fff8c5; color:#946200; }}
    .status-fail {{ background:#ffe3e3; color:#c92a2a; }}
  </style>
</head>
<body>
  <h1>Microtainer Validation Report</h1>
  <p><strong>Generated:</strong> {generated_at}</p>
  <h2>Study Information</h2>
  {_metadata_to_html(metadata)}
  <h2>Executive Summary</h2>
  {microtainer_executive_summary_html(overall_summary, decision, formatted_criteria)}
  {format_microtainer_overall_summary(overall_summary).to_html(index=False)}
  <h2>Acceptance Criteria</h2>
  <p>Maximum absolute percent bias {criteria['Maximum Absolute Percent Bias']:.2f}%; recovery {criteria['Recovery Lower Limit']:.2f}% to {criteria['Recovery Upper Limit']:.2f}%; minimum R² {criteria['Minimum R²']:.3f}; maximum mean difference {criteria['Maximum Mean Difference']:.3f}; borderline zone {criteria['Borderline Zone']:.2f}%.</p>
  <h2>Statistical Results</h2>
  <h3>Bias Analysis</h3>{format_microtainer_table(bias_summary).to_html(index=False)}
  <h3>Recovery Analysis</h3>{format_microtainer_table(recovery_summary).to_html(index=False)}
  <h3>Correlation Analysis</h3>{format_microtainer_table(correlation_summary).to_html(index=False)}
  <h3>Agreement Analysis</h3>{format_microtainer_table(agreement_summary).to_html(index=False)}
  <h2>Acceptance Criteria Results</h2>
  {criteria_table_to_badged_html(formatted_criteria)}
  <h2>Sample-Level Review</h2>{format_microtainer_table(sample_review.head(10)).to_html(index=False)}
  <h2>Samples Requiring Review</h2>{review_html}
  <h2>Visualizations</h2>{figures_html}
  <h2>Final Conclusion</h2><p class="note">{interpretation_html}</p>
  <h2>Signature Section</h2>
  <p>Prepared By: ________________________________</p>
  <p>Reviewed By: ________________________________</p>
  <p>Approved By: ________________________________</p>
</body>
</html>"""


def build_microtainer_pdf_report(
    study_summary: pd.DataFrame,
    bias_summary: pd.DataFrame,
    recovery_summary: pd.DataFrame,
    correlation_summary: pd.DataFrame,
    agreement_summary: pd.DataFrame,
    volume_summary: pd.DataFrame,
    delay_summary: pd.DataFrame,
    instrument_summary: pd.DataFrame,
    collection_site_summary: pd.DataFrame,
    outlier_review: pd.DataFrame,
    scientific_findings: list[str],
    sample_review: pd.DataFrame,
    overall_summary: dict[str, float | str],
    criteria_table: pd.DataFrame,
    interpretation: str,
    metadata: dict[str, object],
    criteria: dict[str, float],
    decision: str,
) -> bytes:
    """Build a compact Microtainer Validation PDF report."""

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 12, "Microtainer Validation Report", ln=True, align="C")
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align="C")
    _pdf_add_table(pdf, "Study Information", pd.DataFrame([{"Field": k, "Value": v or "Not specified"} for k, v in metadata.items()]))
    _pdf_add_table(pdf, "Executive Summary", format_microtainer_overall_summary(overall_summary))
    _pdf_add_table(pdf, "Acceptance Criteria Results", format_microtainer_criteria_table(criteria_table))
    for title, table in [
        ("Bias Analysis", bias_summary),
        ("Recovery Analysis", recovery_summary),
        ("Correlation Analysis", correlation_summary),
        ("Agreement Analysis", agreement_summary),
        ("Samples Requiring Review", outlier_review.head(8)),
        ("Sample-Level Review", sample_review.head(8)),
    ]:
        if not table.empty:
            _pdf_add_table(pdf, title, format_microtainer_table(table))
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Final Conclusion", ln=True)
    pdf.set_font("Arial", "", 9)
    pdf.multi_cell(0, 5, _pdf_clean(interpretation))
    output = pdf.output(dest="S")
    if isinstance(output, bytes):
        return output
    return output.encode("latin-1")


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
