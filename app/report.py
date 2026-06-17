"""Report and interpretation helpers for validation analytics outputs."""

from __future__ import annotations

from datetime import datetime
from html import escape

import pandas as pd


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
    return criteria_table[
        [
            "Criterion",
            "Observed Value",
            "Acceptance Limit",
            "Pass/Fail Status",
            "Borderline Status",
        ]
    ]


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


def _metadata_to_html(metadata: dict[str, object]) -> str:
    """Render study metadata as HTML rows."""

    rows = []
    for key, value in metadata.items():
        rows.append(
            f"<tr><th>{escape(str(key))}</th><td>{escape(str(value or 'Not specified'))}</td></tr>"
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
    criteria_results_html = criteria_table.to_html(
        index=False, float_format=lambda value: f"{value:.4g}"
    )
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
