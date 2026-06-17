"""Plotly visualization utilities for method-comparison validation studies."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _format_number(value: float, digits: int = 3) -> str:
    """Format numeric labels while handling missing values."""

    if pd.isna(value):
        return "NA"
    return f"{value:.{digits}f}"


def create_scatter_plot(analyzed_data: pd.DataFrame, summary: dict[str, float]) -> go.Figure:
    """Create a reference-vs-candidate scatter plot with regression line."""

    figure = px.scatter(
        analyzed_data,
        x="Reference",
        y="Candidate",
        labels={"Reference": "Reference Result", "Candidate": "Candidate Result"},
        title="Reference vs Candidate Method",
        template="plotly_white",
    )

    slope = summary.get("Slope", np.nan)
    intercept = summary.get("Intercept", np.nan)
    if not pd.isna(slope) and not pd.isna(intercept):
        x_values = np.array(
            [analyzed_data["Reference"].min(), analyzed_data["Reference"].max()]
        )
        y_values = slope * x_values + intercept
        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines",
                name="Regression line",
                line={"color": "#d62728", "width": 2},
            )
        )

    equation = (
        f"y = {_format_number(summary.get('Slope'))}x "
        f"+ {_format_number(summary.get('Intercept'))}"
    )
    r_squared = f"R² = {_format_number(summary.get('R²'))}"
    figure.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        text=f"{equation}<br>{r_squared}",
        showarrow=False,
        align="left",
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#d9d9d9",
        borderwidth=1,
    )
    figure.update_layout(legend_title_text="")
    return figure


def create_percent_bias_histogram(analyzed_data: pd.DataFrame) -> go.Figure:
    """Create a histogram of percent bias values."""

    figure = px.histogram(
        analyzed_data.dropna(subset=["Percent Bias"]),
        x="Percent Bias",
        nbins=20,
        labels={"Percent Bias": "Percent Bias (%)"},
        title="Percent Bias Distribution",
        template="plotly_white",
    )
    figure.update_traces(marker_color="#2a6f97")
    return figure


def create_difference_plot(analyzed_data: pd.DataFrame) -> go.Figure:
    """Create an average-of-methods vs difference plot with LOA lines."""

    mean_difference = analyzed_data["Difference"].mean()
    sd_difference = analyzed_data["Difference"].std(ddof=1)
    upper_loa = mean_difference + (1.96 * sd_difference)
    lower_loa = mean_difference - (1.96 * sd_difference)

    figure = px.scatter(
        analyzed_data,
        x="Average",
        y="Difference",
        labels={
            "Average": "Average of Reference and Candidate",
            "Difference": "Difference (Candidate - Reference)",
        },
        title="Difference Plot",
        template="plotly_white",
    )
    figure.add_hline(
        y=mean_difference,
        line_dash="dash",
        line_color="#d62728",
        annotation_text=f"Mean difference = {mean_difference:.3f}",
        annotation_position="top left",
    )
    figure.add_hline(
        y=upper_loa,
        line_dash="dot",
        line_color="#9467bd",
        annotation_text=f"Upper LOA = {upper_loa:.3f}",
        annotation_position="top left",
    )
    figure.add_hline(
        y=lower_loa,
        line_dash="dot",
        line_color="#9467bd",
        annotation_text=f"Lower LOA = {lower_loa:.3f}",
        annotation_position="bottom left",
    )
    figure.add_hline(y=0, line_color="#808080", line_width=1)
    return figure


def create_precision_run_chart(analyzed_data: pd.DataFrame) -> go.Figure:
    """Create a run chart for repeated precision measurements."""

    hover_columns = [
        column
        for column in ["Sample ID", "Day", "Run", "Replicate", "Result"]
        if column in analyzed_data.columns
    ]
    figure = px.line(
        analyzed_data,
        x="Measurement Order",
        y="Result",
        color="Level",
        markers=True,
        hover_data=hover_columns,
        title="Precision Run Chart",
        labels={"Result": "Result", "Measurement Order": "Measurement Order"},
        template="plotly_white",
    )
    return figure


def create_precision_cv_bar_chart(level_summary: pd.DataFrame) -> go.Figure:
    """Create a precision summary bar chart showing CV% by level."""

    figure = px.bar(
        level_summary,
        x="Level",
        y="CV%",
        text="CV%",
        title="Precision Summary by Level",
        labels={"CV%": "CV%"},
        template="plotly_white",
    )
    figure.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    figure.update_layout(yaxis_title="CV%", uniformtext_minsize=8)
    return figure


def create_precision_box_plot(analyzed_data: pd.DataFrame) -> go.Figure:
    """Create a box plot showing result distribution by level."""

    figure = px.box(
        analyzed_data,
        x="Level",
        y="Result",
        color="Level",
        points="all",
        title="Precision Result Distribution",
        labels={"Result": "Result"},
        template="plotly_white",
    )
    figure.update_layout(showlegend=False)
    return figure


def create_linearity_plot(
    level_summary: pd.DataFrame, regression_summary: dict[str, float | str]
) -> go.Figure:
    """Create expected-vs-observed linearity plot with regression line."""

    figure = px.scatter(
        level_summary,
        x="Expected Result",
        y="Mean Observed Result",
        text="Level",
        title="Linearity Plot",
        labels={
            "Expected Result": "Expected Result",
            "Mean Observed Result": "Mean Observed Result",
        },
        template="plotly_white",
    )
    slope = regression_summary.get("Slope", np.nan)
    intercept = regression_summary.get("Intercept", np.nan)
    if not pd.isna(slope) and not pd.isna(intercept):
        x_values = np.array(
            [level_summary["Expected Result"].min(), level_summary["Expected Result"].max()]
        )
        y_values = slope * x_values + intercept
        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines",
                name="Regression line",
                line={"color": "#d62728", "width": 3},
            )
        )
    annotation = (
        f"y = {_format_number(slope, 4)}x + {_format_number(intercept, 4)}"
        f"<br>R² = {_format_number(regression_summary.get('R²'), 4)}"
    )
    figure.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        text=annotation,
        showarrow=False,
        align="left",
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#d9d9d9",
        borderwidth=1,
        borderpad=8,
        font={"size": 14, "color": "#102a43"},
    )
    figure.update_traces(
        marker={"color": "#2a6f97", "line": {"color": "#102a43", "width": 1}, "size": 10},
        textfont={"size": 12},
        textposition="top center",
    )
    figure.update_layout(
        margin={"l": 70, "r": 40, "t": 80, "b": 70},
        legend_title_text="",
    )
    return figure


def create_linearity_residual_plot(
    level_summary: pd.DataFrame, regression_summary: dict[str, float | str]
) -> go.Figure:
    """Create residual plot from mean observed minus predicted values."""

    slope = regression_summary.get("Slope", np.nan)
    intercept = regression_summary.get("Intercept", np.nan)
    residual_data = level_summary.copy()
    residual_data["Predicted"] = slope * residual_data["Expected Result"] + intercept
    residual_data["Residual"] = (
        residual_data["Mean Observed Result"] - residual_data["Predicted"]
    )
    figure = px.scatter(
        residual_data,
        x="Expected Result",
        y="Residual",
        text="Level",
        title="Residual Plot<br><sup>Observed - Predicted Residuals</sup>",
        labels={
            "Expected Result": "Expected Result",
            "Residual": "Observed - Predicted Residual",
        },
        template="plotly_white",
    )
    figure.add_hline(
        y=0,
        line_color="#808080",
        line_width=1.5,
        line_dash="dash",
        annotation_text="Zero residual",
        annotation_position="top left",
    )
    figure.update_traces(
        marker={"color": "#2a6f97", "symbol": "diamond", "size": 10},
        textposition="top center",
    )
    figure.update_layout(margin={"l": 70, "r": 35, "t": 80, "b": 65})
    return figure


def create_percent_recovery_plot(level_summary: pd.DataFrame) -> go.Figure:
    """Create percent recovery plot with 90%, 100%, and 110% reference lines."""

    figure = px.line(
        level_summary,
        x="Expected Result",
        y="Percent Recovery",
        text="Level",
        markers=True,
        title="Percent Recovery by Linearity Level",
        template="plotly_white",
    )
    for y_value, dash in [(90, "dash"), (100, "solid"), (110, "dash")]:
        figure.add_hline(
            y=y_value,
            line_dash=dash,
            line_color="#808080" if y_value == 100 else "#9467bd",
            annotation_text=f"{y_value}%",
            annotation_position="top left",
        )
    figure.update_traces(textposition="top center")
    return figure
