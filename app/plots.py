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
