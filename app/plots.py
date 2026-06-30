"""Plotly visualization utilities for method-comparison validation studies."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from validation.visualization import STATUS_COLOR_MAP

METHOD_POINT_COLOR = "#4865f4"
METHOD_REGRESSION_COLOR = "#102a43"
METHOD_REFERENCE_COLOR = "#64748b"
METHOD_BIAS_COLOR = "#b42318"
METHOD_LIMIT_COLOR = "#6d5bd0"
METHOD_GRID_COLOR = "#e6ebf2"
METHOD_TEXT_COLOR = "#334155"


def _format_number(value: float, digits: int = 3) -> str:
    """Format numeric labels while handling missing values."""

    if pd.isna(value):
        return "NA"
    return f"{value:.{digits}f}"


def _apply_method_comparison_figure_style(figure: go.Figure, *, height: int = 430) -> go.Figure:
    """Apply the professional Method Comparison chart style."""

    figure.update_layout(
        template="plotly_white",
        height=height,
        margin={"l": 64, "r": 26, "t": 42, "b": 58},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={"family": "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif", "color": METHOD_TEXT_COLOR},
        title={"font": {"size": 16, "color": "#202939"}, "x": 0, "xanchor": "left"},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "font": {"size": 12, "color": METHOD_TEXT_COLOR},
        },
        legend_title_text="",
        hovermode="closest",
    )
    figure.update_xaxes(
        showgrid=True,
        gridcolor=METHOD_GRID_COLOR,
        zeroline=False,
        linecolor="#cbd5e1",
        ticks="outside",
        tickfont={"color": "#64748b"},
        title_font={"color": "#64748b", "size": 13},
    )
    figure.update_yaxes(
        showgrid=True,
        gridcolor=METHOD_GRID_COLOR,
        zeroline=False,
        linecolor="#cbd5e1",
        ticks="outside",
        tickfont={"color": "#64748b"},
        title_font={"color": "#64748b", "size": 13},
    )
    return figure


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
    figure.update_traces(
        marker={
            "color": METHOD_POINT_COLOR,
            "size": 7,
            "opacity": 0.82,
            "line": {"color": "#ffffff", "width": 0.8},
        },
        hovertemplate="Reference=%{x:.3f}<br>Candidate=%{y:.3f}<extra></extra>",
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
                line={"color": METHOD_REGRESSION_COLOR, "width": 2.5},
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
        bgcolor="rgba(248,250,252,0.94)",
        bordercolor="#cbd5e1",
        borderwidth=1,
        borderpad=8,
        font={"size": 12, "color": METHOD_TEXT_COLOR},
    )
    return _apply_method_comparison_figure_style(figure)


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
    figure.update_traces(
        marker={
            "color": METHOD_POINT_COLOR,
            "size": 7,
            "opacity": 0.82,
            "line": {"color": "#ffffff", "width": 0.8},
        },
        hovertemplate="Average=%{x:.3f}<br>Difference=%{y:.3f}<extra></extra>",
    )
    figure.add_hline(
        y=mean_difference,
        line_dash="dash",
        line_color=METHOD_BIAS_COLOR,
        annotation_text=f"Mean difference = {mean_difference:.3f}",
        annotation_position="top left",
    )
    figure.add_hline(
        y=upper_loa,
        line_dash="dot",
        line_color=METHOD_LIMIT_COLOR,
        annotation_text=f"Upper LOA = {upper_loa:.3f}",
        annotation_position="top left",
    )
    figure.add_hline(
        y=lower_loa,
        line_dash="dot",
        line_color=METHOD_LIMIT_COLOR,
        annotation_text=f"Lower LOA = {lower_loa:.3f}",
        annotation_position="bottom left",
    )
    figure.add_hline(y=0, line_color=METHOD_REFERENCE_COLOR, line_width=1.2)
    return _apply_method_comparison_figure_style(figure)


def create_method_comparison_residual_plot(analyzed_data: pd.DataFrame, summary: dict[str, float]) -> go.Figure:
    """Create residual plot for candidate results against fitted regression."""

    data = analyzed_data.copy()
    slope = summary.get("Slope", np.nan)
    intercept = summary.get("Intercept", np.nan)
    if pd.isna(slope) or pd.isna(intercept):
        data["Residual"] = data["Candidate"] - data["Reference"]
    else:
        data["Residual"] = data["Candidate"] - ((float(slope) * data["Reference"]) + float(intercept))
    figure = px.scatter(
        data,
        x="Reference",
        y="Residual",
        labels={"Reference": "Reference Result", "Residual": "Residual"},
        title="Residuals vs Reference Result",
        template="plotly_white",
    )
    figure.update_traces(
        marker={
            "color": METHOD_POINT_COLOR,
            "size": 7,
            "opacity": 0.82,
            "line": {"color": "#ffffff", "width": 0.8},
        },
        hovertemplate="Reference=%{x:.3f}<br>Residual=%{y:.3f}<extra></extra>",
    )
    figure.add_hline(y=0, line_dash="dash", line_color=METHOD_REFERENCE_COLOR, annotation_text="Zero residual", annotation_position="top left")
    figure.update_layout(yaxis_tickformat=".2f")
    return _apply_method_comparison_figure_style(figure)


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


def create_precision_variance_component_plot(variance_components: pd.DataFrame) -> go.Figure:
    """Create a precision component CV% plot by QC level."""

    component_columns = [
        "Within-Run CV%",
        "Between-Run CV%",
        "Between-Day CV%",
        "Total CV%",
    ]
    available_columns = [
        column for column in component_columns if column in variance_components.columns
    ]
    plot_data = variance_components.melt(
        id_vars=["Level"],
        value_vars=available_columns,
        var_name="Precision Component",
        value_name="CV%",
    ).dropna(subset=["CV%"])
    plot_data["Precision Component"] = plot_data["Precision Component"].str.replace(" CV%", "", regex=False)

    figure = px.bar(
        plot_data,
        x="Level",
        y="CV%",
        color="Precision Component",
        barmode="group",
        text="CV%",
        title="Precision Component Summary",
        labels={"CV%": "CV (%)"},
        template="plotly_white",
    )
    figure.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    figure.update_layout(yaxis_title="CV%", uniformtext_minsize=8)
    return figure


def create_precision_run_variation_plot(run_summary: pd.DataFrame) -> go.Figure:
    """Create run-to-run CV% plot by QC level."""

    figure = px.bar(
        run_summary,
        x="Run",
        y="CV%",
        color="Level",
        barmode="group",
        text="CV%",
        title="Run-to-Run Variation",
        labels={"CV%": "CV%", "Run": "Run"},
        template="plotly_white",
    )
    figure.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    figure.update_layout(yaxis_title="CV%", uniformtext_minsize=8, margin={"l": 70, "r": 40, "t": 70, "b": 65})
    return figure


def create_precision_day_variation_plot(day_summary: pd.DataFrame) -> go.Figure:
    """Create day-to-day CV% plot by QC level."""

    figure = px.line(
        day_summary,
        x="Day",
        y="CV%",
        color="Level",
        markers=True,
        title="Day-to-Day Variation",
        labels={"CV%": "CV%", "Day": "Day"},
        template="plotly_white",
    )
    figure.update_traces(line={"width": 2.5}, marker={"size": 9, "line": {"color": "#102a43", "width": 1}})
    figure.update_layout(yaxis_title="CV%", margin={"l": 70, "r": 40, "t": 70, "b": 65})
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


def create_stability_trend_plot(
    timepoint_summary: pd.DataFrame, baseline_mean: float
) -> go.Figure:
    """Create a stability trend plot showing mean result by timepoint."""

    color = "Storage Condition" if "Storage Condition" in timepoint_summary.columns else None
    figure = px.line(
        timepoint_summary,
        x="Timepoint",
        y="Timepoint Mean",
        color=color,
        markers=True,
        title="Stability Trend Plot",
        labels={"Timepoint Mean": "Mean Result", "Timepoint": "Timepoint"},
        template="plotly_white",
    )
    figure.add_hline(
        y=baseline_mean,
        line_dash="dash",
        line_color="#808080",
        annotation_text=f"Baseline mean = {baseline_mean:.2f}",
        annotation_position="top left",
    )
    figure.update_traces(marker={"size": 9})
    figure.update_layout(margin={"l": 70, "r": 40, "t": 70, "b": 65})
    return figure


def create_stability_percent_change_plot(
    timepoint_summary: pd.DataFrame,
    max_percent_change: float,
    borderline_zone: float = 0.0,
) -> go.Figure:
    """Create a percent change plot with acceptance threshold lines."""

    color = "Storage Condition" if "Storage Condition" in timepoint_summary.columns else None
    plot_data = timepoint_summary.copy()
    caution_floor = max(0.0, max_percent_change - borderline_zone)
    plot_data["Review Status"] = np.select(
        [
            plot_data["Mean Percent Change"].abs() >= max_percent_change,
            plot_data["Mean Percent Change"].abs() > caution_floor,
        ],
        ["FAIL", "PASS WITH CAUTION"],
        default="PASS",
    )
    figure = px.line(
        plot_data,
        x="Timepoint",
        y="Mean Percent Change",
        color=color,
        markers=True,
        title="Percent Change from Baseline",
        labels={
            "Timepoint": "Timepoint",
            "Mean Percent Change": "Mean Percent Change (%)",
        },
        template="plotly_white",
    )
    figure.add_trace(
        go.Scatter(
            x=plot_data["Timepoint"],
            y=plot_data["Mean Percent Change"],
            mode="markers",
            name="Review status",
            marker={
                "color": plot_data["Review Status"].map(
                    {"PASS": "#2a6f97", "PASS WITH CAUTION": "#f59f00", "FAIL": "#c92a2a"}
                ),
                "size": 11,
                "line": {"color": "#102a43", "width": 1},
            },
            text=plot_data["Review Status"],
            hovertemplate="Timepoint=%{x}<br>Percent Change=%{y:.2f}%<br>Status=%{text}<extra></extra>",
        )
    )
    for y_value in [max_percent_change, -max_percent_change]:
        figure.add_hline(
            y=y_value,
            line_dash="dash",
            line_color="#9467bd",
            annotation_text=f"{y_value:.1f}%",
            annotation_position="top left" if y_value > 0 else "bottom left",
        )
    figure.add_hline(y=0, line_color="#808080", line_width=1)
    figure.update_traces(marker={"size": 9})
    figure.update_layout(margin={"l": 70, "r": 40, "t": 70, "b": 65})
    return figure


def create_stability_recovery_plot(
    timepoint_summary: pd.DataFrame,
    min_recovery: float,
    borderline_zone: float = 0.0,
) -> go.Figure:
    """Create a recovery plot with recovery threshold lines."""

    color = "Storage Condition" if "Storage Condition" in timepoint_summary.columns else None
    plot_data = timepoint_summary.copy()
    caution_ceiling = min_recovery + borderline_zone
    plot_data["Review Status"] = np.select(
        [
            plot_data["Percent Recovery"] <= min_recovery,
            plot_data["Percent Recovery"] < caution_ceiling,
        ],
        ["FAIL", "PASS WITH CAUTION"],
        default="PASS",
    )
    figure = px.line(
        plot_data,
        x="Timepoint",
        y="Percent Recovery",
        color=color,
        markers=True,
        title="Percent Recovery by Timepoint",
        labels={"Timepoint": "Timepoint", "Percent Recovery": "Percent Recovery (%)"},
        template="plotly_white",
    )
    figure.add_trace(
        go.Scatter(
            x=plot_data["Timepoint"],
            y=plot_data["Percent Recovery"],
            mode="markers",
            name="Review status",
            marker={
                "color": plot_data["Review Status"].map(
                    {"PASS": "#2a6f97", "PASS WITH CAUTION": "#f59f00", "FAIL": "#c92a2a"}
                ),
                "size": 11,
                "line": {"color": "#102a43", "width": 1},
            },
            text=plot_data["Review Status"],
            hovertemplate="Timepoint=%{x}<br>Recovery=%{y:.2f}%<br>Status=%{text}<extra></extra>",
        )
    )
    figure.add_hline(
        y=100,
        line_color="#808080",
        line_width=1,
        annotation_text="100%",
        annotation_position="top left",
    )
    figure.add_hline(
        y=min_recovery,
        line_dash="dash",
        line_color="#9467bd",
        annotation_text=f"Minimum recovery = {min_recovery:.1f}%",
        annotation_position="bottom left",
    )
    figure.update_traces(marker={"size": 9})
    figure.update_layout(margin={"l": 70, "r": 40, "t": 70, "b": 65})
    return figure


def create_stability_bias_plot(
    timepoint_summary: pd.DataFrame,
    max_abs_bias: float,
    borderline_zone: float = 0.0,
) -> go.Figure:
    """Create a bias-over-time plot with acceptance limit lines."""

    color = "Storage Condition" if "Storage Condition" in timepoint_summary.columns else None
    plot_data = timepoint_summary.copy()
    bias_zone = max_abs_bias * (borderline_zone / 100)
    caution_floor = max(0.0, max_abs_bias - bias_zone)
    plot_data["Review Status"] = np.select(
        [
            plot_data["Bias"].abs() >= max_abs_bias,
            plot_data["Bias"].abs() > caution_floor,
        ],
        ["FAIL", "PASS WITH CAUTION"],
        default="PASS",
    )
    figure = px.line(
        plot_data,
        x="Timepoint",
        y="Bias",
        color=color,
        markers=True,
        title="Bias Over Time",
        labels={"Timepoint": "Timepoint", "Bias": "Bias"},
        template="plotly_white",
    )
    for y_value in [max_abs_bias, -max_abs_bias]:
        figure.add_hline(
            y=y_value,
            line_dash="dash",
            line_color="#9467bd",
            annotation_text=f"{y_value:.2f}",
            annotation_position="top left" if y_value > 0 else "bottom left",
        )
    figure.add_hline(y=0, line_color="#808080", line_width=1)
    figure.add_trace(
        go.Scatter(
            x=plot_data["Timepoint"],
            y=plot_data["Bias"],
            mode="markers",
            name="Review status",
            marker={
                "color": plot_data["Review Status"].map(
                    {"PASS": "#2a6f97", "PASS WITH CAUTION": "#f59f00", "FAIL": "#c92a2a"}
                ),
                "size": 11,
                "line": {"color": "#102a43", "width": 1},
            },
            text=plot_data["Review Status"],
            hovertemplate="Timepoint=%{x}<br>Bias=%{y:.2f}<br>Status=%{text}<extra></extra>",
        )
    )
    figure.update_layout(margin={"l": 70, "r": 40, "t": 70, "b": 65})
    return figure


def create_condition_difference_plot(
    condition_comparison: pd.DataFrame,
    acceptance_limit: float | None = None,
) -> go.Figure:
    """Create a storage-condition difference plot."""

    figure = px.line(
        condition_comparison,
        x="Timepoint",
        y="Difference",
        markers=True,
        title="Condition Difference Plot",
        labels={
            "Timepoint": "Timepoint",
            "Difference": "Difference Between Conditions",
        },
        template="plotly_white",
    )
    figure.add_hline(
        y=0,
        line_color="#808080",
        line_dash="dash",
        annotation_text="Zero difference",
        annotation_position="top left",
    )
    if acceptance_limit is not None:
        for y_value in [acceptance_limit, -acceptance_limit]:
            figure.add_hline(
                y=y_value,
                line_dash="dash",
                line_color="#9467bd",
                annotation_text=f"{y_value:+.2f} units",
                annotation_position="top left" if y_value > 0 else "bottom left",
            )
    figure.update_traces(
        marker={"color": "#2a6f97", "size": 10, "line": {"color": "#102a43", "width": 1}},
        line={"color": "#2a6f97", "width": 2.5},
        hovertemplate="Timepoint=%{x}<br>Difference=%{y:.2f}<extra></extra>",
    )
    figure.update_layout(margin={"l": 70, "r": 40, "t": 70, "b": 65})
    return figure


def create_individual_stability_plot(analyzed_data: pd.DataFrame) -> go.Figure:
    """Create individual sample stability trajectories over time."""

    hover_columns = [
        column
        for column in ["Sample ID", "Storage Condition", "Replicate", "Result", "Percent Change"]
        if column in analyzed_data.columns
    ]
    figure = px.line(
        analyzed_data,
        x="Timepoint",
        y="Result",
        color="Sample ID",
        line_group="Sample ID",
        markers=True,
        hover_data=hover_columns,
        title="Individual Sample Stability Plot",
        labels={"Result": "Result", "Timepoint": "Timepoint"},
        template="plotly_white",
    )
    figure.update_traces(marker={"size": 7}, line={"width": 1.8})
    figure.update_layout(margin={"l": 70, "r": 40, "t": 70, "b": 65})
    return figure


def create_accuracy_expected_observed_plot(accuracy_summary: pd.DataFrame) -> go.Figure:
    """Create expected-vs-observed accuracy plot with identity line."""

    figure = px.scatter(
        accuracy_summary,
        x="Expected Result",
        y="Mean Observed Result",
        text="Level",
        title="Expected vs Observed Accuracy Plot",
        labels={
            "Expected Result": "Expected Result",
            "Mean Observed Result": "Mean Observed Result",
        },
        template="plotly_white",
    )
    min_value = min(
        accuracy_summary["Expected Result"].min(),
        accuracy_summary["Mean Observed Result"].min(),
    )
    max_value = max(
        accuracy_summary["Expected Result"].max(),
        accuracy_summary["Mean Observed Result"].max(),
    )
    figure.add_trace(
        go.Scatter(
            x=[min_value, max_value],
            y=[min_value, max_value],
            mode="lines",
            name="Identity line",
            line={"color": "#808080", "dash": "dash", "width": 2},
        )
    )
    figure.update_traces(
        marker={"color": "#2a6f97", "size": 10, "line": {"color": "#102a43", "width": 1}},
        textposition="top center",
    )
    figure.update_layout(margin={"l": 70, "r": 40, "t": 70, "b": 65})
    return figure


def create_accuracy_percent_bias_plot(
    accuracy_summary: pd.DataFrame,
    max_abs_percent_bias: float,
    borderline_zone: float = 0.0,
) -> go.Figure:
    """Create percent bias by level plot with threshold lines."""

    plot_data = accuracy_summary.copy()
    caution_floor = max(0.0, max_abs_percent_bias - borderline_zone)
    plot_data["Review Status"] = np.select(
        [
            plot_data["Percent Bias"].abs() >= max_abs_percent_bias,
            plot_data["Percent Bias"].abs() > caution_floor,
        ],
        ["FAIL", "PASS WITH CAUTION"],
        default="PASS",
    )
    figure = px.bar(
        plot_data,
        x="Level",
        y="Percent Bias",
        title="Percent Bias by Level",
        labels={"Percent Bias": "Percent Bias (%)"},
        template="plotly_white",
    )
    figure.update_traces(marker_color="#2a6f97", hovertemplate="Level=%{x}<br>Bias=%{y:.2f}%<extra></extra>")
    for y_value in [max_abs_percent_bias, -max_abs_percent_bias]:
        figure.add_hline(
            y=y_value,
            line_dash="dash",
            line_color="#9467bd",
            annotation_text=f"{y_value:.1f}%",
            annotation_position="top left" if y_value > 0 else "bottom left",
        )
    figure.add_hline(y=0, line_color="#808080", line_width=1)
    figure.update_layout(showlegend=False, margin={"l": 70, "r": 40, "t": 70, "b": 65})
    return figure


def create_accuracy_recovery_plot(
    accuracy_summary: pd.DataFrame,
    min_recovery: float,
    max_recovery: float,
    borderline_zone: float = 0.0,
) -> go.Figure:
    """Create recovery by level plot with reference and threshold lines."""

    plot_data = accuracy_summary.copy()
    plot_data["Review Status"] = np.select(
        [
            (plot_data["Percent Recovery"] <= min_recovery)
            | (plot_data["Percent Recovery"] >= max_recovery),
            (plot_data["Percent Recovery"] < min_recovery + borderline_zone)
            | (plot_data["Percent Recovery"] > max_recovery - borderline_zone),
        ],
        ["FAIL", "PASS WITH CAUTION"],
        default="PASS",
    )
    figure = px.line(
        plot_data,
        x="Level",
        y="Percent Recovery",
        markers=True,
        title="Recovery by Level",
        labels={"Percent Recovery": "Percent Recovery (%)"},
        template="plotly_white",
    )
    figure.add_trace(
        go.Scatter(
            x=plot_data["Level"],
            y=plot_data["Percent Recovery"],
            mode="markers",
            name="Recovery",
            marker={
                "color": plot_data["Review Status"].map(
                    STATUS_COLOR_MAP
                ),
                "size": 11,
                "line": {"color": "#102a43", "width": 1},
            },
            text=plot_data["Review Status"],
            hovertemplate="Level=%{x}<br>Recovery=%{y:.2f}%<br>Status=%{text}<extra></extra>",
        )
    )
    for y_value, dash in [(min_recovery, "dash"), (100, "solid"), (max_recovery, "dash")]:
        figure.add_hline(
            y=y_value,
            line_dash=dash,
            line_color="#808080" if y_value == 100 else "#9467bd",
            annotation_text=f"{y_value:.1f}%",
            annotation_position="top left",
        )
    figure.update_layout(showlegend=False, margin={"l": 70, "r": 40, "t": 70, "b": 65})
    return figure


def create_accuracy_replicate_distribution_plot(analyzed_data: pd.DataFrame) -> go.Figure:
    """Create observed result distribution by accuracy level."""

    hover_columns = [
        column
        for column in ["Sample ID", "Replicate", "Expected Result", "Observed Result"]
        if column in analyzed_data.columns
    ]
    figure = px.box(
        analyzed_data,
        x="Level",
        y="Observed Result",
        color="Level",
        points="all",
        hover_data=hover_columns,
        title="Replicate Distribution by Level",
        labels={"Observed Result": "Observed Result"},
        template="plotly_white",
    )
    figure.update_layout(showlegend=False, margin={"l": 70, "r": 40, "t": 70, "b": 65})
    return figure


def create_accuracy_performance_heatmap(accuracy_summary: pd.DataFrame) -> go.Figure:
    """Create a heatmap for rapid level-by-metric accuracy review."""

    heatmap_data = accuracy_summary[
        ["Level", "Percent Bias", "Percent Recovery", "Absolute Difference"]
    ].copy()
    heatmap_data = heatmap_data.rename(columns={"Absolute Difference": "Absolute Bias"})
    metric_order = ["Percent Bias", "Percent Recovery", "Absolute Bias"]
    matrix = heatmap_data.set_index("Level")[metric_order].T
    figure = go.Figure(
        data=go.Heatmap(
            z=matrix.values,
            x=matrix.columns.astype(str),
            y=matrix.index,
            colorscale="RdYlGn_r",
            colorbar={"title": "Observed value"},
            text=np.vectorize(lambda value: f"{value:.2f}" if not pd.isna(value) else "")(
                matrix.values
            ),
            texttemplate="%{text}",
            hovertemplate="Level=%{x}<br>Metric=%{y}<br>Value=%{z:.2f}<extra></extra>",
        )
    )
    figure.update_layout(
        title="Accuracy Performance Heatmap",
        xaxis_title="Level",
        yaxis_title="Metric",
        margin={"l": 90, "r": 40, "t": 70, "b": 65},
        template="plotly_white",
    )
    return figure


def create_individual_accuracy_bias_plot(
    analyzed_data: pd.DataFrame,
    max_abs_bias: float,
) -> go.Figure:
    """Create sample-level bias plot with mean bias and acceptance limits."""

    plot_data = analyzed_data.copy()
    plot_data["Individual Bias"] = (
        plot_data["Observed Result"] - plot_data["Expected Result"]
    )
    mean_bias = plot_data["Individual Bias"].mean()
    hover_columns = [
        column
        for column in ["Sample ID", "Level", "Replicate", "Observed Result"]
        if column in plot_data.columns
    ]
    figure = px.scatter(
        plot_data,
        x="Expected Result",
        y="Individual Bias",
        color="Level",
        hover_data=hover_columns,
        title="Individual Sample Bias Plot",
        labels={
            "Expected Result": "Expected Result",
            "Individual Bias": "Individual Sample Bias",
        },
        template="plotly_white",
    )
    figure.add_hline(
        y=mean_bias,
        line_color="#2a6f97",
        line_width=2,
        annotation_text=f"Mean bias {mean_bias:.2f}",
        annotation_position="top left",
    )
    for y_value in [max_abs_bias, -max_abs_bias]:
        figure.add_hline(
            y=y_value,
            line_dash="dash",
            line_color="#9467bd",
            annotation_text=f"Limit {y_value:.2f}",
            annotation_position="top left" if y_value > 0 else "bottom left",
        )
    figure.add_hline(y=0, line_color="#808080", line_width=1)
    figure.update_traces(
        marker={"size": 9, "line": {"color": "#102a43", "width": 1}},
    )
    figure.update_layout(margin={"l": 70, "r": 40, "t": 70, "b": 65})
    return figure


def create_blank_distribution_histogram(analyzed_data: pd.DataFrame) -> go.Figure:
    """Create blank replicate histogram for LoB review."""

    blank_data = analyzed_data[
        analyzed_data["Sample Type"].str.lower() == "blank"
    ]
    figure = px.histogram(
        blank_data,
        x="Observed Result",
        nbins=16,
        title="Blank Distribution Histogram",
        labels={"Observed Result": "Blank Observed Result"},
        template="plotly_white",
    )
    figure.update_traces(marker_color="#2a6f97")
    figure.update_layout(margin={"l": 70, "r": 40, "t": 70, "b": 65})
    return figure


def create_blank_replicate_boxplot(analyzed_data: pd.DataFrame) -> go.Figure:
    """Create blank replicate boxplot for LoB review."""

    blank_data = analyzed_data[
        analyzed_data["Sample Type"].str.lower() == "blank"
    ]
    figure = px.box(
        blank_data,
        x="Sample Type",
        y="Observed Result",
        points="all",
        title="Blank Replicate Boxplot",
        labels={"Observed Result": "Observed Result"},
        template="plotly_white",
    )
    figure.update_traces(marker={"color": "#2a6f97"})
    figure.update_layout(showlegend=False, margin={"l": 70, "r": 40, "t": 70, "b": 65})
    return figure


def create_low_level_distribution_plot(analyzed_data: pd.DataFrame) -> go.Figure:
    """Create low-level replicate distribution for LoD review."""

    low_data = analyzed_data[
        analyzed_data["Sample Type"].str.lower() == "low concentration"
    ]
    figure = px.box(
        low_data,
        x="Sample Type",
        y="Observed Result",
        points="all",
        title="Low-Level Replicate Distribution",
        labels={"Observed Result": "Observed Result"},
        template="plotly_white",
    )
    figure.update_traces(marker={"color": "#2a6f97"})
    figure.update_layout(showlegend=False, margin={"l": 70, "r": 40, "t": 70, "b": 65})
    return figure


def create_lob_lod_visualization(lob_summary: pd.DataFrame, lod_summary: pd.DataFrame) -> go.Figure:
    """Create LoB versus LoD bar visualization."""

    plot_data = pd.DataFrame(
        [
            {"Metric": "LoB", "Value": lob_summary["LoB"].iloc[0]},
            {"Metric": "LoD", "Value": lod_summary["LoD"].iloc[0]},
        ]
    )
    figure = px.bar(
        plot_data,
        x="Metric",
        y="Value",
        text="Value",
        title="LoB vs LoD Visualization",
        labels={"Value": "Calculated Limit"},
        template="plotly_white",
    )
    figure.update_traces(
        marker_color=["#2a6f97", "#9467bd"],
        texttemplate="%{text:.3f}",
        textposition="outside",
    )
    figure.update_layout(showlegend=False, margin={"l": 70, "r": 40, "t": 70, "b": 65})
    return figure


def create_loq_cv_plot(loq_summary: pd.DataFrame, target_cv: float) -> go.Figure:
    """Create CV% versus concentration plot for LoQ review."""

    figure = px.line(
        loq_summary,
        x="Concentration Level",
        y="CV%",
        markers=True,
        title="CV% vs Concentration",
        labels={"CV%": "CV%"},
        template="plotly_white",
    )
    figure.add_hline(
        y=target_cv,
        line_dash="dash",
        line_color="#9467bd",
        annotation_text=f"Target CV% {target_cv:.1f}%",
        annotation_position="top left",
    )
    figure.update_traces(marker={"size": 10, "line": {"color": "#102a43", "width": 1}})
    figure.update_layout(margin={"l": 70, "r": 40, "t": 70, "b": 65})
    return figure


def create_loq_recovery_plot(loq_summary: pd.DataFrame) -> go.Figure:
    """Create recovery versus concentration plot for LoQ review."""

    figure = px.line(
        loq_summary,
        x="Concentration Level",
        y="Recovery %",
        markers=True,
        title="Recovery vs Concentration",
        labels={"Recovery %": "Recovery (%)"},
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
    figure.update_traces(marker={"size": 10, "line": {"color": "#102a43", "width": 1}})
    figure.update_layout(margin={"l": 70, "r": 40, "t": 70, "b": 65})
    return figure


def create_loq_decision_plot(loq_summary: pd.DataFrame, target_cv: float) -> go.Figure:
    """Create LoQ decision plot with pass/fail annotations."""

    plot_data = loq_summary.copy()
    figure = px.scatter(
        plot_data,
        x="Concentration Level",
        y="CV%",
        color="Pass/Fail",
        text="Pass/Fail",
        title="LoQ Decision Plot",
        labels={"CV%": "CV%"},
        color_discrete_map={
            "PASS": STATUS_COLOR_MAP["PASS"],
            "FAIL": STATUS_COLOR_MAP["FAIL"],
        },
        template="plotly_white",
    )
    figure.add_hline(
        y=target_cv,
        line_dash="dash",
        line_color="#9467bd",
        annotation_text=f"Target CV% {target_cv:.1f}%",
        annotation_position="top left",
    )
    figure.update_traces(
        marker={"size": 12, "line": {"color": "#102a43", "width": 1}},
        textposition="top center",
    )
    figure.update_layout(margin={"l": 70, "r": 40, "t": 70, "b": 65})
    return figure


def create_detection_replicate_distribution_plot(analyzed_data: pd.DataFrame) -> go.Figure:
    """Create replicate distribution by quantitation concentration."""

    loq_data = analyzed_data[
        analyzed_data["Sample Type"].str.lower() == "quantitation level"
    ]
    figure = px.box(
        loq_data,
        x="Concentration Level",
        y="Observed Result",
        color="Concentration Level",
        points="all",
        title="Replicate Distribution by Concentration",
        labels={"Observed Result": "Observed Result"},
        template="plotly_white",
    )
    figure.update_layout(showlegend=False, margin={"l": 70, "r": 40, "t": 70, "b": 65})
    return figure


def create_detection_replicate_scatter_plot(analyzed_data: pd.DataFrame) -> go.Figure:
    """Create observed result versus replicate number scatter plot."""

    plot_data = analyzed_data.copy()
    if "Replicate" not in plot_data.columns:
        plot_data["Replicate"] = plot_data.groupby("Sample Type").cumcount() + 1
    figure = px.scatter(
        plot_data,
        x="Replicate",
        y="Observed Result",
        color="Sample Type",
        symbol="Sample Type",
        hover_data=["Sample ID", "Concentration Level"],
        title="Replicate Scatter Plot",
        labels={"Observed Result": "Observed Result", "Replicate": "Replicate Number"},
        template="plotly_white",
    )
    figure.update_traces(marker={"size": 9, "line": {"color": "#102a43", "width": 1}})
    figure.update_layout(margin={"l": 70, "r": 40, "t": 70, "b": 65})
    return figure


def create_loq_precision_curve(
    loq_summary: pd.DataFrame,
    target_cv: float,
    operational_loq: float,
) -> go.Figure:
    """Create LoQ precision curve highlighting the operational LoQ."""

    figure = create_loq_cv_plot(loq_summary, target_cv)
    figure.update_layout(title="LoQ Precision Curve")
    if not pd.isna(operational_loq):
        figure.add_vline(
            x=operational_loq,
            line_dash="dot",
            line_color="#1f7a1f",
            annotation_text=f"Operational LoQ {operational_loq:.3f}",
            annotation_position="top right",
        )
    return figure


def create_detection_capability_ladder(
    overall_summary: dict[str, float | str],
) -> go.Figure:
    """Create single-axis Blank-to-LoQ detection capability ladder."""

    ladder_data = pd.DataFrame(
        [
            {"Metric": "Blank", "Value": 0.0},
            {"Metric": "LoB", "Value": overall_summary["LoB"]},
            {"Metric": "LoD", "Value": overall_summary["LoD"]},
            {"Metric": "LoQ", "Value": overall_summary["LoQ"]},
        ]
    )
    figure = px.scatter(
        ladder_data,
        x="Value",
        y=["Detection Capability"] * len(ladder_data),
        text="Metric",
        title="Detection Capability Ladder",
        labels={"Value": "Result / Concentration", "y": ""},
        template="plotly_white",
    )
    figure.add_trace(
        go.Scatter(
            x=ladder_data["Value"],
            y=["Detection Capability"] * len(ladder_data),
            mode="lines",
            line={"color": "#2a6f97", "width": 3},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    figure.update_traces(
        marker={"size": 13, "line": {"color": "#102a43", "width": 1}},
        textposition="top center",
    )
    figure.update_yaxes(showticklabels=False)
    figure.update_layout(margin={"l": 70, "r": 40, "t": 70, "b": 65})
    return figure


def create_detection_density_plot(
    analyzed_data: pd.DataFrame,
    lob: float,
    lod: float,
) -> go.Figure:
    """Create overlaid blank and low-level distribution density plot."""

    plot_data = analyzed_data[
        analyzed_data["Sample Type"].str.lower().isin(["blank", "low concentration"])
    ].copy()
    figure = px.histogram(
        plot_data,
        x="Observed Result",
        color="Sample Type",
        histnorm="probability density",
        barmode="overlay",
        opacity=0.55,
        nbins=20,
        title="Distribution Density Plot",
        labels={"Observed Result": "Observed Result"},
        template="plotly_white",
    )
    figure.add_vline(
        x=lob,
        line_dash="dash",
        line_color="#2a6f97",
        annotation_text=f"LoB {lob:.3f}",
        annotation_position="top right",
    )
    figure.add_vline(
        x=lod,
        line_dash="dash",
        line_color="#9467bd",
        annotation_text=f"LoD {lod:.3f}",
        annotation_position="top right",
    )
    figure.update_layout(margin={"l": 70, "r": 40, "t": 70, "b": 65})
    return figure


def create_dbs_scatter_plot(
    analyzed_data: pd.DataFrame, overall_summary: dict[str, float | str]
) -> go.Figure:
    """Create DBS-vs-reference scatter plot with identity and regression lines."""

    figure = px.scatter(
        analyzed_data,
        x="Reference Result",
        y="DBS Result",
        hover_data=["Sample ID"],
        title="DBS vs Reference Scatter Plot",
        labels={"Reference Result": "Reference Result", "DBS Result": "DBS Result"},
        template="plotly_white",
    )
    min_value = min(analyzed_data["Reference Result"].min(), analyzed_data["DBS Result"].min())
    max_value = max(analyzed_data["Reference Result"].max(), analyzed_data["DBS Result"].max())
    x_values = np.array([min_value, max_value])
    figure.add_trace(
        go.Scatter(
            x=x_values,
            y=x_values,
            mode="lines",
            name="Identity line",
            line={"color": "#808080", "dash": "dash"},
        )
    )
    slope = overall_summary.get("Slope", np.nan)
    intercept = overall_summary.get("Intercept", np.nan)
    if not pd.isna(slope) and not pd.isna(intercept):
        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=(float(slope) * x_values) + float(intercept),
                mode="lines",
                name="Regression line",
                line={"color": "#d62728", "width": 2},
            )
        )
    figure.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        text=(
            f"y = {_format_number(overall_summary.get('Slope'), 4)}x + "
            f"{_format_number(overall_summary.get('Intercept'), 4)}"
            f"<br>R² = {_format_number(overall_summary.get('R²'), 4)}"
        ),
        showarrow=False,
        align="left",
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#d9d9d9",
        borderwidth=1,
        borderpad=8,
    )
    figure.update_traces(marker={"size": 9})
    figure.update_layout(margin={"l": 70, "r": 40, "t": 70, "b": 65})
    return figure


def create_dbs_bland_altman_plot(
    analyzed_data: pd.DataFrame, overall_summary: dict[str, float | str]
) -> go.Figure:
    """Create DBS Bland-Altman agreement plot."""

    figure = px.scatter(
        analyzed_data,
        x="Mean of Methods",
        y="Difference",
        hover_data=["Sample ID"],
        title="DBS Bland-Altman Plot",
        labels={"Mean of Methods": "Average of DBS and Reference", "Difference": "DBS - Reference"},
        template="plotly_white",
    )
    lines = [
        ("Mean difference", overall_summary.get("Mean Difference"), "#2a6f97"),
        ("Upper LoA", overall_summary.get("Upper Limit of Agreement"), "#c92a2a"),
        ("Lower LoA", overall_summary.get("Lower Limit of Agreement"), "#c92a2a"),
    ]
    for label, y_value, color in lines:
        if not pd.isna(y_value):
            figure.add_hline(
                y=float(y_value),
                line_dash="dash" if label != "Mean difference" else "solid",
                line_color=color,
                annotation_text=f"{label}: {float(y_value):.2f}",
                annotation_position="top left",
            )
    figure.update_traces(marker={"color": "#2a6f97", "size": 9})
    return figure


def create_dbs_recovery_plot(
    analyzed_data: pd.DataFrame, min_recovery: float, max_recovery: float
) -> go.Figure:
    """Create DBS sample-level recovery plot."""

    plot_data = analyzed_data.copy()
    figure = px.scatter(
        plot_data,
        x="Sample ID",
        y="Recovery %",
        title="DBS Recovery by Sample",
        labels={"Recovery %": "Recovery (%)"},
        template="plotly_white",
    )
    for y_value, color, dash in [
        (min_recovery, "#c92a2a", "dash"),
        (100, "#808080", "solid"),
        (max_recovery, "#c92a2a", "dash"),
    ]:
        figure.add_hline(
            y=y_value,
            line_color=color,
            line_dash=dash,
            annotation_text=f"{y_value:.1f}%",
            annotation_position="top left",
        )
    figure.update_traces(marker={"color": "#2a6f97", "size": 8})
    figure.update_layout(xaxis_tickangle=-45)
    return figure


def create_dbs_percent_bias_plot(
    analyzed_data: pd.DataFrame, max_percent_bias: float
) -> go.Figure:
    """Create DBS percent-bias-by-sample plot."""

    figure = px.scatter(
        analyzed_data,
        x="Sample ID",
        y="Percent Bias",
        title="DBS Percent Bias by Sample",
        labels={"Percent Bias": "Percent Bias (%)"},
        template="plotly_white",
    )
    for y_value in [-max_percent_bias, 0, max_percent_bias]:
        figure.add_hline(
            y=y_value,
            line_color="#808080" if y_value == 0 else "#c92a2a",
            line_dash="solid" if y_value == 0 else "dash",
            annotation_text=f"{y_value:.1f}%",
            annotation_position="top left",
        )
    figure.update_traces(marker={"color": "#2a6f97", "size": 8})
    figure.update_layout(xaxis_tickangle=-45)
    return figure


def create_dbs_distribution_comparison(analyzed_data: pd.DataFrame) -> go.Figure:
    """Create overlaid reference and DBS result distributions."""

    long_data = analyzed_data.melt(
        value_vars=["Reference Result", "DBS Result"],
        var_name="Specimen Type",
        value_name="Result",
    )
    figure = px.histogram(
        long_data,
        x="Result",
        color="Specimen Type",
        barmode="overlay",
        marginal="box",
        opacity=0.6,
        title="Reference and DBS Distribution Comparison",
        template="plotly_white",
    )
    return figure


def _add_simple_regression_line(figure: go.Figure, data: pd.DataFrame, x: str, y: str) -> go.Figure:
    """Overlay a simple regression line when enough data are available."""

    subset = data[[x, y]].dropna()
    if len(subset) >= 2 and subset[x].nunique() > 1 and subset[y].nunique() > 1:
        slope, intercept = np.polyfit(subset[x], subset[y], 1)
        x_values = np.array([subset[x].min(), subset[x].max()])
        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=(slope * x_values) + intercept,
                mode="lines",
                name="Regression line",
                line={"color": "#d62728", "width": 2},
            )
        )
    return figure


def create_dbs_hematocrit_bias_plot(analyzed_data: pd.DataFrame) -> go.Figure:
    """Create hematocrit vs absolute bias assessment plot."""

    figure = px.scatter(
        analyzed_data.dropna(subset=["Hematocrit", "Bias"]),
        x="Hematocrit",
        y="Bias",
        hover_data=["Sample ID"],
        title="Hematocrit vs Bias",
        template="plotly_white",
    )
    figure = _add_simple_regression_line(figure, analyzed_data, "Hematocrit", "Bias")
    figure.add_hline(y=0, line_dash="dash", line_color="#808080")
    return figure


def create_dbs_hematocrit_percent_bias_plot(analyzed_data: pd.DataFrame) -> go.Figure:
    """Create hematocrit vs percent bias assessment plot."""

    figure = px.scatter(
        analyzed_data.dropna(subset=["Hematocrit", "Percent Bias"]),
        x="Hematocrit",
        y="Percent Bias",
        hover_data=["Sample ID"],
        title="Hematocrit vs Percent Bias",
        labels={"Percent Bias": "Percent Bias (%)"},
        template="plotly_white",
    )
    figure = _add_simple_regression_line(
        figure, analyzed_data, "Hematocrit", "Percent Bias"
    )
    figure.add_hline(y=0, line_dash="dash", line_color="#808080")
    return figure


def create_dbs_delay_bias_plot(analyzed_data: pd.DataFrame) -> go.Figure:
    """Create extraction delay vs bias plot."""

    figure = px.scatter(
        analyzed_data.dropna(subset=["Extraction Delay (days)", "Bias"]),
        x="Extraction Delay (days)",
        y="Bias",
        hover_data=["Sample ID"],
        title="Extraction Delay vs Bias",
        template="plotly_white",
    )
    figure = _add_simple_regression_line(
        figure, analyzed_data, "Extraction Delay (days)", "Bias"
    )
    figure.add_hline(y=0, line_dash="dash", line_color="#808080")
    return figure


def create_dbs_delay_distribution_plot(analyzed_data: pd.DataFrame) -> go.Figure:
    """Create extraction delay distribution histogram."""

    figure = px.histogram(
        analyzed_data.dropna(subset=["Extraction Delay (days)"]),
        x="Extraction Delay (days)",
        nbins=8,
        title="Extraction Delay Distribution",
        template="plotly_white",
    )
    return figure


def create_dbs_delay_category_bias_plot(analyzed_data: pd.DataFrame) -> go.Figure:
    """Create average bias by extraction delay category."""

    if "Delay Category" not in analyzed_data.columns:
        return go.Figure()
    summary = (
        analyzed_data.groupby("Delay Category", dropna=False)["Bias"]
        .mean()
        .reset_index(name="Mean Bias")
    )
    figure = px.bar(
        summary,
        x="Delay Category",
        y="Mean Bias",
        title="Average Bias by Delay Category",
        template="plotly_white",
    )
    figure.add_hline(y=0, line_dash="dash", line_color="#808080")
    return figure


def create_dbs_instrument_bias_plot(instrument_summary: pd.DataFrame) -> go.Figure:
    """Create mean bias by instrument plot."""

    figure = px.bar(
        instrument_summary,
        x="Instrument",
        y="Mean Bias",
        title="Mean Bias by Instrument",
        template="plotly_white",
    )
    figure.add_hline(y=0, line_dash="dash", line_color="#808080")
    return figure


def create_dbs_instrument_recovery_plot(instrument_summary: pd.DataFrame) -> go.Figure:
    """Create mean recovery by instrument plot."""

    figure = px.bar(
        instrument_summary,
        x="Instrument",
        y="Mean Recovery",
        title="Mean Recovery by Instrument",
        template="plotly_white",
    )
    figure.add_hline(y=100, line_dash="dash", line_color="#808080")
    return figure


def _microtainer_as_dbs(analyzed_data: pd.DataFrame) -> pd.DataFrame:
    """Return a plotting copy compatible with paired DBS-style chart helpers."""

    return analyzed_data.rename(columns={"Microtainer Result": "DBS Result"})


def create_microtainer_scatter_plot(
    analyzed_data: pd.DataFrame, overall_summary: dict[str, float | str]
) -> go.Figure:
    """Create Microtainer-vs-reference scatter plot."""

    figure = create_dbs_scatter_plot(_microtainer_as_dbs(analyzed_data), overall_summary)
    figure.update_layout(title="Microtainer vs Reference Scatter Plot")
    figure.update_yaxes(title_text="Microtainer Result")
    return figure


def create_microtainer_bland_altman_plot(
    analyzed_data: pd.DataFrame, overall_summary: dict[str, float | str]
) -> go.Figure:
    """Create Microtainer Bland-Altman plot."""

    figure = create_dbs_bland_altman_plot(_microtainer_as_dbs(analyzed_data), overall_summary)
    figure.update_layout(title="Microtainer Bland-Altman Plot")
    figure.update_yaxes(title_text="Microtainer - Reference")
    return figure


def create_microtainer_recovery_plot(
    analyzed_data: pd.DataFrame, min_recovery: float, max_recovery: float
) -> go.Figure:
    """Create Microtainer recovery by sample plot."""

    figure = create_dbs_recovery_plot(_microtainer_as_dbs(analyzed_data), min_recovery, max_recovery)
    figure.update_layout(title="Microtainer Recovery by Sample")
    return figure


def create_microtainer_percent_bias_plot(
    analyzed_data: pd.DataFrame, max_percent_bias: float
) -> go.Figure:
    """Create Microtainer percent bias by sample plot."""

    figure = create_dbs_percent_bias_plot(_microtainer_as_dbs(analyzed_data), max_percent_bias)
    figure.update_layout(title="Microtainer Percent Bias by Sample")
    return figure


def create_microtainer_distribution_comparison(analyzed_data: pd.DataFrame) -> go.Figure:
    """Create reference and microtainer distribution comparison."""

    long_data = analyzed_data.melt(
        value_vars=["Reference Result", "Microtainer Result"],
        var_name="Specimen Type",
        value_name="Result",
    )
    return px.histogram(
        long_data,
        x="Result",
        color="Specimen Type",
        barmode="overlay",
        marginal="box",
        opacity=0.6,
        title="Reference and Microtainer Distribution Comparison",
        template="plotly_white",
    )


def create_microtainer_volume_impact_plot(analyzed_data: pd.DataFrame) -> go.Figure:
    """Create collection volume vs bias plot."""

    figure = px.scatter(
        analyzed_data.dropna(subset=["Collection Volume", "Bias"]),
        x="Collection Volume",
        y="Bias",
        hover_data=["Sample ID"],
        title="Collection Volume vs Bias",
        template="plotly_white",
    )
    figure = _add_simple_regression_line(figure, analyzed_data, "Collection Volume", "Bias")
    figure.add_hline(y=0, line_dash="dash", line_color="#808080")
    return figure


def create_microtainer_delay_impact_plot(analyzed_data: pd.DataFrame) -> go.Figure:
    """Create processing delay vs bias plot."""

    figure = px.scatter(
        analyzed_data.dropna(subset=["Processing Delay (days)", "Bias"]),
        x="Processing Delay (days)",
        y="Bias",
        hover_data=["Sample ID"],
        title="Processing Delay vs Bias",
        template="plotly_white",
    )
    figure = _add_simple_regression_line(figure, analyzed_data, "Processing Delay (days)", "Bias")
    figure.add_hline(y=0, line_dash="dash", line_color="#808080")
    return figure


def create_microtainer_instrument_comparison_plot(instrument_summary: pd.DataFrame) -> go.Figure:
    """Create microtainer mean bias by instrument plot."""

    return create_dbs_instrument_bias_plot(instrument_summary).update_layout(
        title="Microtainer Mean Bias by Instrument"
    )
