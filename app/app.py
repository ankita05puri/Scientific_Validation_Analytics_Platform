"""Streamlit application for the Scientific Validation Analytics Platform."""

from __future__ import annotations

from io import StringIO
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from analysis import (
    calculate_percent_samples_meeting_agreement,
    evaluate_acceptance_criteria,
    evaluate_precision_criteria,
    get_top_outliers,
    run_precision_study,
    run_method_comparison,
)
from plots import (
    create_difference_plot,
    create_percent_bias_histogram,
    create_precision_box_plot,
    create_precision_cv_bar_chart,
    create_precision_run_chart,
    create_scatter_plot,
)
from report import (
    build_criteria_table,
    build_html_report,
    build_precision_html_report,
    build_summary_table,
    format_precision_criteria_table,
    format_precision_summary_table,
    generate_interpretation,
    generate_precision_interpretation,
)
from studies import get_study_type_config, get_study_type_names


APP_TITLE = "Scientific Validation Analytics Platform"
ROOT_DIR = Path(__file__).resolve().parents[1]
SAMPLE_DATA_PATH = ROOT_DIR / "data" / "sample_data" / "hba1c_method_comparison.csv"
PRECISION_SAMPLE_DATA_PATH = ROOT_DIR / "data" / "sample_data" / "precision_study_hba1c.csv"
DASHBOARD_MODULES = (
    ("Method Comparison", "Paired reference and candidate comparison studies."),
    ("Accuracy Studies", "Bias, recovery, and agreement with expected values."),
    ("Precision Studies", "Intra-assay and inter-assay precision workflows."),
    ("Linearity Studies", "Analytical measurement range and recovery by level."),
    ("Stability Studies", "Specimen, reagent, and timepoint stability review."),
    ("DBS Validation", "Dried blood spot correction and agreement workflows."),
    ("Microtainer Validation", "Small-volume specimen comparison workflows."),
    ("Validation Reports", "Scientific summary reports and review packages."),
)


def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    """Load a CSV or Excel file uploaded through Streamlit."""

    file_name = uploaded_file.name.lower()
    if file_name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if file_name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    raise ValueError("Unsupported file type. Please upload a CSV or Excel file.")


def get_numeric_columns(data: pd.DataFrame) -> list[str]:
    """Return columns that contain at least one numeric value after coercion."""

    numeric_columns: list[str] = []
    for column in data.columns:
        converted = pd.to_numeric(data[column], errors="coerce")
        if converted.notna().any():
            numeric_columns.append(column)
    return numeric_columns


def detect_sample_id_column(data: pd.DataFrame) -> str | None:
    """Detect a likely sample identifier column."""

    possible_names = {"sample id", "sample_id", "sampleid", "id", "specimen id"}
    for column in data.columns:
        normalized = str(column).strip().lower()
        if normalized in possible_names:
            return column
    return None


def render_dashboard() -> None:
    """Render dashboard cards for current and planned validation modules."""

    st.subheader("Validation Dashboard")
    for row_start in range(0, len(DASHBOARD_MODULES), 4):
        columns = st.columns(4)
        for column, (title, description) in zip(
            columns, DASHBOARD_MODULES[row_start : row_start + 4]
        ):
            with column:
                st.markdown(
                    f"""
                    <div style="border:1px solid #d9e2ec; border-radius:8px; padding:14px; min-height:132px;">
                      <h3 style="margin-top:0; font-size:1.05rem;">{title}</h3>
                      <p style="font-size:0.9rem; color:#52606d;">{description}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_study_type_selector() -> str:
    """Render study-type selection and module capabilities."""

    st.subheader("Study Type")
    study_type = st.selectbox("Select validation study type", get_study_type_names())
    config = get_study_type_config(study_type)
    st.caption(config.description)

    with st.expander("Study module framework", expanded=False):
        cols = st.columns(4)
        cols[0].markdown("**Metrics**")
        cols[0].write("\n".join(f"- {item}" for item in config.metrics))
        cols[1].markdown("**Acceptance Criteria**")
        cols[1].write("\n".join(f"- {item}" for item in config.acceptance_criteria))
        cols[2].markdown("**Visualizations**")
        cols[2].write("\n".join(f"- {item}" for item in config.visualizations))
        cols[3].markdown("**Report Sections**")
        cols[3].write("\n".join(f"- {item}" for item in config.report_sections))

    if not config.implemented:
        st.info(
            f"{study_type} is registered in the platform framework. "
            "Its custom metrics, criteria, visualizations, and report sections can be implemented in a future module."
        )
    return study_type


def render_study_documentation(study_type: str) -> dict[str, object]:
    """Render validation study documentation inputs and return collected values."""

    is_precision = study_type == "Precision Study"
    default_study_name = (
        "HbA1c Precision Study" if is_precision else "HbA1c Method Comparison Study"
    )
    default_objective = (
        "Evaluate repeatability of repeated HbA1c measurements."
        if is_precision
        else "Evaluate agreement between candidate and reference results."
    )
    default_design = (
        "Repeated low-level and high-level quality control measurements across multiple days, runs, and replicates."
        if is_precision
        else "Paired specimen comparison using reference and candidate measurements."
    )

    st.subheader("Validation Study Documentation")
    first_row = st.columns(3)
    with first_row[0]:
        study_name = st.text_input(
            "Study Name",
            value=default_study_name,
        )
    with first_row[1]:
        analyst_name = st.text_input("Analyst Name")
    with first_row[2]:
        study_date = st.date_input("Study Date", value=date.today())

    study_objective = st.text_area(
        "Study Objective",
        value=default_objective,
        height=80,
    )
    study_design = st.text_area(
        "Study Design",
        value=default_design,
        height=80,
    )

    second_row = st.columns(2)
    with second_row[0]:
        assay_name = st.text_input("Assay / Biomarker", value="HbA1c")
    with second_row[1]:
        specimen_type = st.text_input("Specimen Type", value="Whole blood")

    reference_method = ""
    candidate_method = ""
    if is_precision:
        units = st.text_input("Units", value="%")
        with st.expander("Laboratory Documentation", expanded=False):
            lab_row_1 = st.columns(3)
            with lab_row_1[0]:
                instrument_name = st.text_input("Instrument Name", key="precision_instrument_name")
            with lab_row_1[1]:
                instrument_id = st.text_input("Instrument ID", key="precision_instrument_id")
            with lab_row_1[2]:
                laboratory_site = st.text_input("Laboratory Site", key="precision_laboratory_site")

            lab_row_2 = st.columns(3)
            with lab_row_2[0]:
                reagent_lot = st.text_input("Reagent Lot Number", key="precision_reagent_lot")
            with lab_row_2[1]:
                calibrator_lot = st.text_input("Calibrator Lot Number", key="precision_calibrator_lot")
            with lab_row_2[2]:
                qc_lot = st.text_input("Quality Control Lot Number", key="precision_qc_lot")

            operator_name = st.text_input("Operator Name", key="precision_operator_name")
    else:
        third_row = st.columns(3)
        with third_row[0]:
            reference_method = st.text_input("Reference Method", value="Reference HbA1c")
        with third_row[1]:
            candidate_method = st.text_input("Candidate Method", value="Candidate HbA1c")
        with third_row[2]:
            units = st.text_input("Units", value="%")

    notes = st.text_area("Notes", height=80)
    deviations = st.text_area("Deviations", height=80)
    conclusions = st.text_area("Conclusions", height=80)

    metadata = {
        "Study Name": study_name,
        "Study Objective": study_objective,
        "Study Design": study_design,
        "Assay / Biomarker": assay_name,
        "Specimen Type": specimen_type,
        "Analyst Name": analyst_name,
        "Study Date": study_date.isoformat(),
        "Sample Count": "Calculated after analysis",
        "Units": units,
        "Notes": notes,
        "Deviations": deviations,
        "Conclusions": conclusions,
    }
    if not is_precision:
        metadata["Reference Method"] = reference_method
        metadata["Candidate Method"] = candidate_method
    else:
        metadata.update(
            {
                "Instrument Name": instrument_name,
                "Instrument ID": instrument_id,
                "Reagent Lot Number": reagent_lot,
                "Calibrator Lot Number": calibrator_lot,
                "Quality Control Lot Number": qc_lot,
                "Operator Name": operator_name,
                "Laboratory Site": laboratory_site,
            }
        )
    return metadata


def render_method_comparison_criteria() -> dict[str, float | None]:
    """Render user-defined preliminary acceptance criteria controls."""

    st.subheader("Acceptance Criteria")
    st.caption(
        "These criteria are preliminary screening thresholds selected by the user."
    )
    criteria_columns = st.columns(4)
    with criteria_columns[0]:
        min_r_squared = st.number_input(
            "Minimum acceptable R²",
            min_value=0.0,
            max_value=1.0,
            value=0.95,
            step=0.01,
            format="%.2f",
        )
        min_correlation = st.number_input(
            "Minimum acceptable correlation coefficient (r)",
            min_value=-1.0,
            max_value=1.0,
            value=0.95,
            step=0.01,
            format="%.2f",
        )
    with criteria_columns[1]:
        slope_lower = st.number_input(
            "Slope lower limit",
            value=0.90,
            step=0.01,
            format="%.2f",
        )
        slope_upper = st.number_input(
            "Slope upper limit",
            value=1.10,
            step=0.01,
            format="%.2f",
        )
    with criteria_columns[2]:
        max_abs_intercept = st.number_input(
            "Maximum acceptable intercept",
            min_value=0.0,
            value=0.50,
            step=0.05,
            format="%.3f",
        )
        max_abs_mean_bias = st.number_input(
            "Maximum acceptable absolute mean bias",
            min_value=0.0,
            value=0.50,
            step=0.05,
            format="%.3f",
        )
    with criteria_columns[3]:
        max_abs_percent_bias = st.number_input(
            "Maximum acceptable absolute mean percent bias",
            min_value=0.0,
            value=5.0,
            step=0.5,
            format="%.2f",
        )
        max_abs_mean_difference = st.number_input(
            "Maximum acceptable absolute mean difference",
            min_value=0.0,
            value=0.50,
            step=0.05,
            format="%.3f",
        )

    agreement_columns = st.columns(2)
    with agreement_columns[0]:
        sample_agreement_limit = st.number_input(
            "Sample agreement percent bias limit",
            min_value=0.0,
            value=5.0,
            step=0.5,
            format="%.2f",
        )
    with agreement_columns[1]:
        min_percent_samples_agree = st.number_input(
            "Minimum percent of samples meeting agreement criteria",
            min_value=0.0,
            max_value=100.0,
            value=95.0,
            step=1.0,
            format="%.1f",
        )

    return {
        "Minimum R²": min_r_squared,
        "Minimum Correlation r": min_correlation,
        "Slope Lower Limit": slope_lower,
        "Slope Upper Limit": slope_upper,
        "Maximum Absolute Intercept": max_abs_intercept,
        "Maximum Absolute Mean Bias": max_abs_mean_bias,
        "Maximum Absolute Mean Percent Bias": max_abs_percent_bias,
        "Maximum Absolute Mean Difference": max_abs_mean_difference,
        "Sample Agreement Percent Bias Limit": sample_agreement_limit,
        "Minimum Percent Samples Meeting Agreement": min_percent_samples_agree,
    }


def render_precision_criteria() -> dict[str, float]:
    """Render user-defined preliminary precision acceptance criteria."""

    st.subheader("Acceptance Criteria")
    st.caption(
        "These criteria are preliminary screening thresholds selected by the user, not regulatory approval."
    )
    max_cv = st.number_input(
        "Maximum acceptable CV%",
        min_value=0.0,
        value=5.0,
        step=0.5,
        format="%.2f",
    )
    return {"Maximum CV%": max_cv}


def optional_column_selectbox(
    label: str, columns: list[str], default_column: str | None = None
) -> str | None:
    """Render a selectbox that can return no selected column."""

    options = ["None"] + columns
    default_index = options.index(default_column) if default_column in options else 0
    selected = st.selectbox(label, options, index=default_index)
    return None if selected == "None" else selected


def show_decision(decision: str) -> None:
    """Display the preliminary decision with simple visual emphasis."""

    if decision == "PASS":
        st.success("Overall preliminary decision: PASS")
    elif decision in {"BORDERLINE", "REVIEW"}:
        st.warning(f"Overall preliminary decision: {decision}")
    else:
        label = "INVESTIGATE" if decision == "INVESTIGATE" else "FAIL"
        st.error(f"Overall preliminary decision: {label}")


def build_precision_summary_csv(
    metadata: dict[str, object],
    criteria_table: pd.DataFrame,
    level_summary: pd.DataFrame,
    day_summary: pd.DataFrame,
    run_summary: pd.DataFrame,
) -> str:
    """Build a sectioned CSV summary for precision study documentation."""

    buffer = StringIO()
    pd.DataFrame(
        [{"Field": key, "Value": value or "Not specified"} for key, value in metadata.items()]
    ).to_csv(buffer, index=False)
    buffer.write("\nAcceptance Criteria Results\n")
    criteria_table.to_csv(buffer, index=False)
    buffer.write("\nPrecision Summary\n")
    level_summary.to_csv(buffer, index=False)
    if not day_summary.empty:
        buffer.write("\nDay-Level Summary\n")
        day_summary.to_csv(buffer, index=False)
    if not run_summary.empty:
        buffer.write("\nRun-Level Summary\n")
        run_summary.to_csv(buffer, index=False)
    return buffer.getvalue()


def render_precision_workspace(metadata: dict[str, object]) -> None:
    """Render the Precision Study workflow."""

    criteria = render_precision_criteria()

    uploaded_file = st.file_uploader(
        "Upload repeated-measurement precision data",
        type=["csv", "xlsx", "xls"],
        help="Upload a file containing repeated results by level, day, run, and replicate.",
    )

    use_sample_data = st.checkbox(
        "Use included sample HbA1c precision study dataset",
        value=uploaded_file is None,
    )

    data = None
    if uploaded_file is not None:
        try:
            data = load_uploaded_file(uploaded_file)
        except Exception as exc:
            st.error(f"Unable to load file: {exc}")
            st.stop()
    elif use_sample_data:
        data = pd.read_csv(PRECISION_SAMPLE_DATA_PATH)

    if data is None:
        st.info("Upload a CSV or Excel file to begin.")
        st.stop()

    metadata["Sample Count"] = len(data)

    st.subheader("Data Preview")
    st.dataframe(data.head(20), width="stretch")

    numeric_columns = get_numeric_columns(data)
    if len(numeric_columns) < 1:
        st.error("At least one numeric result column is required for precision analysis.")
        st.stop()

    all_columns = list(data.columns)
    detected_sample_id = detect_sample_id_column(data)

    st.subheader("Column Selection")
    first_row = st.columns(3)
    with first_row[0]:
        result_column = st.selectbox(
            "Result column",
            numeric_columns,
            index=numeric_columns.index("Result") if "Result" in numeric_columns else 0,
        )
    with first_row[1]:
        level_column = st.selectbox(
            "Level column",
            all_columns,
            index=all_columns.index("Level") if "Level" in all_columns else 0,
        )
    with first_row[2]:
        sample_id_column = optional_column_selectbox(
            "Sample ID column (optional)",
            all_columns,
            detected_sample_id,
        )

    second_row = st.columns(3)
    with second_row[0]:
        day_column = optional_column_selectbox(
            "Day column",
            all_columns,
            "Day" if "Day" in all_columns else None,
        )
    with second_row[1]:
        run_column = optional_column_selectbox(
            "Run column",
            all_columns,
            "Run" if "Run" in all_columns else None,
        )
    with second_row[2]:
        replicate_column = optional_column_selectbox(
            "Replicate column",
            all_columns,
            "Replicate" if "Replicate" in all_columns else None,
        )

    run_analysis = st.button("Run Precision Analysis", type="primary")

    if run_analysis:
        try:
            result = run_precision_study(
                data=data,
                result_column=result_column,
                level_column=level_column,
                day_column=day_column,
                run_column=run_column,
                replicate_column=replicate_column,
                sample_id_column=sample_id_column,
            )
        except Exception as exc:
            st.error(f"Precision analysis could not be completed: {exc}")
            st.stop()

        criteria_result = evaluate_precision_criteria(
            result.level_summary,
            max_acceptable_cv=criteria["Maximum CV%"],
        )
        decision = str(criteria_result["decision"])
        criteria_table = build_criteria_table(criteria_result)
        display_level_summary = format_precision_summary_table(result.level_summary)
        display_day_summary = format_precision_summary_table(result.day_summary)
        display_run_summary = format_precision_summary_table(result.run_summary)
        display_criteria_table = format_precision_criteria_table(criteria_table)
        interpretation = generate_precision_interpretation(
            result.level_summary,
            metadata,
            criteria["Maximum CV%"],
            decision,
        )

        st.subheader("Precision Summary")
        show_decision(decision)
        st.dataframe(display_level_summary, width="stretch")

        st.subheader("Acceptance Criteria Results")
        st.dataframe(display_criteria_table, width="stretch")

        if not result.day_summary.empty:
            st.subheader("Day-Level Precision Summary")
            st.dataframe(display_day_summary, width="stretch")

        if not result.run_summary.empty:
            st.subheader("Run-Level Precision Summary")
            st.dataframe(display_run_summary, width="stretch")

        st.subheader("Interpretation")
        st.info(interpretation)

        st.subheader("Visualizations")
        run_chart = create_precision_run_chart(result.analyzed_data)
        cv_bar_chart = create_precision_cv_bar_chart(result.level_summary)
        box_plot = create_precision_box_plot(result.analyzed_data)
        st.plotly_chart(run_chart, width="stretch")

        chart_left, chart_right = st.columns(2)
        with chart_left:
            st.plotly_chart(cv_bar_chart, width="stretch")
        with chart_right:
            st.plotly_chart(box_plot, width="stretch")

        st.subheader("Analyzed Data")
        st.dataframe(result.analyzed_data, width="stretch")

        html_report = build_precision_html_report(
            level_summary=result.level_summary,
            day_summary=result.day_summary,
            run_summary=result.run_summary,
            analyzed_data=result.analyzed_data,
            criteria_table=criteria_table,
            interpretation=interpretation,
            metadata=metadata,
            max_acceptable_cv=criteria["Maximum CV%"],
            decision=decision,
            visualization_html={
                "Precision Run Chart": run_chart.to_html(
                    full_html=False, include_plotlyjs="cdn"
                ),
                "Precision CV% Bar Chart": cv_bar_chart.to_html(
                    full_html=False, include_plotlyjs=False
                ),
                "Precision Box Plot": box_plot.to_html(
                    full_html=False, include_plotlyjs=False
                ),
            },
        )

        export_left, export_right = st.columns(2)
        with export_left:
            st.download_button(
                "Download precision summary CSV",
                data=build_precision_summary_csv(
                    metadata,
                    display_criteria_table,
                    display_level_summary,
                    display_day_summary,
                    display_run_summary,
                ).encode("utf-8"),
                file_name="precision_summary.csv",
                mime="text/csv",
            )
        with export_right:
            st.download_button(
                "Download precision report HTML",
                data=html_report.encode("utf-8"),
                file_name="precision_study_report.html",
                mime="text/html",
            )


def main() -> None:
    """Render the Streamlit interface and run selected analyses."""

    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption(
        "Validation analytics for assay studies and diagnostic laboratory workflows."
    )

    page = st.sidebar.radio(
        "Navigation",
        ["Dashboard", "Study Workspace", "Validation Reports"],
    )

    if page == "Dashboard":
        render_dashboard()
        st.stop()

    if page == "Validation Reports":
        st.subheader("Validation Reports")
        st.info("Report archive and report management workflows are planned for a future module.")
        st.stop()

    study_type = render_study_type_selector()
    if study_type == "Precision Study":
        metadata = render_study_documentation(study_type)
        render_precision_workspace(metadata)
        st.stop()

    if study_type != "Method Comparison":
        st.stop()

    metadata = render_study_documentation(study_type)
    criteria = render_method_comparison_criteria()

    uploaded_file = st.file_uploader(
        "Upload paired method-comparison data",
        type=["csv", "xlsx", "xls"],
        help="Upload a file containing reference and candidate result columns.",
    )

    use_sample_data = st.checkbox(
        "Use included sample HbA1c method-comparison dataset",
        value=uploaded_file is None,
    )

    data = None
    if uploaded_file is not None:
        try:
            data = load_uploaded_file(uploaded_file)
        except Exception as exc:
            st.error(f"Unable to load file: {exc}")
            st.stop()
    elif use_sample_data:
        data = pd.read_csv(SAMPLE_DATA_PATH)

    if data is None:
        st.info("Upload a CSV or Excel file to begin.")
        st.stop()

    metadata["Sample Count"] = len(data)

    st.subheader("Data Preview")
    st.dataframe(data.head(20), width='stretch')

    numeric_columns = get_numeric_columns(data)
    if len(numeric_columns) < 2:
        st.error("At least two numeric result columns are required for analysis.")
        st.stop()

    st.subheader("Column Selection")
    detected_sample_id = detect_sample_id_column(data)
    sample_id_options = ["None"] + list(data.columns)
    sample_id_default = (
        sample_id_options.index(detected_sample_id)
        if detected_sample_id in sample_id_options
        else 0
    )

    left, middle, right = st.columns(3)
    with left:
        reference_column = st.selectbox(
            "Reference result column",
            numeric_columns,
            index=0,
        )
    with middle:
        default_candidate_index = 1 if len(numeric_columns) > 1 else 0
        candidate_column = st.selectbox(
            "Candidate result column",
            numeric_columns,
            index=default_candidate_index,
        )
    with right:
        sample_id_selection = st.selectbox(
            "Sample ID column (optional)",
            sample_id_options,
            index=sample_id_default,
        )
        sample_id_column = None if sample_id_selection == "None" else sample_id_selection

    if reference_column == candidate_column:
        st.warning("Select two different columns before running the analysis.")

    run_analysis = st.button("Run Analysis", type="primary")

    if run_analysis:
        if reference_column == candidate_column:
            st.stop()

        try:
            result = run_method_comparison(
                data, reference_column, candidate_column, sample_id_column
            )
        except Exception as exc:
            st.error(f"Analysis could not be completed: {exc}")
            st.stop()

        summary_table = build_summary_table(result.summary)
        percent_samples_meeting_agreement = calculate_percent_samples_meeting_agreement(
            result.analyzed_data,
            criteria["Sample Agreement Percent Bias Limit"],
        )
        criteria_result = evaluate_acceptance_criteria(
            result.summary,
            min_r_squared=criteria["Minimum R²"],
            min_correlation_r=criteria["Minimum Correlation r"],
            slope_lower_limit=criteria["Slope Lower Limit"],
            slope_upper_limit=criteria["Slope Upper Limit"],
            max_abs_intercept=criteria["Maximum Absolute Intercept"],
            max_abs_mean_bias=criteria["Maximum Absolute Mean Bias"],
            max_abs_mean_percent_bias=criteria[
                "Maximum Absolute Mean Percent Bias"
            ],
            max_abs_mean_difference=criteria["Maximum Absolute Mean Difference"],
            percent_samples_meeting_agreement=percent_samples_meeting_agreement,
            min_percent_samples_meeting_agreement=criteria[
                "Minimum Percent Samples Meeting Agreement"
            ],
        )
        decision = str(criteria_result["decision"])
        criteria_table = build_criteria_table(criteria_result)
        outlier_table = get_top_outliers(result.analyzed_data, count=5)
        interpretation = generate_interpretation(
            result.summary, metadata, criteria, decision
        )

        st.subheader("Summary Table")
        show_decision(decision)
        st.dataframe(summary_table, width='stretch')

        st.subheader("Acceptance Criteria Results")
        st.dataframe(criteria_table, width='stretch')
        st.metric(
            "Samples meeting agreement criteria",
            f"{percent_samples_meeting_agreement:.1f}%",
        )

        st.subheader("Interpretation")
        st.info(interpretation)

        st.subheader("Bland-Altman Style Difference Statistics")
        ba_columns = st.columns(4)
        ba_columns[0].metric("Mean Difference", f"{result.summary['Mean Difference']:.3f}")
        ba_columns[1].metric("SD Difference", f"{result.summary['SD Difference']:.3f}")
        ba_columns[2].metric(
            "Upper LOA", f"{result.summary['Upper Limit of Agreement']:.3f}"
        )
        ba_columns[3].metric(
            "Lower LOA", f"{result.summary['Lower Limit of Agreement']:.3f}"
        )

        st.subheader("Visualizations")
        scatter_plot = create_scatter_plot(result.analyzed_data, result.summary)
        bias_histogram = create_percent_bias_histogram(result.analyzed_data)
        difference_plot = create_difference_plot(result.analyzed_data)
        st.plotly_chart(
            scatter_plot,
            width='stretch',
        )

        chart_left, chart_right = st.columns(2)
        with chart_left:
            st.plotly_chart(
                bias_histogram,
                width='stretch',
            )
        with chart_right:
            st.plotly_chart(
                difference_plot,
                width='stretch',
            )

        st.subheader("Top Disagreement Samples")
        st.dataframe(outlier_table, width='stretch')

        st.subheader("Analyzed Data")
        st.dataframe(result.analyzed_data, width='stretch')

        html_report = build_html_report(
            summary_table,
            criteria_table,
            outlier_table,
            interpretation,
            metadata,
            criteria,
            decision,
            reference_column,
            candidate_column,
            visualization_html={
                "Reference vs Candidate Scatter Plot": scatter_plot.to_html(
                    full_html=False, include_plotlyjs="cdn"
                ),
                "Percent Bias Histogram": bias_histogram.to_html(
                    full_html=False, include_plotlyjs=False
                ),
                "Difference Plot": difference_plot.to_html(
                    full_html=False, include_plotlyjs=False
                ),
            },
        )

        export_left, export_right = st.columns(2)
        with export_left:
            st.download_button(
                "Download analysis summary CSV",
                data=summary_table.to_csv(index=False).encode("utf-8"),
                file_name="analysis_summary.csv",
                mime="text/csv",
            )
        with export_right:
            st.download_button(
                "Download summary report HTML",
                data=html_report.encode("utf-8"),
                file_name="summary_report.html",
                mime="text/html",
            )


if __name__ == "__main__":
    main()
