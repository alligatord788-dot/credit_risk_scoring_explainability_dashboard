from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.project_utils import (
    ARTIFACTS_DIR,
    CLEAN_DATA_PATH,
    EXPLAINABILITY_PATH,
    METRICS_PATH,
    MODEL_PATH,
    SCHEMA_PATH,
    load_json,
    load_pickle,
    probability_to_decision,
    probability_to_risk_band,
    probability_to_score,
)


SCORED_APPLICANTS_PATH = ARTIFACTS_DIR / "scored_test_applicants.csv"
THRESHOLD_PATH = ARTIFACTS_DIR / "threshold_analysis.csv"


@st.cache_data
def load_clean_data():
    return pd.read_csv(CLEAN_DATA_PATH)


@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


@st.cache_resource
def load_model():
    return load_pickle(MODEL_PATH)


def check_artifacts():
    required_paths = [
        CLEAN_DATA_PATH,
        MODEL_PATH,
        METRICS_PATH,
        SCHEMA_PATH,
        EXPLAINABILITY_PATH,
        SCORED_APPLICANTS_PATH,
        THRESHOLD_PATH,
    ]
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        st.error("Run data cleaning and model training first.")
        for path in missing:
            st.write(path)
        st.stop()


def main():
    st.set_page_config(page_title="Credit Risk Scoring Dashboard", layout="wide")
    check_artifacts()

    df = load_clean_data()
    scored_df = load_csv(SCORED_APPLICANTS_PATH)
    threshold_df = load_csv(THRESHOLD_PATH)
    importance_df = load_csv(EXPLAINABILITY_PATH)
    metrics = load_json(METRICS_PATH)
    schema = load_json(SCHEMA_PATH)
    model = load_model()

    st.title("Credit Risk Scoring & Explainability Dashboard")
    st.caption("Default probability, risk score, risk bands, approval decisions and model explainability")

    best_model = metrics["best_model"]
    best_metrics = metrics["model_results"][best_model]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Records", f"{len(df):,}")
    c2.metric("Default Rate", f"{df['default'].mean():.2%}")
    c3.metric("Best Model", best_model)
    c4.metric("ROC-AUC", best_metrics["roc_auc"])
    c5.metric("Recall", best_metrics["recall"])

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Risk Analytics", "Explainability", "Thresholds", "Applicant Scoring"]
    )

    with tab1:
        left, right = st.columns(2)

        with left:
            fig_band = px.histogram(
                scored_df,
                x="risk_band",
                color="actual_default",
                title="Risk Band Distribution",
                barmode="group",
            )
            st.plotly_chart(fig_band, use_container_width=True)

        with right:
            fig_decision = px.histogram(
                scored_df,
                x="credit_decision",
                color="actual_default",
                title="Approval Decision Distribution",
                barmode="group",
            )
            st.plotly_chart(fig_decision, use_container_width=True)

        numeric_columns = schema["numeric_columns"]
        if numeric_columns:
            selected_numeric = st.selectbox("Select numeric feature", numeric_columns)
            fig_feature = px.box(
                df,
                x="default",
                y=selected_numeric,
                color="default",
                title=f"{selected_numeric} by Default Status",
            )
            st.plotly_chart(fig_feature, use_container_width=True)

    with tab2:
        st.subheader("Top Model Risk Drivers")
        st.dataframe(importance_df, use_container_width=True)

        fig_importance = px.bar(
            importance_df.head(15),
            x="importance",
            y="feature",
            orientation="h",
            title="Top 15 Important Features",
        )
        fig_importance.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_importance, use_container_width=True)

    with tab3:
        st.subheader("Threshold Tradeoff")
        st.write(
            "Lower threshold catches more risky borrowers but may also flag more safe borrowers."
        )
        st.dataframe(threshold_df, use_container_width=True)

        fig_threshold = px.line(
            threshold_df,
            x="threshold",
            y=["precision", "recall", "f1_score"],
            markers=True,
            title="Precision, Recall and F1 Across Thresholds",
        )
        st.plotly_chart(fig_threshold, use_container_width=True)

    with tab4:
        st.subheader("Score New Applicant")

        user_input = {}

        with st.form("credit_risk_form"):
            numeric_columns = schema["numeric_columns"]
            categorical_columns = schema["categorical_columns"]

            for column in numeric_columns:
                user_input[column] = st.number_input(
                    column,
                    value=float(schema["numeric_defaults"][column]),
                )

            for column in categorical_columns:
                options = schema["categorical_options"].get(column, [])
                if options:
                    user_input[column] = st.selectbox(column, options)
                else:
                    user_input[column] = st.text_input(column, "Unknown")

            submitted = st.form_submit_button("Calculate Credit Risk")

        if submitted:
            input_df = pd.DataFrame([user_input], columns=schema["feature_columns"])
            probability = float(model.predict_proba(input_df)[0][1])
            score = probability_to_score(probability)
            band = probability_to_risk_band(probability)
            decision = probability_to_decision(probability)

            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Default Probability", f"{probability:.2%}")
            r2.metric("Risk Score", score)
            r3.metric("Risk Band", band)
            r4.metric("Decision", decision)

            st.info(
                "Interview explanation: the model predicts probability first, then "
                "business rules convert it into a score, risk band and lending decision."
            )


if __name__ == "__main__":
    main()
