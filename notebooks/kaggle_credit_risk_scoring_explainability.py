# %% [markdown]
# # Credit Risk Scoring & Explainability Dashboard
# ### This notebook builds a finance-style credit-risk system: data cleaning, model training, default probability, risk score, risk bands, approval decisions, explainability and dashboard export.

# %%
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import HTML, display
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


# %% [markdown]
# # 1. Load Dataset
# ### Attach the loan dataset in Kaggle Input. This directly loads your LendingClub CSV and shows the first rows.

# %%
possible_paths = list(Path("/kaggle/input").rglob("*.csv"))

print("CSV files found:")
for path in possible_paths:
    print(path)

DATA_PATH = None
for path in possible_paths:
    lower_name = path.name.lower()
    if "accepted" in lower_name or "loan" in lower_name or "credit" in lower_name:
        DATA_PATH = path
        break

if DATA_PATH is None and possible_paths:
    DATA_PATH = possible_paths[0]

print("Using dataset:", DATA_PATH)
df = pd.read_csv(DATA_PATH, nrows=100000, low_memory=False)
df.head()


# %% [markdown]
# # 2. Basic Data Check
# ### Check rows, columns, missing values and data types before cleaning.

# %%
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())
print("\nMissing values:")
print(df.isna().sum().sort_values(ascending=False).head(20))
df.info()


# %% [markdown]
# # 3. Detect Target Column
# ### Loan datasets may use names like `loan_status`, `default` or `target`, so this step detects the target automatically.

# %%
TARGET_CANDIDATES = [
    "default",
    "loan_default",
    "is_default",
    "target",
    "loan_status",
    "status",
    "Loan_Status",
    "Default",
    "TARGET",
]


def simplify_columns(dataframe):
    dataframe = dataframe.copy()
    dataframe.columns = (
        dataframe.columns.str.strip()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.replace("/", "_")
    )
    return dataframe


def find_target_column(dataframe):
    lower_map = {column.lower(): column for column in dataframe.columns}

    for candidate in TARGET_CANDIDATES:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    binary_columns = []
    for column in dataframe.columns:
        if dataframe[column].dropna().nunique() == 2:
            binary_columns.append(column)

    if binary_columns:
        return binary_columns[-1]

    raise ValueError("Target column not found. Rename target to default or loan_status.")


df = simplify_columns(df)
target_column = find_target_column(df)
print("Detected target column:", target_column)
df[target_column].value_counts(dropna=False).head(20)


# %% [markdown]
# # 4. Convert Target Into 0 And 1
# ### Classification needs a numeric target. Here `1` means default/high-risk and `0` means non-default/low-risk. Current/unresolved loans are removed because their final outcome is unknown.

# %%
def convert_target_to_binary(series):
    if pd.api.types.is_numeric_dtype(series):
        values = sorted(series.dropna().unique())
        if len(values) != 2:
            raise ValueError("Target must contain exactly two classes.")
        return series.map({values[0]: 0, values[1]: 1}).astype(int)

    def to_binary(value):
        text = str(value).strip().lower()

        if text in {"yes", "y", "1", "true", "bad"}:
            return 1
        if text in {"no", "n", "0", "false", "good"}:
            return 0
        if "fully paid" in text or text == "paid":
            return 0
        if "charged off" in text or "default" in text or "late" in text or "not paid" in text:
            return 1
        if "current" in text or "grace" in text:
            return np.nan

        return np.nan

    return series.apply(to_binary)


df = df.dropna(subset=[target_column]).copy()
y = convert_target_to_binary(df[target_column])
valid_target_mask = y.notna()
df = df.loc[valid_target_mask].copy()
y = y.loc[valid_target_mask].astype(int)

print("Default distribution:")
print(y.value_counts(normalize=True).round(4))


# %% [markdown]
# # 5. Create Clean Feature Table
# ### Remove target leakage columns. Payment, recovery, settlement and last-payment fields are future information, so using them would make the model unrealistic.

# %%
DROP_KEYWORDS = [
    "id",
    "name",
    "url",
    "desc",
    "title",
    "pymnt",
    "recover",
    "collection",
    "settlement",
    "hardship",
    "last_",
    "next_",
    "out_prncp",
    "total_rec",
    "debt_settlement",
]

SELECTED_FEATURES = [
    "loan_amnt",
    "term",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "purpose",
    "dti",
    "delinq_2yrs",
    "fico_range_low",
    "fico_range_high",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "application_type",
    "mort_acc",
    "pub_rec_bankruptcies",
]

X = df.drop(columns=[target_column]).copy()

columns_to_drop = []
for column in X.columns:
    lower_name = column.lower()
    if any(keyword in lower_name for keyword in DROP_KEYWORDS):
        columns_to_drop.append(column)
    elif X[column].nunique(dropna=True) <= 1:
        columns_to_drop.append(column)

X = X.drop(columns=columns_to_drop, errors="ignore")

available_features = [column for column in SELECTED_FEATURES if column in X.columns]
if available_features:
    X = X[available_features].copy()

for column in X.columns:
    if X[column].dtype == "object":
        X[column] = X[column].astype(str).str.strip()

cleaned_df = X.copy()
cleaned_df["default"] = y.values

print("Cleaned shape:", cleaned_df.shape)
cleaned_df.head()


# %% [markdown]
# # 6. Split Numeric And Categorical Columns
# ### Numeric features are scaled. Categorical features are one-hot encoded.

# %%
feature_columns = [column for column in cleaned_df.columns if column != "default"]
numeric_columns = cleaned_df[feature_columns].select_dtypes(include=np.number).columns.tolist()
categorical_columns = [column for column in feature_columns if column not in numeric_columns]

print("Numeric columns:", numeric_columns)
print("Categorical columns:", categorical_columns)


# %% [markdown]
# # 7. Train-Test Split
# ### Stratified split keeps the default ratio similar in training and testing data.

# %%
X = cleaned_df[feature_columns]
y = cleaned_df["default"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("Training rows:", X_train.shape[0])
print("Testing rows:", X_test.shape[0])


# %% [markdown]
# # 8. Build Preprocessing And Model Pipeline
# ### A pipeline ensures that the same preprocessing is used during training and prediction.

# %%
def create_pipeline(model):
    numeric_processor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_processor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_processor, numeric_columns),
            ("categorical", categorical_processor, categorical_columns),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


# %% [markdown]
# # 9. Train And Compare Models
# ### Logistic Regression is interpretable. Decision Tree and Random Forest capture non-linear patterns.

# %%
def evaluate_model(model, X_test, y_test, threshold=0.50):
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    return {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
    }


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, class_weight="balanced"),
}

results = {}
trained_models = {}

for name, model in models.items():
    pipeline = create_pipeline(model)
    pipeline.fit(X_train, y_train)
    results[name] = evaluate_model(pipeline, X_test, y_test)
    trained_models[name] = pipeline

results_df = pd.DataFrame(results).T.sort_values("roc_auc", ascending=False)
results_df


# %% [markdown]
# # 10. Select Best Model
# ### Choose the model with highest ROC-AUC because it separates risky and safe borrowers best across thresholds.

# %%
best_model_name = max(results, key=lambda name: results[name]["roc_auc"])
best_model = trained_models[best_model_name]

print("Best model:", best_model_name)
print("Best model metrics:", results[best_model_name])


# %% [markdown]
# # 11. Convert Probability Into Risk Score, Risk Band And Decision
# ### This is the business layer. It converts model probability into an interpretable credit-risk output.

# %%
def probability_to_score(probability):
    return int(round(850 - probability * 550))


def probability_to_risk_band(probability):
    if probability < 0.20:
        return "Low Risk"
    if probability < 0.40:
        return "Medium Risk"
    if probability < 0.60:
        return "High Risk"
    return "Very High Risk"


def probability_to_decision(probability):
    if probability < 0.25:
        return "Approve"
    if probability < 0.50:
        return "Manual Review"
    return "Reject"


test_probabilities = best_model.predict_proba(X_test)[:, 1]

scored_test_df = X_test.reset_index(drop=True).copy()
scored_test_df["default_probability"] = test_probabilities
scored_test_df["risk_score"] = scored_test_df["default_probability"].apply(probability_to_score)
scored_test_df["risk_band"] = scored_test_df["default_probability"].apply(probability_to_risk_band)
scored_test_df["credit_decision"] = scored_test_df["default_probability"].apply(probability_to_decision)
scored_test_df["actual_default"] = y_test.reset_index(drop=True).values

scored_test_df[["default_probability", "risk_score", "risk_band", "credit_decision", "actual_default"]].head()


# %% [markdown]
# # 12. Threshold Analysis
# ### Lower thresholds catch more defaulters but also flag more safe borrowers. This is important in credit-risk decisions.

# %%
threshold_rows = []

for threshold in [0.20, 0.30, 0.40, 0.50, 0.60]:
    predictions = (test_probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
    threshold_rows.append(
        {
            "threshold": threshold,
            "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
            "approved_high_risk_missed": int(fn),
            "safe_borrowers_flagged": int(fp),
            "true_defaults_caught": int(tp),
            "true_safe_approved": int(tn),
        }
    )

threshold_df = pd.DataFrame(threshold_rows)
threshold_df


# %% [markdown]
# # 13. Feature Importance
# ### Explainability shows which features influence the model most. For Logistic Regression, we use coefficient magnitude.

# %%
def get_feature_importance(model_pipeline):
    model = model_pipeline.named_steps["model"]
    preprocessor = model_pipeline.named_steps["preprocessor"]
    transformed_names = preprocessor.get_feature_names_out()

    if hasattr(model, "coef_"):
        coefficients = model.coef_[0]
        importance_df = pd.DataFrame(
            {
                "feature": transformed_names,
                "importance": np.abs(coefficients),
                "effect": coefficients,
            }
        )
        return importance_df.sort_values("importance", ascending=False).head(25)

    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame(
            {
                "feature": transformed_names,
                "importance": model.feature_importances_,
                "effect": model.feature_importances_,
            }
        )
        return importance_df.sort_values("importance", ascending=False).head(25)

    return pd.DataFrame()


importance_df = get_feature_importance(best_model)
importance_df.head(15)


# %% [markdown]
# # 14. Save Artifacts
# ### Save model, metrics, schema, scored applicants and explainability files to Kaggle output.

# %%
OUTPUT_DIR = Path("/kaggle/working/credit_risk_artifacts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

metrics = {
    "best_model": best_model_name,
    "model_results": results,
    "target_mean_default_rate": round(float(y.mean()), 4),
    "training_rows": int(X_train.shape[0]),
    "testing_rows": int(X_test.shape[0]),
    "risk_band_rules": {
        "Low Risk": "probability < 0.20",
        "Medium Risk": "0.20 <= probability < 0.40",
        "High Risk": "0.40 <= probability < 0.60",
        "Very High Risk": "probability >= 0.60",
    },
    "decision_rules": {
        "Approve": "probability < 0.25",
        "Manual Review": "0.25 <= probability < 0.50",
        "Reject": "probability >= 0.50",
    },
    "threshold_analysis": threshold_df.to_dict(orient="records"),
}

numeric_defaults = {}
for column in numeric_columns:
    numeric_defaults[column] = float(cleaned_df[column].median())

categorical_options = {}
for column in categorical_columns:
    categorical_options[column] = cleaned_df[column].dropna().astype(str).value_counts().head(30).index.tolist()

schema = {
    "feature_columns": feature_columns,
    "numeric_columns": numeric_columns,
    "categorical_columns": categorical_columns,
    "numeric_defaults": numeric_defaults,
    "categorical_options": categorical_options,
    "metrics": metrics,
}

with open(OUTPUT_DIR / "model.pkl", "wb") as file:
    pickle.dump(best_model, file)

with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=2)

with open(OUTPUT_DIR / "input_schema.json", "w", encoding="utf-8") as file:
    json.dump(schema, file, indent=2)

cleaned_df.to_csv(OUTPUT_DIR / "cleaned_credit_risk_data.csv", index=False)
scored_test_df.to_csv(OUTPUT_DIR / "scored_test_applicants.csv", index=False)
threshold_df.to_csv(OUTPUT_DIR / "threshold_analysis.csv", index=False)
importance_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

print("Artifacts saved to:", OUTPUT_DIR)


# %% [markdown]
# # 15. Sample Applicant Scoring
# ### Build one applicant from median/common values and produce default probability, score, band and decision.

# %%
sample_input = {}

for column in numeric_columns:
    sample_input[column] = numeric_defaults[column]

for column in categorical_columns:
    options = categorical_options.get(column, [])
    sample_input[column] = options[0] if options else "Unknown"

sample_df = pd.DataFrame([sample_input], columns=feature_columns)
sample_probability = float(best_model.predict_proba(sample_df)[0][1])

print("Default Probability:", round(sample_probability, 4))
print("Risk Score:", probability_to_score(sample_probability))
print("Risk Band:", probability_to_risk_band(sample_probability))
print("Credit Decision:", probability_to_decision(sample_probability))


# %% [markdown]
# # 16. Dashboard Charts
# ### Create charts for model performance, risk bands, threshold analysis and feature importance.

# %%
import plotly.express as px
import plotly.io as pio


def show_chart(fig):
    html = fig.to_html(full_html=False, include_plotlyjs=True)
    display(HTML(html))


model_results_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "model"})

fig_model = px.bar(
    model_results_df,
    x="model",
    y="roc_auc",
    color="model",
    text="roc_auc",
    title="Model Comparison by ROC-AUC",
)
show_chart(fig_model)


# %%
fig_band = px.histogram(
    scored_test_df,
    x="risk_band",
    color="actual_default",
    barmode="group",
    title="Risk Band Distribution",
)
show_chart(fig_band)


# %%
fig_threshold = px.line(
    threshold_df,
    x="threshold",
    y=["precision", "recall", "f1_score"],
    markers=True,
    title="Threshold Tradeoff",
)
show_chart(fig_threshold)


# %%
fig_importance = px.bar(
    importance_df.head(15),
    x="importance",
    y="feature",
    orientation="h",
    title="Top Risk Drivers",
)
fig_importance.update_layout(yaxis={"categoryorder": "total ascending"})
show_chart(fig_importance)


# %% [markdown]
# # 17. Save Dashboard HTML
# ### Export the dashboard as one downloadable HTML file.

# %%
for fig in [fig_model, fig_band, fig_threshold, fig_importance]:
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#111827", font={"color": "#e5edf7"})

dashboard_html_path = Path("/kaggle/working/credit_risk_dashboard.html")

html_parts = [
    """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Credit Risk Scoring Dashboard</title>
    <style>
        body { margin: 0; font-family: Arial, sans-serif; background: #0f172a; color: #e5edf7; }
        .page { max-width: 1150px; margin: 0 auto; padding: 28px 20px 44px; }
        .section { margin-top: 22px; padding: 18px; background: #111827; border: 1px solid #334155; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 10px; border-bottom: 1px solid #334155; text-align: left; }
        th { background: #1f2937; }
    </style>
</head>
<body>
<main class="page">
<h1>Credit Risk Scoring & Explainability Dashboard</h1>
<p>Default probability, risk score, risk bands, approval decision, threshold analysis and feature importance.</p>
""",
    "<div class='section'><h2>Model Metrics</h2>",
    model_results_df.to_html(index=False),
    "</div>",
    "<div class='section'><h2>Threshold Analysis</h2>",
    threshold_df.to_html(index=False),
    "</div>",
    "<div class='section'><h2>Model Comparison</h2>",
    pio.to_html(fig_model, full_html=False, include_plotlyjs="cdn"),
    "</div>",
    "<div class='section'><h2>Risk Bands</h2>",
    pio.to_html(fig_band, full_html=False, include_plotlyjs=False),
    "</div>",
    "<div class='section'><h2>Threshold Tradeoff</h2>",
    pio.to_html(fig_threshold, full_html=False, include_plotlyjs=False),
    "</div>",
    "<div class='section'><h2>Top Risk Drivers</h2>",
    pio.to_html(fig_importance, full_html=False, include_plotlyjs=False),
    "</div>",
    "</main></body></html>",
]

dashboard_html_path.write_text("\n".join(html_parts), encoding="utf-8")
print("Saved dashboard HTML:", dashboard_html_path)


# %% [markdown]
# # 18. Save Streamlit App
# ### Download this app with the artifacts and run it locally.

# %%
streamlit_app_code = r'''
import json
import pickle
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_DIR / "credit_risk_artifacts"

def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)

def load_pickle(path):
    with open(path, "rb") as file:
        return pickle.load(file)

def probability_to_score(probability):
    return int(round(850 - probability * 550))

def probability_to_risk_band(probability):
    if probability < 0.20:
        return "Low Risk"
    if probability < 0.40:
        return "Medium Risk"
    if probability < 0.60:
        return "High Risk"
    return "Very High Risk"

def probability_to_decision(probability):
    if probability < 0.25:
        return "Approve"
    if probability < 0.50:
        return "Manual Review"
    return "Reject"

st.set_page_config(page_title="Credit Risk Scoring", layout="wide")
st.title("Credit Risk Scoring & Explainability Dashboard")

model = load_pickle(ARTIFACTS_DIR / "model.pkl")
metrics = load_json(ARTIFACTS_DIR / "metrics.json")
schema = load_json(ARTIFACTS_DIR / "input_schema.json")
df = pd.read_csv(ARTIFACTS_DIR / "cleaned_credit_risk_data.csv")
scored_df = pd.read_csv(ARTIFACTS_DIR / "scored_test_applicants.csv")
threshold_df = pd.read_csv(ARTIFACTS_DIR / "threshold_analysis.csv")
importance_df = pd.read_csv(ARTIFACTS_DIR / "feature_importance.csv")

best_model = metrics["best_model"]
best_metrics = metrics["model_results"][best_model]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Records", f"{len(df):,}")
c2.metric("Default Rate", f"{df['default'].mean():.2%}")
c3.metric("Best Model", best_model)
c4.metric("ROC-AUC", best_metrics["roc_auc"])

tab1, tab2, tab3 = st.tabs(["Analytics", "Explainability", "Predict"])

with tab1:
    fig_band = px.histogram(scored_df, x="risk_band", color="actual_default", barmode="group")
    st.plotly_chart(fig_band, use_container_width=True)
    fig_threshold = px.line(threshold_df, x="threshold", y=["precision", "recall", "f1_score"], markers=True)
    st.plotly_chart(fig_threshold, use_container_width=True)

with tab2:
    st.dataframe(importance_df, use_container_width=True)
    fig_importance = px.bar(importance_df.head(15), x="importance", y="feature", orientation="h")
    fig_importance.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_importance, use_container_width=True)

with tab3:
    user_input = {}
    for column in schema["numeric_columns"]:
        user_input[column] = st.number_input(column, value=float(schema["numeric_defaults"][column]))
    for column in schema["categorical_columns"]:
        options = schema["categorical_options"].get(column, [])
        user_input[column] = st.selectbox(column, options) if options else st.text_input(column, "Unknown")

    if st.button("Calculate Credit Risk"):
        input_df = pd.DataFrame([user_input], columns=schema["feature_columns"])
        probability = float(model.predict_proba(input_df)[0][1])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Default Probability", f"{probability:.2%}")
        c2.metric("Risk Score", probability_to_score(probability))
        c3.metric("Risk Band", probability_to_risk_band(probability))
        c4.metric("Decision", probability_to_decision(probability))
'''

Path("/kaggle/working/dashboard_app.py").write_text(streamlit_app_code, encoding="utf-8")
print("Saved Streamlit app:", "/kaggle/working/dashboard_app.py")


# %% [markdown]
# # 19. Final Output Paths
# ### Download these files from Kaggle output after running all cells.

# %%
print("Kaggle outputs created:")
print("- /kaggle/working/credit_risk_artifacts/model.pkl")
print("- /kaggle/working/credit_risk_artifacts/metrics.json")
print("- /kaggle/working/credit_risk_artifacts/input_schema.json")
print("- /kaggle/working/credit_risk_artifacts/cleaned_credit_risk_data.csv")
print("- /kaggle/working/credit_risk_artifacts/feature_importance.csv")
print("- /kaggle/working/credit_risk_artifacts/threshold_analysis.csv")
print("- /kaggle/working/credit_risk_artifacts/scored_test_applicants.csv")
print("- /kaggle/working/credit_risk_dashboard.html")
print("- /kaggle/working/dashboard_app.py")
