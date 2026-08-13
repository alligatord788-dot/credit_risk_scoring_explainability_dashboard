import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


# Common project paths used by all scripts. Keeping paths here avoids repeating
# folder logic in cleaning, training, prediction and dashboard files.
PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"

# Main outputs created by the pipeline. The dashboard and web app read these
# saved artifacts instead of retraining the model.
CLEAN_DATA_PATH = PROCESSED_DATA_DIR / "cleaned_credit_risk_data.csv"
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
SCHEMA_PATH = ARTIFACTS_DIR / "input_schema.json"
EXPLAINABILITY_PATH = ARTIFACTS_DIR / "feature_importance.csv"
SAMPLE_ROWS = 100000

# Possible target names used in loan datasets. This lets the code work with
# different public datasets without manually renaming columns first.
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

# Keywords for columns we should remove. Many of these are IDs or future loan
# behavior like payments/recoveries, which would create target leakage.
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

# A compact set of borrower-level features. This keeps the model explainable
# and makes the dashboard/app form easier for interviews.
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


# Find the CSV file. First check this project's data/raw folder; if empty, reuse
# the dataset from the previous loan_default_risk_prediction project.
def find_dataset_path():
    csv_files = sorted(RAW_DATA_DIR.glob("*.csv"))
    if csv_files:
        return csv_files[0]

    sibling_raw = PROJECT_DIR.parent / "loan_default_risk_prediction" / "data" / "raw"
    sibling_csv_files = sorted(sibling_raw.glob("*.csv"))
    if sibling_csv_files:
        return sibling_csv_files[0]

    raise FileNotFoundError(
        "No CSV found. Put the loan dataset in data/raw/ or keep the previous "
        "loan_default_risk_prediction dataset folder beside this project."
    )


# Read only the first SAMPLE_ROWS rows because the LendingClub file is large.
# This keeps the project fast enough for local runs and Kaggle notebooks.
def read_credit_csv(path=None, sample_rows=SAMPLE_ROWS):
    path = path or find_dataset_path()
    return pd.read_csv(path, nrows=sample_rows, low_memory=False)


# Standardize column names so feature selection works reliably.
def simplify_columns(df):
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.replace("/", "_")
    )
    return df


# Detect which column should be predicted. For this project, the target is
# whether a loan is default/high-risk or non-default/low-risk.
def find_target_column(df):
    lower_map = {column.lower(): column for column in df.columns}

    for candidate in TARGET_CANDIDATES:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    binary_columns = []
    for column in df.columns:
        if df[column].dropna().nunique() == 2:
            binary_columns.append(column)

    if binary_columns:
        return binary_columns[-1]

    raise ValueError("Target column not found. Rename target to default or loan_status.")


# Convert loan status text into a binary target. 1 means default/high-risk and
# 0 means fully paid/low-risk. Current loans are removed because their outcome
# is not yet known.
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


# Full cleaning pipeline: simplify columns, create target, remove leakage,
# select useful features and return a model-ready table.
def clean_raw_data(df):
    # Step 1: clean column names and find the loan outcome column.
    df = simplify_columns(df)
    target_column = find_target_column(df)

    # Step 2: convert target values into 0/1 and remove unresolved loans.
    df = df.dropna(subset=[target_column]).copy()
    y = convert_target_to_binary(df[target_column])
    valid_target_mask = y.notna()
    df = df.loc[valid_target_mask].copy()
    y = y.loc[valid_target_mask].astype(int)

    # Step 3: remove target from features before training.
    X = df.drop(columns=[target_column]).copy()

    # Step 4: remove leakage, ID-like and constant columns.
    columns_to_drop = []
    for column in X.columns:
        lower_name = column.lower()
        if any(keyword in lower_name for keyword in DROP_KEYWORDS):
            columns_to_drop.append(column)
        elif X[column].nunique(dropna=True) <= 1:
            columns_to_drop.append(column)

    X = X.drop(columns=columns_to_drop, errors="ignore")

    # Step 5: keep selected borrower-level features if available.
    available_features = [column for column in SELECTED_FEATURES if column in X.columns]
    if available_features:
        X = X[available_features].copy()

    # Step 6: clean extra spaces in categorical text values.
    for column in X.columns:
        if X[column].dtype == "object":
            X[column] = X[column].astype(str).str.strip()

    # Step 7: add final binary target column.
    cleaned_df = X.copy()
    cleaned_df["default"] = y.values
    return cleaned_df, target_column, columns_to_drop


# Split feature names into numeric and categorical groups because scikit-learn
# uses different preprocessing for each type.
def get_feature_columns(df):
    feature_columns = [column for column in df.columns if column != "default"]
    numeric_columns = df[feature_columns].select_dtypes(include=np.number).columns.tolist()
    categorical_columns = [column for column in feature_columns if column not in numeric_columns]
    return feature_columns, numeric_columns, categorical_columns


# Convert default probability into a simple scorecard-style number. Higher
# probability means lower credit score.
def probability_to_score(probability):
    return int(round(850 - probability * 550))


# Convert probability into business-friendly risk categories.
def probability_to_risk_band(probability):
    if probability < 0.20:
        return "Low Risk"
    if probability < 0.40:
        return "Medium Risk"
    if probability < 0.60:
        return "High Risk"
    return "Very High Risk"


# Convert probability into a lending decision. This is the business decision
# layer that makes the project more advanced than simple prediction.
def probability_to_decision(probability):
    if probability < 0.25:
        return "Approve"
    if probability < 0.50:
        return "Manual Review"
    return "Reject"


# Add score, risk band and decision columns to a DataFrame of applicants.
def add_risk_outputs(df, probabilities):
    result_df = df.copy()
    result_df["default_probability"] = probabilities
    result_df["risk_score"] = result_df["default_probability"].apply(probability_to_score)
    result_df["risk_band"] = result_df["default_probability"].apply(probability_to_risk_band)
    result_df["credit_decision"] = result_df["default_probability"].apply(probability_to_decision)
    return result_df


# Build a schema for prediction forms. It stores feature order, numeric default
# values and dropdown options for categorical fields.
def build_input_schema(df, metrics):
    feature_columns, numeric_columns, categorical_columns = get_feature_columns(df)

    numeric_defaults = {}
    for column in numeric_columns:
        numeric_defaults[column] = float(df[column].median())

    categorical_options = {}
    for column in categorical_columns:
        values = df[column].dropna().astype(str).value_counts().head(30).index.tolist()
        categorical_options[column] = values

    return {
        "feature_columns": feature_columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "numeric_defaults": numeric_defaults,
        "categorical_options": categorical_options,
        "metrics": metrics,
    }


# Save dictionary-like data such as metrics and schema as JSON.
def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


# Load JSON files created during training.
def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


# Save trained model pipeline as a pickle file.
def save_pickle(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(data, file)


# Load saved model pipeline for prediction and dashboards.
def load_pickle(path):
    with open(path, "rb") as file:
        return pickle.load(file)
