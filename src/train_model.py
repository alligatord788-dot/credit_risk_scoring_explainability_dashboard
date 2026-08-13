import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from project_utils import (
    ARTIFACTS_DIR,
    CLEAN_DATA_PATH,
    EXPLAINABILITY_PATH,
    METRICS_PATH,
    MODEL_PATH,
    SCHEMA_PATH,
    add_risk_outputs,
    build_input_schema,
    get_feature_columns,
    load_json,
    probability_to_decision,
    probability_to_risk_band,
    probability_to_score,
    save_json,
    save_pickle,
)


# Create one complete scikit-learn pipeline. It combines preprocessing and the
# ML model, so the same transformations happen during training and prediction.
def create_pipeline(model, numeric_columns, categorical_columns):
    # Numeric features get median imputation for missing values and scaling for
    # models like Logistic Regression.
    numeric_processor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical features get missing-value filling and one-hot encoding.
    # handle_unknown="ignore" prevents errors for unseen categories in the app.
    categorical_processor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # ColumnTransformer applies the correct preprocessing to each column group.
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_processor, numeric_columns),
            ("categorical", categorical_processor, categorical_columns),
        ]
    )

    # Final pipeline: preprocess first, then fit/predict with the model.
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


# Calculate model performance on test data. Threshold lets us decide what
# probability should be treated as default/high-risk.
def evaluate_model(model, X_test, y_test, threshold=0.50):
    # predict_proba returns default probability for each applicant.
    probabilities = model.predict_proba(X_test)[:, 1]

    # Convert probability into 0/1 prediction using the selected threshold.
    predictions = (probabilities >= threshold).astype(int)

    return {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
    }


# Build a threshold table to show business tradeoffs. Lower threshold catches
# more defaults, but also flags more safe borrowers.
def create_threshold_table(model, X_test, y_test):
    probabilities = model.predict_proba(X_test)[:, 1]
    rows = []

    for threshold in [0.20, 0.30, 0.40, 0.50, 0.60]:
        # For each threshold, create predictions and confusion matrix values.
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
        rows.append(
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

    return pd.DataFrame(rows)


# Extract important features from the best model. This gives an explainability
# layer so we can discuss which factors influence credit risk.
def get_feature_importance(best_model, feature_columns):
    model = best_model.named_steps["model"]

    # Logistic Regression uses coefficients. Larger absolute coefficient means
    # stronger influence on the prediction.
    if hasattr(model, "coef_"):
        preprocessor = best_model.named_steps["preprocessor"]
        transformed_names = preprocessor.get_feature_names_out()
        coefficients = model.coef_[0]

        importance_df = pd.DataFrame(
            {
                "feature": transformed_names,
                "importance": np.abs(coefficients),
                "effect": coefficients,
            }
        )
        return importance_df.sort_values("importance", ascending=False).head(25)

    # Tree models use feature_importances_.
    if hasattr(model, "feature_importances_"):
        preprocessor = best_model.named_steps["preprocessor"]
        transformed_names = preprocessor.get_feature_names_out()
        importance_df = pd.DataFrame(
            {
                "feature": transformed_names,
                "importance": model.feature_importances_,
                "effect": model.feature_importances_,
            }
        )
        return importance_df.sort_values("importance", ascending=False).head(25)

    return pd.DataFrame({"feature": feature_columns, "importance": 0.0, "effect": 0.0})


# Main workflow: load cleaned data, train models, create scorecard outputs,
# save artifacts and print results.
def main():
    # Training starts from the cleaned CSV so raw cleaning and modeling stay
    # separate and easy to explain.
    if not CLEAN_DATA_PATH.exists():
        raise FileNotFoundError("Run python src/data_cleaning.py first.")

    df = pd.read_csv(CLEAN_DATA_PATH)
    feature_columns, numeric_columns, categorical_columns = get_feature_columns(df)

    # X contains borrower features; y contains the default target.
    X = df[feature_columns]
    y = df["default"]

    # Stratified split keeps default/non-default ratio similar in train and test.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # Three simple models are trained for comparison. class_weight="balanced"
    # helps because default cases are fewer than non-default cases.
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=42,
            class_weight="balanced",
        ),
    }

    results = {}
    trained_models = {}

    # Train each model pipeline and store its metrics.
    for name, model in models.items():
        pipeline = create_pipeline(model, numeric_columns, categorical_columns)
        pipeline.fit(X_train, y_train)
        results[name] = evaluate_model(pipeline, X_test, y_test)
        trained_models[name] = pipeline

    # Choose best model using ROC-AUC because it measures ranking quality across
    # all possible thresholds.
    best_model_name = max(results, key=lambda name: results[name]["roc_auc"])
    best_model = trained_models[best_model_name]

    # Generate probabilities on test applicants and convert them into score,
    # risk band and credit decision.
    test_probabilities = best_model.predict_proba(X_test)[:, 1]
    scored_test_df = add_risk_outputs(X_test.reset_index(drop=True), test_probabilities)
    scored_test_df["actual_default"] = y_test.reset_index(drop=True).values

    # Create advanced project outputs: threshold tradeoff and feature importance.
    threshold_df = create_threshold_table(best_model, X_test, y_test)
    feature_importance_df = get_feature_importance(best_model, feature_columns)

    # Store all model/business outputs in one metrics dictionary for dashboard.
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

    # Schema is used by prediction apps to create valid input forms.
    schema = build_input_schema(df, metrics)

    # Save all artifacts needed by dashboard, web app and future prediction.
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    save_pickle(best_model, MODEL_PATH)
    save_json(metrics, METRICS_PATH)
    save_json(schema, SCHEMA_PATH)
    feature_importance_df.to_csv(EXPLAINABILITY_PATH, index=False)

    scored_path = ARTIFACTS_DIR / "scored_test_applicants.csv"
    threshold_path = ARTIFACTS_DIR / "threshold_analysis.csv"
    scored_test_df.to_csv(scored_path, index=False)
    threshold_df.to_csv(threshold_path, index=False)

    # Print key outputs so the terminal run itself is explainable.
    print("Best model:", best_model_name)
    print("\nModel results:")
    print(pd.DataFrame(results).T)
    print("\nThreshold analysis:")
    print(threshold_df)
    print("\nTop risk drivers:")
    print(feature_importance_df.head(10))
    print("\nSaved model:", MODEL_PATH)
    print("Saved metrics:", METRICS_PATH)
    print("Saved schema:", SCHEMA_PATH)
    print("Saved feature importance:", EXPLAINABILITY_PATH)
    print("Saved scored applicants:", scored_path)
    print("Saved threshold analysis:", threshold_path)


# This lets us run the file directly with: python src/train_model.py
if __name__ == "__main__":
    main()
