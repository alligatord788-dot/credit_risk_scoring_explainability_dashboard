# Credit Risk Scoring & Explainability Dashboard

## Project Goal

This project upgrades a simple loan default model into a finance-style credit-risk decision system.

Instead of only predicting `default` or `non-default`, it produces:

- default probability
- credit risk score
- risk band
- approval decision
- threshold tradeoff analysis
- feature importance / risk drivers
- dashboard for business interpretation

## Why This Is More Advanced Than Basic Loan Prediction

Basic loan default project:

```text
borrower data -> model -> default / non-default
```

This project:

```text
borrower data -> default probability -> risk score -> risk band -> credit decision -> explanation
```

That makes it closer to how data science is used in banking and fintech.

## Data

Use a public LendingClub / loan default dataset.

Put the CSV here:

```text
data/raw/
```

For convenience, this project also detects the dataset from the previous folder:

```text
../loan_default_risk_prediction/data/raw/
```

The scripts load the first `100,000` rows by default so that training is fast and easy to explain.

## Current Result On Your Dataset

- Dataset used: LendingClub accepted loans CSV
- Rows loaded: `100,000`
- Cleaned resolved loans: `88,399`
- Default rate: `20.49%`
- Best model: `Logistic Regression`
- ROC-AUC: `0.7398`
- Recall: `0.6739`
- Sample applicant output: `37.29%` default probability, risk score `645`, `Medium Risk`, `Manual Review`

## Pipeline

### 1. Explore Data

```bash
python src/explore_data.py
```

### 2. Clean Data

```bash
python src/data_cleaning.py
```

This step:

- detects the target column
- converts loan status into binary default target
- removes unresolved/current loans
- removes leakage columns such as payment, recovery and settlement fields
- keeps clean borrower-level features

### 3. Train Scoring Model

```bash
python src/train_model.py
```

Models:

- Logistic Regression
- Decision Tree
- Random Forest

Outputs:

- `model.pkl`
- `metrics.json`
- `input_schema.json`
- `feature_importance.csv`
- `threshold_analysis.csv`
- `scored_test_applicants.csv`

### 4. Predict One Applicant

```bash
python src/predict.py
```

Output example:

```text
Default probability: 0.3729
Risk score: 645
Risk band: Medium Risk
Credit decision: Manual Review
```

### 5. Run Dashboard

```bash
python -m streamlit run dashboard_app.py
```

### 6. Run the dashboard on PowerShell

```bash
cd E:\Internship\projects\credit_risk_scoring_explainability_dashboard
..\.venv\Scripts\python.exe -m streamlit run dashboard_app.py --server.port 8502
```

## Risk Band Logic

```text
Low Risk: probability < 0.20
Medium Risk: 0.20 to 0.40
High Risk: 0.40 to 0.60
Very High Risk: probability >= 0.60
```

## Decision Logic

```text
Approve: probability < 0.25
Manual Review: 0.25 to 0.50
Reject: probability >= 0.50
```

## Interview Explanation

I built a credit-risk scoring system, not just a binary classifier. The model first predicts probability of default. Then I convert that probability into a risk score, risk band and lending decision. I also added threshold analysis to show the tradeoff between catching risky borrowers and wrongly flagging safe borrowers. Finally, I used feature importance to explain which borrower attributes influence risk.

## Folder Structure

```text
credit_risk_scoring_explainability_dashboard/
  data/
    raw/
    processed/
  src/
    explore_data.py
    data_cleaning.py
    train_model.py
    predict.py
    project_utils.py
  notebooks/
    kaggle_credit_risk_scoring_explainability.py
  artifacts/
  dashboard_app.py
  README.md
```
