import pandas as pd

from project_utils import (
    MODEL_PATH,
    SCHEMA_PATH,
    load_json,
    load_pickle,
    probability_to_decision,
    probability_to_risk_band,
    probability_to_score,
)


def build_sample_input(schema):
    sample = {}

    for column in schema["numeric_columns"]:
        sample[column] = schema["numeric_defaults"][column]

    for column in schema["categorical_columns"]:
        options = schema["categorical_options"].get(column, [])
        sample[column] = options[0] if options else "Unknown"

    return sample


def main():
    model = load_pickle(MODEL_PATH)
    schema = load_json(SCHEMA_PATH)

    sample = build_sample_input(schema)
    input_df = pd.DataFrame([sample], columns=schema["feature_columns"])

    probability = float(model.predict_proba(input_df)[0][1])
    risk_score = probability_to_score(probability)
    risk_band = probability_to_risk_band(probability)
    decision = probability_to_decision(probability)

    print("Sample applicant:")
    for key, value in sample.items():
        print(f"{key}: {value}")

    print("\nCredit risk output:")
    print("Default probability:", round(probability, 4))
    print("Risk score:", risk_score)
    print("Risk band:", risk_band)
    print("Credit decision:", decision)


if __name__ == "__main__":
    main()
