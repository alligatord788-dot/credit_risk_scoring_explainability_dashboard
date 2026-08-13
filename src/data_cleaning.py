from project_utils import CLEAN_DATA_PATH, clean_raw_data, find_dataset_path, read_credit_csv


def main():
    dataset_path = find_dataset_path()
    raw_df = read_credit_csv(dataset_path)
    cleaned_df, target_column, columns_to_drop = clean_raw_data(raw_df)

    CLEAN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(CLEAN_DATA_PATH, index=False)

    print("Dataset path:", dataset_path)
    print("Target column:", target_column)
    print("Cleaned shape:", cleaned_df.shape)
    print("\nDefault distribution:")
    print(cleaned_df["default"].value_counts(normalize=True).round(4))
    print("\nColumns dropped due to leakage/id/constant values:")
    print(columns_to_drop)
    print("\nSaved cleaned data to:", CLEAN_DATA_PATH)


if __name__ == "__main__":
    main()
