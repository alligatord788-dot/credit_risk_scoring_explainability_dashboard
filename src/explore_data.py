from project_utils import find_dataset_path, read_credit_csv


def main():
    dataset_path = find_dataset_path()
    df = read_credit_csv(dataset_path)

    print("Dataset path:", dataset_path)
    print("Shape:", df.shape)
    print("\nFirst five rows:")
    print(df.head())

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isna().sum().sort_values(ascending=False).head(20))

    print("\nData types:")
    print(df.dtypes)


if __name__ == "__main__":
    main()
