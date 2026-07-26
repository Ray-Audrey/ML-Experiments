import os
import pandas as pd


def load_dataset(filename):
    """
    Load a CSV dataset from datasets/raw.
    """

    current_dir = os.path.dirname(os.path.abspath(__file__))

    dataset_path = os.path.join(
        current_dir,
        "..",
        "datasets",
        "raw",
        filename
    )

    return pd.read_csv(dataset_path)


def dataset_info(df):
    """
    Display basic information about the dataset.
    """

    print("=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)

    print(f"\nShape : {df.shape}")

    print("\nColumns")
    print(df.columns.tolist())

    print("\nData Types")
    print(df.dtypes)

    print("\nMissing Values")
    print(df.isnull().sum())

    print("\nDuplicate Rows")
    print(df.duplicated().sum())

    print("\nSummary Statistics")
    print(df.describe(include="all"))


def print_shape(df):
    """
    Print the dataset shape.
    """

    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")


def save_dataset(df, filename):
    """
    Save dataframe to datasets/processed.
    """

    current_dir = os.path.dirname(os.path.abspath(__file__))

    output_dir = os.path.join(
        current_dir,
        "..",
        "datasets",
        "processed"
    )

    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, filename)

    df.to_csv(output_path, index=False)

    print(f"Dataset saved to:\n{output_path}")
