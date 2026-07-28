import os
import pandas as pd



# ==========================================================
# LOAD DATASET
# ==========================================================

def load_dataset(
    filename,
    folder="raw"
):
    """
    Load CSV dataset from datasets folder.
    """



    current_dir = os.path.dirname(
        os.path.abspath(__file__)
    )



    path = os.path.join(

        current_dir,

        "..",

        "datasets",

        folder,

        filename

    )



    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )



    return pd.read_csv(path)





# ==========================================================
# DATASET INFORMATION
# ==========================================================

def dataset_info(df):


    print("="*60)
    print("DATASET INFORMATION")
    print("="*60)



    print(
        "\nShape:",
        df.shape
    )


    print(
        "\nColumns:"
    )

    print(
        list(df.columns)
    )


    print(
        "\nData Types:"
    )

    print(
        df.dtypes
    )


    print(
        "\nMissing Values:"
    )

    print(
        df.isnull().sum()
    )


    print(
        "\nDuplicate Rows:",
        df.duplicated().sum()
    )


    print(
        "\nStatistics:"
    )

    print(
        df.describe(
            include="all"
        )
    )





# ==========================================================
# PRINT SHAPE
# ==========================================================

def print_shape(df):

    print(
        "Rows:",
        df.shape[0]
    )

    print(
        "Columns:",
        df.shape[1]
    )





# ==========================================================
# SAVE DATASET
# ==========================================================

def save_dataset(
    df,
    filename
):


    current_dir = os.path.dirname(
        os.path.abspath(__file__)
    )


    output_dir = os.path.join(

        current_dir,

        "..",

        "datasets",

        "processed"

    )


    os.makedirs(
        output_dir,
        exist_ok=True
    )



    path = os.path.join(
        output_dir,
        filename
    )


    df.to_csv(
        path,
        index=False
    )


    print(
        f"Dataset saved: {path}"
    )