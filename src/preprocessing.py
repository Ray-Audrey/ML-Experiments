import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


# ==========================================================
# REMOVE DUPLICATES
# ==========================================================

def remove_duplicates(df):
    """
    Remove duplicate rows.
    """

    return df.drop_duplicates()


# ==========================================================
# DROP UNWANTED COLUMNS
# ==========================================================

def drop_columns(df, columns=None):
    """
    Drop unwanted columns if provided.
    """

    if columns is None:
        return df

    return df.drop(columns=columns)


# ==========================================================
# HANDLE MISSING VALUES
# ==========================================================

def handle_missing_values(df):
    """
    Fill missing values.

    Numeric Columns      -> Mean

    Categorical Columns  -> Mode
    """

    df = df.copy()

    for column in df.columns:

        if pd.api.types.is_numeric_dtype(df[column]):

            df[column] = df[column].fillna(
                df[column].mean()
            )

        else:

            df[column] = df[column].fillna(
                df[column].mode()[0]
            )

    return df


# ==========================================================
# ENCODE CATEGORICAL COLUMNS
# ==========================================================

def encode_categorical(df):
    """
    Encode only categorical columns.
    """

    df = df.copy()

    encoders = {}

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns

    for column in categorical_columns:

        encoder = LabelEncoder()

        df[column] = encoder.fit_transform(df[column])

        encoders[column] = encoder

    return df, encoders


# ==========================================================
# SPLIT FEATURES AND TARGET
# ==========================================================

def split_features_target(df, target_column):
    """
    Split dataset into X and y.
    """

    X = df.drop(columns=[target_column])

    y = df[target_column]

    return X, y


# ==========================================================
# SCALE FEATURES
# ==========================================================

def scale_features(X):
    """
    Standardize feature columns.
    """

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler


# ==========================================================
# COMPLETE PREPROCESSING PIPELINE
# ==========================================================

def preprocess_data(
    df,
    target_column,
    drop_cols=None,
    encode=True,
    scale=True,
    stratify=True,
    test_size=0.2,
    random_state=42
):
    """
    Complete preprocessing pipeline.
    """

    # ------------------------------
    # Remove duplicates
    # ------------------------------

    df = remove_duplicates(df)

    # ------------------------------
    # Drop unwanted columns
    # ------------------------------

    df = drop_columns(df, drop_cols)

    # ------------------------------
    # Handle missing values
    # ------------------------------

    df = handle_missing_values(df)

    # ------------------------------
    # Encode categorical columns
    # ------------------------------

    encoders = {}

    if encode:

        df, encoders = encode_categorical(df)

    # ------------------------------
    # Split X and y
    # ------------------------------

    X, y = split_features_target(
        df,
        target_column
    )

    # ------------------------------
    # Scale Features
    # ------------------------------

    scaler = None

    if scale:

        X, scaler = scale_features(X)

    # ------------------------------
    # Train Test Split
    # ------------------------------

    if stratify:

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

    else:

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state
        )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        encoders
    )

