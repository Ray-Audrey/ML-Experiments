from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

import numpy as np
import pandas as pd


# ==========================================================
# SINGLE MODEL CROSS VALIDATION
# ==========================================================

def perform_cross_validation(
    model,
    X,
    y,
    folds=5,
    scoring="accuracy"
):
    """
    Perform K-Fold Cross Validation.
    """

    kfold = KFold(
        n_splits=folds,
        shuffle=True,
        random_state=42
    )


    scores = cross_val_score(
        model,
        X,
        y,
        cv=kfold,
        scoring=scoring
    )


    print("=" * 50)
    print("CROSS VALIDATION")
    print("=" * 50)


    for i, score in enumerate(scores):

        print(
            f"Fold {i+1}: {score:.4f}"
        )


    print()

    print(
        "Average :",
        round(np.mean(scores),4)
    )

    print(
        "Std Dev :",
        round(np.std(scores),4)
    )


    return scores



# ==========================================================
# MULTI MODEL CROSS VALIDATION COMPARISON
# ==========================================================

def compare_cross_validation(
    models,
    X,
    y,
    folds=5,
    scoring="accuracy"
):
    """
    Compare multiple classifiers using K-Fold CV.

    Parameters
    ----------
    models : dictionary

        Example:
        {
        "Gaussian NB": gaussian_nb,
        "Best KNN": knn
        }

    Returns
    -------
    DataFrame

        Fold-wise accuracy comparison
    """


    kfold = KFold(
        n_splits=folds,
        shuffle=True,
        random_state=42
    )


    results = {}


    for name, model in models.items():


        scores = cross_val_score(
            model,
            X,
            y,
            cv=kfold,
            scoring=scoring
        )


        results[name] = scores



    dataframe = pd.DataFrame(
        results
    )


    dataframe.index = [
        f"Fold {i+1}"
        for i in range(folds)
    ]


    dataframe.loc["Average"] = (
        dataframe.mean()
    )


    print("=" * 50)
    print("MODEL CROSS VALIDATION COMPARISON")
    print("=" * 50)


    print(dataframe)


    return dataframe