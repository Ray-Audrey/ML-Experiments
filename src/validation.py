from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

import numpy as np


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

        print(f"Fold {i+1}: {score:.4f}")

    print()

    print("Average :", round(np.mean(scores),4))
    print("Std Dev :", round(np.std(scores),4))

    return scores