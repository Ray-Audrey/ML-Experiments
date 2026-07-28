from sklearn.model_selection import (
    KFold,
    cross_val_score,
    cross_validate,
    GridSearchCV,
    RandomizedSearchCV
)

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
    scoring="r2"
):

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


    print("="*50)
    print("CROSS VALIDATION RESULTS")
    print("="*50)


    for i, score in enumerate(scores):

        print(
            f"Fold {i+1}: {score:.4f}"
        )


    print()

    print(
        "Average R2:",
        round(
            np.mean(scores),
            4
        )
    )


    print(
        "Std Dev:",
        round(
            np.std(scores),
            4
        )
    )


    return scores





# ==========================================================
# MULTIPLE REGRESSION MODEL CROSS VALIDATION
# ==========================================================

def compare_cross_validation(
    models,
    X,
    y,
    folds=5
):

    kfold = KFold(
        n_splits=folds,
        shuffle=True,
        random_state=42
    )


    results = []



    for name, model in models.items():


        scores = cross_validate(

            model,

            X,

            y,

            cv=kfold,

            scoring={

                "mae":
                "neg_mean_absolute_error",

                "mse":
                "neg_mean_squared_error",

                "rmse":
                "neg_root_mean_squared_error",

                "r2":
                "r2"

            },

            return_train_score=True

        )



        results.append({

            "Model":
            name,


            "MAE":
            -scores["test_mae"].mean(),


            "MSE":
            -scores["test_mse"].mean(),


            "RMSE":
            -scores["test_rmse"].mean(),


            "R2 Score":
            scores["test_r2"].mean(),


            "Training Error":
            -scores["train_mse"].mean(),


            "Validation Error":
            -scores["test_mse"].mean()

        })



    dataframe = pd.DataFrame(results)


    print("="*60)
    print("5-FOLD CROSS VALIDATION PERFORMANCE")
    print("="*60)

    print(dataframe)


    return dataframe





# ==========================================================
# GRID SEARCH
# ==========================================================

def perform_grid_search(
    model,
    param_grid,
    X,
    y,
    folds=5,
    scoring="r2"
):


    cv = KFold(

        n_splits=folds,

        shuffle=True,

        random_state=42

    )


    grid = GridSearchCV(

        estimator=model,

        param_grid=param_grid,

        cv=cv,

        scoring=scoring,

        n_jobs=-1,

        return_train_score=True

    )


    grid.fit(
        X,
        y
    )


    print("="*50)
    print("GRID SEARCH RESULTS")
    print("="*50)

    print(
        "Best Parameters:",
        grid.best_params_
    )

    print(
        "Best CV R2:",
        round(
            grid.best_score_,
            4
        )
    )


    return grid





# ==========================================================
# RANDOM SEARCH
# ==========================================================

def perform_random_search(
    model,
    param_distributions,
    X,
    y,
    folds=5,
    scoring="r2",
    n_iter=20
):


    cv = KFold(

        n_splits=folds,

        shuffle=True,

        random_state=42

    )


    search = RandomizedSearchCV(

        estimator=model,

        param_distributions=param_distributions,

        cv=cv,

        scoring=scoring,

        n_iter=n_iter,

        n_jobs=-1,

        random_state=42,

        return_train_score=True

    )


    search.fit(
        X,
        y
    )


    print("="*50)
    print("RANDOM SEARCH RESULTS")
    print("="*50)

    print(
        "Best Parameters:",
        search.best_params_
    )

    print(
        "Best CV R2:",
        round(
            search.best_score_,
            4
        )
    )


    return search