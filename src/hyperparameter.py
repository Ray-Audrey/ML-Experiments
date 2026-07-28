#hyperparameter.py
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV
)



# ==========================================================
# GRID SEARCH
# ==========================================================

def grid_search(
    model,
    parameters,
    X_train,
    y_train,
    cv=5,
    scoring="r2"
):
    """
    Generic Grid Search.

    Default scoring:
    Regression -> r2

    Classification:
    pass scoring="accuracy"
    """



    search = GridSearchCV(
        estimator=model,
        param_grid=parameters,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )



    search.fit(
        X_train,
        y_train
    )



    print("=" * 50)
    print("GRID SEARCH RESULTS")
    print("=" * 50)

    print(
        "Best Parameters:",
        search.best_params_
    )

    print(
        "Best Score:",
        round(search.best_score_,4)
    )


    return search





# ==========================================================
# RANDOMIZED SEARCH
# ==========================================================

def randomized_search(
    model,
    parameters,
    X_train,
    y_train,
    cv=5,
    scoring="r2",
    n_iter=20,
    random_state=42
):
    """
    Generic Randomized Search.
    """



    search = RandomizedSearchCV(

        estimator=model,

        param_distributions=parameters,

        n_iter=n_iter,

        cv=cv,

        scoring=scoring,

        random_state=random_state,

        n_jobs=-1

    )



    search.fit(
        X_train,
        y_train
    )



    print("=" * 50)
    print("RANDOMIZED SEARCH RESULTS")
    print("=" * 50)

    print(
        "Best Parameters:",
        search.best_params_
    )

    print(
        "Best Score:",
        round(search.best_score_,4)
    )


    return search