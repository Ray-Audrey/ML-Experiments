from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def grid_search(
    model,
    parameters,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy"
):
    """
    Perform GridSearchCV.
    """

    search = GridSearchCV(
        estimator=model,
        param_grid=parameters,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    print("=" * 50)
    print("GRID SEARCH RESULTS")
    print("=" * 50)
    print("Best Parameters :", search.best_params_)
    print("Best Score      :", round(search.best_score_, 4))

    return search


def randomized_search(
    model,
    parameters,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy",
    n_iter=20,
    random_state=42
):
    """
    Perform RandomizedSearchCV.
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

    search.fit(X_train, y_train)

    print("=" * 50)
    print("RANDOMIZED SEARCH RESULTS")
    print("=" * 50)
    print("Best Parameters :", search.best_params_)
    print("Best Score      :", round(search.best_score_, 4))

    return search