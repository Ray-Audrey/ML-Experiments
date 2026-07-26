src/regression.py
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


# ==========================================================
# TRAIN REGRESSION MODEL
# ==========================================================

def train_regression(
    X_train,
    y_train,
    algorithm="linear_regression",
    random_state=42
):
    """
    Train a regression model.

    Supported Algorithms
    --------------------
    linear_regression
    decision_tree
    random_forest
    svr
    """

    models = {

        "linear_regression":
            LinearRegression(),

        "decision_tree":
            DecisionTreeRegressor(
                random_state=random_state
            ),

        "random_forest":
            RandomForestRegressor(
                random_state=random_state
            ),

        "svr":
            SVR()

    }

    if algorithm not in models:

        raise ValueError(
            f"Unsupported Algorithm : {algorithm}"
        )

    model = models[algorithm]

    model.fit(
        X_train,
        y_train
    )

    return model
