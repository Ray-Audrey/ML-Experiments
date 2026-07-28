from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)

from sklearn.model_selection import GridSearchCV

import pandas as pd
import numpy as np



# ==========================================================
# TRAIN REGRESSION MODEL
# ==========================================================

def train_regression(
    X_train,
    y_train,
    algorithm="linear_regression",
    alpha=1.0,
    l1_ratio=0.5,
    max_iter=10000
):

    algorithm = algorithm.lower()


    if algorithm == "linear_regression":

        model = LinearRegression()



    elif algorithm == "ridge":

        model = Ridge(
            alpha=alpha
        )



    elif algorithm == "lasso":

        model = Lasso(
            alpha=alpha,
            max_iter=max_iter
        )



    elif algorithm == "elastic_net":

        model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=max_iter
        )



    else:

        raise ValueError(
            f"Unsupported model: {algorithm}"
        )



    model.fit(
        X_train,
        y_train
    )


    return model




# ==========================================================
# HYPERPARAMETER TUNING
# ==========================================================

def tune_regression_model(
    X_train,
    y_train,
    algorithm="ridge",
    cv=5
):

    algorithm = algorithm.lower()



    if algorithm == "ridge":

        model = Ridge()


        params = {

            "alpha":
            [
                0.01,
                0.1,
                1,
                10,
                100
            ]

        }



    elif algorithm == "lasso":

        model = Lasso(
            max_iter=10000
        )


        params = {

            "alpha":
            [
                0.001,
                0.01,
                0.1,
                1,
                10
            ]

        }



    elif algorithm == "elastic_net":

        model = ElasticNet(
            max_iter=10000
        )


        params = {

            "alpha":
            [
                0.01,
                0.1,
                1,
                10
            ],

            "l1_ratio":
            [
                0.2,
                0.5,
                0.8
            ]

        }



    else:

        raise ValueError(
            "Only Ridge, Lasso and ElasticNet support tuning"
        )



    grid = GridSearchCV(

        estimator=model,

        param_grid=params,

        cv=cv,

        scoring="r2",

        n_jobs=-1,

        return_train_score=True

    )


    grid.fit(
        X_train,
        y_train
    )


    results = pd.DataFrame(
        grid.cv_results_
    )


    return (

        grid.best_estimator_,

        grid.best_params_,

        grid.best_score_,

        results

    )





# ==========================================================
# PREDICT USING MODEL
# ==========================================================

def predict_model(
    model,
    X
):

    return model.predict(X)





# ==========================================================
# SINGLE MODEL COEFFICIENTS
# ==========================================================

def get_coefficients(
    model,
    feature_names
):

    coefficients = model.coef_


    if hasattr(
        coefficients,
        "toarray"
    ):

        coefficients = coefficients.toarray().flatten()



    return pd.DataFrame({

        "Feature":
        feature_names,

        "Coefficient":
        coefficients

    })





# ==========================================================
# COEFFICIENT COMPARISON TABLE
# ==========================================================

def compare_coefficients(
    models,
    feature_names
):


    dataframe = pd.DataFrame({

        "Feature":
        feature_names

    })


    for name, model in models.items():

        coef = model.coef_


        if hasattr(
            coef,
            "toarray"
        ):

            coef = coef.toarray().flatten()


        dataframe[name] = coef



    return dataframe





# ==========================================================
# TRAINING VS VALIDATION ERROR
# ==========================================================

def get_training_validation_error(
    cv_results,
    parameter="param_alpha"
):


    dataframe = cv_results.copy()



    output = dataframe[

        [
            parameter,
            "mean_train_score",
            "mean_test_score"
        ]

    ]



    output.columns = [

        "Parameter",
        "Training Score",
        "Validation Score"

    ]



    return output





# ==========================================================
# HYPERPARAMETER SUMMARY TABLE
# ==========================================================

def hyperparameter_summary(
    best_params,
    best_score,
    model_name
):


    return pd.DataFrame({

        "Model":
        [
            model_name
        ],

        "Best Parameters":
        [
            str(best_params)
        ],

        "Best CV R2":
        [
            best_score
        ]

    })