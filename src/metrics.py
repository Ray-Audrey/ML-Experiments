import pandas as pd
import time


from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)



# ==========================================================
# REGRESSION METRICS
# ==========================================================

def regression_metrics(
    model,
    X_test,
    y_test,
    model_name="Model"
):


    predictions = model.predict(
        X_test
    )


    mae = mean_absolute_error(
        y_test,
        predictions
    )


    mse = mean_squared_error(
        y_test,
        predictions
    )


    rmse = mse ** 0.5


    r2 = r2_score(
        y_test,
        predictions
    )


    return {


        "Model":
        model_name,


        "MAE":
        mae,


        "MSE":
        mse,


        "RMSE":
        rmse,


        "R2 Score":
        r2,


        "Predictions":
        predictions,


        "Actual":
        y_test

    }





# ==========================================================
# COMPARE REGRESSION MODELS
# ==========================================================

def compare_regression_models(
    models,
    X_test,
    y_test
):


    results=[]



    for name, model in models.items():


        predictions=model.predict(
            X_test
        )


        results.append({

            "Model":
            name,


            "MAE":
            mean_absolute_error(
                y_test,
                predictions
            ),


            "MSE":
            mean_squared_error(
                y_test,
                predictions
            ),


            "RMSE":
            mean_squared_error(
                y_test,
                predictions
            )**0.5,


            "R2 Score":
            r2_score(
                y_test,
                predictions
            )

        })


    dataframe=pd.DataFrame(
        results
    )


    return dataframe