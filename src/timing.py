import time
import pandas as pd

from sklearn.base import clone



# ==========================================================
# MEASURE TRAINING AND PREDICTION TIME
# ==========================================================

def measure_time(
    model,
    X_train,
    y_train,
    X_test
):
    """
    Measure training and prediction time.

    Returns
    -------
    predictions
    training_time
    prediction_time
    """



    # create independent copy

    model = clone(model)



    # -------------------------------
    # Training Time
    # -------------------------------

    start = time.perf_counter()


    model.fit(
        X_train,
        y_train
    )


    training_time = (
        time.perf_counter()
        -
        start
    )



    # -------------------------------
    # Prediction Time
    # -------------------------------

    start = time.perf_counter()


    predictions = model.predict(
        X_test
    )


    prediction_time = (
        time.perf_counter()
        -
        start
    )



    return (

        predictions,

        training_time,

        prediction_time

    )





# ==========================================================
# COMPARE MODEL EXECUTION TIME
# ==========================================================

def compare_training_prediction_time(
    models,
    X_train,
    y_train,
    X_test,
    sort_by="Training Time"
):
    """
    Compare execution time of regression models.

    Models:
    - Linear Regression
    - Ridge
    - Lasso
    - Elastic Net
    """



    results = []



    for name, model in models.items():


        _, train_time, pred_time = measure_time(

            model,

            X_train,

            y_train,

            X_test

        )



        results.append({

            "Model":
            name,


            "Training Time":
            train_time,


            "Prediction Time":
            pred_time

        })



    dataframe = pd.DataFrame(
        results
    )



    if sort_by in dataframe.columns:

        dataframe = dataframe.sort_values(
            by=sort_by
        ).reset_index(
            drop=True
        )



    print("="*60)
    print("TRAINING AND PREDICTION TIME COMPARISON")
    print("="*60)

    print(dataframe)



    return dataframe