import time


def measure_time(
    model,
    X_train,
    y_train,
    X_test
):

    start=time.perf_counter()

    model.fit(
        X_train,
        y_train
    )

    training_time=time.perf_counter()-start


    start=time.perf_counter()

    predictions=model.predict(
        X_test
    )

    prediction_time=time.perf_counter()-start


    return (
        predictions,
        training_time,
        prediction_time
    )



def compare_training_prediction_time(
    models,
    X_train,
    y_train,
    X_test
):

    results=[]


    for name,model in models.items():

        _,train_time,pred_time = measure_time(
            model,
            X_train,
            y_train,
            X_test
        )


        results.append(
            {
            "Algorithm":name,
            "Training Time":train_time,
            "Prediction Time":pred_time
            }
        )


    import pandas as pd

    df=pd.DataFrame(results)

    print(df)

    return df