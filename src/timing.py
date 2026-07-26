import time


def measure_time(
    model,
    X_train,
    y_train,
    X_test
):
    """
    Measure training and prediction time.
    """

    start = time.perf_counter()

    model.fit(X_train, y_train)

    training_time = time.perf_counter() - start

    start = time.perf_counter()

    predictions = model.predict(X_test)

    prediction_time = time.perf_counter() - start

    print("=" * 50)
    print("TIME ANALYSIS")
    print("=" * 50)

    print(f"Training Time : {training_time:.6f} sec")
    print(f"Prediction Time : {prediction_time:.6f} sec")

    return predictions, training_time, prediction_time