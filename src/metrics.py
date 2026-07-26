import os
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


# ==========================================================
# CLASSIFICATION METRICS
# ==========================================================

def classification_metrics(
    model,
    X_test,
    y_test,
    average="weighted",
    show_plots=True,
    save_plots=True,
    save_dir="../figures",
    model_name="model"
):

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 15

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average=average,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average=average,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average=average,
        zero_division=0
    )

    print("\n========== Classification Metrics ==========\n")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    cm = confusion_matrix(
        y_test,
        predictions
    )

    print("\nConfusion Matrix\n")
    print(cm)

    print("\nClassification Report\n")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    # ------------------------------------------------------
    # ROC AUC
    # ------------------------------------------------------

    try:

        probabilities = model.predict_proba(X_test)

        if probabilities.shape[1] == 2:

            roc_auc = roc_auc_score(
                y_test,
                probabilities[:, 1]
            )

        else:

            roc_auc = roc_auc_score(
                y_test,
                probabilities,
                multi_class="ovr"
            )

        print(f"ROC AUC : {roc_auc:.4f}")

    except:

        print("ROC AUC : Not Available")

    if save_plots:
        os.makedirs(save_dir, exist_ok=True)

    # ------------------------------------------------------
    # Confusion Matrix
    # ------------------------------------------------------

    if show_plots:

        fig, ax = plt.subplots(figsize=(6, 5))

        ConfusionMatrixDisplay.from_predictions(
            y_test,
            predictions,
            cmap="Blues",
            ax=ax
        )

        ax.set_title(
        f"{model_name.replace('_',' ').title()} Confusion Matrix",
        fontweight="bold"
    )

        if save_plots:

            fig.savefig(
                os.path.join(
                    save_dir,
                    f"{model_name}_confusion_matrix.png"
                ),
                dpi=600,
                bbox_inches="tight"
            )

        plt.show()
        plt.close(fig)

   # ------------------------------------------------------
    # ROC Curve
    # ------------------------------------------------------

    try:

        fig, ax = plt.subplots(figsize=(6,5))

        RocCurveDisplay.from_estimator(
            model,
            X_test,
            y_test,
            ax=ax
        )

        ax.set_title(
            f"{model_name.replace('_',' ').title()} ROC Curve",
            fontweight="bold"
        )

        if save_plots:

            fig.savefig(
                os.path.join(
                    save_dir,
                    f"{model_name}_roc_curve.png"
                ),
                dpi=600,
                bbox_inches="tight"
            )

        plt.show()
        plt.close(fig)

    except:
        pass

    # ------------------------------------------------------
    # Precision Recall Curve
    # ------------------------------------------------------

    try:

        fig, ax = plt.subplots(figsize=(6,5))

        PrecisionRecallDisplay.from_estimator(
            model,
            X_test,
            y_test,
            ax=ax
        )

        ax.set_title(
            f"{model_name.replace('_',' ').title()} Precision-Recall Curve",
            fontweight="bold"
        )

        if save_plots:

            fig.savefig(
                os.path.join(
                    save_dir,
                    f"{model_name}_precision_recall_curve.png"
                ),
                dpi=600,
                bbox_inches="tight"
            )

        plt.show()
        plt.close(fig)

    except:
        pass

    return predictions


# ==========================================================
# REGRESSION METRICS
# ==========================================================

def regression_metrics(
    model,
    X_test,
    y_test,
    show_plots=True,
    save_plots=True,
    save_dir="../figures",
    model_name="regression_model"
):

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 15

    predictions = model.predict(X_test)

    mse = mean_squared_error(
        y_test,
        predictions
    )

    rmse = mse ** 0.5

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    print("\n========== Regression Metrics ==========\n")

    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"R²   : {r2:.4f}")

    if save_plots:
        os.makedirs(save_dir, exist_ok=True)

    # ------------------------------------------------------
    # Actual vs Predicted Plot
    # ------------------------------------------------------

    if show_plots:

        fig, ax = plt.subplots(figsize=(6,5))

        ax.scatter(
            y_test,
            predictions,
            edgecolors="black"
        )

        min_val = min(min(y_test), min(predictions))
        max_val = max(max(y_test), max(predictions))

        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            color="red",
            linestyle="--"
        )

        ax.set_title(
            "Actual vs Predicted",
            fontweight="bold"
        )

        ax.set_xlabel(
            "Actual Values",
            fontweight="bold"
        )

        ax.set_ylabel(
            "Predicted Values",
            fontweight="bold"
        )

        if save_plots:

            fig.savefig(
                os.path.join(
                    save_dir,
                    f"{model_name}_actual_vs_predicted.png"
                ),
                dpi=600,
                bbox_inches="tight"
            )

        plt.show()
        plt.close(fig)

    # ------------------------------------------------------
    # Residual Plot
    # ------------------------------------------------------

    if show_plots:

        residuals = y_test - predictions

        fig, ax = plt.subplots(figsize=(6,5))

        ax.scatter(
            predictions,
            residuals,
            edgecolors="black"
        )

        ax.axhline(
            y=0,
            color="red",
            linestyle="--"
        )

        ax.set_title(
            "Residual Plot",
            fontweight="bold"
        )

        ax.set_xlabel(
            "Predicted Values",
            fontweight="bold"
        )

        ax.set_ylabel(
            "Residuals",
            fontweight="bold"
        )

        if save_plots:

            fig.savefig(
                os.path.join(
                    save_dir,
                    f"{model_name}_residual_plot.png"
                ),
                dpi=600,
                bbox_inches="tight"
            )

        plt.show()
        plt.close(fig)

    return {
    "predictions": predictions,
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "roc_auc": roc_auc
    }