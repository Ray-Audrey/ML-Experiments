#visualization.py
import os
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc
)


FONT = "Times New Roman"
SIZE = 15



# ==========================================================
# COMMON FORMAT
# ==========================================================

def format_plot(
    title,
    xlabel,
    ylabel
):

    plt.title(
        title,
        fontsize=SIZE,
        fontname=FONT,
        fontweight="bold"
    )

    plt.xlabel(
        xlabel,
        fontsize=SIZE,
        fontname=FONT,
        fontweight="bold"
    )

    plt.ylabel(
        ylabel,
        fontsize=SIZE,
        fontname=FONT,
        fontweight="bold"
    )



# ==========================================================
# TARGET DISTRIBUTION
# ==========================================================

def plot_target_distribution(
    target,
    target_name="Loan Sanction Amount",
    save_dir="../figures"
):

    os.makedirs(
        save_dir,
        exist_ok=True
    )


    plt.figure(
        figsize=(7,5)
    )


    plt.hist(
        target,
        bins=30,
        edgecolor="black"
    )


    format_plot(
        "Target Variable Distribution",
        target_name,
        "Frequency"
    )


    plt.grid(True)


    plt.tight_layout()


    plt.savefig(
        os.path.join(
            save_dir,
            "target_distribution.png"
        ),
        dpi=300
    )


    plt.show()



# ==========================================================
# FEATURE VS TARGET
# ==========================================================

def plot_feature_vs_target(
    dataframe,
    feature,
    target
):

    plt.figure(
        figsize=(7,5)
    )


    plt.scatter(
        dataframe[feature],
        dataframe[target],
        s=20
    )


    format_plot(
        f"{feature} vs {target}",
        feature,
        target
    )


    plt.grid(True)


    plt.tight_layout()


    plt.show()



# ==========================================================
# ACTUAL VS PREDICTED
# ==========================================================

def plot_actual_vs_predicted(
    actual,
    predicted,
    model_name="model",
    save_dir="../figures"
):

    os.makedirs(
        save_dir,
        exist_ok=True
    )


    plt.figure(
        figsize=(7,5)
    )


    plt.scatter(
        actual,
        predicted
    )


    minimum = min(
        actual.min(),
        predicted.min()
    )


    maximum = max(
        actual.max(),
        predicted.max()
    )


    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--"
    )


    format_plot(
        "Actual vs Predicted",
        "Actual Values",
        "Predicted Values"
    )


    plt.grid(True)


    plt.tight_layout()


    plt.savefig(
        os.path.join(
            save_dir,
            f"{model_name}_actual_vs_predicted.png"
        ),
        dpi=300
    )


    plt.show()



# ==========================================================
# RESIDUAL PLOT
# ==========================================================

def plot_residuals(
    actual,
    predicted,
    model_name="model",
    save_dir="../figures"
):

    os.makedirs(
        save_dir,
        exist_ok=True
    )


    residuals = actual - predicted


    plt.figure(
        figsize=(7,5)
    )


    plt.scatter(
        predicted,
        residuals
    )


    plt.axhline(
        0,
        linestyle="--"
    )


    format_plot(
        "Residual Plot",
        "Predicted Values",
        "Residuals"
    )


    plt.grid(True)


    plt.tight_layout()


    plt.savefig(
        os.path.join(
            save_dir,
            f"{model_name}_residual_plot.png"
        ),
        dpi=300
    )


    plt.show()


# ==========================================================
# CROSS VALIDATION SCORES
# ==========================================================

def plot_regression_cv_scores(
    scores,
    save_dir="../figures"
):

    os.makedirs(
        save_dir,
        exist_ok=True
    )


    plt.figure(
        figsize=(7,5)
    )


    plt.plot(
        range(1,len(scores)+1),
        scores,
        marker="o"
    )


    format_plot(
        "5 Fold Cross Validation R2 Score",
        "Fold",
        "R2 Score"
    )


    plt.grid(True)


    plt.tight_layout()


    plt.savefig(
        os.path.join(
            save_dir,
            "cross_validation_scores.png"
        ),
        dpi=300
    )


    plt.show()



# ==========================================================
# TRAINING VS VALIDATION ERROR
# ==========================================================

def plot_training_validation_error(
    parameters,
    train_error,
    validation_error,
    xlabel="Alpha",
    save_dir="../figures"
):

    os.makedirs(
        save_dir,
        exist_ok=True
    )


    plt.figure(
        figsize=(7,5)
    )


    plt.plot(
        parameters,
        train_error,
        marker="o",
        label="Training Error"
    )


    plt.plot(
        parameters,
        validation_error,
        marker="s",
        label="Validation Error"
    )


    plt.xscale(
        "log"
    )


    format_plot(
        "Training vs Validation Error",
        xlabel,
        "Mean Squared Error"
    )


    plt.legend(
        prop={
            "family":FONT,
            "size":SIZE
        }
    )


    plt.grid(True)


    plt.tight_layout()


    plt.savefig(
        os.path.join(
            save_dir,
            "training_validation_error.png"
        ),
        dpi=300
    )


    plt.show()



# ==========================================================
# REGULARIZATION EFFECT
# ==========================================================

def plot_regularization_effect(
    alpha_values,
    train_scores,
    validation_scores,
    title="Regularization Effect",
    save_dir="../figures"
):

    os.makedirs(
        save_dir,
        exist_ok=True
    )


    plt.figure(
        figsize=(7,5)
    )


    plt.plot(
        alpha_values,
        train_scores,
        marker="o",
        label="Training R2"
    )


    plt.plot(
        alpha_values,
        validation_scores,
        marker="s",
        label="Validation R2"
    )


    plt.xscale(
        "log"
    )


    format_plot(
        title,
        "Alpha",
        "R2 Score"
    )


    plt.legend()


    plt.grid(True)


    plt.tight_layout()


    plt.savefig(
        os.path.join(
            save_dir,
            "regularization_effect.png"
        ),
        dpi=300
    )


    plt.show()



# ==========================================================
# COEFFICIENT COMPARISON
# ==========================================================

def plot_coefficients(
    coefficient_dataframe,
    save_dir="../figures"
):

    os.makedirs(
        save_dir,
        exist_ok=True
    )


    ax = coefficient_dataframe.plot(
        kind="bar",
        figsize=(12,6)
    )


    ax.set_title(
        "Coefficient Comparison",
        fontsize=SIZE,
        fontname=FONT,
        fontweight="bold"
    )


    ax.set_xlabel(
        "Features",
        fontsize=SIZE,
        fontname=FONT,
        fontweight="bold"
    )


    ax.set_ylabel(
        "Coefficient Value",
        fontsize=SIZE,
        fontname=FONT,
        fontweight="bold"
    )


    plt.xticks(
        rotation=90,
        fontsize=10
    )


    plt.grid(True)


    plt.tight_layout()


    plt.savefig(
        os.path.join(
            save_dir,
            "coefficient_comparison.png"
        ),
        dpi=300
    )


    plt.show()



# ==========================================================
# TRAINING VS PREDICTION TIME
# ==========================================================

def plot_regression_time(
    names,
    training_times,
    prediction_times,
    save_dir="../figures"
):

    os.makedirs(
        save_dir,
        exist_ok=True
    )


    x = np.arange(
        len(names)
    )

    width = 0.35


    plt.figure(
        figsize=(9,5)
    )


    plt.bar(
        x-width/2,
        training_times,
        width,
        label="Training Time"
    )


    plt.bar(
        x+width/2,
        prediction_times,
        width,
        label="Prediction Time"
    )


    plt.xticks(
        x,
        names,
        rotation=20
    )


    format_plot(
        "Training and Prediction Time",
        "Models",
        "Time Seconds"
    )


    plt.legend()


    plt.grid(True)


    plt.tight_layout()


    plt.savefig(
        os.path.join(
            save_dir,
            "training_prediction_time.png"
        ),
        dpi=300
    )


    plt.show()



# ==========================================================
# CORRELATION HEATMAP
# ==========================================================

def plot_correlation_heatmap(
    dataframe,
    save_dir="../figures"
):

    import seaborn as sns


    os.makedirs(
        save_dir,
        exist_ok=True
    )


    plt.figure(
        figsize=(10,8)
    )


    sns.heatmap(
        dataframe.corr(numeric_only=True),
        cmap="coolwarm"
    )


    plt.title(
        "Correlation Heatmap",
        fontsize=SIZE,
        fontname=FONT,
        fontweight="bold"
    )


    plt.tight_layout()


    plt.savefig(
        os.path.join(
            save_dir,
            "correlation_heatmap.png"
        ),
        dpi=300
    )


    plt.show()
# ==========================================================
# REGRESSION MODEL COMPARISON
# ==========================================================

def plot_regression_scores(
    names,
    scores,
    ylabel="R2 Score"
):

    plt.figure(
        figsize=(8,5)
    )


    plt.bar(
        names,
        scores
    )


    format_plot(
        "Regression Model Comparison",
        "Regression Model",
        ylabel
    )


    plt.xticks(
        rotation=20,
        fontsize=12,
        fontname=FONT
    )


    plt.yticks(
        fontsize=12,
        fontname=FONT
    )


    plt.grid(True)

    plt.tight_layout()

    plt.show()



# ==========================================================
# CONFUSION MATRIX (Experiment 4)
# ==========================================================

def plot_confusion_matrix(
    actual,
    predicted,
    model_name="model",
    save_dir="../figures"
):

    os.makedirs(
        save_dir,
        exist_ok=True
    )


    cm = confusion_matrix(
        actual,
        predicted
    )


    plt.figure(
        figsize=(6,5)
    )


    plt.imshow(
        cm,
        cmap="Blues"
    )


    plt.colorbar()


    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            plt.text(
                j,
                i,
                cm[i,j],
                ha="center",
                va="center",
                fontsize=SIZE
            )


    plt.xticks(
        [0,1],
        ["Ham","Spam"]
    )


    plt.yticks(
        [0,1],
        ["Ham","Spam"]
    )


    format_plot(
        f"{model_name} Confusion Matrix",
        "Predicted Label",
        "Actual Label"
    )


    plt.tight_layout()


    plt.savefig(
        os.path.join(
            save_dir,
            f"{model_name}_confusion_matrix.png"
        ),
        dpi=300
    )


    plt.show()



# ==========================================================
# ROC CURVE (Experiment 4)
# ==========================================================

def plot_roc_curve(
    models_probabilities,
    y_test,
    save_dir="../figures"
):
    """
    models_probabilities : dict {model_name: predicted_probability_of_spam}
    """

    os.makedirs(
        save_dir,
        exist_ok=True
    )


    plt.figure(
        figsize=(7,6)
    )


    for name, probabilities in models_probabilities.items():

        fpr, tpr, _ = roc_curve(
            y_test,
            probabilities
        )

        roc_auc = auc(
            fpr,
            tpr
        )

        plt.plot(
            fpr,
            tpr,
            label=f"{name} (AUC = {roc_auc:.2f})"
        )


    plt.plot(
        [0,1],
        [0,1],
        linestyle="--",
        color="black"
    )


    format_plot(
        "ROC Curve",
        "False Positive Rate",
        "True Positive Rate"
    )


    plt.legend()


    plt.grid(True)


    plt.tight_layout()


    plt.savefig(
        os.path.join(
            save_dir,
            "roc_curve.png"
        ),
        dpi=300
    )


    plt.show()



# ==========================================================
# CLASSIFICATION MODEL COMPARISON (Experiment 4)
# ==========================================================

def plot_classification_scores(
    names,
    scores,
    ylabel="Accuracy",
    save_dir="../figures"
):

    os.makedirs(
        save_dir,
        exist_ok=True
    )


    plt.figure(
        figsize=(8,5)
    )


    plt.bar(
        names,
        scores
    )


    format_plot(
        "Classification Model Comparison",
        "Model",
        ylabel
    )


    plt.xticks(
        rotation=20,
        fontsize=12,
        fontname=FONT
    )


    plt.yticks(
        fontsize=12,
        fontname=FONT
    )


    plt.grid(True)


    plt.tight_layout()


    plt.savefig(
        os.path.join(
            save_dir,
            "classification_model_comparison.png"
        ),
        dpi=300
    )


    plt.show()



# ==========================================================
# CLASSIFICATION CROSS VALIDATION SCORES (Experiment 4)
# ==========================================================

def plot_classification_cv_scores(
    fold_scores,
    save_dir="../figures"
):
    """
    fold_scores : dict {model_name: array of per-fold accuracy}
    """

    os.makedirs(
        save_dir,
        exist_ok=True
    )


    plt.figure(
        figsize=(7,5)
    )


    for name, scores in fold_scores.items():

        plt.plot(
            range(1,len(scores)+1),
            scores,
            marker="o",
            label=name
        )


    format_plot(
        "5 Fold Cross Validation Accuracy",
        "Fold",
        "Accuracy"
    )


    plt.legend()


    plt.grid(True)


    plt.tight_layout()


    plt.savefig(
        os.path.join(
            save_dir,
            "classification_cv_scores.png"
        ),
        dpi=300
    )


    plt.show()