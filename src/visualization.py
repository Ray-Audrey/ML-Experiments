import matplotlib.pyplot as plt
import numpy as np



FONT = "Times New Roman"
SIZE = 15



def format_plot(title, xlabel, ylabel):

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
# ACCURACY VS K
# ==========================================================


def plot_accuracy_vs_k(
    k_values,
    scores
):

    plt.figure(
        figsize=(7,5)
    )


    plt.plot(
        k_values,
        scores,
        marker="o"
    )


    format_plot(
        "Accuracy vs K",
        "K Value",
        "Accuracy"
    )


    plt.grid(True)

    plt.show()



# ==========================================================
# CLASSIFIER COMPARISON
# ==========================================================


def plot_classifier_comparison(
    names,
    scores
):

    plt.figure(
        figsize=(8,5)
    )


    plt.bar(
        names,
        scores
    )


    format_plot(
        "Classifier Comparison",
        "Classifier",
        "Accuracy"
    )


    plt.xticks(
        rotation=20
    )

    plt.show()



# ==========================================================
# TRAINING TIME
# ==========================================================


def plot_training_time(
    names,
    times
):

    plt.figure(
        figsize=(8,5)
    )


    plt.bar(
        names,
        times
    )


    format_plot(
        "Training Time Comparison",
        "Algorithm",
        "Time (Seconds)"
    )


    plt.xticks(
        rotation=20
    )

    plt.show()



# ==========================================================
# PREDICTION TIME
# ==========================================================


def plot_prediction_time(
    names,
    times
):

    plt.figure(
        figsize=(8,5)
    )


    plt.bar(
        names,
        times
    )


    format_plot(
        "Prediction Time Comparison",
        "Algorithm",
        "Time (Seconds)"
    )


    plt.xticks(
        rotation=20
    )

    plt.show()



# ==========================================================
# CROSS VALIDATION SCORES
# ==========================================================


def plot_cv_scores(
    scores
):

    plt.figure(
        figsize=(7,5)
    )


    plt.plot(
        np.arange(1,len(scores)+1),
        scores,
        marker="o"
    )


    format_plot(
        "Cross Validation Accuracy",
        "Fold",
        "Accuracy"
    )


    plt.grid(True)

    plt.show()



# ==========================================================
# GRID SEARCH HEATMAP
# ==========================================================


def plot_grid_search_heatmap(
    results
):

    pivot = results.pivot_table(
        values="mean_test_score",
        index="param_n_neighbors",
        columns="param_weights"
    )


    plt.figure(
        figsize=(8,5)
    )


    plt.imshow(
        pivot,
        cmap="viridis"
    )


    plt.colorbar()


    plt.xticks(
        range(len(pivot.columns)),
        pivot.columns
    )


    plt.yticks(
        range(len(pivot.index)),
        pivot.index
    )


    format_plot(
        "GridSearchCV Heatmap",
        "Weights",
        "K Value"
    )


    plt.show()



# ==========================================================
# RANDOM SEARCH DISTRIBUTION
# ==========================================================


def plot_random_search_distribution(
    results
):

    plt.figure(
        figsize=(7,5)
    )


    plt.hist(
        results["mean_test_score"],
        bins=10
    )


    format_plot(
        "RandomizedSearchCV Score Distribution",
        "Accuracy Score",
        "Frequency"
    )


    plt.show()