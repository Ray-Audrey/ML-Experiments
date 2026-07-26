import matplotlib.pyplot as plt
import numpy as np


def plot_accuracy_vs_k(k_values, scores):

    plt.figure(figsize=(7,5))

    plt.plot(
        k_values,
        scores,
        marker="o"
    )

    plt.title("Accuracy vs K")

    plt.xlabel("K")

    plt.ylabel("Accuracy")

    plt.grid(True)

    plt.show()


def plot_classifier_comparison(names, scores):

    plt.figure(figsize=(8,5))

    plt.bar(names, scores)

    plt.title("Classifier Comparison")

    plt.ylabel("Accuracy")

    plt.xticks(rotation=20)

    plt.show()


def plot_training_time(names, times):

    plt.figure(figsize=(8,5))

    plt.bar(names, times)

    plt.title("Training Time")

    plt.ylabel("Seconds")

    plt.show()


def plot_prediction_time(names, times):

    plt.figure(figsize=(8,5))

    plt.bar(names, times)

    plt.title("Prediction Time")

    plt.ylabel("Seconds")

    plt.show()


def plot_cv_scores(scores):

    plt.figure(figsize=(7,5))

    plt.plot(
        np.arange(1, len(scores)+1),
        scores,
        marker="o"
    )

    plt.title("Cross Validation Accuracy")

    plt.xlabel("Fold")

    plt.ylabel("Accuracy")

    plt.grid(True)

    plt.show()