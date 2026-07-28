#classification.py
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import (
    GaussianNB,
    MultinomialNB,
    BernoulliNB
)
from sklearn.svm import SVC


def train_classification(
    X_train,
    y_train,
    algorithm="random_forest",
    k=5,
    algorithm_type="auto",
    metric="minkowski",
    weights="uniform",
    random_state=42,
    C=1.0,
    penalty="l2",
    solver="lbfgs",
    max_iter=1000,
    kernel="rbf",
    gamma="scale",
    degree=3
):
    """
    Train a classification model.

    Supported Algorithms
    --------------------
    logistic_regression
    decision_tree
    random_forest
    knn
    gaussian_nb
    multinomial_nb
    bernoulli_nb
    svm

    Extra Parameters (Experiment 4)
    --------------------------------
    Logistic Regression -> C, penalty, solver, max_iter
    SVM                  -> C, kernel, gamma, degree
    """

    algorithm = algorithm.lower()

    if algorithm == "logistic_regression":

        model = LogisticRegression(
            C=C,
            penalty=penalty,
            solver=solver,
            max_iter=max_iter,
            random_state=random_state
        )

    elif algorithm == "decision_tree":

        model = DecisionTreeClassifier(
            random_state=random_state
        )

    elif algorithm == "random_forest":

        model = RandomForestClassifier(
            random_state=random_state
        )

    elif algorithm == "knn":

        model = KNeighborsClassifier(
            n_neighbors=k,
            algorithm=algorithm_type,
            metric=metric,
            weights=weights
        )

    elif algorithm == "gaussian_nb":

        model = GaussianNB()

    elif algorithm == "multinomial_nb":

        model = MultinomialNB()

    elif algorithm == "bernoulli_nb":

        model = BernoulliNB()

    elif algorithm == "svm":

        model = SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            degree=degree,
            probability=True,
            random_state=random_state
        )

    else:

        raise ValueError(
            f"Unsupported algorithm: {algorithm}"
        )

    model.fit(X_train, y_train)

    return model