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
    random_state=42
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
    """

    algorithm = algorithm.lower()

    if algorithm == "logistic_regression":

        model = LogisticRegression(
            random_state=random_state,
            max_iter=1000
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
            probability=True,
            random_state=random_state
        )

    else:

        raise ValueError(
            f"Unsupported algorithm: {algorithm}"
        )

    model.fit(X_train, y_train)

    return model