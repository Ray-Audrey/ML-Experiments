#feature_selection.py
import pandas as pd

from sklearn.feature_selection import (
    SelectKBest,
    chi2,
    f_classif,
    mutual_info_classif,
    f_regression,
    mutual_info_regression
)



# ==========================================================
# FEATURE SELECTION
# ==========================================================

def select_features(
    X_train,
    X_test,
    y_train,
    method="anova",
    k="all",
    task="classification"
):
    """
    Feature selection using SelectKBest.

    Supported:

    Classification:
        anova
        chi2
        mutual_info

    Regression:
        anova
        mutual_info

    Returns:
        Selected training data
        Selected testing data
        selector object
    """


    task = task.lower()
    method = method.lower()



    # ------------------------------------------------------
    # Select Scoring Function
    # ------------------------------------------------------

    if task == "classification":

        methods = {

            "anova": f_classif,

            "chi2": chi2,

            "mutual_info": mutual_info_classif

        }



    elif task == "regression":

        methods = {

            "anova": f_regression,

            "mutual_info": mutual_info_regression

        }



    else:

        raise ValueError(
            "Task must be classification or regression"
        )



    if method not in methods:

        raise ValueError(
            f"{method} is not supported for {task}"
        )



    selector = SelectKBest(
        score_func=methods[method],
        k=k
    )



    X_train_selected = selector.fit_transform(
        X_train,
        y_train
    )


    X_test_selected = selector.transform(
        X_test
    )



    return (
        X_train_selected,
        X_test_selected,
        selector
    )





# ==========================================================
# FEATURE SCORE TABLE
# ==========================================================

def get_feature_scores(
    selector,
    feature_names
):
    """
    Generate feature importance score table.
    """

    scores = pd.DataFrame({

        "Feature": feature_names,

        "Score": selector.scores_

    })


    scores = scores.sort_values(
        by="Score",
        ascending=False
    )


    scores.reset_index(
        drop=True,
        inplace=True
    )


    return scores