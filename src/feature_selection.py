from sklearn.feature_selection import (
    SelectKBest,
    chi2,
    f_classif,
    mutual_info_classif
)


# ==========================================================
# FEATURE SELECTION
# ==========================================================

def select_features(
    X_train,
    X_test,
    y_train,
    method="anova",
    k="all"
):
    """
    Select important features.

    Parameters
    ----------
    method :
        "anova"
        "chi2"
        "mutual_info"

    k :
        Number of features to keep.
    """

    methods = {

        "anova": f_classif,

        "chi2": chi2,

        "mutual_info": mutual_info_classif

    }

    if method not in methods:

        raise ValueError(
            f"Unknown method : {method}"
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
