from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    train_err = []
    test_err = []
    X_folds = np.array_split(X, cv)
    y_folds = np.array_split(y, cv)
    for i in range(cv):
        X_train = np.concatenate(X_folds[:i] + X_folds[i + 1:], axis=0)
        y_train = np.concatenate(y_folds[:i] + y_folds[i + 1:], axis=0)
        X_test, y_test = X_folds[i], y_folds[i]
        model = estimator.fit(X_train, y_train)
        train_err.append(scoring(y_train, model.predict(X_train)))
        test_err.append(scoring(y_test, model.predict(X_test)))
    return np.mean(train_err), np.mean(test_err)
