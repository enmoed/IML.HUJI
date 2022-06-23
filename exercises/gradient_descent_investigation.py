import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.metrics import misclassification_error
from IMLearn.model_selection import cross_validate
from IMLearn.utils import split_train_test
from sklearn.metrics import roc_curve

import plotly.graph_objects as go


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange,
                                       density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1],
                                 mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[
    Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values = []
    weights = []

    def callback(**kwargs):
        values.append(kwargs["val"])
        weights.append(kwargs["weights"])

    return callback, values, weights


def compare_fixed_learning_rates(
        init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
        etas: Tuple[float] = (1, .1, .01, .001)):
    for module in (L1, L2):
        min_loss = module(init).compute_output()
        fig = go.Figure(layout=go.Layout(
            title=f"Convergence Rate of {module.__name__}",
            xaxis_title="GD Iteration",
            yaxis_title="Norm",
            height=400))
        for eta in etas:
            callback, values, weights = get_gd_state_recorder_callback()
            lr = FixedLR(eta)
            L = module(init)
            gd = GradientDescent(learning_rate=lr, callback=callback)
            gd.fit(L, np.ones(1), np.ones(1))
            plot_descent_path(module, np.array(weights),
                              title=f"ETA={eta}").show()
            min_loss = min(min_loss, min(values))
            fig.add_traces([go.Scatter(
                x=np.linspace(0, len(values), len(values)),
                y=values, mode='markers', name=f'Eta = {eta}')])
        fig.show()
        print(f"Min loss for {module.__name__} is {min_loss}")


def compare_exponential_decay_rates(
        init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
        eta: float = .1,
        gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    fig = go.Figure(layout=go.Layout(
        title=f"L1 Convergence Rate of Exponential Decay",
        xaxis_title="GD Iteration",
        yaxis_title="Norm", height=400))
    min_w = np.inf
    weights_95 = []
    for gamma in gammas:
        callback, values, weights = get_gd_state_recorder_callback()
        if gamma == .95:
            weights_95 = weights
        lr = ExponentialLR(eta, gamma)
        L = L1(init)
        gd = GradientDescent(learning_rate=lr, callback=callback)
        gd.fit(L, np.ones(1), np.ones(1))
        min_w = min(min_w, min(values))
        fig.add_traces([go.Scatter(
            x=np.linspace(0, len(values), len(values)),
            y=values, mode='markers+lines', name=f'Gamma = {gamma}')])

    print(f"Min loss for Exponential Decay is {min_w}")

    # Plot algorithm's convergence for the different values of gamma
    fig.show()

    # Plot descent path for gamma=0.95
    plot_descent_path(L1, np.array(weights_95), title="Gamma = 0.95").show()


def load_data(path: str = "../datasets/SAheart.data",
              train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd,
                            train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()
    X_train, y_train, X_test, y_test = X_train.to_numpy(), y_train.to_numpy(), \
                                       X_test.to_numpy(), y_test.to_numpy()

    # Plotting convergence rate of logistic regression over SA heart disease data
    solver = GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20000)
    L = LogisticRegression(solver=solver)
    L.fit(X_train, y_train)
    y_proba = L.predict_proba(X_train)
    fpr, tpr, thresholds = roc_curve(y_train, y_proba)
    roc_fig = go.Figure([go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                    line=dict(color="black", dash='dash'),
                                    name="Random Class Assignment"),
                         go.Scatter(x=fpr, y=tpr, mode='markers+lines',
                                    text=thresholds,
                                    name="", showlegend=False,
                                    hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
                        layout=go.Layout(title='ROC Curve of Logistic Regression over dataset',
                                         xaxis_title='False Positive Rate (FPR)',
                                         yaxis_title='True Positive Rate (TPR)'))
    roc_fig.show()
    roc_vals = tpr - fpr
    thresholds = np.round(thresholds, 2)
    best = thresholds[np.argmax(roc_vals)]
    print(f"Best alpha for Logistic Regression is {best}")
    best_logistic_loss = LogisticRegression(alpha=best, solver=solver).fit(
        X_train, y_train).loss(X_test, y_test)
    print(f"Best loss result for Logistic Regression is {best_logistic_loss}")


    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    lam_space = np.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1])
    train_losses_l1, test_losses_l1 = np.zeros(1), np.zeros(len(lam_space))
    train_losses_l2, test_losses_l2 = np.zeros(1), np.zeros(len(lam_space))
    for i, lam in enumerate(lam_space):
        train_losses_l1, test_losses_l1[i] = cross_validate(
            estimator=LogisticRegression(solver=solver, penalty='l1', lam=lam),
            X=X_train, y=y_train, scoring=misclassification_error)
        train_losses_l2, test_losses_l2[i] = cross_validate(
            estimator=LogisticRegression(solver=solver, penalty='l2', lam=lam),
            X=X_train, y=y_train, scoring=misclassification_error)
    best_lam_l1 = lam_space[np.argmin(test_losses_l1)]
    test_loss_l1 = LogisticRegression(solver=solver, penalty='l1', \
                                                     lam=best_lam_l1).fit(
        X_train, y_train).loss(X_test, y_test)
    print(f"Best lambda for L1 is {best_lam_l1}")
    print(f"Test error is {test_loss_l1}")
    best_lam_l2 = lam_space[np.argmin(test_losses_l2)]
    test_loss_l2 = LogisticRegression(solver=solver, penalty='l2', \
                                                     lam=best_lam_l2).fit(
        X_train, y_train).loss(X_test, y_test)
    print(f"Best lambda for L2 is {best_lam_l2}")
    print(f"Test error is {test_loss_l2}")


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    fit_logistic_regression()
