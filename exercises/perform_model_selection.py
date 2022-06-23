from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    f = lambda x: (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
    X = np.linspace(-1.2, 2, n_samples)
    y = f(X) + np.random.normal(0, noise, size=n_samples)
    X_train, y_train, X_test, y_test = split_train_test(pd.DataFrame(X),
                                                        pd.Series(y),
                                                        2 / 3)
    X_train, X_test, y_train, y_test = np.array(X_train).flatten(), \
                                       np.array(X_test).flatten(), \
                                       np.array(y_train).flatten(), \
                                       np.array(y_test).flatten()
    make_subplots(1, 1).add_traces([go.Scatter(x=X, y=f(X), mode="markers",
                                marker=dict(color="black", opacity=.7),
                                showlegend=False),
                     go.Scatter(x=X_train, y=y_train, mode="markers",
                                marker=dict(color="blue", opacity=.7),
                                showlegend=False),
                     go.Scatter(x=X_test, y=y_test, mode="markers",
                                marker=dict(color="red", opacity=.7),
                                showlegend=False)],).update_layout(
                    title="Polynomial Fitting",
                    margin=dict(t=100)).show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    fig = go.Figure()
    test_errors = np.zeros(11)
    train_errors = np.zeros(11)
    for degree in range(11):
        train_errors[degree], test_errors[degree] = cross_validate(PolynomialFitting(degree), X_train, y_train, mean_square_error, 5)
    x_axis = np.linspace(0, 10, 11)
    fig.add_trace(go.Scatter(x=x_axis, y=train_errors, mode="markers+lines",
                            name="Train Error",
                            line=dict(color="blue", width=2)))
    fig.add_trace(go.Scatter(x=x_axis, y=test_errors, mode="markers+lines",
                            name="Test Error",
                            line=dict(color="red", width=2)))
    fig.update_layout(title="Fitting Polynomials of Different Degrees",
                        margin=dict(t=100))
    fig.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    best_degree = np.argmin(test_errors)
    model = PolynomialFitting(int(best_degree)).fit(X_train, y_train)
    model_y_test = model.predict(X_test)
    print(f"Test error for {best_degree}-degree polynomial model: {mean_square_error(y_test, model_y_test)}")



def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    X_train, y_train, X_test, y_test = X[:n_samples], y[:n_samples], X[n_samples:], y[n_samples:]


    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    fig = go.Figure()
    rig_test_errors = np.zeros(n_evaluations)
    rig_train_errors = np.zeros(n_evaluations)
    las_test_errors = np.zeros(n_evaluations)
    las_train_errors = np.zeros(n_evaluations)
    for i in range(n_evaluations):
        reg_param = 2*i / n_evaluations
        rig_train_errors[i], rig_test_errors[i] = cross_validate(RidgeRegression(reg_param), X_train, y_train, mean_square_error, 5)
        las_train_errors[i], las_test_errors[i] = cross_validate(Lasso(reg_param), X_train, y_train, mean_square_error, 5)
    x_axis = np.linspace(0, 2, n_evaluations)
    fig.add_trace(go.Scatter(x=x_axis, y=rig_train_errors,
                             mode="lines",name="Ridge Train Error",
                                line=dict(color="blue", width=2)))
    fig.add_trace(go.Scatter(x=x_axis, y=rig_test_errors,
                                mode="lines",name="Ridge Test Error",
                                line=dict(color="red", width=2)))
    fig.add_trace(go.Scatter(x=x_axis, y=las_train_errors,
                                mode="lines",name="Lasso Train Error",
                                line=dict(color="green", width=2)))
    fig.add_trace(go.Scatter(x=x_axis, y=las_test_errors,
                                mode="lines",name="Lasso Test Error",
                                line=dict(color="orange", width=2)))
    fig.update_layout(title="Fitting Ridge and Lasso Regressions",
                        margin=dict(t=100))
    fig.show()


    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    reg_param_func = lambda x: 2*x / n_evaluations
    best_reg_param = reg_param_func(np.argmin(rig_test_errors))
    best_lasso_param = reg_param_func(np.argmin(las_test_errors))
    reg = RidgeRegression(float(best_reg_param)).fit(X_train, y_train)
    lasso = Lasso(best_lasso_param).fit(X_train, y_train)
    LS = LinearRegression().fit(X_train, y_train)
    print(f"Ridge Regression: {mean_square_error(y_test, reg.predict(X_test))}")
    print(f"Lasso Regression: {mean_square_error(y_test, lasso.predict(X_test))}")
    print(f"Least Squares: {mean_square_error(y_test, LS.predict(X_test))}")


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(100, 0)
    select_polynomial_degree(1500, 10)
    select_regularization_parameter()
