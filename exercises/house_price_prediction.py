from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename)
    df.drop(columns=["id", "lat", "long"], inplace=True)
    df = df.dropna()
    df.drop(df[df.date == "0"].index, inplace=True)
    df.drop(df[df.price <= 0].index, inplace=True)
    df.drop(df[df.floors <= 0].index, inplace=True)
    df["year_sold"] = pd.to_datetime(df["date"], format="%Y%m%dT%f").dt.year
    df.drop(columns=["date"], inplace=True)
    df1 = df.groupby("zipcode", as_index=False)["grade"].mean()
    df["zipcode"] = df["zipcode"].apply(lambda x: df1[df1.zipcode ==
                                                    x].grade.values[0])
    df.rename(columns={"zipcode": "sqft_of_location"}, inplace=True)
    y = df["price"]
    X = df.drop(columns=["price"])
    return X, y





def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str =
".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for col in X.columns:
        fig = px.scatter(X, x=col, y=y,
                         title=f"Pearson Correlation between {col} and price "
                               f"= {X[col].cov(y)/(X[col].std()*y.std())}")
        fig.update_layout(xaxis_title=col,
                          yaxis_title="price")
        fig.write_image(f"{output_path}/{col}.png")

if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data("/Users/eitanmoed/Documents/Hebrew University/Classes/Year 2/Semester 2/Intro to Machine Learning/IML.HUJI/datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y, output_path="/Users/eitanmoed/Documents/Hebrew University/Classes/Year 2/Semester 2/Intro to Machine Learning/IML.HUJI/")

    # Question 3 - Split samples into training- and testing sets.
    X_train, y_train, X_test, y_test = split_train_test(X, y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    # Save plot to file
    x = np.linspace(10, 101, 90)
    mean_pred, var_pred, mean_conf_plus, mean_conf_minus =\
        np.array([]), np.array([]), np.array([]), np.array([])
    for p in range(10, 101):
        loss_total = np.array([])
        for i in range(10):
            X_train_p = X_train.sample(frac=p/100)
            y_train_p = y_train.loc[X_train_p.index]
            reg = LinearRegression(include_intercept=True)
            reg.fit(X_train_p, y_train_p)
            loss_total = np.append(loss_total, reg._loss(X_test, y_test))
        mean_pred = np.append(mean_pred, np.mean(loss_total))
        var_pred = np.append(var_pred, np.std(loss_total))
        mean_conf_plus = np.append(mean_conf_plus, mean_pred[-1] +
                                   2*var_pred[-1])
        mean_conf_minus = np.append(mean_conf_minus, mean_pred[-1] -
                                    2*var_pred[-1])
    go.Figure([go.Scatter(x=x, y=mean_pred, mode="markers+lines",
                                    name="Mean Prediction",
                                    line=dict(dash="dash"),
                                    marker=dict(color="green",
                                                opacity=.7)),
    go.Scatter(x=x, y=mean_conf_minus,
                                    fill=None, mode="lines",
                                    line=dict(color="lightgrey"),
                                    showlegend=False),
    go.Scatter(x=x, y=mean_conf_plus,
                                    fill='tonexty', mode="lines",
                                    line=dict(color="lightgrey"),
                                    showlegend=False)]).show()

