import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=["Date"]).dropna().drop_duplicates()
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df = df.drop(df[df["Temp"].astype(int) <= -70].index)

    return df

if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    data = load_data("/Users/eitanmoed/Documents/Hebrew "
                   "University/Classes/Year 2/Semester 2/Intro to Machine "
                     "Learning/IML.HUJI/datasets/City_Temperature.csv")


    # Question 2 - Exploring data for specific country
    data_israel = data[data["Country"] == "Israel"]
    data_israel["Year"] = data_israel["Year"].astype(str)
    px.scatter(data_israel, x="DayOfYear", y="Temp", color="Year",
               title="Temp as function of DayOfYear",).show()
    data_israel_month = data_israel.groupby("Month", as_index=False).agg(
        Temp_std=("Temp", "std"))
    px.bar(data_israel_month, x="Month", y="Temp_std",
           title="Standard deviation of the daily temperatures").show()

    # Question 3 - Exploring differences between countries
    data_country_month = data.groupby(["Country", "Month"], as_index=False)\
        .agg(Temp_std=("Temp", "std"), Temp_mean=("Temp", "mean"))
    px.line(data_country_month, x="Month", y="Temp_mean",
            error_y="Temp_std", color="Country",).show()


    # Question 4 - Fitting model for different values of `k`
    data_israel_k = data_israel
    response = data_israel_k.pop("Temp")
    train_X, train_y, test_X, test_y = split_train_test(data_israel_k, response)
    loss = np.array([])
    for k in range(1, 11):
        polyfit = PolynomialFitting(k)
        polyfit.fit(train_X["DayOfYear"], train_y)
        loss = np.append(loss, round(polyfit.loss(test_X["DayOfYear"],
                                                 test_y), 2))
    print(loss)
    px.bar(data_israel, x=range(1, 11), y=loss,
           labels={"x": "K", "y": "Loss"},
           title="The loss value according to the degree").show()

    # Question 5 - Evaluating fitted model on different countries
    polyfit = PolynomialFitting(5)
    polyfit.fit(data_israel_k["DayOfYear"], response)
    error = np.array([])
    countries = data["Country"].unique()
    countries = countries[countries != ["Israel"]]
    for country in countries:
        temp = data[data["Country"] == country]
        error = np.append(error, polyfit.loss(temp["DayOfYear"], temp.pop(
            "Temp")))
    df = pd.DataFrame({"Country": countries, "Test Loss": error})
    px.bar(df, x="Country", y="Test Loss",
           title="Evaluating fitted model on different countries").show()