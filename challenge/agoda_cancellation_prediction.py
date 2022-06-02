# from challenge.agoda_cancellation_estimator import AgodaCancellationEstimator
# from IMLearn.utils import split_train_test
#
# import numpy as np
# import pandas as pd
#
#
# def load_data(filename: str):
#     """
#     Load Agoda booking cancellation dataset
#     Parameters
#     ----------
#     filename: str
#         Path to house prices dataset
#
#     Returns
#     -------
#     Design matrix and response vector in either of the following formats:
#     1) Single dataframe with last column representing the response
#     2) Tuple of pandas.DataFrame and Series
#     3) Tuple of ndarray of shape (n_samples, n_features) and ndarray of shape (n_samples,)
#     """
#     # TODO - replace below code with any desired preprocessing
#     full_data = pd.read_csv(filename).dropna().drop_duplicates()
#     features = full_data[["h_booking_id",
#                           "hotel_id",
#                           "accommadation_type_name",
#                           "hotel_star_rating",
#                           "customer_nationality"]]
#     labels = full_data["cancellation_datetime"]
#
#     return features, labels
#
#
# def evaluate_and_export(estimator: BaseEstimator, X: np.ndarray, filename: str):
#     """
#     Export to specified file the prediction results of given estimator on given testset.
#
#     File saved is in csv format with a single column named 'predicted_values' and n_samples rows containing
#     predicted values.
#
#     Parameters
#     ----------
#     estimator: BaseEstimator or any object implementing predict() method as in BaseEstimator (for example sklearn)
#         Fitted estimator to use for prediction
#
#     X: ndarray of shape (n_samples, n_features)
#         Test design matrix to predict its responses
#
#     filename:
#         path to store file at
#
#     """
#     pd.DataFrame(estimator.predict(X), columns=["predicted_values"]).to_csv(filename, index=False)
#
#
# if __name__ == '__main__':
#     np.random.seed(0)
#
#     # Load data
#     df, cancellation_labels = load_data("../datasets/agoda_cancellation_train.csv")
#     train_X, train_y, test_X, test_y = split_train_test(df, cancellation_labels)
#
#     # Fit model over data
#     estimator = AgodaCancellationEstimator().fit(train_X, train_y)
#
#     # Store model predictions over test set
#     evaluate_and_export(estimator, test_X, "id1_id2_id3.csv")
from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression
import re
from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str):

    def calc_cancel():
        reg = "(\d*)D(\d*[N,P])"
        all_cancel_cost = []
        for ind, word in enumerate(df['cancellation_policy_code']):
            result = re.search(reg, word)
            if result == None:
                all_cancel_cost.append(df["original_selling_amount"][ind])
            else:
                days = max(df['res_checkin_delta'][ind] - int(result.group(1)), 1)

                cost_part = int(result.group(2)[:-1])
                part_cancel = 0
                if result.group(2)[-1] == "N":
                    part_cancel = 0.001*df["original_selling_amount"][ind] * cost_part*days
                elif result.group(2)[-1] == "P":
                    part_cancel = 0.001*df["original_selling_amount"][ind] * cost_part/100 * \
                                  df["checkin_checkout_delta"][ind]
                all_cancel_cost.append(days * part_cancel)
        df['cancel_cost'] = all_cancel_cost
    def first_world():
        first_world = ['United States of America', 'United Kingdom', 'Canada',
                   'Australia', 'New Zealand', 'India']
        for first in first_world:
            df["customer_nationality"][df["customer_nationality"] == first] = 1
        df["customer_nationality"][df["customer_nationality"] != 1] = 0
        df["First World Customer"] = df['customer_nationality']
        df.drop(["customer_nationality"], axis=1, inplace=True)
    df = pd.read_csv(filename)
    df.drop_duplicates()
    df['booking_datetime'] = [x.split()[0] for x in df['booking_datetime']]
    df['checkin_date'] = [x.split()[0] for x in df['checkin_date']]
    df['checkout_date'] = [x.split()[0] for x in df['checkout_date']]

    df["res"] = pd.to_datetime(df['booking_datetime'], errors='coerce')
    df["checkin"] = pd.to_datetime(df['checkin_date'], errors='coerce')
    df['res_checkin_delta'] = (df['checkin'] - df['res']).array.days

    df["checkout"] = pd.to_datetime(df['checkout_date'], errors='coerce')
    df['checkin_checkout_delta'] = (df['checkout'] - df['checkin']).array.days


    calc_cancel()
    # data_frame = data_frame.drop(['h_booking_id', 'booking_datetime', 'checkin_date', 'checkout_date',
    #                               'hotel_live_date', 'customer_nationality', 'language', 'original_payment_method',
    #                               'original_payment_currency','original_payment_type', 'is_first_booking', 'request_nonesmoke',
    #                               'cancellation_policy_code'], axis=1)
    # data_frame = data_frame.drop(['request_highfloor','request_largebed','request_twinbeds','request_airport',
    #                               'request_earlycheckin','request_latecheckin','h_customer_id','accommadation_type_name','hotel_country_code',
    #                               'guest_nationality_country_name','origin_country_code','hotel_brand_code',
    #                               'hotel_area_code','hotel_chain_code','hotel_city_code'], axis=1)

    # df['charge_option'][df['charge_option']=="Pay Now"] =0
    # df['charge_option'][df['charge_option'] == "Pay Later"] = 1
    # df['charge_option'][df['charge_option'] == "Pay at Check-in"] = 2

    df = pd.concat([df, pd.get_dummies(df['charge_option'])], axis=1)
    df.drop(['charge_option', 'Pay at Check-in'], axis=1, inplace=True)
    df['is_user_logged_in'][df['is_user_logged_in'] == True] = 1
    df['is_user_logged_in'][df['is_user_logged_in'] == False] = 0
    df['Checkin'] = df['checkin'].dt.dayofyear
    df['Checkout'] = df['checkout'].dt.dayofyear
    df['res'] = df['res'].dt.dayofyear
    df["cancellation_datetime"] = df["cancellation_datetime"].fillna(0)
    df["cancellation_datetime"][df["cancellation_datetime"] != 0] = 1
    amount = df.groupby(['h_customer_id']).size()
    df['Amount of Bookings'] = df['h_customer_id'].map(amount)
    df.drop(["is_first_booking"], axis=1, inplace=True)
    df["Foreigner"] = [1 if df["origin_country_code"][i] != df[
        "hotel_country_code"][i] else 0 for i in range(len(df))]
    df["Special Requests"] = df["request_nonesmoke"] + df[
        "request_earlycheckin"]\
                        + \
                     df["request_latecheckin"] + \
                        df["request_twinbeds"] + df["request_largebed"] + df["request_highfloor"] + \
                        df["request_airport"]
    df.drop(['request_nonesmoke', 'request_earlycheckin', 'request_latecheckin', 'request_twinbeds',
                'request_largebed', 'request_highfloor', 'request_airport'], axis=1, inplace=True)
    df.drop(['hotel_brand_code', 'hotel_area_code', 'hotel_chain_code'],
            axis=1, inplace=True)
    df.drop(['hotel_country_code', 'origin_country_code'], axis=1, inplace=True)
    df.drop(['hotel_city_code'], axis=1, inplace=True)
    df.drop(['hotel_live_date'], axis=1, inplace=True)
    df.drop(['h_booking_id', 'booking_datetime', 'checkin_date',
             'checkout_date','hotel_id', 'h_customer_id',
             'guest_nationality_country_name'] , axis=1,
            inplace=True)

    df["Special Requests"] = df["Special Requests"].fillna(0)
    first_world()
    df = pd.concat([df, pd.get_dummies(df['accommadation_type_name'])], axis=1)
    df.drop(['accommadation_type_name'], axis=1, inplace=True)


    return df





if __name__ == '__main__':
    np.random.seed(0)
    data_path = "/Users/eitanmoed/Documents/Hebrew University/Classes/Year " \
                "2/Semester 2/Intro to Machine Learning/IML.HUJI/datasets/agoda_cancellation_train.csv"
    data = load_data(data_path)
    print(data['res_checkin_delta'])
    print(data['checkin_checkout_delta'])
