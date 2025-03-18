# The following guide was used when writing this code: https://www.analyticsvidhya.com/blog/2021/06/random-forest-for-time-series-forecasting/

from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


CSV_PATH = "seattle_map_aggregated_data_withCRA_ID_filtered(in).csv"
METRIC_NAME  = "crime_rate_per_1000"

def get_epoch_seconds(timestamp):
    epoch  = datetime(1970, 1, 1)
    return (timestamp - epoch).total_seconds()

neighborhoods = {}
neighborhood_id = 0

def convert_neighborhood(neighborhood):
    global neighborhood_id, neighborhoods
    if neighborhood not in neighborhoods.keys():
        neighborhoods[neighborhood] = neighborhood_id
        neighborhood_id += 1
    return neighborhoods[neighborhood]

def convert_back(neightborhood_num):
    global neighborhood_id, neighborhoods
    inv_map = {v: k for k, v in neighborhoods.items()}
    return inv_map[neightborhood_num]


if __name__ == "__main__":
    data = pd.read_csv(CSV_PATH)
    data['Year'] = data['Year'].astype(int)
    data['CRA_ID'] = data['CRA_ID'].astype(float)
    data = data.sort_values(['Year'])
    data.to_csv('sorted.csv', index=False)
    end_point = data.shape[0]
    test_length = 80
    x = end_point - test_length
    finaldf_train = data.loc[:x - 1, :]
    finaldf_test = data.loc[x:, :]
    finaldf_test_x = finaldf_test[['Year', 'CRA_ID']]
    finaldf_test_y = finaldf_test[METRIC_NAME]
    finaldf_train_x = finaldf_train[['Year', 'CRA_ID']]
    finaldf_train_y = finaldf_train[METRIC_NAME]
    print("Starting model train..")
    rfe = RandomForestRegressor(n_estimators=100, oob_score=True)
    fit = rfe.fit(finaldf_train_x, finaldf_train_y)
    y_pred = fit.predict(finaldf_test_x)

    y_true = np.array(finaldf_test_y)
    sumvalue = np.sum(y_true)
    mape = np.sum(np.abs((y_true - y_pred))) / sumvalue * 100

    accuracy = 100 - mape
    print('Accuracy Random Tree:', round(accuracy, 2), '%.')
    print('Out of bag score: ', rfe.oob_score_)
    print('Score: ', rfe.score(finaldf_test_x, finaldf_test_y))

    inputs_2025 = pd.DataFrame()
    inputs_2025['Year'] = pd.Series(2025).repeat(len(data['CRA_ID'].unique()))
    inputs_2025['CRA_ID'] = data['CRA_ID'].unique()
    inputs_2025[METRIC_NAME] = fit.predict(inputs_2025)
    inputs_2025.to_csv('test_results.csv', index=False)

    lr = LinearRegression()
    fit = lr.fit(finaldf_train_x, finaldf_train_y)
    y_pred_lr = fit.predict(finaldf_test_x)
    mape = np.sum(np.abs((y_true - y_pred_lr))) / sumvalue * 100
    accuracy = 100 - mape
    print('Accuracy Linear Regression:', round(accuracy, 2), '%.')

