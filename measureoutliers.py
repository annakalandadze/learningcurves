# import required libraries
import openml
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# set the dataset id from OpenML
dataset_ids = []
arr = []
for dataset_id in dataset_ids:
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, categorical_indicator, attribute_names = dataset.get_data(target=dataset.default_target_attribute)
    data = pd.DataFrame(X, columns=attribute_names)
    cat_attributes = list(X.select_dtypes(include=['category', 'object']))
    data = pd.get_dummies(X, columns=cat_attributes)
    # data = pd.get_dummies(data, columns=attribute_names)

    # calculate the IQR for each column
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1

    # define the threshold for outliers
    threshold = 1.5

    # calculate the upper and lower bounds for outliers
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr

    # count the number of outliers
    num_outliers = ((data < lower_bound) | (data > upper_bound)).sum().sum()

    # calculate the percentage of outliers
    percentage_outliers = num_outliers / data.size * 100
    arr.append((dataset_id, percentage_outliers))

df = pd.DataFrame(arr)
df.to_csv('perc_of_outliers.csv', mode='a', index=False)

