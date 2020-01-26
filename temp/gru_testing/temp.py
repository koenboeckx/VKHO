"""
From https://blog.floydhub.com/gru-with-pytorch/
"""

import os
import time

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

# Define data root directory
data_dir = "./temp/gru_testing/data"

# the scaler objects are stored in this dict so that our output test data from the model can be re-scaled during evaluation
label_scalers = {}

train_x = []
test_x = {}
test_y = {}

for file in tqdm(os.listdir(data_dir)):
    # skipping the files we're not using
    if file[-4:] != '.csv' or file == "pjm_hourly_est.csv":
        continue

    # store csv file in a pandas dataframe
    df = pd.read_csv(f'{data_dir}/{file}', parse_dates=[0])
    # processing time data into suitable input formats
    df['hour'] = df.apply(lambda x: x['Datetime'].hour, axis=1)
    df['dayofweek'] = df.apply(lambda x: x['Datetime'].dayofweek, axis=1)
    df['month'] = df.apply(lambda x: x['Datetime'].month, axis=1)
    df['dayofyear'] = df.apply(lambda x: x['Datetime'].dayofyear, axis=1)
    df = df.sort_values("Datetime").drop("Datetime", axis=1)

    # scaling the input data
    sc = MinMaxScaler()
    label_sc = MinMaxScaler()
    data = sc.fit_transform(df.values)

    # Obtaining the Scale for the labels(usage data) so that output can be re-scaled to actual value during evaluation
    label_sc.fit(df.iloc[:,0].values.reshape(-1, 1))
    label_scalers[file] = label_sc

    # define lookback period and split inputs/labels
    lookback = 90
    inputs = np.zeros((len(data)-lookback, lookback, df.shape[1]))
    labels = np.zeros(len(data)-lookback)

    for i in range(lookback, len(data)):
        inputs[i-lookback] = data[i-lookback:i]
        labels[i-lookback] = data[i,0]
    inputs = inputs.reshape(-1, lookback, df.shape[1])
    labels = labels.reshape(-1, 1)

    # Split data into train/test portions and combining all data from different files into a single array
    test_portion = int(0.1*len(inputs))
    if len(train_x) == 0:
        train_x = inputs[:-test_portion]
        train_y = labels[:-test_portion]
    else:
        train_x = np.concatenate((train_x,inputs[:-test_portion]))
        train_y = np.concatenate((train_y,labels[:-test_portion]))
    test_x[file] = (inputs[-test_portion:])
    test_y[file] = (labels[-test_portion:])

batch_size = 1024
train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True