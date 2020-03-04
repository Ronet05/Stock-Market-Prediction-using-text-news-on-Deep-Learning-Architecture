import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, Normalizer
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def getData():
    with open('process/process_file_3.csv') as csv_file:
        reader = csv.reader(csv_file)
        data = list(reader)

    return data

def preprocessData(df):
    labelencoder = LabelEncoder()
    one_hot_encoder = OneHotEncoder()
    normalizer = Normalizer()

    # create a copy for one hot encoding
    df_ohe = df

    df[:,0] = labelencoder.fit_transform(df[:,0])
    df = df.astype(float)
    label_processdf = df[:, 2:6]
    label_processdf = normalizer.fit_transform(label_processdf)
    np.random.shuffle(label_processdf)

    ohe_processed = one_hot_encoder.fit_transform(df_ohe[:,0:2]).toarray()
    ohe_processdf = np.append(ohe_processed, df_ohe[2:6], 1)
    np.random.shuffle(ohe_processdf)

    return label_processdf, ohe_processdf


def learn(data):
    labeldf, ohedf = preprocessData(data)

    X_labeled = labeldf[:, 0:labeldf.shape[1]-1]
    Y_labeled = labeldf[:, -1].reshape(-1,1)

    X_ohe = labeldf[:, 0:labeldf.shape[1] - 1]
    Y_ohe = labeldf[:, -1].reshape(-1, 1)

    # Created 2 sets of data to train with each model, will increase time duration but also give wider results
    x_l_train, x_l_test, y_l_train, y_l_test = train_test_split(X_labeled, Y_labeled, train_size=0.7)
    x_o_train, x_o_test, y_o_train, y_o_test = train_test_split(X_ohe, Y_ohe, train_size=0.7)

    svr_l = SVR(kernel='linear')
    svr_o = SVR(kernel='rbf')
    lr_l = LinearRegression()
    lr_o = LinearRegression()
    rf_l = RandomForestRegressor()
    rf_o = RandomForestRegressor()
    gb_l = GradientBoostingRegressor()
    gb_o = GradientBoostingRegressor()

    # fitting for simple label encoded
    svr_l.fit(x_l_train, y_l_train)
    lr_l.fit(x_l_train, y_l_train)
    rf_l.fit(x_l_train, y_l_train)
    gb_l.fit(x_l_train, y_l_train)

    # fitting for one hot encoded
    svr_o.fit(x_o_train, y_o_train)
    lr_o.fit(x_o_train, y_o_train)
    rf_o.fit(x_o_train, y_o_train)
    gb_o.fit(x_o_train, y_o_train)










