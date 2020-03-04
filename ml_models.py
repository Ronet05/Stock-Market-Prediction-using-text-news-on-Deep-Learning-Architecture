import csv
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, Normalizer
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, roc_auc_score


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

    df[:, 0] = labelencoder.fit_transform(df[:, 0])
    df = df.astype(float)
    label_processdf = df[:, 2:6]
    label_processdf = normalizer.fit_transform(label_processdf)
    np.random.shuffle(label_processdf)

    ohe_processed = one_hot_encoder.fit_transform(df_ohe[:, 0:2]).toarray()
    ohe_processdf = np.append(ohe_processed, df_ohe[2:6], 1)
    np.random.shuffle(ohe_processdf)

    return label_processdf, ohe_processdf


def learn(data):
    labeldf, ohedf = preprocessData(data)

    X_labeled = labeldf[:, 0:labeldf.shape[1] - 1]
    Y_labeled = labeldf[:, -1].reshape(-1, 1)

    X_ohe = labeldf[:, 0:labeldf.shape[1] - 1]
    Y_ohe = labeldf[:, -1].reshape(-1, 1)

    # Created 2 sets of data to train with each model, will increase time duration but also give wider results
    x_l_train, x_l_test, y_l_train, y_l_test = train_test_split(X_labeled, Y_labeled, train_size=0.7)
    x_o_train, x_o_test, y_o_train, y_o_test = train_test_split(X_ohe, Y_ohe, train_size=0.7)

    svr_l = SVR(kernel='linear')
    svr_o = SVR(kernel='rbf')
    lr_l = LinearRegression(n_jobs=4)
    lr_o = LinearRegression(n_jobs=4)
    rf_l = RandomForestRegressor(n_jobs=4)
    rf_o = RandomForestRegressor(n_jobs=4)
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

    # results in both
    svr_l_predict = svr_l.predict(x_l_test)
    lr_l_predict = lr_l.predict(x_l_test)
    rf_l_predict = rf_l.predict(x_l_test)
    gb_l_predict = gb_l.predict(x_l_test)

    svr_o_predict = svr_o.predict(x_o_test)
    lr_o_predict = lr_o.predict(x_o_test)
    rf_o_predict = rf_o.predict(x_o_test)
    gb_o_predict = gb_o.predict(x_o_test)

    # evaluate for label encode
    svr_l_result = np.sqrt(mean_squared_error(y_l_test, svr_l_predict))
    lr_l_result = np.sqrt(mean_squared_error(y_l_test, lr_l_predict))
    rf_l_result = np.sqrt(mean_squared_error(y_l_test, rf_l_predict))
    gb_l_result = np.sqrt(mean_squared_error(y_l_test, gb_l_predict))

    # evaluate for one hot encode
    svr_o_result = np.sqrt(mean_squared_error(y_o_test, svr_o_predict))
    lr_o_result = np.sqrt(mean_squared_error(y_o_test, lr_o_predict))
    rf_o_result = np.sqrt(mean_squared_error(y_o_test, rf_o_predict))
    gb_o_result = np.sqrt(mean_squared_error(y_o_test, gb_o_predict))

    print("Printing for LabelEncoded Data")
    print("Test Error for SVR: ", svr_l_result)
    print("Test Error for LR: ", lr_l_result)
    print("Test Error for RFR: ", rf_l_result)
    print("Test Error for GBR: ", gb_l_result)

    print("Printing for OneHot Encoded")
    print("Test Error for SVR: ", svr_o_result)
    print("Test Error for SVR: ", lr_o_result)
    print("Test Error for SVR: ", rf_o_result)
    print("Test Error for SVR: ", gb_o_result)

def main():
    df = np.array(getData())
    learn(data)

main()