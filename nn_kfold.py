# NOTE: Validation error is very low due to the fact that we have also introduced dropout layer along with k-fold
# cross-validation

import csv
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn import preprocessing
import pickle
import tensorflow as tf
import tensorflow.contrib.slim as slim


def getData():
    with open('process/process_file_3.csv') as csv_file:
        reader = csv.reader(csv_file)
        data = list(reader)

    return data


def preprocessData(data):
    # check different file
    label_encoder = preprocessing.LabelEncoder()

    data[:, 0] = label_encoder.fit_transform(data[:, 0])
    data = data.astype(float)
    processed_data = data[:, 2:6]

    processed_data = preprocessing.normalize(processed_data)
    np.random.shuffle(processed_data)

    return processed_data


def learn(data):
    data = preprocessData(data)
    num_params = data.shape[1] - 1

    X = data[:, 0:num_params]
    Y = data[:, num_params].reshape(-1, 1)

    # split it into train and test first, then into 5 fold cross validation

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
    # X_test and Y_test will be true unseen test material, where as performing
    # k-fold on training sample will give us a validation set
    kf = KFold(n_splits=5, shuffle=False, random_state=None)
    kf.get_n_splits(X_train)
    test_results = []

    # Define Network
    # Get the initial stock prices for computing the relative cost
    stock_data = tf.placeholder(tf.float32, [None, num_params])
    opening_price = tf.placeholder(tf.float32, [None, 1])
    stock_price = tf.placeholder(tf.float32, [None, 1])

    # Number of neurons in the hidden layer
    n_hidden_1 = 3
    n_hidden_2 = 3

    weights = {
        'out': tf.Variable(tf.random_normal([n_hidden_2, 1]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([1]))
    }

    # Implement dropout to reduce overfitting
    keep_prob_input = tf.placeholder(tf.float32)
    keep_prob_hidden = tf.placeholder(tf.float32)

    # Hidden layers
    input_dropout = tf.nn.dropout(stock_data, keep_prob_input)
    layer_1 = slim.fully_connected(input_dropout, n_hidden_1, biases_initializer=None, activation_fn=tf.nn.relu)
    layer_1_dropout = tf.nn.dropout(layer_1, keep_prob_hidden)
    layer_2 = slim.fully_connected(input_dropout, n_hidden_1, biases_initializer=None, activation_fn=tf.nn.relu)
    layer_2_dropout = tf.nn.dropout(layer_2, keep_prob_hidden)
    # regression  layer = (w'x+b)

    output_layer = tf.add(tf.matmul(layer_2_dropout, weights['out']), biases['out'])

    learning_rate = 1e-4
    cost_function = tf.reduce_mean(tf.pow(tf.div(tf.subtract(stock_price, output_layer), opening_price), 2))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost_function)

    last_cost = 0
    tolerance = 1e-6
    epochs = 1
    max_epochs = 1e6

    train_dict = {}
    test_dict = {}
    validation_dict = {}
    fold_num = 0
    for train_index, test_index in kf.split(X_train):
        train_X, test_X = X_train[train_index], X_train[test_index]
        train_Y, test_Y = Y_train[train_index], Y_train[test_index]

        train_opening_price = train_X[:, num_params - 1].reshape(-1, 1)
        validation_opening_price = test_X[:, num_params - 1].reshape(-1, 1)
        test_opening_price = X_test[:, num_params - 1].reshape(-1, 1)

        sess = tf.Session()
        with sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            history_train = []
            validation_test = []

            while True:
                sess.run(optimizer,
                         feed_dict={stock_data: train_X, opening_price: train_opening_price, stock_price: train_Y,
                                    keep_prob_input: 0.8, keep_prob_hidden: 0.5})

                if epochs % 100 == 0:
                    cost = sess.run(cost_function, feed_dict={stock_data: train_X, opening_price: train_opening_price,
                                                              stock_price: train_Y, keep_prob_input: 0.8,
                                                              keep_prob_hidden: 0.5})
                    history_train.append(cost)
                    print("Epoch: %d: Training Error: %f" % (epochs, cost))

                    if abs(cost - last_cost) <= tolerance or epochs > max_epochs:
                        print("Converged.")
                        break

                    last_cost = cost

                    print("Validation error: ",
                          sess.run(cost_function,
                                   feed_dict={stock_data: test_X, opening_price: validation_opening_price,
                                              stock_price: test_Y, keep_prob_input: 1.0,
                                              keep_prob_hidden: 1.0}))
                    validation_results = sess.run(output_layer,
                                                  feed_dict={stock_data: test_X, stock_price: test_Y,
                                                             keep_prob_input: 1.0,
                                                             keep_prob_hidden: 1.0})
                    validation_test.append(validation_results)

                epochs += 1

            # check with True test results
            print("Test error: ",
                  sess.run(cost_function, feed_dict={stock_data: X_test, opening_price: test_opening_price,
                                                     stock_price: Y_test, keep_prob_input: 1.0,
                                                     keep_prob_hidden: 1.0}))
            test_result = sess.run(output_layer,
                                   feed_dict={stock_data: X_test, stock_price: Y_test, keep_prob_input: 1.0,
                                              keep_prob_hidden: 1.0})
            test_results.append(test_result)

        train_dict[fold_num] = history_train
        validation_dict[fold_num] = validation_test
        test_dict[fold_num] = test_results

        avg_perc_error = 0
        max_perc_error = 0
        mei = 0
        for i in range(len(Y_test)):
            # Actual stock market change between opening and closing prices
            actual_change = abs(Y_test[i][0] - X_test[i][num_params - 1]) / X_test[i][num_params - 1]
            predicted_change = abs(test_result[i][0] - X_test[i][num_params - 1]) / X_test[i][num_params - 1]
            delta = abs(actual_change - predicted_change)
            avg_perc_error = avg_perc_error + delta
            if delta > max_perc_error:
                max_perc_error = delta
                mei = i

        avg_perc_error = (avg_perc_error * 100) / len(test_Y)
        max_perc_error *= 100
        print("Maximum percentage error: %f\nAverage percentage error: %f\n" % (max_perc_error, avg_perc_error))

        fold_num += 1

    print('Average Test Error from all folds = ', np.mean(test_results))

    # Dumping to pickle for future use, plotting
    pk_train = open('Train_Error_KFold_LabelEncode_3.pkl', 'ab')
    pk_val = open('Validation_Error_KFold_LabelEncode_3.pkl', 'ab')
    pk_test = open('Test_Error_KFold_LabelEncode_3.pkl', 'ab')
    pickle.dump(train_dict, pk_train)
    pickle.dump(test_dict, pk_test)
    pickle.dump(validation_dict, pk_val)
    pk_test.close()
    pk_train.close()
    pk_val.close()

def main():
    data = np.array(getData())
    learn(data)


main()
