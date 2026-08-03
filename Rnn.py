#!/usr/bin/env python
# coding: utf-8

# ### MasterCard Stock Price Prediction Using LSTM
# We use Kaggle's MasterCard stock dataset from May-25-2006 to Oct-11-2021 and train an LSTM model to forecast the stock price.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.utils import set_random_seed

DATA_PATH = "data/Mastercard_stock_history.csv"
TICKER = "MA"
START = "2010-01-01"
TSTART = 2016
TEND = 2020
N_STEPS = 60
FEATURES = 1
EPOCHS = 50
BATCH_SIZE = 32
SEED = 455


def download_stock_data(ticker=TICKER, start=START, end=None, save_path=DATA_PATH):
    """Download historical stock data and save it in the format the model expects."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    stock = yf.Ticker(ticker)
    df = stock.history(start=start, end=end, actions=True)
    if df.empty:
        raise ValueError(
            f"No data returned for ticker '{ticker}' — check the symbol or date range."
        )
    df.index.name = "Date"
    df.to_csv(save_path)
    print(f"Saved {len(df)} rows to {save_path}")
    return df


def load_dataset(path=DATA_PATH):
    """Load the CSV and drop the columns the model does not use."""
    return pd.read_csv(path, index_col="Date", parse_dates=["Date"]).drop(
        ["Dividends", "Stock Splits"], axis=1
    )


def train_test_plot(dataset, tstart=TSTART, tend=TEND):
    """Plot train vs test split of the High column."""
    dataset.loc[f"{tstart}":f"{tend}", "High"].plot(figsize=(16, 4), legend=True)
    dataset.loc[f"{tend+1}":, "High"].plot(figsize=(16, 4), legend=True)
    plt.legend([f"Train (Before {tend+1})", f"Test ({tend+1} and beyond)"])
    plt.title("MasterCard stock price")
    plt.show()


def train_test_split(dataset, tstart=TSTART, tend=TEND):
    """Split the High column into train (tstart..tend) and test (tend+1..) arrays."""
    train = dataset.loc[f"{tstart}":f"{tend}", "High"].values
    test = dataset.loc[f"{tend+1}":, "High"].values
    return train, test


def split_sequence(sequence, n_steps):
    """Turn a 1D sequence into (X, y) supervised-learning samples."""
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def build_lstm_model(n_steps, features):
    """Build and compile the LSTM model."""
    model = Sequential()
    model.add(LSTM(units=125, activation="tanh", input_shape=(n_steps, features)))
    model.add(Dense(units=1))
    model.compile(optimizer="RMSprop", loss="mse")
    return model


def predict_test(model, dataset, scaler, test_set):
    """Build the test samples and return inverse-transformed predictions."""
    dataset_total = dataset.loc[:, "High"]
    inputs = dataset_total[len(dataset_total) - len(test_set) - N_STEPS :].values
    inputs = inputs.reshape(-1, 1)
    inputs = scaler.transform(inputs)
    X_test, _ = split_sequence(inputs, N_STEPS)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], FEATURES)
    predicted = model.predict(X_test)
    return scaler.inverse_transform(predicted)


def plot_predictions(test, predicted):
    plt.plot(test, color="gray", label="Real")
    plt.plot(predicted, color="red", label="Predicted")
    plt.title("MasterCard Stock Price Prediction")
    plt.xlabel("Time")
    plt.ylabel("MasterCard Stock Price")
    plt.legend()
    plt.show()


def return_rmse(test, predicted):
    rmse = np.sqrt(mean_squared_error(test, predicted))
    print("The root mean squared error is {:.2f}.".format(rmse))


def main():
    set_random_seed(SEED)
    np.random.seed(SEED)

    if not os.path.exists(DATA_PATH):
        download_stock_data()

    dataset = load_dataset()
    print(dataset.head())
    print(dataset.describe())
    print(dataset.isna().sum())

    train_test_plot(dataset)

    training_set, test_set = train_test_split(dataset)

    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set.reshape(-1, 1))

    X_train, y_train = split_sequence(training_set_scaled, N_STEPS)
    X_train = X_train.reshape(X_train.shape[0], N_STEPS, FEATURES)

    model = build_lstm_model(N_STEPS, FEATURES)
    model.summary()

    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)

    predicted_stock_price = predict_test(model, dataset, sc, test_set)

    plot_predictions(test_set, predicted_stock_price)
    return_rmse(test_set, predicted_stock_price)


if __name__ == "__main__":
    main()
