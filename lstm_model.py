import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import sqlite3

def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        X.append(a)
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

def train_lstm_model(symbol):
    conn = sqlite3.connect('trading_bot.db')
    cursor = conn.cursor()

    cursor.execute('''
        SELECT date, close FROM FinancialData
        WHERE symbol = ?
        ORDER BY date ASC
    ''', (symbol,))
    data = cursor.fetchall()
    conn.close()

    if len(data) > 60:  # Ensure there's enough data
        df = pd.DataFrame(data, columns=['date', 'close'])
        close_data = df['close'].values.reshape(-1, 1)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_data)

        time_step = 60
        X, Y = create_dataset(scaled_data, time_step)

        # Reshape input to be [samples, time steps, features]
        X = X.reshape(X.shape[0], X.shape[1], 1)

        # Split the data into training and testing sets
        train_size = int(len(X) * 0.7)
        X_train, Y_train = X[:train_size], Y[:train_size]

        # Build the LSTM model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model.fit(X_train, Y_train, batch_size=1, epochs=1)

        # Save the model
        model.save(f'{symbol}_lstm_model.h5')
        return model, scaler, time_step

    return None, None, None

def predict_stock_price(symbol, sentiment_score):
    model, scaler, time_step = train_lstm_model(symbol)

    if model:
        conn = sqlite3.connect('trading_bot.db')
        cursor = conn.cursor()

        cursor.execute('''
            SELECT date, close FROM FinancialData
            WHERE symbol = ?
            ORDER BY date DESC
            LIMIT ?
        ''', (symbol, time_step))
        recent_data = cursor.fetchall()
        conn.close()

        if len(recent_data) == time_step:
            df = pd.DataFrame(recent_data, columns=['date', 'close'])
            close_data = df['close'].values.reshape(-1, 1)
            scaled_data = scaler.transform(close_data)

            X_input = np.append(scaled_data, [[sentiment_score]], axis=0).reshape(1, -1, 1)

            predicted_price = model.predict(X_input)
            predicted_price = scaler.inverse_transform(predicted_price)
            return predicted_price[0][0]

    return None
