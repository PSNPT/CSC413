# LINKING 
from dotenv import load_dotenv
import os
load_dotenv()
ROOT = os.getenv("ROOT")
import sys
sys.path.append(ROOT)
# LINKING
FEATURES = ['Close','Open', 'High', 'Low', 'Volume', 'SMA_30', 'EMA_14',
       'BBANDS_H', 'BBANDS_L', 'BBANDS_MAVG', 'VWAP', 'AROON_25', 'AROONU_25',
       'AROOND_25', 'RSI_14', 'StochRSI', 'MFI', 'Williams_R',
       '1-Year Real Interest Rate', '1-Year Expected Inflation',
       'Personal Saving Rate', 'Sticky Price Consumer Price Index',
       'Unemployment Rate', '5-Year Forward Inflation Expectation Rate',
       'sentiment', "resid_63","seasonal_63","trend_63","resid_252","seasonal_252","trend_252"]

from DataCombiner.combiner import join_dataframes
from DataGetters.economic import fetch_economic_data
from DataGetters.market import get_historical_data
from DataProcessors.sentiment import analyze_subreddit_sentiment
from DataProcessors.technical_indicators import get_selected_technical_indicators

import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import tensorflow as tf
from keras.layers import Layer
from tqdm import tqdm
from keras.models import Model, load_model
import pickle
import statsmodels.api as sm

class PositionalEncoding(Layer):
    def __init__(self, position, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.position = position
        self.d_model = d_model
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                     np.arange(d_model)[np.newaxis, :],
                                     d_model)
        # Apply sine to even indices in the array
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        # Apply cosine to odd indices in the array
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "position": self.position,
            "d_model": self.d_model
        })
        return config

class KerasTranspose(Layer):
    def call(self, x):
        return tf.transpose(x, perm=[0, 2, 1])
    
if __name__ == "__main__":
    ticker_symbol = input("Please input a ticker to predict on\n")
    wanted_predictions = int(input("Number of prior predictions you want statistics for\n"))

    print("Retrieving market data...")
    market, flag = get_historical_data(ticker_symbol, period='1y', interval='1d')
    print("Done...\n")
    if flag:
        print("Error with retrieving historical data")
        exit(1)
    
    print("Computing technical indicators data...")
    technical = get_selected_technical_indicators(market)
    print("Done...\n")

    # Retrieve both from cache if possible
    with open(ROOT + "Cache/date.txt", 'r') as f:
        date = f.readline().strip()
        timestamp = datetime.strptime(date, '%Y-%m-%d').date()
        print(f'Cache was updated on {timestamp}')

    if timestamp == datetime.now().date():
        print("Loading cached economic and sentiment data...")
        economic = pd.read_csv(ROOT + 'Cache/economic.csv', index_col='Date')
        sentiment = pd.read_csv(ROOT + 'Cache/sentiment.csv', index_col='Date')

        economic.index = pd.to_datetime(economic.index).date
        sentiment.index = pd.to_datetime(sentiment.index).date

        print("Done...\n")
    else:
        print("Retrieving economic data...")
        economic = fetch_economic_data(start_date = market.index[0], end_date=market.index[-1])
        print("Done...\n")

        print("Retrieving sentiment data...")
        sentiment =  analyze_subreddit_sentiment('finance')
        print("Done...\n")
        
        # Cache economic and sentiment
        economic.to_csv(ROOT + 'Cache/economic.csv')
        sentiment.to_csv(ROOT + 'Cache/sentiment.csv')
        
        with open(ROOT + "Cache/date.txt", 'w') as f:
            date = datetime.now().date()
            f.write(date.strftime('%Y-%m-%d'))

    # Convert market to percent change
    for name in market.columns:
        market[name] = market[name].pct_change() * 100

    # Join DataFrames
    combined_df = join_dataframes(market, economic, technical, sentiment)
    combined_df.ffill(inplace=True)
    combined_df.bfill(inplace=True)

    # Add Season, Trend, Residual
    temp, _ = get_historical_data(ticker_symbol, period='5y', interval='1d')
    for name in temp.columns:
        temp[name] = temp[name].pct_change() * 100
    temp.ffill(inplace=True)
    temp.bfill(inplace=True)
    
    for period in [63,252]:
        stl = sm.tsa.STL(temp['Close'], period=period)
        result = stl.fit()
        combined_df[f'resid_{period}'] = result.resid[-len(combined_df.index):]
        combined_df[f'seasonal_{period}'] = result.seasonal[-len(combined_df.index):]
        combined_df[f'trend_{period}'] = result.trend[-len(combined_df.index):]

    # Fix order
    combined_df = combined_df[FEATURES]

    # Convert to tensor
    data = combined_df.to_numpy()
    combined_df.to_csv('t')

    # Transform
    with open(ROOT + f'Scalers/scaler_{len(FEATURES)}_IQR_ROBUST.pkl', 'rb') as file:
        scaler = pickle.load(file)

    data = scaler.transform(data)

    # Keep relevant parts
    current = np.array([data[-120:]])
    
    history_x = []
    history_y = []
    for i in range(1,wanted_predictions+1):
        window = data[-120 - i: -i]

        history_x.append(window)
    history_y = (data[-wanted_predictions:, 0] > 0).astype(int)
    history_y = np.array(history_y)
    history_x = np.array(history_x)
    print(history_x.shape, history_y.shape)
    # Load models
    LSTM = load_model(ROOT + "Models/LSTM.keras")
    CNN = load_model(ROOT + "Models/CNN.keras")
    TRANSFORMER = load_model(ROOT + "Models/TRANSFORMER.keras", custom_objects={'PositionalEncoding': PositionalEncoding, 'KerasTranspose': KerasTranspose})
    SPACE = load_model(ROOT + "Models/SPACE.keras", custom_objects={'PositionalEncoding': PositionalEncoding, 'KerasTranspose': KerasTranspose})

    MODELS = {"LSTM":LSTM, "CNN": CNN, "TRANSFORMER": TRANSFORMER, "SPACE": SPACE}

    for key in MODELS:
        print(f"Using {key}")
        prediction = MODELS[key].predict(current)
        print(f"Current prediction : {prediction[0]}")

        if wanted_predictions > 0:
            mets = MODELS[key].evaluate(history_x, history_y)
            print(f"Historical Accuracy : {mets[1]}")
