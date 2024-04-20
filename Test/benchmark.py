import os


# LINKING 
from dotenv import load_dotenv
load_dotenv()
ROOT = os.getenv("ROOT")
import sys
sys.path.append(ROOT)
# LINKING

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tqdm import tqdm
from tensorflow.keras.models import Model, load_model
import pickle

FEATURES = ['Close','Open', 'High', 'Low', 'Volume', 'SMA_30', 'EMA_14',
       'BBANDS_H', 'BBANDS_L', 'BBANDS_MAVG', 'VWAP', 'AROON_25', 'AROONU_25',
       'AROOND_25', 'RSI_14', 'StochRSI', 'MFI', 'Williams_R',
       '1-Year Real Interest Rate', '1-Year Expected Inflation',
       'Personal Saving Rate', 'Sticky Price Consumer Price Index',
       'Unemployment Rate', '5-Year Forward Inflation Expectation Rate',
       'sentiment', "resid_63","seasonal_63","trend_63","resid_252","seasonal_252","trend_252"]

def construct_dataset(FEATURES,  seq_length, TICKERS):

    X, Y = [], []

    for ticker_symbol in tqdm(TICKERS):
        data = pd.read_csv(ROOT + f"Data/TEMP/{ticker_symbol}.csv", index_col=0)
        data = data[FEATURES]
        xs = []
        ys = []

        # Check for infinite values
        if data.isin([np.inf, -np.inf]).any().any():
            print(f"\nSkipping {ticker_symbol}, inf detected\n")
            continue

        for i in range(len(data)-seq_length-1):
            x = data.iloc[i:(i+seq_length)]
            y = (data.iloc[i+seq_length, data.columns.get_loc('Close')] > 0).astype(int)

            xs.append(x)
            ys.append(y)

        X.extend(np.array(xs))
        Y.extend(np.array(ys))


    return np.array(X), np.array(Y)

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
    
def print_named_statistics(model_name, model, statistics, metrics):
    print(f"Statistics for {model_name}")
    for i, name in enumerate(metrics):
        print(f"{name} : {statistics[i]}")
    print("\n")

if __name__ == "__main__":
    TIME_STEP = 120
    METRICS = ['loss', 'accuracy', "AUC", "precision", "recall"]

    with open(ROOT + f'Scalers/scaler_{len(FEATURES)}_IQR_ROBUST.pkl', 'rb') as file:
        scaler = pickle.load(file)
    print(f"Constructing Dataset")
    for TICKER in ['^GSPC']:
        print(TICKER)
        TICKERS = [TICKER]
        X, y = construct_dataset(FEATURES, TIME_STEP, TICKERS)
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}\n")

        flattened_X =  X.reshape(-1, X.shape[2])
        flattened_X = scaler.transform(flattened_X)
        X = flattened_X.reshape(y.shape[0], TIME_STEP, len(FEATURES))

        print(f"Tranforming and scaling using: Scalers/scaler_{len(FEATURES)}_IQR_ROBUST.pkl")

        LSTM = load_model(ROOT + "Models/LSTM.keras")
        CNN = load_model(ROOT + "Models/CNN.keras")
        TRANSFORMER = load_model(ROOT + "Models/TRANSFORMER.keras", custom_objects={'PositionalEncoding': PositionalEncoding, 'KerasTranspose': KerasTranspose})
        SPACE = load_model(ROOT + "Models/SPACE.keras", custom_objects={'PositionalEncoding': PositionalEncoding, 'KerasTranspose': KerasTranspose})
        
        STATISTICS = []
        NAMES = ["LSTM", "CNN", "TRANSFORMER", "SPACE"]
        MODELS = [LSTM, CNN, TRANSFORMER, SPACE]
        STATISTICS.append(LSTM.evaluate(X, y))
        STATISTICS.append(CNN.evaluate(X, y))
        STATISTICS.append(TRANSFORMER.evaluate(X, y))
        STATISTICS.append(SPACE.evaluate(X, y))

        for NAME, MODEL, STAT in zip(NAMES, MODELS, STATISTICS):
            print_named_statistics(NAME, MODEL, STAT, METRICS)
        
        # Export table
        df = pd.DataFrame(index=NAMES, columns=METRICS)
        for i, name in enumerate(NAMES):
            df.loc[name] = STATISTICS[i]

        s = df.style \
        .set_caption("Performance Metrics") \
        .format(precision=3, thousands=".", decimal=",") \
        .format_index(str.upper, axis=1)

        s.to_latex(ROOT + "Test/pdlatex")

        print(df)
