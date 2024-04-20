#############################################################################################
#############################################################################################
#############################################################################################
# IMPORTS
#############################################################################################
#############################################################################################
#############################################################################################

# LINKING 
from dotenv import load_dotenv
import os
load_dotenv()
ROOT = os.getenv("ROOT")
import sys
sys.path.append(ROOT)
# LINKING

import tensorflow as tf
from tensorflow import keras
from keras.callbacks import ModelCheckpoint
import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dropout, Dense, LayerNormalization, MultiHeadAttention, Add, Flatten, Activation
from keras.optimizers import Adam
import tensorflow as tf
from keras.layers import Layer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.layers import Conv1D
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from keras import layers, models
import tensorflow as tf
from keras import layers, models
from keras.layers import BatchNormalization
from keras.models import load_model
from tqdm import tqdm
from scipy import stats
#############################################################################################
#############################################################################################
#############################################################################################
# IMPORTS
#############################################################################################
#############################################################################################
#############################################################################################

def culled_data(data, threshold=5):
    # Calculate Z-scores for each column
    z_scores = stats.zscore(data, axis=0)

    # Find indices where Z-score is greater than 3 in absolute value
    outlier_indices = np.any(z_scores > threshold, axis=1)

    # Print number of rows with outliers
    outliers = sum(outlier_indices)
    total = data.shape[0]

    pct = np.round((outliers/total) * 100, 2)
    print(f"Percentage of outliers : {pct}")


    # Filter out the rows with outliers
    return data[~outlier_indices]


def culled_data_iqr(data, threshold=3):
    """
    Remove rows with outliers based on the Interquartile Range (IQR) method.

    Parameters:
    data (numpy.ndarray): Input data array.
    threshold (float): Multiplier for IQR to adjust strictness of outlier definition. Default is 1.5.

    Returns:
    numpy.ndarray: Data array after removing rows with outliers.
    """
    # Calculate Q1 (25th percentile) and Q3 (75th percentile) for each column
    Q1 = np.percentile(data, 25, axis=0)
    Q3 = np.percentile(data, 75, axis=0)
    
    # Calculate the IQR (interquartile range)
    IQR = Q3 - Q1
    
    # Calculate outlier cutoffs
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    

    outliers = np.sum((data < lower_bound) | (data > upper_bound) ,axis=0)
    total = data.shape[0]
    pct_outliers = np.round((outliers / total) * 100, 2)

    stats_df = pd.DataFrame({
    'IQR': IQR,
    'LB': lower_bound,
    'UB': upper_bound,
    'OUT' : pct_outliers,
    }, index=FEATURES)

    # Print the DataFrame
    print(stats_df)
    
    # Determine a mask for rows without outliers
    no_outlier_mask = ~((data < lower_bound) | (data > upper_bound)).any(axis=1)

    # Count outliers and print statistics
    outliers = np.invert(no_outlier_mask).sum()
    total = data.shape[0]
    pct_outliers = np.round((outliers / total) * 100, 2)
    print(f"Percentage of outlier rows: {pct_outliers}%")

    # Filter and return data without outliers
    return data[no_outlier_mask]

#############################################################################################
#############################################################################################
#############################################################################################
# HELPER
#############################################################################################
#############################################################################################
#############################################################################################
print(tf.config.list_physical_devices('GPU'))

print(f"Constructing Dataset")
TICKERS = []
TRAIN_X = []
TRAIN_Y = []
TEST_X = []
TEST_Y = []
USED = []

TIME_STEP = 1

for file in os.listdir(ROOT + f"Data/Combined"):
    TICKERS.append(file[:-4])
print(len(TICKERS))

FEATURES = ['Close', 'Open', 'High', 'Low','Volume',
            'SMA_30', 'EMA_14','BBANDS_H', 'BBANDS_L',
            'BBANDS_MAVG', 'VWAP', 'AROON_25', 'AROONU_25', 'AROOND_25', 'RSI_14', 'StochRSI', 'MFI', 'Williams_R',
            '1-Year Real Interest Rate', '1-Year Expected Inflation',
            'Personal Saving Rate', 'Sticky Price Consumer Price Index',
            'Unemployment Rate', '5-Year Forward Inflation Expectation Rate',
            'sentiment',
            "resid_63","seasonal_63","trend_63","resid_252","seasonal_252","trend_252"]

# FEATURES = [
#     'Close',
#     'SMA_30', 'BBANDS_MAVG', 'VWAP', 'RSI_14', 'StochRSI',
#     '1-Year Real Interest Rate', '1-Year Expected Inflation','Personal Saving Rate','Unemployment Rate',
#     'sentiment',
#     "seasonal_63", "trend_63", "trend_252"] # 14

for ticker_symbol in tqdm(TICKERS):

    data_path = f"{ROOT}/Data/TEMP/{ticker_symbol}.csv"
    data = pd.read_csv(data_path, index_col=0)
    if data.isin([np.inf, -np.inf]).any().any():
        print(f"\nSkipping {ticker_symbol}, inf detected\n")
        continue
    data = data[FEATURES]
    
    USED.append(ticker_symbol)
    # Keep the first 90% of the rows for train
    num_rows = int(len(data) * 0.9)
    
    train = data.iloc[:num_rows]
    TRAIN_X.append(train.to_numpy())
    TRAIN_Y.append((train.iloc[TIME_STEP:, data.columns.get_loc('Close')] > 0).astype(int).to_numpy())

    test = data.iloc[num_rows:]
    TEST_X.append(test.to_numpy())
    TEST_Y.append((test.iloc[TIME_STEP:, data.columns.get_loc('Close')] > 0).astype(int).to_numpy())
    
print(f"USED: {len(USED)}")
print(f"Stacking and scaling")
X_train = np.vstack(TRAIN_X)
X_test = np.vstack(TEST_X)
print(X_train.shape)

BASE  = X_train
CULLED_IQR = culled_data_iqr(X_train)
CULLED =  culled_data(X_train)

MM  = MinMaxScaler((-1,1))
ST  = StandardScaler()
RB = RobustScaler()

SRCS = {'BASE': BASE, 'IQR': CULLED_IQR, 'Z': CULLED}
SCLS = {"MINMAX" : MM, "STANDARD" : ST, "ROBUST": RB}

import pickle
for SRC_KEY in SRCS:
    for SCL_KEY in SCLS:
        SRC = SRCS[SRC_KEY]
        SCL = SCLS[SCL_KEY]

        FITTED = SCL.fit(SRC)

        with open(ROOT + f'Scalers/scaler_{len(FEATURES)}_{SRC_KEY}_{SCL_KEY}.pkl', 'wb') as file:
            pickle.dump(FITTED, file)
