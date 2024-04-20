# LINKING 
from dotenv import load_dotenv
import os
load_dotenv()
ROOT = os.getenv("ROOT")
import sys
sys.path.append(ROOT)
# LINKING 

import pandas as pd
from ta.momentum import RSIIndicator, StochRSIIndicator, WilliamsRIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator, AroonIndicator
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice, MFIIndicator
import numpy as np
from DataGetters.market import get_historical_data
import matplotlib.pyplot as plt

def get_selected_technical_indicators(df):
    """
    Returns a DataFrame with selected technical indicators using daily data.
    
    Parameters:
    - df (pd.DataFrame): DataFrame with columns 'Open', 'High', 'Low', 'Close', 'Volume'.
    
    Returns:
    - df (pd.DataFrame): DataFrame with selected technical indicators.
    """

    # Ensure the DataFrame has required columns
    if not set(['Open', 'High', 'Low', 'Close', 'Volume']).issubset(df.columns):
        raise ValueError("DataFrame must contain Open, High, Low, Close, Volume columns.")
    
    # Empty df with orignals index
    dfr = pd.DataFrame(index=df.index)
    
    # Add Simple Moving Average (SMA)
    dfr['SMA_30'] = SMAIndicator(close=df['Close'], window=30).sma_indicator()
    
    # Add Exponential Moving Average (EMA)
    dfr['EMA_14'] = EMAIndicator(close=df['Close'], window=14).ema_indicator()

    # Add Bollinger Bands
    bb = BollingerBands(close=df['Close'])
    dfr['BBANDS_H'] = bb.bollinger_hband()
    dfr['BBANDS_L'] = bb.bollinger_lband()
    dfr['BBANDS_MAVG'] = bb.bollinger_mavg()
    
    # Add Volume Weighted Average Price (VWAP) - Daily
    dfr['VWAP'] = VolumeWeightedAveragePrice(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'], window=14).volume_weighted_average_price()
    
    # Calculate percent change for each indicator and replace
    dfr.bfill(inplace=True)
    for name in dfr.columns:
        dfr[name] = dfr[name].pct_change() * 100

    # Add AROON
    AROON = AroonIndicator(high=df['High'], low=df['Low'], window=25)
    dfr['AROON_25'] = AROON.aroon_indicator()
    dfr['AROONU_25'] = AROON.aroon_down()
    dfr['AROOND_25'] = AROON.aroon_up()

    # Add Relative Strength Index (RSI)
    dfr['RSI_14'] = RSIIndicator(close=df['Close'], window=14).rsi()

    
    # Add Stochastic RSI (StochRSI)
    stoch_rsi = StochRSIIndicator(close=df['Close'], window=14, smooth1=3, smooth2=3)
    dfr['StochRSI'] = stoch_rsi.stochrsi()
    
    # Add MFI
    dfr['MFI'] = MFIIndicator(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'], window=14).money_flow_index()
    
    # Add Williams %R
    dfr['Williams_R'] = WilliamsRIndicator(high=df['High'], low=df['Low'], close=df['Close'], lbp=14).williams_r()

    # Fill NaN values
    dfr.bfill(inplace=True)
    
    return dfr

# Example Usage
if __name__ == "__main__":

    # Get raw data
    ticker_symbol = "AAPL"
    data = get_historical_data(ticker_symbol, period="3mo", interval="1d")
    print(data)

    # Get technical indicators
    ti_df = get_selected_technical_indicators(data)
    print(ti_df)
