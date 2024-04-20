# Import necessary libraries
import yfinance as yf
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()
ROOT = os.getenv("ROOT")

def get_historical_data(ticker_symbol, period='1mo', interval='1d', start_date=None, end_date=None):
    """
    Fetches historical stock data for a given ticker symbol.

    Current API provider: Yahoo Finance

    Parameters:
    - ticker_symbol (str): The ticker symbol of the stock (e.g., 'AAPL' for Apple Inc.).
    - start_date (str): The start date for the data retrieval in 'YYYY-MM-DD' format.
    - end_date (str): The end date for the data retrieval in 'YYYY-MM-DD' format.

    Returns:
    - pandas.DataFrame: Historical stock data including Open, High, Low, Close, and Volume.
    """
    
    ticker = yf.Ticker(ticker_symbol)
    hist_data = ticker.history(period=period, interval=interval, start=start_date, end=end_date)
    try:
        hist_data.index = hist_data.index.date
    except:
        return None, True
    return hist_data[['Open', 'High', 'Low', 'Close', 'Volume']], False

# Examples
if __name__ == "__main__":

    ticker_symbol = "AAPL"

    data = get_historical_data(ticker_symbol, period='1y', interval='1d')
    print(data.index)
    print(data)
