# LINKING 
from dotenv import load_dotenv
import os
load_dotenv()
ROOT = os.getenv("ROOT")
import sys
sys.path.append(ROOT)
# LINKING 

from DataGetters.economic import fetch_economic_data
from DataGetters.market import get_historical_data
from DataProcessors.sentiment import analyze_subreddit_sentiment
from DataProcessors.technical_indicators import get_selected_technical_indicators

import pandas as pd

def join_dataframes(market_df, economic_df, technical_df, sentiment_df):
    """
    Joins four data-indexed DataFrames (Market, Economic, Technical, Sentiment)
    into a single DataFrame based on their indices. The resulting DataFrame
    contains only the indices present in both Market and Technical DataFrames.
    
    Parameters:
    - market_df (pd.DataFrame): DataFrame with market data.
    - economic_df (pd.DataFrame): DataFrame with economic data.
    - technical_df (pd.DataFrame): DataFrame with technical indicators.
    - sentiment_df (pd.DataFrame): DataFrame with sentiment analysis results.
    
    Returns:
    - pd.DataFrame: A DataFrame containing merged data from all input DataFrames,
                    limited to the indices present in Market and Technical DataFrames.
    """
    # Perform inner joins
    # Start with Market and Technical as they define the indices to keep
    result_df = market_df.join(technical_df, how='inner')
    result_df = result_df.join(economic_df, how='inner')
    result_df = result_df.join(sentiment_df, how='inner')

    return result_df

# Example usage
if __name__ == "__main__":
    
    ticker_symbol = "AAPL"

    market = get_historical_data(ticker_symbol, interval="1d", start_date = '2024-01-01', end_date='2024-04-01')
    technical = get_selected_technical_indicators(market)
    economic = fetch_economic_data(start_date = '2024-01-01', end_date='2024-04-01')
    sentiment =  analyze_subreddit_sentiment('apple')

    # Join DataFrames
    combined_df = join_dataframes(market, economic, technical, sentiment)
    print(combined_df.columns)
    print(market.index)
    print(sentiment.index)
    print(combined_df)