# LINKING 
from dotenv import load_dotenv
import os
load_dotenv()
ROOT = os.getenv("ROOT")
import sys
sys.path.append(ROOT)
# LINKING

from DataCombiner.combiner import join_dataframes
from DataGetters.economic import fetch_economic_data
from DataGetters.market import get_historical_data
from DataProcessors.sentiment import analyze_headlines_sentiment, average_sentiment, setup_sentiment_model
from DataProcessors.technical_indicators import get_selected_technical_indicators
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from tqdm import tqdm
import statsmodels.api as sm
def utc_to_date(utc):
    return datetime.utcfromtimestamp(utc).strftime('%Y-%m-%d')

#TICKERS = ['MSFT', 'AAPL', 'NVDA', 'GOOG', 'AMD', 'AMZN', 'INTC']
#TICKERS = ['^GSPC']
#TICKERS = ['AVGO','JPM','XOM','V', 'MA', 'PG', 'HD', 'COST', 'PEP', 'KO' , 'ADBE', 'CSCO']
tickers = pd.read_html(
    'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
TICKERS = tickers['Symbol'].tolist()
BAD = []
print(TICKERS)
SUBREDDIT = 'finance'
START_DATE = '2010-01-01'
END_DATE = '2020-12-31'

def save_market():

    for ticker_symbol in TICKERS:
        print(f"Retrieving market data for {ticker_symbol}...")
        market, flag = get_historical_data(ticker_symbol, interval='1d', start_date=START_DATE, end_date=END_DATE)
        if flag:
            print(f"No data {ticker_symbol}")
            BAD.append(ticker_symbol)
            continue
        if market.index[0] > datetime.strptime('2010-01-04', '%Y-%m-%d').date():
            print(f"{market.index[0]} was the earliest. {ticker_symbol}")
            BAD.append(ticker_symbol)
            continue
        market.to_csv(ROOT + f'Data/Historical/{ticker_symbol}.csv')
        print("Done...\n")

        save_technical(market, ticker_symbol)

def save_technical(market, ticker_symbol):

    print(f"Computing technical indicators data for {ticker_symbol}...")
    technical = get_selected_technical_indicators(market)
    technical.to_csv(ROOT + f'Data/Technical/{ticker_symbol}.csv')
    print("Done...\n")

def save_economic():
    print("Retrieving economic data...")
    economic = fetch_economic_data(start_date = START_DATE, end_date=END_DATE)
    economic.to_csv(ROOT + f'Data/Economic/economic.csv')
    print("Done...\n")


def save_semantic():

    # Generate the index
    dates_i = sorted(pd.date_range(start=START_DATE, end=END_DATE, freq='D').date)
    dates = pd.date_range(start=START_DATE, end=END_DATE, freq='D').strftime('%Y-%m-%d')

    # Initialize date-submissions dictionary
    submissions_by_date = {date: [] for date in dates}

    # Read relevant submissions file 
    print("Loading submissions...\n")
    with open(f"./Data/Reddit/{SUBREDDIT}_submissions", 'r', encoding="utf8") as f:
        for line in f:
            obj = json.loads(line)
            date = utc_to_date(int(obj['created_utc']))
            title = obj['title']

            if date in submissions_by_date:
                submissions_by_date[date].append(title)

    # Setup model once
    print("Setting up FinBERT...\n")
    sentiment_pipeline = setup_sentiment_model('yiyanghkust/finbert-tone')

    df = pd.DataFrame(index=dates_i)
    df['sentiment'] = pd.NA
    df.index.name = 'Date'

    averages = []
    print("Computing sentiment scores...\n")
    empty = 0
    for date in tqdm(sorted(dates), total=len(dates)):

        if submissions_by_date[date] == []:
            empty += 1
            sentiments = []
            averages.append(0)
        else:
            sentiments = sentiment_pipeline(submissions_by_date[date])
            averages.append(average_sentiment(sentiments))

        # Calculate the trailing average sentiment for the last 3 days
        trailing_avg = np.mean(averages[-3:])

        # Add to df
        timestamp = datetime.strptime(date, '%Y-%m-%d').date()
        df.loc[timestamp, 'sentiment'] = trailing_avg

    # Estimate missing values
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    print("Saving...\n")
    # Saving
    df.to_csv(ROOT + f'Data/Sentiment/{SUBREDDIT}.csv')

def convert_index(df):
    df.index = pd.to_datetime(df.index).date
    df.index.name = 'Date'

def load_and_combine():
    for ticker_symbol in TICKERS:
        if ticker_symbol in BAD:
            print(f"Skipping {ticker_symbol}")
            continue
        # Load in all
        print(f"Combining and saving for {ticker_symbol}...\n")

        market = pd.read_csv(ROOT + f'Data/Historical/{ticker_symbol}.csv', index_col=0)
        convert_index(market)

        technical = pd.read_csv(ROOT + f'Data/Technical/{ticker_symbol}.csv', index_col=0)
        convert_index(technical)

        economic = pd.read_csv(ROOT + f'Data/Economic/economic.csv', index_col=0)
        convert_index(economic)

        sentiment = pd.read_csv(ROOT + f'Data/Sentiment/{SUBREDDIT}.csv', index_col=0)
        convert_index(sentiment)

        # Convert market to percent change
        for name in market.columns:
            market[name] = market[name].pct_change() * 100

        # Join
        combined = join_dataframes(market, economic, technical, sentiment)

        # 2011-2019
        start = datetime.strptime('2011-01-03', '%Y-%m-%d').date()
        end = datetime.strptime('2019-12-31', '%Y-%m-%d').date()
        combined = combined.loc[start:end]
        combined.to_csv(ROOT + f'Data/Combined/{ticker_symbol}.csv')

    for file in os.listdir(ROOT + f"Data/Combined"):
        df = pd.read_csv(ROOT + f"Data/Combined/{file}", index_col=0)
        for period in [63,252]:
            stl = sm.tsa.STL(df['Close'], period=period)
            result = stl.fit()
            df[f'resid_{period}'] = result.resid
            df[f'seasonal_{period}'] = result.seasonal
            df[f'trend_{period}'] = result.trend
        df.to_csv(ROOT+f"Data/TEMP/{file}")

if __name__ == '__main__':
    save_market()
    save_economic()
    save_semantic()
    load_and_combine()