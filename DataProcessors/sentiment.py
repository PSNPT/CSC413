# LINKING 
from dotenv import load_dotenv
import os
load_dotenv()
ROOT = os.getenv("ROOT")
import sys
sys.path.append(ROOT)
# LINKING 
  
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import torch
from DataGetters.reddit import fetch_monthly_top_posts_and_comments, fetch_top_posts_and_comments, fetch_yearly_top_posts_and_comments
import numpy as np
from datetime import datetime, timedelta
import pandas as pd

def setup_sentiment_model(model_name='yiyanghkust/finbert-tone'):
    """
    Initializes and returns a sentiment analysis pipeline with the specified model.
    
    Parameters:
    - model_name (str): The model identifier on Hugging Face's Model Hub.

    Returns:
    - A Hugging Face pipeline object for sentiment analysis.
    """
    # Load thee tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=device)
    
    return sentiment_pipeline

PIPELINE = setup_sentiment_model('yiyanghkust/finbert-tone')



def average_sentiment(sentiments):
    """
    Calculates the average sentiment from a list of sentiment tuples.
    
    Parameters:
    - sentiments (list of tuple): Each tuple contains ('label', score).
    
    Returns:
    - float: The average sentiment score.
    """
    # Convert sentiment labels to numerical values (+1, 0, -1)
    label_to_num = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
    
    # Calculate weighted sentiment score
    weighted_scores = [label_to_num[s['label']] * s['score'] for s in sentiments]
    
    # Compute average if the list is not empty
    if weighted_scores:
        avg_sentiment = sum(weighted_scores) / len(weighted_scores)
    else:
        avg_sentiment = 0  # Default to neutral (0) if no sentiments provided
    
    return avg_sentiment


def analyze_subreddit_sentiment(subreddit_name):
    """
    Fetches yearly top posts and comments from a specified subreddit, analyzes their sentiment,
    and prints the average trailing sentiment for each day.
    
    Parameters:
    - subreddit_name (str): The name of the subreddit to analyze.
    """
    # Calculate the dates for one month prior
    end_date = datetime.now()
    start_date = end_date - pd.DateOffset(years=1)

    # Generate a DatetimeIndex from start_date to end_date
    dates = pd.date_range(start=start_date, end=end_date, freq='D').date

    # Create an empty DataFrame with this DatetimeIndex
    df = pd.DataFrame(index=dates)
    df['sentiment'] = pd.NA

    df.index.name = 'Date'

    submissions_by_date = fetch_top_posts_and_comments(subreddit_name)
    averages = []
    
    for key in sorted(submissions_by_date):
        sentiments = PIPELINE(submissions_by_date[key])
        averages.append(average_sentiment(sentiments))
        # Calculate the trailing average sentiment for the last 3 days
        trailing_avg = np.mean(averages[-3:])
        print(f"Day {key}: Average Trailing Sentiment {trailing_avg}")
        
        # Add to df
        timestamp = datetime.strptime(key, '%Y-%m-%d').date()
        df.loc[timestamp, 'sentiment'] = trailing_avg

    # Estimate missing values
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    return df


# Examples
if __name__ == "__main__":
    # Subreddit and date for data fetching
    SUBREDDIT_NAME = 'finance'  # Example subreddit
    
    df = analyze_subreddit_sentiment(SUBREDDIT_NAME)
    print(df)