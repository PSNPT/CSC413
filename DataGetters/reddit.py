import praw
from datetime import datetime, timedelta
import pytz
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
USER_AGENT = os.getenv("USER_AGENT")

def setup_praw():
    """
    Initializes and returns a PRAW instance.
    """
    return praw.Reddit(client_id=CLIENT_ID,
                       client_secret=CLIENT_SECRET,
                       user_agent=USER_AGENT)

def utc_to_date(utc):
    return datetime.utcfromtimestamp(utc).strftime('%Y-%m-%d')

def fetch_yearly_top_posts_and_comments(subreddit_name):
    """
    Fetches upto 1000 of the top posts within the year
    
    Parameters:
    - reddit: The PRAW Reddit instance.
    - subreddit_name (str): The name of the subreddit.
    
    Returns:
    - A dictionary where dates are keys and the submissions on those days
        are values
    """
    # Get submissions 
    reddit = setup_praw()

    subreddit = reddit.subreddit(subreddit_name)
    
    submissions = subreddit.top(time_filter="year", limit=1000)
    
    # Group submissions by date
    submissions_by_date = {}
    for submission in submissions:
        date = utc_to_date(submission.created_utc)
        if date in submissions_by_date:
            submissions_by_date[date].append(submission.title)
        else:
            submissions_by_date[date] = [submission.title]
    
    return submissions_by_date

def fetch_monthly_top_posts_and_comments(subreddit_name):
    """
    Fetches upto 1000 of the top posts within the month
    
    Parameters:
    - reddit: The PRAW Reddit instance.
    - subreddit_name (str): The name of the subreddit.
    
    Returns:
    - A dictionary where dates are keys and the submissions on those days
        are values
    """
    # Get submissions 
    reddit = setup_praw()

    subreddit = reddit.subreddit(subreddit_name)
    
    submissions = subreddit.top(time_filter="month", limit=1000)
    
    # Group submissions by date
    submissions_by_date = {}
    for submission in submissions:
        date = utc_to_date(submission.created_utc)
        if date in submissions_by_date:
            submissions_by_date[date].append(submission.title)
        else:
            submissions_by_date[date] = [submission.title]
    
    return submissions_by_date

def fetch_top_posts_and_comments(subreddit_name):
    print("Fetching yearly posts")
    yearly = fetch_yearly_top_posts_and_comments(subreddit_name)
    print("Fetching monthly posts")
    monthly = fetch_monthly_top_posts_and_comments(subreddit_name)

    combined = {}
    valid_keys = set(yearly.keys()).union(monthly.keys())
    for date in list(valid_keys):
        y = set()
        m = set()

        if date in yearly:
            y = set(yearly[date])
        if date in monthly:
            m = set(monthly[date])

        combined[date] = list(y.union(m))

    return combined

# Examples
if __name__ == "__main__":
    
    # Subreddit and date for data fetching
    SUBREDDIT_NAME = 'apple'  # Example subreddit
    
    submissions_by_date = fetch_top_posts_and_comments(SUBREDDIT_NAME)
    
    for key in sorted(submissions_by_date):
        print(f"{key}: {len(submissions_by_date[key])}")