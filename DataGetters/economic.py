import pandas as pd
import pandas_datareader.data as web
from datetime import datetime

def fetch_economic_data(start_date, end_date):
    """
    Fetches economic data from FRED and returns a datetime indexed DataFrame.
    
    Parameters:
    - start_date (str): The start date for the data retrieval in 'YYYY-MM-DD' format.
    - end_date (str): The end date for the data retrieval in 'YYYY-MM-DD' format.
    
    Returns:
    - pd.DataFrame: DataFrame containing the economic data indexed by date.
    """
    
    # Define the economic indicators to fetch
    indicators = {
        'REAINTRATREARAT1YE' : '1-Year Real Interest Rate',
        'EXPINF1YR' : '1-Year Expected Inflation',
        'PSAVERT' : 'Personal Saving Rate',
        'STICKCPIM157SFRBATL' : 'Sticky Price Consumer Price Index',
        'UNRATE': 'Unemployment Rate',
        'T5YIFR': '5-Year Forward Inflation Expectation Rate'
    }
    
    # Initialize an empty DataFrame
    df_economic = pd.DataFrame()
    
    for code, name in indicators.items():
        df_temp = web.DataReader(code, 'fred', start_date, end_date)
        df_temp.rename(columns={code: name}, inplace=True)
        if df_economic.empty:
            df_economic = df_temp
        else:
            df_economic = df_economic.join(df_temp, how='outer')
    
    # Set daily frequency and forward-fill missing values
    df_economic = df_economic.asfreq('D').ffill()
    df_economic.index = df_economic.index.date
    df_economic.index.name = 'Date'
    
    return df_economic

# Example usage
if __name__ == "__main__":
    start_date = '2020-01-01'
    end_date = str(datetime.now().date())  # Today's date for end date
    df_economic_data = fetch_economic_data(start_date, end_date)
    print(df_economic_data.tail())
    print(df_economic_data.index)
    