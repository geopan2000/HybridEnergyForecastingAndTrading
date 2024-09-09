from requests import Session
import requests
import pandas as pd
import datetime
import warnings

class RebaseAPI:
  
  challenge_id = 'heftcom2024'
  base_url = 'https://api.rebase.energy'

  def __init__(
    self,
    api_key = open("team_key.txt").read()
    ):
    self.api_key = api_key
    self.headers = {
      'Authorization': f"Bearer {api_key}"
      }
    self.session = Session()
    self.session.headers = self.headers


  def get_variable(
      self,
      day: str,
      variable: ["market_index",
                 "day_ahead_price",
                 "imbalance_price",
                 "wind_total_production",
                 "solar_total_production",
                 "solar_and_wind_forecast"
                 ], # type: ignore
                 ):
    url = f"{self.base_url}/challenges/data/{variable}"
    params = {'day': day}
    resp = self.session.get(url, params=params)

    data = resp.json()
    df = pd.DataFrame(data)
    return df


  # Solar and wind forecast
  def get_solar_wind_forecast(self,day):
    url = f"{self.base_url}/challenges/data/solar_and_wind_forecast"
    params = {'day': day}
    resp = self.session.get(url, params=params)
    data = resp.json()
    df = pd.DataFrame(data)
    return df


  # Day ahead demand forecast
  def get_day_ahead_demand_forecast(self):
    url = f"{self.base_url}/challenges/data/day_ahead_demand"
    resp = self.session.get(url)
    print(resp)
    return resp.json()


  # Margin forecast
  def get_margin_forecast(self):
    url = f"{self.base_url}/challenges/data/margin_forecast"
    resp = self.session.get(url)
    print(resp)
    return resp.json()


  def query_weather_latest(self,model, lats, lons, variables, query_type):
    url = f"{self.base_url}/weather/v2/query"

    body = {
        'model': model,
        'latitude': lats,
        'longitude': lons,
        'variables': variables,
        'type': query_type,
        'output-format': 'json',
        'forecast-horizon': 'latest'
    }

    resp = requests.post(url, json=body, headers={'Authorization': self.api_key})
    print(resp.status_code)

    return resp.json()


  def query_weather_latest_points(self,model, lats, lons, variables):
    # Data here is returned a list
    data = self.query_weather_latest(model, lats, lons, variables, 'points')

    df = pd.DataFrame()
    for point in range(len(data)):
      new_df = pd.DataFrame(data[point])
      new_df["point"] = point
      new_df["latitude"] = lats[point]
      new_df["longitude"] = lons[point]
      df = pd.concat([df,new_df])

    return df


  def query_weather_latest_grid(self,model, lats, lons, variables):
    # Data here is returned as a flattened
    data = self.query_weather_latest(model, lats, lons, variables, 'grid')
    df = pd.DataFrame(data)

    return df


  # To query Hornsea project 1 DWD_ICON-EU grid
  def get_hornsea_dwd(self):
    # As a 6x6 grid
    lats = [53.77, 53.84, 53.9, 53.97, 54.03, 54.1]
    lons = [1.702, 1.767, 1.832, 1.897, 1.962, 2.027]

    variables = 'WindSpeed, WindSpeed:100, WindDirection, WindDirection:100, Temperature, RelativeHumidity'
    return self.query_weather_latest_grid('DWD_ICON-EU', lats, lons, variables)


  # To query Hornsea project 1 GFS grid
  def get_hornsea_gfs(self):
    # As a 3x3 grid
    lats = [53.59, 53.84, 54.09]
    lons = [1.522, 1.772, 2.022]

    variables = 'WindSpeed, WindSpeed:100, WindDirection, WindDirection:100, Temperature, RelativeHumidity'
    return self.query_weather_latest_grid('NCEP_GFS', lats, lons, variables)


  def get_pes10_nwp(self,model):
    # As a list of points
    lats = [52.4872562, 52.8776682, 52.1354277, 52.4880497, 51.9563696, 52.2499177, 52.6416477, 52.2700912, 52.1960768, 52.7082618, 52.4043468, 52.0679429, 52.024023, 52.7681276, 51.8750506, 52.5582373, 52.4478922, 52.5214863, 52.8776682, 52.0780721]
    lons = [0.4012455, 0.7906532, -0.2640343, -0.1267052, 0.6588173, 1.3894081, 1.3509559, 0.7082557, 0.1534462, 0.7302284, 1.0762977, 1.1751747, 0.2962684, 0.1699257, 0.9115028, 0.7137489, 0.1204872, 1.5706825, 1.1916542, -0.0113488]

    variables = 'SolarDownwardRadiation, CloudCover, Temperature'
    return self.query_weather_latest_points(model, lats, lons, variables)


  def get_demand_nwp(self,model):
    # As list of points
    lats = [51.479, 51.453, 52.449, 53.175, 55.86, 53.875, 54.297]
    lons = [-0.451, -2.6, -1.926, -2.986, -4.264, -0.442, -1.533]

    variables = 'Temperature, WindSpeed, WindDirection, TotalPrecipitation, RelativeHumidity'
    return self.query_weather_latest_points(model, lats, lons, variables)


  def submit(self,data):

    url = f"{self.base_url}/challenges/{self.challenge_id}/submit"

    resp = self.session.post(url,headers=self.headers, json=data)
    
    print(resp)
    print(resp.text)

    # Write log file
    text_file = open(f"logs/sub_{pd.Timestamp('today').strftime('%Y%m%d-%H%M%S')}.txt", "w")
    text_file.write(resp.text)
    text_file.close()



# Convert nwp data frame to xarray
def weather_df_to_xr(weather_data):
  
  weather_data["ref_datetime"] = pd.to_datetime(weather_data["ref_datetime"],utc=True)
  weather_data["valid_datetime"] = pd.to_datetime(weather_data["valid_datetime"],utc=True)

  
  if 'point' in weather_data.columns:
    weather_data = weather_data.set_index(["ref_datetime",
                                          "valid_datetime",
                                          "point"])
  else:
      weather_data = pd.melt(weather_data,id_vars=["ref_datetime","valid_datetime"])
  
      weather_data = pd.concat([weather_data,
                            weather_data["variable"].str.split("_",expand=True)],
                            axis=1).drop(['variable',1,3], axis=1)
  
      weather_data.rename(columns={0:"variable",2:"latitude",4:"longitude"},inplace=True)
  
      weather_data = weather_data.set_index(["ref_datetime",
                                          "valid_datetime",
                                          "longitude",
                                          "latitude"])
      weather_data = weather_data.pivot(columns="variable",values="value")
  
  weather_data = weather_data.to_xarray()

  weather_data['ref_datetime'] = pd.DatetimeIndex(weather_data['ref_datetime'].values,tz="UTC")
  weather_data['valid_datetime'] = pd.DatetimeIndex(weather_data['valid_datetime'].values,tz="UTC")

  return weather_data


def day_ahead_market_times(today_date=pd.to_datetime('today')):

  tomorrow_date = today_date + pd.Timedelta(1,unit="day")
  DA_Market = [pd.Timestamp(datetime.datetime(today_date.year,today_date.month,today_date.day,23,0,0),
                          tz="Europe/London"),
              pd.Timestamp(datetime.datetime(tomorrow_date.year,tomorrow_date.month,tomorrow_date.day,22,30,0),
              tz="Europe/London")]

  DA_Market = pd.date_range(start=DA_Market[0], end=DA_Market[1],
                  freq=pd.Timedelta(30,unit="minute"))
  
  return DA_Market


def prep_submission_in_json_format(submission_data,market_day=pd.to_datetime('today') + pd.Timedelta(1,unit="day")):
  submission = []

  if any(submission_data["market_bid"]<0):
    submission_data.loc[submission_data["market_bid"]<0,"market_bid"] = 0
    warnings.warn("Warning...Some market bids were less than 0 and have been set to 0")

  if any(submission_data["market_bid"]>1800):
    submission_data.loc[submission_data["market_bid"]>1800,"market_bid"] = 1800
    warnings.warn("Warning...Some market bids were greater than 1800 and have been set to 1800")

  for i in range(len(submission_data.index)):
      submission.append({
          'timestamp': submission_data["datetime"][i].isoformat(),
          'market_bid': submission_data["market_bid"][i],
          'probabilistic_forecast': {
              10: submission_data["q10"][i],
              20: submission_data["q20"][i],
              30: submission_data["q30"][i],
              40: submission_data["q40"][i],
              50: submission_data["q50"][i],
              60: submission_data["q60"][i],
              70: submission_data["q70"][i],
              80: submission_data["q80"][i],
              90: submission_data["q90"][i],
          }
      })

  data = {
      'market_day': market_day.strftime("%Y-%m-%d"),
      'submission': submission
  }
  
  return data

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt

# Function to calculate pinball loss
def pinball_loss(y_true, y_pred, quantile):
    """Calculate the Pinball Loss for a single quantile."""
    delta = y_true - y_pred
    return np.where(delta > 0, quantile * delta, (quantile - 1) * delta)

# Function to calculate average pinball loss at each step
def calculate_average_pinball_loss_at_each_step(dataframe, target_column):
    """Calculate the average Pinball Loss for each time step across all quantiles."""
    losses_df = pd.DataFrame(index=dataframe.index)
    
    for quantile in range(10, 100, 10):
        q_col = f'q{quantile}'
        if q_col in dataframe.columns:
            quantile_value = quantile / 100.0
            losses_df[q_col] = pinball_loss(dataframe[target_column], dataframe[q_col], quantile_value)
    
    dataframe['average_pinball_loss'] = losses_df.mean(axis=1)

def plot_quantiles_target_and_average_loss_interactive(dataframe, target_column, test_times, save_path, title="Hybrid Production Forecast"):
    """Plot quantiles, target, and average pinball loss interactively with customizable title."""
    calculate_average_pinball_loss_at_each_step(dataframe, target_column)
    dataframe['time'] = test_times
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dataframe['time'], y=dataframe[target_column], mode='lines', name='Observations', line=dict(color='black', width=3)))
    
    quantile_colors = px.colors.qualitative.Plotly
    for idx, quantile in enumerate(range(10, 100, 10)):
        q_col = f'q{quantile}'
        if q_col in dataframe.columns:
            fig.add_trace(go.Scatter(x=dataframe['time'], y=dataframe[q_col], mode='lines', name=f'Q{quantile}', line=dict(color=quantile_colors[idx % len(quantile_colors)])))
    
    # Add trace for average pinball loss with a secondary y-axis
    fig.add_trace(go.Scatter(
        x=dataframe['time'], 
        y=dataframe['average_pinball_loss'], 
        mode='lines', 
        name='Avg Pinball Loss', 
        line=dict(color='rgba(255,0,0,0.5)', width=2, dash='dot'),
        yaxis='y2'  # This assigns the trace to the secondary y-axis
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Time',
        yaxis=dict(title='MWh', side='left', rangemode='tozero'),
        yaxis2=dict(title='Average Pinball Loss', overlaying='y', side='right', rangemode='tozero'),
        legend=dict(x=0.01, y=0.99),
        width=1000,
        height=600
    )
    
    fig.update_layout(xaxis=dict(rangeslider=dict(visible=True), type='date'))
    fig.write_html(save_path)

# Function to calculate pinball loss for each quantile
def pinball(y, q, alpha):
    """Calculate the Pinball loss between predictions and actual values."""
    return (y-q)*alpha*(y>=q) + (q-y)*(1-alpha)*(y<q)

# Function to calculate average pinball loss across quantiles
def pinball_score(df, target_col='total_generation_MWh'):
    """Compute the average Pinball loss across a range of quantiles for a given DataFrame."""
    score = []
    for qu in range(10, 100, 10):
        alpha = qu / 100
        score.append(df.apply(lambda x: pinball(x[target_col], x[f"q{qu}"], alpha), axis=1).mean())
    return sum(score) / len(score)

# Function to plot quantile statistics
def quantile_stats(quantile_predictions_df, target_column='total_generation_MWh'):
    """Plot the percentage of actual values below and above each quantile."""
    quantiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    statistics = []

    for quantile in quantiles:
        q_col = f'q{quantile}'
        below = np.mean(quantile_predictions_df[target_column] < quantile_predictions_df[q_col]) * 100
        above = 100 - below
        statistics.append({'Quantile': f'Q{quantile}', 'Below (%)': below, 'Above (%)': above})

    statistics_df = pd.DataFrame(statistics)

    # Plotting
    fig, ax = plt.subplots(figsize=(9, 5))
    bars_below = ax.bar(statistics_df['Quantile'], statistics_df['Below (%)'], color='skyblue', label='Below Quantile')
    bars_above = ax.bar(statistics_df['Quantile'], statistics_df['Above (%)'], bottom=statistics_df['Below (%)'], color='orange', label='Above Quantile')

    ax.set_ylabel('Percentage')
    ax.set_xlabel('Quantile')
    ax.set_title('Percentage of Actual Values Below and Above Each Quantile')
    ax.legend()

    # Adding the percentage text inside the bars
    for bar in bars_below:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    for bar, pct_below in zip(bars_above, statistics_df['Below (%)']):
        height = bar.get_height() + pct_below
        ax.annotate(f'{bar.get_height():.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    

def calculate_average_quantile_loss(dataframe, target_column):
    """Calculate the average Pinball Loss for each quantile."""
    quantiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    average_losses = []

    for quantile in quantiles:
        q_col = f'q{quantile}'
        if q_col in dataframe.columns:
            quantile_value = quantile / 100.0
            loss = pinball_loss(dataframe[target_column], dataframe[q_col], quantile_value)
            average_loss = np.mean(loss)
            average_losses.append({'Quantile': quantile, 'Average Pinball Loss': average_loss})

    return pd.DataFrame(average_losses)



import matplotlib.pyplot as plt

def plot_average_quantile_loss(average_losses_df, plot_title="Average Pinball Loss for Each Quantile"):
    """Plot the average Pinball Loss for each quantile as a bar chart."""
    fig, ax = plt.subplots(figsize=(8, 6))  # Adjusted figure size for better readability
    # Ensure 'Quantile' data is appropriate for plotting
    average_losses_df['Quantile'] = average_losses_df['Quantile'].astype(str)
    bars = ax.bar(average_losses_df['Quantile'], average_losses_df['Average Pinball Loss'], color='red', width=0.8)
    ax.set_xlabel('Quantile')
    ax.set_ylabel('Average Pinball Loss')
    ax.set_title(plot_title)  # Use the custom title
    plt.xticks(rotation=45)  # Adjust rotation if necessary

    # Annotate each bar with its height
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

def calculate_and_plot_average_quantile_loss(dataframe, target_column, plot_title="Average Pinball Loss for Each Quantile"):
    """Calculate and plot the average Pinball Loss for each quantile with a customizable title."""
    average_losses_df = calculate_average_quantile_loss(dataframe, target_column)
    plot_average_quantile_loss(average_losses_df, plot_title)


def add_cyclic_features(table):
    # Year as a numeric feature
    table['year'] = table['valid_time'].dt.year
    # Cyclic encoding for month
    table['month_sin'] = np.sin(2 * np.pi * table['valid_time'].dt.month / 12)
    table['month_cos'] = np.cos(2 * np.pi * table['valid_time'].dt.month / 12)
    # Cyclic encoding for day
    max_day = 31  # You might adjust this based on your data, e.g., using the actual max day of each month
    table['day_sin'] = np.sin(2 * np.pi * table['valid_time'].dt.day / max_day)
    table['day_cos'] = np.cos(2 * np.pi * table['valid_time'].dt.day / max_day)
    # Cyclic encoding for hour
    table['hour_sin'] = np.sin(2 * np.pi * table['valid_time'].dt.hour / 24)
    table['hour_cos'] = np.cos(2 * np.pi * table['valid_time'].dt.hour / 24)
    # Cyclic encoding for minute
    table['minute_sin'] = np.sin(2 * np.pi * table['valid_time'].dt.minute / 60)
    table['minute_cos'] = np.cos(2 * np.pi * table['valid_time'].dt.minute / 60)
    return table

def add_cyclic_features_V2(table):
    # Year as a numeric feature
    table['year'] = table['valid_time'].dt.year

    # Cyclic encoding for month
    table['month_sin'] = np.sin(2 * np.pi * table['valid_time'].dt.month / 12)
    table['month_cos'] = np.cos(2 * np.pi * table['valid_time'].dt.month / 12)

    # Cyclic encoding for day
    max_day = 31  # You might adjust this based on your data, e.g., using the actual max day of each month
    table['day_sin'] = np.sin(2 * np.pi * table['valid_time'].dt.day / max_day)
    table['day_cos'] = np.cos(2 * np.pi * table['valid_time'].dt.day / max_day)

    # Combine hour and minute into total minutes since the start of the day
    total_minutes = table['valid_time'].dt.hour * 60 + table['valid_time'].dt.minute
    max_minutes = 24 * 60  # Total minutes in a day

    # Cyclic encoding for combined hours and minutes
    table['time_sin'] = np.sin(2 * np.pi * total_minutes / max_minutes)
    table['time_cos'] = np.cos(2 * np.pi * total_minutes / max_minutes)

    return table



from dateutil import parser

# Function to parse dates flexibly
def parse_date(date_str):
    try:
        return parser.parse(date_str)
    except ValueError:
        return None
    
from sp2ts import sp2ts, from_unixtime
 
def convert_to_unix_timestamp(row):
    date = row['SETTLEMENT_DATE'].date()  # Get the date part of the timestamp
    sp = row['SETTLEMENT_PERIOD']
    timestamp = sp2ts(date, sp)
    return timestamp

# Function to convert Unix timestamp to UTC datetime
def convert_to_utc_from_unix(unix_timestamp):
    return from_unixtime(unix_timestamp)


def sort_quantiles(df, quantile_columns):
    """
    This function iterates through each row of the dataframe and sorts the quantile columns
    to ensure that q10 <= q20 <= q30 <= ... <= q90.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the quantile predictions.
    quantile_columns (list): List of column names corresponding to the quantiles.
    
    Returns:
    pd.DataFrame: DataFrame with sorted quantile columns for each row.
    """
    sorted_quantiles = df[quantile_columns].apply(lambda row: sorted(row), axis=1)
    sorted_quantiles_df = pd.DataFrame(sorted_quantiles.tolist(), columns=quantile_columns, index=df.index)
    df[quantile_columns] = sorted_quantiles_df
    return df
  

import pandas as pd

def sort_quantilesVersion2(df, quantile_columns):
    """
    This function iterates through each row of the DataFrame and sorts the quantile columns
    to ensure that q10 <= q20 <= ... <= q90, with the condition that q50 is in the middle
    and quantiles on either side do not cross over q50.

    Parameters:
    df (pd.DataFrame): DataFrame containing the quantile predictions.
    quantile_columns (list): List of column names corresponding to the quantiles.

    Returns:
    pd.DataFrame: DataFrame with sorted quantile columns for each row.
    """
    for index, row in df.iterrows():
        q50 = row['q50']
        # Sorting lower half: q10 to q40
        lower_half = sorted(row[col] for col in quantile_columns if int(col[1:]) < 50)
        lower_half = [min(q50, x) for x in lower_half]  # Ensure they do not exceed q50
        
        # Sorting upper half: q60 to q90
        upper_half = sorted(row[col] for col in quantile_columns if int(col[1:]) > 50)
        upper_half = [max(q50, x) for x in upper_half]  # Ensure they are not below q50

        # Reconstruct the row with sorted values
        sorted_row = lower_half + [q50] + upper_half
        sorted_index = [col for col in quantile_columns if int(col[1:]) < 50] + ['q50'] + \
                       [col for col in quantile_columns if int(col[1:]) > 50]
        
        # Place sorted values back into the DataFrame
        df.loc[index, sorted_index] = sorted_row

    return df
