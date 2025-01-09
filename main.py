import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import plotly.express as px
import re

# Load and process data
raw_df = pd.read_csv("cyclist_data.csv")
df = pd.DataFrame(raw_df)

# Split Position into Latitude and Longitude
def split_position(position):
    try:
        first_dot_index = position.find('.')
        second_dot_index = position.find('.', first_dot_index + 1)
        if first_dot_index != -1 and second_dot_index != -1:
            latitude = float(position[:second_dot_index])
            longitude = float(position[second_dot_index - 1:])
            return latitude, longitude
    except ValueError:
        pass
    return None, None

# Fix unrealistic coordinates
latitude_threshold = 0.1
longitude_threshold = 0.1
def fix_unrealistic_changes(df):
    for i in range(1, len(df)):
        if abs(df['Latitude'].iloc[i] - df['Latitude'].iloc[i - 1]) > latitude_threshold:
            df['Latitude'].iloc[i] = (df['Latitude'].iloc[i - 1] + df['Latitude'].iloc[i + 1]) / 2 if i + 1 < len(df) else df['Latitude'].iloc[i - 1]
        if abs(df['Longitude'].iloc[i] - df['Longitude'].iloc[i - 1]) > longitude_threshold:
            df['Longitude'].iloc[i] = (df['Longitude'].iloc[i - 1] + df['Longitude'].iloc[i + 1]) / 2 if i + 1 < len(df) else df['Longitude'].iloc[i - 1]
    return df

raw_df['Latitude'], raw_df['Longitude'] = zip(*raw_df['Position'].map(split_position))
df = raw_df.drop(columns=['Position'])
df = fix_unrealistic_changes(df)

# Calc speed
df['Time'] = pd.to_datetime(df['Time'])
df['Speed'] = df['DistanceMeters'].diff() / df['Time'].diff().dt.total_seconds()

# Average Speed Calc
total_distance = df['DistanceMeters'].iloc[-1] - df['DistanceMeters'].iloc[0]
total_time = (df['Time'].iloc[-1] - df['Time'].iloc[0]).total_seconds()
average_speed = total_distance / total_time
print(f"Average speed: {average_speed} m/s")

# Drop Na
df = df.dropna(subset=['Speed'])

# Plott
if "DistanceMeters" in raw_df.columns:
    raw_df = raw_df.dropna(subset=["DistanceMeters"])
    fig = px.scatter(
        raw_df,
        x="DistanceMeters",
        y="Time",
        title="Distribution of Distance Meters",
        labels={"DistanceMeters": "Distance in Meters"},
        color_discrete_sequence=["red"]
    )
    fig.update_layout(title_font_size=20)
    fig.show()

# General Path Scatter Plot
fig = px.scatter(df, x="Longitude", y="Latitude", title="General Path", color_discrete_sequence=["red"])
fig.update_layout(title_font_size=20)
fig.show()

# Speed Over Time
fig = px.line(df, x='Time', y='Speed', title="Speed Over Time", labels={"Speed": "Speed (m/s)", "Time": "Time"}, color_discrete_sequence=["blue"])
fig.update_layout(title_font_size=20)
fig.show()

# Cumulative Distance Over Time
df['CumulativeDistance'] = df['DistanceMeters'] - df['DistanceMeters'].iloc[0]
fig = px.line(df, x='Time', y='CumulativeDistance', title="Cumulative Distance Over Time", labels={"CumulativeDistance": "Distance (meters)", "Time": "Time"}, color_discrete_sequence=["green"])
fig.update_layout(title_font_size=20)
fig.show()

#Average Speed Over Time
df['RollingSpeed'] = df['Speed'].rolling(window=30).mean()
fig = px.line(df, x='Time', y='RollingSpeed', title="Average Speed Over Time", labels={"RollingSpeed": "Speed (m/s)", "Time": "Time"}, color_discrete_sequence=["red"])
fig.update_layout(title_font_size=20)
fig.show()

print(df)

