import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import plotly.express as px
import plotly.graph_objects as go
import re
import numpy as np
from scipy.stats import alpha
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

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

# General Path Scatter
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



# Altitude
fig = px.scatter(df, x="Time", y="AltitudeMeters", title="Altitude Over Time", color_discrete_sequence=["purple"])
fig.update_layout(title_font_size=20)
fig.show()

fig = px.scatter(df, x="Time", y="HeartRateBpm", title="Heart Rate Over Time", color_discrete_sequence=["orange"])
fig.update_layout(title_font_size=20)
fig.show()

correlation_matrix = df[['Speed', 'HeartRateBpm', 'AltitudeMeters']].corr()
print(correlation_matrix)

# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
# plt.title("Correlation Heatmap: Speed, HeartRate, Altitude")
# # plt.show()


df['RollingSpeed'] = df['Speed'].rolling(window=10).mean()
df['RollingHeartRate'] = df['HeartRateBpm'].rolling(window=10).mean()
df['RollingAltitude'] = df['AltitudeMeters'].rolling(window=10).mean()

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df['Time'],
    y=df['RollingSpeed'],
    mode='lines',
    name='Speed (m/s)',
    line=dict(color='blue')
))

fig.add_trace(go.Scatter(
    x=df['Time'],
    y=df['RollingHeartRate'],
    mode='lines',
    name='Heart Rate (bpm)',
    line=dict(color='green')
))

fig.add_trace(go.Scatter(
    x=df['Time'],
    y=df['RollingAltitude'],
    mode='lines',
    name='Altitude (m)',
    line=dict(color='red')
))

fig.update_layout(
    title="Speed, Heart Rate, and Altitude Over Time with Trend Lines",
    xaxis_title="Time",
    yaxis_title="Value",
    title_font_size=20,
    legend_title="Legend"
)
fig.show()

df = df.drop(columns=['Extensions', 'SensorState'])
df['Cadence'] = df['Cadence'].fillna(df['Cadence'].mean())

print(df.info(max_cols=len(df)))

fig = px.line(df, x='Time', y='AltitudeMeters',
              title='Cadence vs Altitude Over Time',
              labels={'AltitudeMeters': 'Altitude (m)', 'Time': 'Time'},
              line_shape='linear')

fig.add_scatter(x=df['Time'], y=df['Cadence'], mode='lines',
                line=dict( width=2),
                name='Cadence (rpm)', showlegend=False)

fig.update_layout(
    title_font_size=20,
    xaxis_title="Time",
    yaxis_title="Altitude (m)",
    legend_title="Legend",
    coloraxis_colorbar_title="Cadence (rpm)",
)

fig.show()

# machine learning start.....

features = ['Speed', 'AltitudeMeters', 'DistanceMeters', 'Cadence']
target = 'HeartRateBpm'


X = df[features]
y = df[target]

# Train
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

df['PredictedHeartRate'] = model.predict(X)

# Plot the actual and predicted
fig = px.line(df, x='Time', y=['HeartRateBpm', 'PredictedHeartRate'],
              title='Actual vs Predicted Heart Rate Over Time',
              labels={'HeartRateBpm': 'Actual Heart Rate (bpm)', 'PredictedHeartRate': 'Predicted Heart Rate (bpm)'})

fig.update_traces(opacity=0.7)

fig.update_layout(title_font_size=20)
fig.show()


