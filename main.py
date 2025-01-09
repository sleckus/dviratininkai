import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import plotly.express as px
import re

raw_df = pd.read_csv("cyclist_data.csv")

print(raw_df.info(max_cols=len(raw_df)))

df = pd.DataFrame(raw_df)
# fixing the position by spliting it into cordinates.
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

# fixing unrealistic cordinates

latitude_threshold = 0.1
longitude_threshold = 0.1

def fix_unrealistic_changes(df):
    for i in range(1, len(df)):
        # Check if latitude change exceeds trhe4sh
        if abs(df['Latitude'].iloc[i] - df['Latitude'].iloc[i - 1]) > latitude_threshold:
            if i + 1 < len(df):
                df['Latitude'].iloc[i] = (df['Latitude'].iloc[i - 1] + df['Latitude'].iloc[i + 1]) / 2
            else:
                df['Latitude'].iloc[i] = df['Latitude'].iloc[i - 1]

        if abs(df['Longitude'].iloc[i] - df['Longitude'].iloc[i - 1]) > longitude_threshold:

            if i + 1 < len(df):
                df['Longitude'].iloc[i] = (df['Longitude'].iloc[i - 1] + df['Longitude'].iloc[i + 1]) / 2
            else:
                df['Longitude'].iloc[i] = df['Longitude'].iloc[i - 1]

    return df

raw_df['Latitude'], raw_df['Longitude'] = zip(*raw_df['Position'].map(split_position))


# Drop og position
df = raw_df.drop(columns=['Position'])
df = fix_unrealistic_changes(df)
print(df.head())

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

print(df.info(max_cols=len(df)))

# general path drawing.
fig = px.scatter(
    df,
    x="Longitude",
    y="Latitude",
    title="General Path",
    color_discrete_sequence=["red"]
)

fig.update_layout(title_font_size=20)
fig.show()