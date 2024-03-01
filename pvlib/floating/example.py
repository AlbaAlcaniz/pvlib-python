import pandas as pd
import numpy as np
from wind2inclination import wind2inclination

filename = "KNW-1.0_H37-ERA_NL-064-116.csv"
df = pd.read_csv(
    filename,
    skiprows=26312,
    nrows=8760,
    usecols=[0, 1, 2, 3],
    names=["date", "hour", "wind_speed", "wind_dir"],
)

wind_speed = np.array(df['wind_speed'])
wind_dir = np.array(df['wind_dir'])
facing_dir = 180

tilt,azimuth = wind2inclination(wind_speed[:100], wind_dir[:100], facing_dir)