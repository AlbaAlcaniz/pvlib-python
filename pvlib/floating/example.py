import pandas as pd
import numpy as np
from wind2inclination import wind2inclination
from wind2inclination_pvl import wind2inclination_pvl
import os

filename = os.path.dirname(__file__) + "\\..\\..\\..\\KNW-1.0_H37-ERA_NL-064-116.csv"
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
t_max = 50

# Do not touch! Serves as back up
tilt,azimuth = wind2inclination(wind_speed[:t_max], wind_dir[:t_max], facing_dir)

# Improvements
tilt_pvl,azimuth_pvl = wind2inclination_pvl(wind_speed[:t_max], wind_dir[:t_max], facing_dir)

print(np.all(np.equal(tilt,tilt_pvl)))
print(np.all(np.equal(azimuth,azimuth_pvl)))