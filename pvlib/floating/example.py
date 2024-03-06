import pandas as pd
import numpy as np
from wind2inclination import wind2inclination
from floating import decompose_wind_speed, Sea, Floater
from sizes_and_weights import sizes_and_weights

#### Load the data to test the code
df = pd.read_csv('wind_data_validation.csv',index_col=0)
wind_speed = np.array(df['wind_speed'])
wind_dir = np.array(df['wind_dir'])
facing_dir = 180
# Limit the number of points for the checking
t_max = 50
ws = wind_speed[:t_max]
wd = wind_dir[:t_max]


#### Code translated from Matlab to python. Serves as back up
tilt_matlab, azimuth_matlab = wind2inclination(ws, wd, facing_dir)


#### How it would work in PVlib
# The user defines the floater characteristics. Here using the ones from the paper
m_tot, wid, leng, thick, area_pontoon = sizes_and_weights()

# Initialization of the classes
north_sea = Sea()
pontoon = Floater(m_tot, wid, leng, thick, facing_dir)

# Decompose the wind speed
wind_speed_x, wind_speed_y = decompose_wind_speed(ws, wd, pontoon.orientation)

# Get the amplitudes of the decomposed components at x = 0
sea_amplitudes = north_sea.get_sea_amplitudes(wind_speed_x, wind_speed_y, 0,
                                              'JONSWAP')

# Get the tilt and azimuth
tilt, azimuth = pontoon.get_tilt_azimuth(north_sea, sea_amplitudes)


#### Ugly and temporal test
print(max(abs(tilt_matlab - tilt)) < 1e-5)
print(max(abs(azimuth_matlab - azimuth)) < 1e-5)


## Validation that the user could perform
# For validation purposes, estimate the significant wave height from the
# elevation and compare it with the considered one
# Compute the surface elevation at x = 0
eta = north_sea.surface_elevation(wind_speed, 0, 'JONSWAP')[0]
H_from_eta = np.mean(np.sort(eta - np.min(eta,axis=1,keepdims=True))[:,-int(eta.shape[1]/3):],axis=1)

## Check that the user could perform
# Wavelength analysis: find the moments when the limit of wavelength is
# smaller than the pontoon size
g = 9.806 # Gravitational acceleration [m/s^2]
# Compute the peak angular frequency from the JONSWAP spectrum
wp = north_sea.jonswap_spectrum(ws)[1]
wavelength = np.pi*g/(0.15*wp**2)
limit = (wavelength < pontoon.length).sum()/wp.size
print(
    str(limit * 100)
    + "%% of the time the wavelength is smaller than the beam length"
)