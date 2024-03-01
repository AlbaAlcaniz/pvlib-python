import numpy as np
from jonswap_spect import jonswap_spect
from compute_surface_elevation import compute_surface_elevation
from sizes_and_weights import sizes_and_weights
from pressure_methodology import pressure_methodology
from coordinate_change import coordinate_change

# Gravitational acceleration [m/s^2]
g = 9.806
# Seawater density [kg/m^3]
density = 1029


def wind2inclination(
    wind_speed, wind_dir, facing_dir, dw=0.01, w_max=8, dt=1, t_dur=3600
):
    """Determine beam inclination caused by wind-generated waves in the North Sea
        Syntax:
    %   [tilt, azimuth] = wind2inclination(wind_speed, wind_dir, facing_dir)
    %   [tilt, azimuth] = wind2inclination(wind_speed, wind_dir, facing_dir, dw)
    %   [tilt, azimuth] = wind2inclination(wind_speed, wind_dir, facing_dir, dw, w_max)
    %   [tilt, azimuth] = wind2inclination(wind_speed, wind_dir, facing_dir, dw, w_max, dt)
    %   [tilt, azimuth] = wind2inclination(wind_speed, wind_dir, facing_dir, dw, w_max, dt, t_dur)
    %   [tilt, azimuth] = wind2inclination(wind_speed, wind_dir, facing_dir, 'dt', dt, 't_dur', t_dur)
    %   [tilt, azimuth] = wind2inclination(wind_speed, wind_dir, facing_dir, 'w_max', w_max, 't_dur', t_dur)
    %
    % Description:
    %     This is the main code employed in the article "Offshore floating PV -
    %     DC and AC yield analysis considering wave effects"  by A. Alcañiz et
    %     al. published in Energy Conversion and Management. This code
    %     calculates the tilt and the azimuth of a rigid beam with PV modules
    %     on top. The change in angles is caused by the water movement due to
    %     wind-generated waves. Therefore, the only inputs to this code are the
    %     wind and the beam characteristics. The beam is assumed to be a rigid
    %     rectangular cuboid. Detailed explanation of the steps undertaken can
    %     be found in the manuscript.
    %
    % Inputs:
    %     wind_speed - time series of wind speed in m/s. wind_speed(i)
    %       corresponding to wind_dir(i). Data assumed to be hourly by default
    %     wind_dir - scalar or vector of wind direction in degrees. The
    %       convention exployed is South: 0°, East: 90°, North: 180°, West:
    %       270°. Data assumed to be hourly by default
    %     facing_dir - facing direction of the pontoon. Employed for the wind
    %       decomposition. To be considered when providing the dimensions of
    %       the pontoon. Same convention for the angles as wind_dir
    %
    % Optional inputs:
    %     dw - step in the angular frequency in rad/s (0.01 by default)
    %     w_max - maximum angular frequency in rad/s (8 by default)
    %     dt - time step in seconds which provides the time resolution of the
    %       final angle outputs (1 by default)
    %     t_dur - number of dt that are in each wind_speed value. For instance,
    %       if the wind data has a resolution of 1 hour and dt = 1 second,
    %       t_dur should be 3600 (3600 by default)
    %
    % Outputs:
    %     tilt - time-resolved tilt of the pontoon in degrees with resolution
    %       provided by dt
    %     azimuth - time-resolved azimuth of the pontoon in degrees with
    %       resolution provided by dt. Same convention for the angles as
    %       wind_dir and facing_dir
    %
    % Cite as:
    %     A. Alcañiz et al. "Offshore floating PV - DC and AC yield analysis
    %     considering wave effects" in Energy Conversion and Management

        Args:
            wind_speed (_type_): _description_
            wind_dir (_type_): _description_
            facing_dir (_type_): _description_
    """
    # Read and check the format of the variables

    # Precalculations
    # Angular frequency [rad/s]
    w = np.arange(dw, w_max + dw, dw)
    # Time period [s]
    t = np.arange(dt, t_dur + dt, dt)
    # Dispersion relation to obtain the wave number [1/m]
    k = w**2 / g

    # Decompose the wind speed into components
    wind_speed_x = abs(wind_speed * np.cos(np.deg2rad(wind_dir - facing_dir)))
    wind_speed_y = abs(wind_speed * np.sin(np.deg2rad(wind_dir - facing_dir)))

    # Compute the JONSWAP spectrum for each component
    S, wp = jonswap_spect(w, wind_speed, g)
    S_x = jonswap_spect(w, wind_speed_x, g)[0]
    S_y = jonswap_spect(w, wind_speed_y, g)[0]

    # Compute the surface elevation for each component at x = 0
    eta = compute_surface_elevation(S, w, t, k, 0)[0]
    AmpA_x, AmpB_x = compute_surface_elevation(S_x, w, t, k, 0)[1:3]
    AmpA_y, AmpB_y = compute_surface_elevation(S_y, w, t, k, 0)[1:3]

    # For validation purposes, estimate the significant wave height from the
    # elevation and compare it with the considered one
    H_from_eta = np.mean(np.sort(eta - np.min(eta,axis=1,keepdims=True))[:,-int(eta.shape[1]/3):],axis=1)

    # Sizing and inertia
    # Consider the geometries and masses of each component in the beam. This
    # function should be adapted to each individual case
    m_tot, wid, leng, thick, area_pontoon = sizes_and_weights()

    # Calculate the moments of inertia assuming a rectangular cuboid beam
    # x-axis inertia - roll
    I_x = 1/12*m_tot*(leng**2 + thick**2)
    # y-axis inertia - pitch
    I_y = 1/12*m_tot*(wid**2 + thick**2)

    # Wavelength analysis: find the moments when the limit of wavelength is
    # smaller than the pontoon size
    wavelength = np.pi*g/(0.15*wp**2)
    limit = (wavelength < leng).sum()/wp.size
    print(
        str(limit * 100)
        + "%% of the time the wavelength is smaller than the beam length"
    )

    # Angles calculation: Pitch and roll
    roll = pressure_methodology(I_x, density, g, wid, k, leng, w, t, AmpA_y, AmpB_y)
    pitch = pressure_methodology(I_y, density, g, leng, k, wid, w, t, AmpA_x, AmpB_x)

    # Coordinate change. From "roof" coordinates to horizontal ones
    tilt, azimuth = coordinate_change(pitch, roll)
    return tilt, azimuth
