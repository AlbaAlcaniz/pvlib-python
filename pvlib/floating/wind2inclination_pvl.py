import numpy as np
from floating import decompose_wind_speed, Sea, Floater
from sizes_and_weights import sizes_and_weights
from pressure_methodology import pressure_methodology
from coordinate_change import coordinate_change

# Gravitational acceleration [m/s^2]
g = 9.806
# Seawater density [kg/m^3]
density = 1029


def wind2inclination_pvl(
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

    # Initialization
    north_sea = Sea()
    w = north_sea.ang_freq
    t = north_sea.time
    k = north_sea.wave_num
    m_tot, wid, leng, thick, area_pontoon = sizes_and_weights()
    pontoon = Floater(m_tot, wid, leng, thick, facing_dir)

    # Decompose the wind speed into components
    wind_speed_x, wind_speed_y = decompose_wind_speed(wind_speed, wind_dir, facing_dir)

    # Compute the JONSWAP spectrum and peak angular frequency
    S, wp = north_sea.jonswap_spectrum(wind_speed)

    # Compute the surface elevation at x = 0
    eta = north_sea.surface_elevation('JONSWAP', wind_speed, 0)[0]

    # For validation purposes, estimate the significant wave height from the
    # elevation and compare it with the considered one
    H_from_eta = np.mean(np.sort(eta - np.min(eta,axis=1,keepdims=True))[:,-int(eta.shape[1]/3):],axis=1)

    # Get the amplitudes of the decomposed components
    AmpA_x, AmpB_x = north_sea.surface_elevation('JONSWAP', wind_speed_x, 0)[1:3]
    AmpA_y, AmpB_y = north_sea.surface_elevation('JONSWAP', wind_speed_y, 0)[1:3]    

    # Calculate the moments of inertia assuming a rectangular cuboid beam
    roll = pontoon.get_inclination_angles('x', north_sea, AmpA_y, AmpB_y)
    pitch = pontoon.get_inclination_angles('y', north_sea, AmpA_x, AmpB_x)

    # Wavelength analysis: find the moments when the limit of wavelength is
    # smaller than the pontoon size
    wavelength = np.pi*g/(0.15*wp**2)
    limit = (wavelength < leng).sum()/wp.size
    print(
        str(limit * 100)
        + "%% of the time the wavelength is smaller than the beam length"
    )

    # Coordinate change. From "roof" coordinates to horizontal ones
    tilt, azimuth = coordinate_change(pitch, roll)
    return tilt, azimuth
