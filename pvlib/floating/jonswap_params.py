import numpy as np

# Literature data from [1]
WindSpeed = np.arange(0, 91, 10) / 3.6  # m/s
WaveHeight = [0, 0.25, 0.5, 1.2, 2.5, 4.5, 7.1, 10.3, 14.3, 19.3]  # m
AveragePeriod = [0, 1.6, 3.2, 4.6, 6.2, 7.7, 9.9, 10.8, 12.4, 13.9]  # s
# Extract the parameters of the interpolation
c_wh = np.polyfit(WindSpeed, WaveHeight, 3)
c_avp = np.polyfit(WindSpeed, AveragePeriod, 1)

def jonswap_params(wind_speed):
    """JONSWAP_PARAMS Obtain the needed JONSWAP parameters from the wind speed
    %
    % Syntax:
    %   [H,T] = jonswap_params(wind_speed)
    %
    % Description:
    %   Obtain the significant wave height and the average period, needed to
    %   calculate the JONSWAP spectra, from the wind speed via interpolation of
    %   data from the literature [1]
    %
    % Citations:
    %   [1] P. R. Pinet "Essential invitation to oceanography" Jones & Bartlett
    %     Publishers, 2014
    %
    % Inputs:
    %   wind_speed - time-resolved wind speed data in m/S
    %
    % Outputs:
    %   H - time-resolved significant wave height [m]
    %   T - time-resolved average period [s]

        Args:
            wind_speed (_type_): _description_

        Returns:
            _type_: _description_
    """

    # Perform the interpolation with the input wind speed data
    H = np.polyval(c_wh, wind_speed)
    T = np.polyval(c_avp, wind_speed)

    return H, T
