import numpy as np
from jonswap_params import jonswap_params

# Default values
# Peakedness parameter
Gamma = 3.3
Beta = 5 / 4
# Spectral width parameter
SigmaA = 0.07
SigmaB = 0.09


def jonswap_spect(Omega, wind_speed, g):
    """JONSWAP_SPECT Create the JONSWAP spectra
    %
    % Syntax:
    %   [S,Omegam] = jonswap_spect(Omega, wind_speed, g)
    %
    % Description:
    %   Create the JONSWAP spectra from the time-resolved wind speed and
    %   angular frequency. Since the fetch and period are hard to measure, the
    %   equations of the spectra have been modified to use only the significant
    %   wave height and the average period, which can be extracted from the
    %   wind speed via interpolation of the literature data
    %
    % Note:
    %   Function adapted from the one created by Sayyed Mohsen Vazirizade from
    %   the University of Arizona
    %   smvazirizade@email.arizona.edu
    %
    % Inputs:
    %   Omega - angular frequency over which the spectra is to be calculated in
    %     rad/s
    %   wind_speed - time series of wind speed in m/s
    %   g - gravitational acceleration in m/s^2
    %
    % Outputs:
    %   S - spectra for every time series and angular frequency in m^2*s
    %   Omegam - peak angular frequency in rad/s

        Args:
            Omega (_type_): _description_
            wind_speed (_type_): _description_
            g (_type_): _description_

        Returns:
            _type_: _description_
    """

    # Compute the significant wave height [m] and the average period [s] for
    # each component as a function of wind speed
    Hs, Tz = jonswap_params(wind_speed)

    # Convert wave period Tz to peak wave period Tm
    Tm = Tz * (0.327 * np.exp(-0.315 * Gamma) + 1.17)

    # Compute peak angular frequency
    Omegam = 2 * np.pi / Tm

    # Jonswap spectrum
    Omega = Omega.reshape(1,len(Omega))
    Omegam = Omegam.reshape(len(Omegam),1)
    sigma = (Omega <= Omegam) * SigmaA + (Omega > Omegam) * SigmaB
    A = np.exp(-(((Omega / Omegam - 1) / (sigma * np.sqrt(2))) ** 2))
    # Modified Phillips constant
    alphabar = 5.058 * (1 - 0.287 * np.log(Gamma)) * (Hs / Tm**2) ** 2
    alphabar = alphabar.reshape(len(alphabar),1)
    # Spectra m^2.s
    S = alphabar * g**2 * Omega**-5 * np.exp(-(Beta * (Omega / Omegam) ** -4)) * Gamma**A

    return S, Omegam
