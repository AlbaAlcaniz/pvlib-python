import numpy as np

# Default values for Jonswap spectrum
# Literature data from [1]
WindSpeed = np.arange(0, 91, 10) / 3.6  # m/s
WaveHeight = [0, 0.25, 0.5, 1.2, 2.5, 4.5, 7.1, 10.3, 14.3, 19.3]  # m
AveragePeriod = [0, 1.6, 3.2, 4.6, 6.2, 7.7, 9.9, 10.8, 12.4, 13.9]  # s
# Extract the parameters of the interpolation
c_wh = np.polyfit(WindSpeed, WaveHeight, 3)
c_avp = np.polyfit(WindSpeed, AveragePeriod, 1)
# Peakedness parameter
Gamma = 3.3
Beta = 5 / 4
# Spectral width parameter
SigmaA = 0.07
SigmaB = 0.09

# Gravitational acceleration [m/s^2]
g = 9.806

def decompose_wind_speed(wind_speed, wind_dir, facing_dir):
    """_summary_

    Args:
        wind_speed (_type_): _description_
        wind_dir (_type_): _description_
        float_orient (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Decompose the wind speed into components
    wind_speed_x = abs(wind_speed * np.cos(np.deg2rad(wind_dir - facing_dir)))
    wind_speed_y = abs(wind_speed * np.sin(np.deg2rad(wind_dir - facing_dir)))    
    return wind_speed_x, wind_speed_y


class Sea:    
    def __init__(self, dw=0.01, w_max=8, dt=1, t_dur=3600, density=1029):
        # Angular frequency [rad/s]
        self.ang_freq = np.arange(dw, w_max + dw, dw)
        # Time period [s]
        self.time = np.arange(dt, t_dur + dt, dt)
        # Dispersion relation to obtain the wave number [1/m]
        self.wave_num = self.ang_freq**2 / g
        # Sea water density
        self.density = density
          
    def jonswap_spectrum(self, wind_speed):
        """JONSWAP_SPECT Create the JONSWAP spectra
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
        %   wind_speed - time series of wind speed in m/s
        %
        % Outputs:
        %   S - spectra for every time series and angular frequency in m^2*s
        %   peak_ang_freq - peak angular frequency in rad/s

            Args:
                ang_freq (_type_): _description_
                wind_speed (_type_): _description_
                g (_type_): _description_

            Returns:
                _type_: _description_
        """
        # Compute the significant wave height [m] and the average period [s] for
        # each component as a function of wind speed
        Hs, Tz = self.jonswap_params(wind_speed)

        # Convert wave period Tz to peak wave period Tm
        Tm = Tz * (0.327 * np.exp(-0.315 * Gamma) + 1.17)

        # Compute peak angular frequency
        peak_ang_freq = 2 * np.pi / Tm
        peak_ang_freq = peak_ang_freq.reshape(len(peak_ang_freq),1)

        # Jonswap spectrum
        ang_freq = self.ang_freq.reshape(1,len(self.ang_freq))
        sigma = (ang_freq <= peak_ang_freq) * SigmaA + (ang_freq > peak_ang_freq) * SigmaB
        A = np.exp(-(((ang_freq / peak_ang_freq - 1) / (sigma * np.sqrt(2))) ** 2))
        # Modified Phillips constant
        alphabar = 5.058 * (1 - 0.287 * np.log(Gamma)) * (Hs / Tm**2) ** 2
        alphabar = alphabar.reshape(len(alphabar),1)
        # Spectra m^2.s
        spectrum = alphabar * g**2 * ang_freq**-5 * np.exp(-(Beta * (ang_freq / peak_ang_freq) ** -4)) * Gamma**A

        return spectrum, peak_ang_freq
    

    def jonswap_params(self,wind_speed):
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

    
    def pierson_moskowitz_spectrum(self, wind_speed):
        spectrum = wind_speed
        return spectrum
    

    def surface_elevation(self, type_spectrum, wind_speed, x):
        if type_spectrum == 'JONSWAP':
            spectrum = self.jonswap_spectrum(wind_speed)[0]
        elif type_spectrum == 'Pierson-Moskowitz':
            spectrum = self.pierson_moskowitz_spectrum(wind_speed)
        
        n_w = len(self.ang_freq)
        dw = self.ang_freq[1] - self.ang_freq[0]
        X = x * np.ones(self.time.shape)
        # AmpA = (spectrum*dw)**0.5 * np.random.normal(size=(1,n_w))
        # AmpB = (spectrum*dw)**0.5 * np.random.normal(size=(1,n_w))
        AmpA = (spectrum*dw)**0.5 * np.ones((1,n_w))
        AmpB = (spectrum*dw)**0.5 * np.ones((1,n_w))
        arg_rad = (self.wave_num.reshape(len(self.wave_num),1))@(X.reshape(1,len(X))) - (self.ang_freq.reshape(len(self.ang_freq),1))@(self.time.reshape(1,len(self.time)))
        elevation = AmpA@np.cos(arg_rad) + AmpB@np.sin(arg_rad)

        return elevation, AmpA, AmpB
    
    def get_sea_amplitudes(self, type_spectrum, wind_speed_x, wind_speed_y, x):
        AmpA_x, AmpB_x = self.surface_elevation(type_spectrum, wind_speed_x, x)[1:3]
        AmpA_y, AmpB_y = self.surface_elevation(type_spectrum, wind_speed_y, x)[1:3]
        return [AmpA_x, AmpB_x, AmpA_y, AmpB_y]
    

class Floater:
    def __init__(self, mass, width, length, thickness, orientation):
        self.mass = mass
        self.width = width
        self.length = length
        self.thickness = thickness
        self.orientation = orientation
    
    def compute_inertia_moment(self, axis):
        # This function assumes a rectangular floater
        # Could be upgraded to include inertia moments for different shapes
        if axis == 'x':
            I = 1/12*self.mass*(self.length**2 + self.thickness**2)
        elif axis == 'y':
            I = 1/12*self.mass*(self.width**2 + self.thickness**2)

        return I
    
    def get_inclination_angles(self, axis, Sea, AmpA, AmpB):
        I = self.compute_inertia_moment(axis)
        if axis == 'x':
            wid = self.width; leng = self.length
        elif axis == 'y':
            wid = self.length; leng = self.width
        fact = 2 * Sea.density * g * wid / I
        xx = Sea.wave_num * leng / 2
        aux1 = 1 / (Sea.wave_num**2 * Sea.ang_freq**2) * (np.sin(xx) - xx * np.cos(xx))
        aux1 = aux1.reshape(len(aux1),1)
        # shift = np.random.uniform(size=(len(Sea.ang_freq),1)) * np.ones((1,len(Sea.time)))
        shift = np.ones((len(Sea.ang_freq),1)) * np.ones((1,len(Sea.time)))
        arg_rad = (Sea.ang_freq.reshape(len(Sea.ang_freq),1))@(Sea.time.reshape(1,len(Sea.time))) + shift
        aux2 = AmpA @ (aux1*np.sin(arg_rad)) + AmpB @ (aux1*np.cos(arg_rad))
        theta = fact * aux2
        return theta
    
    def get_tilt_azimuth(self, Sea, sea_amplitudes):
        # Get the inclination angles for both axes
        AmpA_x, AmpB_x, AmpA_y, AmpB_y = sea_amplitudes
        roll = self.get_inclination_angles('x', Sea, AmpA_y, AmpB_y)
        pitch = self.get_inclination_angles('y', Sea, AmpA_x, AmpB_x)

        # Transformation to horizontal coordinates
        tilt = np.rad2deg(np.arccos(np.cos(np.deg2rad(pitch)) * np.cos(np.deg2rad(roll))))
        tan_Am = np.sin(np.deg2rad(pitch)) / np.tan(np.deg2rad(roll))
        Am = np.rad2deg(np.arctan(tan_Am))
        # Ensure that the azimuth always stays within [0, 360)
        Am[(90-pitch)>90] = Am[(90-pitch)>90] + 180
        Am[Am<0] = Am[Am<0] + 360
        # Set the azimuth to 0 degrees when the floater is flat
        Am[(pitch==0) & (roll==0)] = 0

        # Convert from matrix to vectors
        tilt = tilt.reshape(tilt.size)
        Am = Am.reshape(Am.size)
        return tilt, Am