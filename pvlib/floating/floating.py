"""
The ``floating`` module contains methods to calculate the varying inclinations
of a floater on sea
Methodology developed in A. Alcañiz et al. "Offshore floating PV - DC and AC 
yield analysis considering wave effects" in Energy Conversion and Management

TODO: ensure that it works also with pandas inputs
TODO: include the pierson-moskowitz spectrum
"""

# Alba Alcañiz, Delft University of Technology 2024

import numpy as np
from pvlib.tools import sind, cosd, tand, acosd, atand

def decompose_wind_speed(wind_speed, wind_dir, facing_dir):
    """
    Decompose the wind speed into x and y components following a determined 
    direction

    Parameters
    ----------
    wind_speed : numeric
        Wind speed [m/s]
    wind_dir : numeric
        Wind direction [degrees]
    facing_dir : numeric
        Facing direction of the floater over which the wind is decomposed. Same 
        convention for the angles as the wind direction

    Returns
    -------
    wind_speed_x : numeric
        Wind speed parallel to the facing direction [m/s]
    wind_speed_y : numeric
        Wind speed perpendicular to the facing direction [m/s]
    """
    # Decompose the wind speed into components
    wind_speed_x = abs(wind_speed * cosd(wind_dir - facing_dir))
    wind_speed_y = abs(wind_speed * sind(wind_dir - facing_dir))
    return wind_speed_x, wind_speed_y


class Sea:
    '''
    Sea objects are containers for spectra and surface elevation associated to 
    a particular water body.

    Parameters
    ----------
    dw : float
        Step in the angular frequency, default 0.01 [rad/s]
    w_max: float
        Maximum angular frequency, default 8 [rad/s]
    dt : float
        Time step that provides the time resolution of the final angle outputs,
        default 1 [s]
    t_dur : float
        Number of dt that are in each wind_speed value, e.g. if the wind data 
        has hourly resolution and dt = 1 s, t_dur should be 3600, default 3600
    density : float
        Sea water density, default 1029 [kg/m3]
    '''
    def __init__(self, dw=0.01, w_max=8, dt=1, t_dur=3600, density=1029):
        '''
        ang_freq : numeric
            Range of angular frequencies over which the spectrum is computed
        time : numeric
            Time period indicating the correspondence in resolution between the 
            input wind data and the output results
        wave_num : numeric
            Wave number associated to the angular frequency [1/m]
        '''
        self.ang_freq = np.arange(dw, w_max + dw, dw)
        self.time = np.arange(dt, t_dur + dt, dt)
        self.wave_num = self.ang_freq**2 / 9.806
        self.density = density
          
    def jonswap_spectrum(self, wind_speed):
        '''
        Obtain the JONSWAP spectra [1] from the time-resolved wind speed and 
        angular frequency. Since the fetch and period are hard to measure, the
        equations of the spectra have been modified to use only the significant
        wave height and the average period, which can be extracted from the
        wind speed via interpolation of literature data.
        
        Parameters
        ----------
        wind_speed : numeric
            Wind speed [m/s]

        Returns
        -------
        spectrum : numeric
            Spectra for every time step and angular frequency [m2*s]
        peak_ang_freq : numeric
            Peak angular frequency [rad/s]

        Notes
        -----
        Function adapted from the one created by Sayyed Mohsen Vazirizade from
        the University of Arizona
        smvazirizade@email.arizona.edu
        
        References
        ----------
        .. [1] Yu, Y., Pei, H. & Xu, C. "Parameter identification of JONSWAP 
           spectrum acquired by airborne LIDAR" J. Ocean Univ. China 16, 
           998–1002 (2017) https://doi.org/10.1007/s11802-017-3271-2
        '''
        sign_wave_height, avg_period = self.jonswap_params(wind_speed)

        peak_period = avg_period * (0.327 * np.exp(-1.0395) + 1.17)
        peak_ang_freq = 2 * np.pi / peak_period
        peak_ang_freq = peak_ang_freq.reshape(len(peak_ang_freq),1)

        ang_freq = self.ang_freq.reshape(1,len(self.ang_freq))
        sigma = (0.07 * (ang_freq <= peak_ang_freq) + 
                 0.09 * (ang_freq > peak_ang_freq))
        peak_enh_fact = 3.3**np.exp(-((ang_freq / peak_ang_freq - 1) / 
                                      (sigma * np.sqrt(2))) ** 2)
        mod_phillips_ct = 3.32484722 * (sign_wave_height / peak_period**2) ** 2
        mod_phillips_ct = mod_phillips_ct.reshape(len(mod_phillips_ct),1)
        spectrum = (mod_phillips_ct * 96.15764 * ang_freq ** -5 *
                    np.exp(-(1.25 * (ang_freq / peak_ang_freq) ** -4)) * 
                    peak_enh_fact)

        return spectrum, peak_ang_freq
    

    def jonswap_params(self,wind_speed):
        '''
        Obtain the significant wave height and the average period, needed to
        calculate the JONSWAP spectra, from the wind speed via interpolation of
        data from the literature [1].

        Parameters
        ----------
        wind_speed : numeric
            Wind speed [m/s]

        Returns
        -------
        sign_wave_height : numeric
            Significant wave height [m]
        avg_period : numeric
            Average wave period [s]
    
        References
        ----------
        .. [1] P. R. Pinet "Essential invitation to oceanography" Jones & 
           Bartlett Publishers, 2014
        '''
        WIND_SPEED_LIT = np.arange(0, 91, 10) / 3.6
        WAVE_HEIGHT_LIT = [0, 0.25, 0.5, 1.2, 2.5, 4.5, 7.1, 10.3, 14.3, 19.3]
        AVG_PERIOD_LIT = [0, 1.6, 3.2, 4.6, 6.2, 7.7, 9.9, 10.8, 12.4, 13.9]
        c_wh = np.polyfit(WIND_SPEED_LIT, WAVE_HEIGHT_LIT, 3)
        c_avp = np.polyfit(WIND_SPEED_LIT, AVG_PERIOD_LIT, 1)

        sign_wave_height = np.polyval(c_wh, wind_speed)
        avg_period = np.polyval(c_avp, wind_speed)
        return sign_wave_height, avg_period

    
    def pierson_moskowitz_spectrum(self, wind_speed):
        spectrum = wind_speed
        return spectrum
    

    def surface_elevation(self, wind_speed, x, type_spectrum='JONSWAP'):
        '''
        Generate a statistically possible sea surface elevation from the sea
        spectrum.

        Parameter ``type_spectrum`` allows selection of different sea spectra.

        Parameters
        ----------
        wind_speed : numeric
            Wind speed [m/s]
        x : float
            Location at which the wave is evaluated [m]
        type_spectrum : string, default 'JONSWAP'
            Available spectra include the following:

            * 'JONSWAP' - Spectrum for the North Sea [1]
            * 'Pierson-Moskowitz' - TODO: Under development

        Returns
        -------
        elevation : numeric
            Time-resolved statistically possible sea surface elevation [m]
        AmpA : numeric
            Amplitude of the surface elevation related to the cosine [m]
        AmpB : numeric
            Amplitude of the surface elevation related to the sine [m]

        Notes
        -----
        Function adapted from the one created by Sayyed Mohsen Vazirizade from
        the University of Arizona
        smvazirizade@email.arizona.edu
        
        References
        ----------
        .. [1] Yu, Y., Pei, H. & Xu, C. "Parameter identification of JONSWAP 
           spectrum acquired by airborne LIDAR" J. Ocean Univ. China 16, 
           998–1002 (2017) https://doi.org/10.1007/s11802-017-3271-2
        '''
        if type_spectrum == 'JONSWAP':
            spectrum = self.jonswap_spectrum(wind_speed)[0]
        # elif type_spectrum == 'Pierson-Moskowitz':
        #     spectrum = self.pierson_moskowitz_spectrum(wind_speed)
        else:
            raise ValueError('%s is not a valid model for sea spectrum', 
                             type_spectrum)
        
        n_w = len(self.ang_freq)
        dw = self.ang_freq[1] - self.ang_freq[0]
        X = x * np.ones(self.time.shape)
        # AmpA = (spectrum * dw) ** 0.5 * np.random.normal(size=(1, n_w))
        # AmpB = (spectrum * dw) ** 0.5 * np.random.normal(size=(1, n_w))
        AmpA = (spectrum * dw) ** 0.5 * np.ones((1, n_w))
        AmpB = (spectrum * dw) ** 0.5 * np.ones((1, n_w))
        arg_rad = ((self.wave_num.reshape(len(self.wave_num),1)) @
                   (X.reshape(1, len(X))) - 
                   (self.ang_freq.reshape(len(self.ang_freq),1)) @
                   (self.time.reshape(1, len(self.time))))
        elevation = AmpA @ np.cos(arg_rad) + AmpB @ np.sin(arg_rad)

        return elevation, AmpA, AmpB
    

    def get_sea_amplitudes(self, wind_speed_x, wind_speed_y, x, 
                           type_spectrum='JONSWAP'):
        '''
        Get the sea amplitudes at a specified location given the wind speed.

        Parameters
        ----------
        wind_speed_x : numeric
            Wind speed in the x-direction [m/s]
        wind_speed_y : numeric
            Wind speed in the perpendicular direction of ``wind_speed_x`` [m/s]
        x : float
            Location at which the wave is evaluated [m]
        type_spectrum : string, default 'JONSWAP'
            Available spectra include the following:

            * 'JONSWAP' - Spectrum for the North Sea [1]
            * 'Pierson-Moskowitz' - TODO: Under development

        Returns
        -------
        amplitudes : list
            Amplitudes in the x and y directions and for the sine and cosine 
            components of the sea elevation
        '''
        AmpA_x, AmpB_x = self.surface_elevation(wind_speed_x, x,
                                                type_spectrum)[1:3]
        AmpA_y, AmpB_y = self.surface_elevation(wind_speed_y, x,
                                                type_spectrum)[1:3]
        amplitudes = [AmpA_x, AmpB_x, AmpA_y, AmpB_y]
        return amplitudes
    

class Floater:
    '''
    Floater objects contain main dimensions and characteristics of the floating
    body over which PV modules are placed.

    Parameters
    ----------
    mass : float
        Mass of the floater including the PV system components [kg]
    width : float
        Width of the floater [m]
    length : float
        Length of the floater [m]
    thickness : float
        Thickness of the floater [m]
    orientation : float
        Facing direction of the floater [degree]
    '''
    def __init__(self, mass, width, length, thickness, orientation):
        self.mass = mass
        self.width = width
        self.length = length
        self.thickness = thickness
        self.orientation = orientation
    
    def compute_inertia_moment(self, axis):
        """
        Compute the inertia moment for a rectangular floater
        TODO: upgrade to include inertia moments for different shapes

        Parameters
        ----------
        axis : string
            Axis over which the inertia is computed. Options are 'x' and 'y'

        Returns
        -------
        I : float
            Inertia moment
        """
        if axis == 'x':
            I = 1/12*self.mass*(self.length**2 + self.thickness**2)
        elif axis == 'y':
            I = 1/12*self.mass*(self.width**2 + self.thickness**2)
        else:
            raise ValueError('%s is not a valid axis for the inertia', axis)

        return I
    
    def get_inclination_angles(self, axis, Sea, AmpA, AmpB):
        """
        Compute the rotational angle of a rigid floater considering the 
        pressure exerted by the waves on the floater.

        Parameters
        ----------
        axis : string
            Axis of the rotational angle. Options are 'x' and 'y'
        Sea : object
            Container of the characteristics of the sea where the floater is
        AmpA : numeric
            Amplitude of the surface elevation related to the cosine [m]
        AmpB : numeric
            Amplitude of the surface elevation related to the sine [m]

        Returns
        -------
        theta : numeric
            Inclination angle [degree]
        """
        I = self.compute_inertia_moment(axis)
        if axis == 'x':
            wid = self.width; leng = self.length
        elif axis == 'y':
            wid = self.length; leng = self.width
        fact = 19.612 * Sea.density * wid / I
        xx = Sea.wave_num * leng / 2
        aux1 = (1 / (Sea.wave_num**2 * Sea.ang_freq**2) * 
                (np.sin(xx) - xx * np.cos(xx)))
        aux1 = aux1.reshape(len(aux1),1)
        # shift = np.random.uniform(size=(len(Sea.ang_freq),1)) * np.ones((1,len(Sea.time)))
        shift = np.ones((len(Sea.ang_freq), 1)) * np.ones((1, len(Sea.time)))
        arg_rad = ((Sea.ang_freq.reshape(len(Sea.ang_freq), 1)) @ 
                   (Sea.time.reshape(1,len(Sea.time))) + shift)
        aux2 = (AmpA @ (aux1 * np.sin(arg_rad)) + 
                AmpB @ (aux1 * np.cos(arg_rad)))
        theta = fact * aux2
        return theta
    
    def get_tilt_azimuth(self, Sea, sea_amplitudes):
        """
        Get the tilt and azimuth of the floater. First compute the rotational 
        angles of the floater (the pitch and roll) and then perform a 
        coordinate change to azimuth and tilt. 

        Parameters
        ----------
        Sea : object
            Container of the characteristics of the sea where the floater is
        sea_amplitudes : list
            Amplitudes in the x and y directions and for the sine and cosine 
            components of the sea elevation

        Returns
        -------
        tilt : numeric
            Tilt of the floater [degree]
        azimuth : numeric
            Azimuth of the floater [degree]

        Notes
        -----
        The coordinate change is based on the roof to horizontal coordinate 
        change described in Appendix E.4 from the "Solar Energy" book from 
        A. Smets et al.
        """
        AmpA_x, AmpB_x, AmpA_y, AmpB_y = sea_amplitudes
        roll = self.get_inclination_angles('x', Sea, AmpA_y, AmpB_y)
        pitch = self.get_inclination_angles('y', Sea, AmpA_x, AmpB_x)

        tilt = acosd(cosd(pitch) * cosd(roll))
        azimuth = atand(sind(pitch) / tand(roll))
        azimuth[(90-pitch)>90] = azimuth[(90-pitch)>90] + 180
        azimuth[azimuth<0] = azimuth[azimuth<0] + 360
        azimuth[(pitch==0) & (roll==0)] = 0

        tilt = tilt.reshape(tilt.size)
        azimuth = azimuth.reshape(azimuth.size)
        return tilt, azimuth