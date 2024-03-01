import numpy as np


def pressure_methodology(I, rho, g, wid, k, leng, w, t, AmpA, AmpB):
    """PRESSURE_METHODOLOGY Calculate the rotational angle from wave pressure
    %
    % Syntax:
    %   theta = pressure_methodology(I, rho, g, wid, k, leng, w, t, AmpA, AmpB)
    %
    % Description:
    %   Compute the time-resolved rotational angle of a rigid floater
    %   considering the pressure exerted by the waves on the floater.
    %
    % Inputs:
    %   I - moment of inertia around the axis of rotation [kg*m^2]
    %   rho - seawater density [kg/m^3]
    %   g - gravitational acceleration [m/s^2]
    %   wid - dimension of the floater parallel to the axis of rotation [m]
    %   leng - dimension of the floater perpendicular to the axis of rotation
    %     [m]
    %   w - angular frequency of the spectra [rad/s]
    %   t - time for every time step in the spectra [s]
    %   AmpA - amplitude of the surface elevation related to the cosine of the
    %     elevation [m]
    %   AmpB - amplitude of the surface elevation related to the sine of the
    %     elevation [m]
    %
    % Outputs:
    %   theta - time-resolved rotational angle [degrees]

        Args:
            I (_type_): _description_
            rho (_type_): _description_
            g (_type_): _description_
            wid (_type_): _description_
            k (_type_): _description_
            leng (_type_): _description_
            w (_type_): _description_
            t (_type_): _description_
            AmpA (_type_): _description_
            AmpB (_type_): _description_

        Returns:
            _type_: _description_
    """

    fact = 2 * rho * g * wid / I
    xx = k * leng / 2
    aux1 = 1 / (k**2 * w**2) * (np.sin(xx) - xx * np.cos(xx))
    aux1 = aux1.reshape(len(aux1),1)
    # shift = np.random.uniform(size=(len(w),1)) * np.ones((1,len(t)))
    shift = np.ones((len(w),1)) * np.ones((1,len(t)))
    arg_rad = (w.reshape(len(w),1))@(t.reshape(1,len(t))) + shift
    aux2 = AmpA @ (aux1*np.sin(arg_rad)) + AmpB @ (aux1*np.cos(arg_rad))
    theta = fact * aux2
    return theta
