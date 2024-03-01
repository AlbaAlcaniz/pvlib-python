import numpy as np


def compute_surface_elevation(S, w, t, k, x):
    """COMPUTE_SURFACE_ELEVATION Compute the sea surface elevation
    %
    % Syntax:
    %   [eta, AmpA, AmpB] = compute_surface_elevation(S, w, t, k, x)
    %
    % Description:
    %     Generate a statistically possible sea surface elevation from the sea
    %     spectrum.
    %
    % Note:
    %     Function adapted from the one created by Sayyed Mohsen Vazirizade
    %     from the University of Arizona
    %     smvazirizade@email.arizona.edu
    %
    % Inputs:
    %   S - time-resolved spectra for every angular frequency in m^2*s
    %   w - angular frequency of the spectra in rad/s
    %   t - time for every time step in the spectra in s
    %   w - wave number in 1/m
    %   x - location at which the wave is evaluated in m
    %
    % Outputs:
    %   eta - time-resolved statistically possible sea surface elevation in m
    %   AmpA - amplitude of the surface elevation related to the cosine in m
    %   AmpB - amplitude of the surface elevation related to the sine in m

        Args:
            S (_type_): _description_
            w (_type_): _description_
            t (_type_): _description_
            k (_type_): _description_
            x (_type_): _description_

        Returns:
            _type_: _description_
    """

    n_w = len(w)
    dw = w[1] - w[0]
    X = x * np.ones(t.shape)
    # AmpA = (S*dw)**0.5 * np.random.normal(size=(1,n_w))
    # AmpB = (S*dw)**0.5 * np.random.normal(size=(1,n_w))
    AmpA = (S*dw)**0.5 * np.ones((1,n_w))
    AmpB = (S*dw)**0.5 * np.ones((1,n_w))
    arg_rad = (k.reshape(len(k),1))@(X.reshape(1,len(X))) - (w.reshape(len(w),1))@(t.reshape(1,len(t)))
    eta = AmpA@np.cos(arg_rad) + AmpB@np.sin(arg_rad)
    return eta, AmpA, AmpB
