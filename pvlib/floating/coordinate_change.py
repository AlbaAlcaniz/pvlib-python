import numpy as np


def coordinate_change(pitch, roll):
    """COORDINATE_CHANGE Convert the pitch and roll into azimuth and tilt
    %
    % Syntax:
    %   [Am,tilt] = coordinate_change(pitch,roll)
    %
    % Description:
    %   Perform a coordinate change from pitch and roll, which are the
    %   rotational angles in the x and y directions, to azimuth and tilt. Based
    %   on the roof to horizontal coordinate change described in Appendix E.4
    %   from the "Solar Energy" book from A. Smets et al.
    %
    % Inputs:
    %   pitch: time-resolved pitch (rotation around y axis) in degrees
    %   roll: time-resolved roll (rotation around x axis) in degrees
    %
    % Outputs:
    %   Am: time-resolved azimuth in degrees. The convention exployed is South:
    %     0°, East: 90°, North: 180°, West: 270°
    %   tilt: time-resolved tilt in degrees

        Args:
            pitch (_type_): _description_
            roll (_type_): _description_

        Returns:
            _type_: _description_
    """

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
