import numpy as np

def cosd(angle):
    """
    Trigonometric cosine with angle input in degrees.

    Parameters
    ----------
    angle : float or array-like
        Angle in degrees

    Returns
    -------
    result : float or array-like
        Cosine of the angle
    """
    res = np.cos(np.radians(angle))
    return res


def sind(angle):
    """
    Trigonometric sine with angle input in degrees.

    Parameters
    ----------
    angle : float
        Angle in degrees

    Returns
    -------
    result : float
        Sin of the angle
    """
    res = np.sin(np.radians(angle))
    return res


def tand(angle):
    """
    Trigonometric tangent with angle input in degrees.

    Parameters
    ----------
    angle : float
        Angle in degrees

    Returns
    -------
    result : float
        Tan of the angle
    """
    res = np.tan(np.radians(angle))
    return res


def asind(number):
    """
    Trigonometric inverse sine returning an angle in degrees.

    Parameters
    ----------
    number : float
        Input number

    Returns
    -------
    result : float
        arcsin result
    """
    res = np.degrees(np.arcsin(number))
    return res


def acosd(number):
    """
    Trigonometric inverse cosine returning an angle in degrees.

    Parameters
    ----------
    number : float
        Input number

    Returns
    -------
    result : float
        arccos result
    """
    res = np.degrees(np.arccos(number))
    return res


def atand(number):
    """
    Trigonometric inverse tangent returning an angle in degrees.

    Parameters
    ----------
    number : float
        Input number

    Returns
    -------
    result : float
        arctan result
    """
    res = np.degrees(np.arctan(number))
    return res