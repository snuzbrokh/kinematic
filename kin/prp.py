import numpy as np


def PRA_from_DCM(dcm):
    """
    Returns the principal rotation angle from a DCM matrix
    """

    return np.arccos(0.5*(dcm[0][0]+dcm[1][1]+dcm[2][2]-1))


def PRV_from_DCM(dcm):
    """
    Returns the principal rotation vector from a DCM matrix
    """
    pra = PRA_from_DCM(dcm)
    coef = 1/(2*np.sin(pra))
    return coef*np.array([
        dcm[1][2]-dcm[2][1],
        dcm[2][0]-dcm[0][2],
        dcm[0][1]-dcm[1][0],
    ])


def DCM_from_PRP(ang, e):
    """
    Returns the DCM from a principal rotation parameters set.
    """
    s = 1 - np.cos(ang)
    return np.array([
        [
            np.sqrt(e[0])*s + np.cos(ang),
            e[0]*[e][1]*s + e[2]*np.sin(ang),
            e[0]*e[2]*s - e[1]*np.sin(ang)
        ], [
            e[1]*e[0]*s - e[2]*np.sin(ang),
            np.sqrt(e[1])*s + np.cos(ang),
            e[1]*e[2]*s + e[0]*np.sin(ang)
        ], [
            e[2]*e[0]*s - e[1]*np.sin(ang),
            e[2]*e[1]*s - e[0]*np.sin(ang),
            np.sqrt(e[2])*s + np.cos(ang)
        ]
    ])

