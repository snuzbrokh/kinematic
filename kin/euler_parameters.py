import numpy as np


def EP_to_DCM(b0, b1, b2, b3):
    """
    Returns a DCM from a set of Euler parameters
    """
    return np.array([
        [
            b0**2 + b1**2 - b2**2 - b3**2,
            2*(b1*b2 + b0*b3),
            2*(b1*b3 - b0*b2),
        ], [
            2*(b1*b2 - b0*b3),
            b0**2 - b1**2 + b2**2 - b3**2,
            2*(b2*b3 + b0*b1),

        ], [
            2*(b1*b3 + b0*b2),
            2*(b2*b3 - b0*b1),
            b0**2 - b1**2 - b2**2 + b3**2,
        ]
    ])


def DCM_to_EP(dcm):
    """
    Returns a set of euler parameters from a DCM
    """
    tentative_betas = {
        'b0': np.sqrt(0.25 * (1 + np.trace(dcm))),
        'b1': np.sqrt(0.25 * (1 + 2*dcm[0][0] - np.trace(dcm))),
        'b2': np.sqrt(0.25 * (1 + 2*dcm[1][1] - np.trace(dcm))),
        'b3': np.sqrt(0.25 * (1 + 2*dcm[2][2] - np.trace(dcm))),
    }

    beta, value = [b for b in tentative_betas.items()
                   if b[1] == max(tentative_betas.values())].pop()
    if beta == 'b0':
        b_0 = value
        b_1 = (dcm[1][2] - dcm[2][1]) / (4.0*b_0)
        b_2 = (dcm[2][0] - dcm[0][2]) / (4.0*b_0)
        b_3 = (dcm[0][1] - dcm[1][0]) / (4.0*b_0)
    elif beta == 'b1':
        b_1 = value
        b_0 = (dcm[1][2] - dcm[2][1]) / (4.0*b_1)
        b_2 = (dcm[2][0] - dcm[0][2]) / (4.0*b_0)
        b_3 = (dcm[0][1] - dcm[1][0]) / (4.0*b_0)
    elif beta == 'b2':
        b_2 = value
        b_0 = (dcm[2][0] - dcm[0][2]) / (4.0*b_2)
        b_1 = (dcm[1][2] - dcm[2][1]) / (4.0*b_0)
        b_3 = (dcm[0][1] - dcm[1][0]) / (4.0*b_0)
    else:
        b_3 = value
        b_0 = (dcm[0][1] - dcm[1][0]) / (4.0*b_3)
        b_1 = (dcm[1][2] - dcm[2][1]) / (4.0*b_0)
        b_2 = (dcm[2][0] - dcm[0][2]) / (4.0*b_0)
    return (b_0, b_1, b_2, b_3)


# TODO: refactor this two metods to EP_add and EP_sustract.
def EP_addition_matrix(eparams):
    return np.array([
        [eparams[0], -eparams[1], -eparams[2], -eparams[3]],
        [eparams[1], eparams[0], -eparams[3], eparams[2]],
        [eparams[2], eparams[3], eparams[0], -eparams[1]],
        [eparams[3], -eparams[2], eparams[1], eparams[0]]
    ])


def EP_subtraction_matris(eparams):
    return np.array([
        [eparams[0], -eparams[1], -eparams[2], -eparams[3]],
        [eparams[1], eparams[0], eparams[3], -eparams[2]],
        [eparams[2], -eparams[3], eparams[0], eparams[1]],
        [eparams[3], eparams[2], -eparams[1], eparams[0]]
    ])

