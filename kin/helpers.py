import numpy as np


def tilde(x):
    return np.array([
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0]
    ])

def are_radians_close(a, b, tolerance=10e+7):
    return abs(a % np.pi - b % np.pi)/np.pi <= tolerance

simplify_radians = lambda x: (x % (np.pi*2))*np.sign(x)
