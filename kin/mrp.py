import numpy as np
from .quaternion import Quaterenion
from .helpers import tilde


class MRP:

    def __init__(self, s1, s2, s3):
        self.vector = np.array([s1, s2, s3])
        self.dcm = None
        self.B = None
        self.invB = None

    def as_short_rotation(self):
        norm = np.linalg.norm(self.vector)
        if norm >= 1:
            self.vector = -self.vector/norm**2

    def compute_dcm(self):
        s_tilde = tilde(self.vector)
        norm_2 = np.linalg.norm(self.vector)**2
        self.dcm = np.identity(3) + \
            (8*s_tilde @ s_tilde - 4*(1 - norm_2)*s_tilde) / (1 + norm_2)**2

    def as_quaternion(self):
        """
        Returns a set of Euler Parametters from a set of Modified Rodirgues Parametters
        """
        mrp_2 = self.vector @ self.vector
        b0 = (1-mrp_2) / (1+mrp_2)
        return Quaterenion(b0, *(2*self.vector / (1 + mrp_2)))

    @classmethod
    def from_principal_rotation(cls, pr):
        return cls(*(np.tan(pr.angle/4)*pr.vector))

    @classmethod
    def from_quaternion(cls, quaternion):
        """
        Returns a set of Modified Rodirgues Parametters from a set of Euler Parametters
        """
        s = quaternion.vector[1:] / (1 + quaternion.vector[0])
        return cls(*s)

    @classmethod
    def from_dcm(cls, dcm):
        trace_term = (np.matrix.trace(dcm) + 1)**0.5
        denom = (trace_term*(trace_term+2))
        num = np.array([
            dcm[1][2] - dcm[2][1],
            dcm[2][0] - dcm[0][2],
            dcm[0][1] - dcm[1][0]
        ])
        return cls(*(num/denom))

def MRP_add(mrp2, mrp1):
    """
    Adds two sets of Modified Rodirgues Parametters
    """
    mrp1_sqrt = np.linalg.norm(mrp1)**2
    mrp2_sqrt = np.linalg.norm(mrp2)**2

    numerator = (1 - mrp1_sqrt)*mrp2 + (1 - mrp2_sqrt)*mrp1 - 2 * np.cross(mrp2, mrp1)
    denominator = 1 + mrp1_sqrt * mrp2_sqrt - 2 * np.dot(mrp1, mrp2)
    s = numerator/denominator
    return s


def MRP_subtract(mrp, mrp1):
    """
    Subtracts two sets of Modified Rodirgues Parametters
    """
    mrp1_sqrt = np.linalg.norm(mrp1)**2
    mrp_sqrt = np.linalg.norm(mrp)**2

    numerator = (1 - mrp1_sqrt)*mrp - (1 - mrp_sqrt)*mrp1 + 2 * np.cross(mrp, mrp1)
    denominator = 1 + mrp1_sqrt * mrp_sqrt + 2 * np.dot(mrp1, mrp)
    s = numerator/denominator
    return s
