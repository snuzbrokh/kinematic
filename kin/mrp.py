import numpy as np

import kin.quaternion

from .helpers import tilde


class MRP:

    def __init__(self, s1, s2, s3):
        self.vector = np.array([s1, s2, s3])
        self._dcm = None
        self._rotation_matrix = None
        self._inverse_rotation_matrix = None

    @property
    def dcm(self):
        if self._dcm is None:
            s_tilde = tilde(self.vector)
            norm_2 = np.linalg.norm(self.vector)**2
            self._dcm = np.identity(3) + \
                (8*s_tilde @ s_tilde - 4*(1 - norm_2)*s_tilde) / (1 + norm_2)**2
        return self._dcm

    @property
    def rotation_matrix(self):
        if self._rotation_matrix is None:
            norm_2 = np.linalg.norm(self.vector)**2
            self._rotation_matrix = 0.25*((1 - norm_2)*np.eye(3) + \
                                          2*tilde(self.vector) + \
                                          2*np.outer(self.vector, self.vector))
        return self._rotation_matrix

    @property
    def inverse_rotation_matrix(self):
        if self._inverse_rotation_matrix is None:
            norm_2 = np.linalg.norm(self.vector)**2
            self._inverse_rotation_matrix = np.transpose(4*self.rotation_matrix) * \
                4/ (1 + norm_2)**2
        return self._inverse_rotation_matrix

    def as_short_rotation(self):
        norm = np.linalg.norm(self.vector)
        if norm >= 1:
            self.vector = -self.vector/norm**2

    def as_quaternion(self):
        """
        Returns a set of Euler Parametters from a set of Modified Rodirgues Parametters
        """
        mrp_2 = self.vector @ self.vector
        b0 = (1-mrp_2) / (1+mrp_2)
        return kin.quaternion.Quaternion(b0, *(2*self.vector / (1 + mrp_2)))

    def add(self, mrp):
        """
        Return a `MRP` instance that is the composite rotation of the
        current orientation and the rotation determined by `mrp`

        :param pr: Rotation to be added to the current `MRP` instance.
        :type pr: `MRP` instance.
        """
        mrp1_2 = np.linalg.norm(mrp.vector)**2
        mrp2_2 = np.linalg.norm(self.vector)**2

        numerator = (1 - mrp1_2)*self.vector + (1 - mrp2_2)*mrp.vector - \
            2 * np.cross(self.vector, mrp.vector)
        denominator = 1 + mrp1_2 * mrp2_2 - 2 * np.dot(self.vector, mrp.vector)
        return MRP(*(numerator/denominator))

    def subtract(self, mrp):
        """
        Return a `MRP` instance that is the decomposed rotation of the
        current orientation and the rotation determined by `mrp`

        :param pr: Rotation to be subtracted to the current `MRP` instance.
        :type pr: `MRP` instance.
        """
        mrp1_2 = np.linalg.norm(mrp.vector)**2
        mrp2_2 = np.linalg.norm(self.vector)**2

        numerator = (1 - mrp1_2)*self.vector - (1 - mrp2_2)*mrp.vector + \
            2 * np.cross(self.vector, mrp.vector)
        denominator = 1 + mrp1_2 * mrp2_2 + 2 * np.dot(self.vector, mrp.vector)
        return MRP(*(numerator/denominator))

    @classmethod
    def from_dcm(cls, dcm):
        if np.isclose(np.trace(dcm), -1.0):
            raise ValueError("Can not compute modified Rodrigues parametters from a " \
                             "DCM that represent a 180º principal rotation.")
        trace_term = (np.trace(dcm) + 1)**0.5
        denom = (trace_term*(trace_term+2))
        num = np.array([
            dcm[1][2] - dcm[2][1],
            dcm[2][0] - dcm[0][2],
            dcm[0][1] - dcm[1][0]
        ])
        return cls(*(num/denom))

    def __repr__(self):
        return f"MRP<{self.vector}>"

    def __add__(self, o):
        """
        Implements modified Rodrigues parametters direct addition. This is faster than
        the `MRP.add` method but is not that precise for long rotation as is the
        linearized version of MRP addition.
        """
        if not isinstance(o, MRP):
            raise ValueError(f"Can not add MRP and {o}")
        return MRP(*(self.vector + o.vector))

    def __sub__(self, o):
        """
        Implements modified Rodrigues parametters direct subtraction. This is faster than
        the `MRP.subtract` method but is not that precise for long rotation as is the
        linearized version of MRP subtraction.
        """
        if not isinstance(o, MRP):
            raise ValueError(f"Can not subtract MRP and {o}")
        return MRP(*(self.vector - o.vector))
