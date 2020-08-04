import numpy as np

from kin.quaternion import Quaternion
from kin.principal_rotation import PrincipalRotation

from .helpers import tilde


class CRP:

    def __init__(self, q1, q2, q3):
        self.vector = np.array([q1, q2, q3])
        self._dcm = None
        self._rotation_matrix = None
        self._inverse_rotation_matrix = None

    @property
    def dcm(self):
        if self._dcm is None:
            q2 = self.vector @ self.vector
            self._dcm = ((1-q2)*np.eye(3) + 2*np.outer(self.vector, self.vector) - \
                         2*tilde(self.vector))/(1 + q2)
        return self._dcm

    @property
    def rotation_matrix(self):
        if self._rotation_matrix is None:
            self._rotation_matrix = 0.5*(np.eye(3) + tilde(self.vector) + \
                                         np.outer(self.vector, self.vector))
        return self._rotation_matrix

    @property
    def inverse_rotation_matrix(self):
        if self._inverse_rotation_matrix is None:
            self._inverse_rotation_matrix = 2*(np.eye(3) - tilde(self.vector)) \
                /(1 + self.vector @ self.vector)
        return self._inverse_rotation_matrix

    def as_quaternion(self):
        q0 = 1 / np.sqrt(1 + np.transpose(self.vector) @ self.vector)
        return Quaternion(q0, *(self.vector*q0))

    def as_principal_rotation(self):
        norm = np.linalg.norm(self.vector)
        angle = np.arctan(norm)*2
        vector = self.vector/norm
        return PrincipalRotation(vector, angle)

    def add(self, q):
        """
        Adds two sets of Classic Rodirgues Parametters
        """
        q_cross = np.cross(self.vector, q.vector)
        q_dot = self.vector @ q.vector
        return CRP(*((self.vector + q.vector - q_cross) / (1 - q_dot)))

    def subtract(self, q):
        """
        Subtracts two sets of Classic Rodirgues Parametters
        """
        q_cross = np.cross(self.vector, q.vector)
        q_dot = self.vector @ q.vector
        return CRP(*((self.vector - q.vector + q_cross) / (1 + q_dot)))

    @classmethod
    def from_dcm(cls, dcm):
        raise NotImplementedError(
            "The simplest way to extract the classical Rodrigues parametters from " + \
            "a given direction cosine matrix is to determine the quaternion first " + \
            "then use `from_quaternion` classmethod to find the corresponding " + \
            "Rodrigues parametters.")

    @classmethod
    def from_principal_rotation(cls, pr):
        if np.isclose(pr.angle % np.pi, 0.0):
            raise ValueError()
        return cls(*(np.tan(pr.angle)*pr.vector))

    def __add__(self, o):
        """
        Implements classical Rodrigues parametters direct addition. This is faster than
        the `CRP.add` method but is not that precise for long rotation as is the
        linearized version of CRP addition.
        """
        if not isinstance(o, CRP):
            raise ValueError(f"Can not add CRP and {o}")
        return CRP(*(self.vector + o.vector))

    def __sub__(self, o):
        """
        Implements classical Rodrigues parametters direct subtraction. This is faster than
        the `CRP.subtract` method but is not that precise for long rotation as is the
        linearized version of CRP subtraction.
        """
        if not isinstance(o, CRP):
            raise ValueError(f"Can not subtract CRP and {o}")
        return CRP(*(self.vector - o.vector))

    def __repr__(self):
        return f"CRP<{self.vector}>"
