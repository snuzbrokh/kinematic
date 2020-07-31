import numpy as np
from .quaternion import Quaterenion
from .helpers import tilde


class CRP:

    def __init__(self, q1, q2, q3):
        self.vector = np.array([q1, q2, q3])
        self.dcm = None
        self.B = None
        self.invB = None

    def _reset_matrices(self):
        self.dcm = None
        self.B = None
        self.invB = None

    def compute_dcm(self):
        crp2 = np.transpose(self.vector) @ self.vector
        return ((1 - crp2)*np.eye(3) + 2*crp2 - 2*tilde[self.vector]) / (1 + crp2)

    def compute_B_matrix(self):
        raise NotImplementedError("B matrix not implemented for " + \
                                  "Classic Rodrigues Parametters")

    def as_quaternion(self):
        q0 = 1 / np.sqrt(1 + np.transpose(self.vector) @ self.vector)
        return Quaterenion(q0, *(self.vector*q0))

    def add(self, q):
        """
        Adds two sets of Classic Rodirgues Parametters
        """
        q_cross = np.cross(self.vector, q.vector)
        q_dot = self.vector @ q.vector
        self.vector = (self.vector + q.vector - q_cross) / (1 - q_dot)
        self._reset_matrices()

    def subtracts(self, q):
        """
        Subtracts two sets of Classic Rodirgues Parametters
        """
        q_cross = np.cross(self.vector, q.vector)
        q_dot = self.vector @ q.vector
        self.vector = (self.vector - q.vector + q_cross) / (1 + q_dot)
        self._reset_matrices()

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

    @classmethod
    def from_quaternion(cls, quaternion):
        if np.isclose(quaternion.vector[0], 0.0):
            raise ValueError()
        return cls(*(quaternion.vector[1:]/quaternion.vector[0]))
