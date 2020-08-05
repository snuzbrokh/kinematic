"""
Euler's Principal Rotation: A rigid body coordinate reference frame can be brought from
and arbitrary initial orientation to an arbitrary final orientation by a single rigid
rotation thorugh a principal angle $\Phi$ about the principal axis :math:`\hat {\bf e}` (...)
TODO: fix citation [1](HS)
"""

import numpy as np
from kin.helpers import tilde, round_radians
import kin.quaternion


class PrincipalRotation:

    def __init__(self, e, angle):
        """
        :param e: Principal rotation vector. `np.arrray` with `shape` equal to (1, 3).
            Must be unique length to be valid
        :param angle: Principal rotation angle in radians
        """
        self.angle = round_radians(angle)
        self.vector = e
        self._dcm = None
        self._rotation_matrix = None
        self._inverse_rotation_matrix = None

    @classmethod
    def from_dcm(cls, dcm):
        """
        Compute principal rotation from dcm.
        This has singularities on angle % np.pi == 0

        It will always return the principal rotation angle as positive,
        but regarding the fact that :math:`(\Phi, e) = (-\Phi, -e)` the rotation
        is not affected.
        """
        # always return the short rotation for the angle
        angle = np.arccos(0.5*(dcm[0][0]+dcm[1][1]+dcm[2][2]-1))

        if np.isnan(angle) or np.isclose(angle, np.pi) or np.isclose(angle, 0.0):
            raise ValueError("Mapping from DCM to Principal Rotation is not " + \
                             f"uniquely defined for a rotation angle of {angle}.")

        vector = np.array([
            dcm[1][2]-dcm[2][1],
            dcm[2][0]-dcm[0][2],
            dcm[0][1]-dcm[1][0],
        ])/(2*np.sin(angle))
        rotation = cls(vector, angle)
        return rotation

    @property
    def dcm(self):
        if self._dcm is None:
            s = 1 - np.cos(self.angle)
            self._dcm = np.array([
                [
                    self.vector[0]**2*s + np.cos(self.angle),
                    self.vector[0]*self.vector[1]*s + self.vector[2]*np.sin(self.angle),
                    self.vector[0]*self.vector[2]*s - self.vector[1]*np.sin(self.angle)
                ], [
                    self.vector[1]*self.vector[0]*s - self.vector[2]*np.sin(self.angle),
                    self.vector[1]**2*s + np.cos(self.angle),
                    self.vector[1]*self.vector[2]*s + self.vector[0]*np.sin(self.angle)
                ], [
                    self.vector[2]*self.vector[0]*s + self.vector[1]*np.sin(self.angle),
                    self.vector[2]*self.vector[1]*s - self.vector[0]*np.sin(self.angle),
                    self.vector[2]**2*s + np.cos(self.angle)
                ]
            ])
        return self._dcm

    @property
    def rotation_matrix(self):
        # TODO: This is not passing tests
        if self._rotation_matrix is None:
            gamma = self.vector*self.angle
            tilde_gamma = tilde(gamma)
            cot_half_gamma = np.tan(np.pi/2 - self.angle/2)
            self._rotation_matrix = np.eye(3) + 0.5*tilde_gamma + \
                (1/self.angle**2)*(1 - self.angle*0.5*cot_half_gamma)*tilde_gamma**2
        return self._rotation_matrix

    @property
    def inverse_rotation_matrix(self):
        # TODO: This is not passing tests
        if self._inverse_rotation_matrix is None:
            gamma = self.vector*self.angle
            tilde_gamma = tilde(gamma)
            self._inverse_rotation_matrix = np.eye(3) - \
                ((1-np.cos(self.angle))/self.angle**2)*tilde_gamma + \
                ((self.angle - np.sin(self.angle))/self.angle**3)*tilde_gamma**2
        return self._inverse_rotation_matrix

    def as_quaternion(self):
        scalar = np.cos(0.5*self.angle)
        s_half_angle = np.sin(0.5*self.angle)
        vector = self.vector*s_half_angle
        return kin.quaternion.Quaternion(scalar, *vector)

    def as_crp(self):
        if np.isclose(self.angle, 0.0):
            raise ValueError("Can not compute classical Rodirgues parametters from " + \
                             f"a zero rotation: pr_vector={self.vector}, " + \
                             f"pr_angle={self.angle}")

        if np.isclose(np.abs(self.angle), np.pi):
            raise ValueError("Classical Rodirgues parametters goes singular for " + \
                             f"±180º rotations: pr_vector={self.vector}, " + \
                             f"pr_angle={self.angle}")
        return kin.crp.CRP(*(np.tan(self.angle/2)*self.vector))

    def __eq__(self, o):
        if not isinstance(o, PrincipalRotation):
            return False

        if np.isclose(self.vector, o.vector).all() and \
                (np.isclose(self.angle, o.angle) or
                 np.isclose(self.angle, o.angle - np.pi*2) or
                 np.isclose(self.angle - np.pi*2, o.angle)):
            return True

        if np.isclose(self.vector, -o.vector).all() and \
                (np.isclose(self.angle, -o.angle) or
                 np.isclose(self.angle, -o.angle - np.pi*2) or
                 np.isclose(self.angle - np.pi*2, -o.angle)):
            return True

        return False

    def __repr__(self):
        return f"PrincipalRotation<angle={self.angle}, vector={self.vector}>"


