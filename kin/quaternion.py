import numpy as np
import kin.principal_rotation


class Quaternion:

    def __init__(self, q0, q1, q2, q3):
        # The first component define if is a long (> 180) or short rotation (< 180)
        if q0 > 0:
            self.vector = np.array([q0, q1, q2, q3])
            self.lvector = np.array([-q0, q1, q2, q3])
        else:
            self.vector = np.array([-q0, q1, q2, q3])
            self.lvector = np.array([q0, q1, q2, q3])

        self._dcm = None
        self._rotation_matrix = None
        self._inverse_rotation_matrix = None
        self._addition_matrix = None
        self._subtraction_matrix = None

    @property
    def dcm(self):
        if self._dcm is None:
            q0 = self.vector[0]
            q1 = self.vector[1]
            q2 = self.vector[2]
            q3 = self.vector[3]
            self._dcm = np.array([
                [
                    q0**2 + q1**2 - q2**2 - q3**2,
                    2*(q1*q2 + q0*q3),
                    2*(q1*q3 - q0*q2),
                ], [
                    2*(q1*q2 - q0*q3),
                    q0**2 - q1**2 + q2**2 - q3**2,
                    2*(q2*q3 + q0*q1),

                ], [
                    2*(q1*q3 + q0*q2),
                    2*(q2*q3 - q0*q1),
                    q0**2 - q1**2 - q2**2 + q3**2,
                ]
            ])
        return self._dcm

    @property
    def rotation_matrix(self):
        if self._rotation_matrix is None:
            q0 = self.vector[0]
            q1 = self.vector[1]
            q2 = self.vector[2]
            q3 = self.vector[3]
            self._rotation_matrix = 0.5*np.array([
                [q0, -q1, -q2, -q3],
                [q1, q0, -q3, q2],
                [q2, q3, q0, -q1],
                [q3, -q2, q1, q0],
            ])
        return self._rotation_matrix

    @property
    def inverse_rotation_matrix(self):
        if self._inverse_rotation_matrix is None:
            q0 = self.vector[0]
            q1 = self.vector[1]
            q2 = self.vector[2]
            q3 = self.vector[3]
            self._inverse_rotation_matrix = 0.5*np.array([
                [q0, q1, q2, q3],
                [-q1, q0, q3, -q2],
                [-q2, -q3, q0, q1],
                [-q3, q2, -q1, q0],
            ])
        return self._inverse_rotation_matrix

    @property
    def addition_matrix(self):
        if self._addition_matrix is None:
            q0 = self.vector[0]
            q1 = self.vector[1]
            q2 = self.vector[2]
            q3 = self.vector[3]
            self._addition_matrix = np.array([
                [q0, -q1, -q2, -q3],
                [q1, q0, q3, -q2],
                [q2, -q3, q0, q1],
                [q3, q2, -q1, q0]
            ])
        return self._addition_matrix

    def add(self, q):
        """
        Return a `Quaternion` instance that is the composed rotation of the current
        quaternion and the rotation determined by `q`

        :param q: Rotation to be added to the current queaternion instance.
        :type q: `Quaternion` instance.
        """
        composed_vector = self.addition_matrix @ q.vector
        return Quaternion(*composed_vector)

    def subtract(self, q):
        """
        Return a `Quaternion` instance that is the decomposed rotation of the current
        quaternion and the rotation determined by `q`

        :param q: Rotation to be added to the current queaternion instance.
        :type q: `Quaternion` instance.
        """
        decomposed_vector = np.transpose(self.addition_matrix) @ q.vector
        return Quaternion(*decomposed_vector)

    def as_principal_rotation(self):
        angle = np.arccos(self.vector[0])*2
        if np.isclose(angle, 0.0) or np.isclose(angle, np.pi*2):
            raise ValueError("Impossible to compute principal rotation vector for " + \
                             "a zero rotation")
        s_half_angle = np.sin(angle*0.5)
        vector = self.vector[1:]/s_half_angle
        return kin.principal_rotation.PrincipalRotation(vector, angle)

    @classmethod
    def from_dcm(cls, dcm):
        trace = np.trace(dcm)
        scores = (
            0.25 * (1 + trace),
            0.25 * (1 + 2*dcm[0][0] - trace),
            0.25 * (1 + 2*dcm[1][1] - trace),
            0.25 * (1 + 2*dcm[2][2] - trace),
        )

        q = max(scores)
        if scores.index(q) == 0:
            q0 = np.sqrt(q)
            q1 = 0.25*(dcm[1][2] - dcm[2][1]) / q0
            q2 = 0.25*(dcm[2][0] - dcm[0][2]) / q0
            q3 = 0.25*(dcm[0][1] - dcm[1][0]) / q0
        elif scores.index(q) == 1:
            q1 = np.sqrt(q)
            q0 = 0.25*(dcm[1][2] - dcm[2][1]) / q1
            q2 = 0.25*(dcm[0][1] + dcm[1][0]) / q1
            q3 = 0.25*(dcm[2][0] + dcm[0][2]) / q1
        elif scores.index(q) == 2:
            q2 = np.sqrt(q)
            q0 = 0.25*(dcm[2][0] - dcm[0][2]) / q2
            q1 = 0.25*(dcm[0][1] + dcm[1][0]) / q2
            q3 = 0.25*(dcm[1][2] + dcm[2][1]) / q2
        else:
            q3 = np.sqrt(q)
            q0 = 0.25*(dcm[0][1] - dcm[1][0]) / q3
            q1 = 0.25*(dcm[2][0] + dcm[0][2]) / q3
            q2 = 0.25*(dcm[1][2] + dcm[2][1]) / q3
        quaternion = cls(q0, q1, q2, q3)
        return quaternion

    def __repr__(self):
        return f"Quaternion<{self.vector}>"
