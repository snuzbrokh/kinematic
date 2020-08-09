import math
import numpy as np

import pytest
from kin import EulerAngles

from kin import PrincipalRotation
from kin.helpers import round_radians
from .cfg import ALL_ANGLES


# Singularities in Euler angles ocur when:
# - the order is simetric and the pitch angle is multiple of np.pi
# - the order is not simetric and the pitch angle is multiple of np.pi/2

non_singular_params = [([r1, r2, r3], order) for r1 in ALL_ANGLES
                       for r2 in ALL_ANGLES
                       for r3 in ALL_ANGLES
                       for order in EulerAngles.allowed_orders[10:11]
                       if sum([r1, r2, r3]) and ((order[0] != order[-1] and \
                           not np.isclose(np.abs(round_radians(r2)), np.pi/2))
                       or (order[0] == order[-1] and \
                           not np.isclose(round_radians(r2), 0.0) and \
                           not np.isclose(round_radians(r2), np.pi)))]


@pytest.mark.parametrize('angles,order', non_singular_params)
def test_rotation_matrices(angles, order):
    """
    Test that $[B] . [B]^{-1} = [I_{3 \times 3}]$
    """
    euler = EulerAngles(*angles, order)
    # if the determinant is close to 1.0 the test passes
    assert np.isclose(euler.inverse_rotation_matrix @ euler.rotation_matrix,
                      np.eye(3)).all()
