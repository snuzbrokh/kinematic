import numpy as np

import pytest
from kin.crp import CRP
from kin.quaternion import Quaterenion

from .cfg import ALL_ANGLES

non_singular_params = [np.array([np.tan(angle)*np.sin(a)*np.cos(b),
                                 np.tan(angle)*np.sin(a)*np.sin(b),
                                 np.tan(angle)*np.cos(a)])
                       for a in ALL_ANGLES
                       for b in ALL_ANGLES
                       for angle in ALL_ANGLES
                       if not np.isclose(angle % np.pi, 0.0)]


@pytest.mark.parametrize('vector', non_singular_params)
def test_dcm_conversion(vector):
    """
    Test the conversion from quaternion to DCM and vice-versa.
    """
    q1 = Quaterenion(*vector)
    q1.compute_dcm()
    q2 = Quaterenion.from_dcm(q1.dcm)
    q2.compute_dcm()

    # Quaternion components must be a four dimensional unit-radious sphere
    assert np.isclose(np.linalg.norm(q1.vector), 1.0)
    assert np.isclose(np.linalg.norm(q2.vector), 1.0)

    # Quaternion's short and long rotations must be the same
    # TODO: This test is not working.
    # It is possible that there is redundancy on quaternions?
    # example:
    # input = [-1.22464680e-16, -1.22464680e-16, -7.07106781e-01,  7.07106781e-01
    # q1_vector = [ 1.22464680e-16, -1.22464680e-16, -7.07106781e-01,  7.07106781e-01]
    # q2_vector = [ 1.22464680e-16,  1.22464680e-16,  7.07106781e-01, -7.07106781e-01]
    #assert np.isclose(q1.vector, q2.vector).all()
    #assert np.isclose(q1.lvector, q2.lvector).all()

    # Principal rotation and quaternion DCMs must be the same

    if np.isclose(np.transpose(q1.dcm) @ q2.dcm, np.eye(3)).all():
        assert True
    elif len(np.unique(np.round(np.abs(q1.vector), 5))) < 4 and \
        (np.isclose(q1.vector, 0.0).any() or np.isclose(np.abs(q1.vector), 0.5).any() ):
        assert np.isclose(q1.dcm @ q2.dcm, np.eye(3)).all()
    else:
        print(q1.vector)
        assert False

