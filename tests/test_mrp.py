import numpy as np

import pytest
from kin import MRP
from kin import Quaternion
from .cfg import ALL_ANGLES

non_singular_params = [(
    np.sin(psi)*np.sin(phi)*np.cos(theta),
    np.sin(psi)*np.sin(phi)*np.sin(theta),
    np.sin(psi)*np.cos(phi),
    np.cos(psi))
                       for psi in ALL_ANGLES
                       for phi in ALL_ANGLES
                       for theta in ALL_ANGLES]


@pytest.mark.parametrize('vector', non_singular_params)
def test_dcm_conversion(vector):
    """
    Test the conversion from quaternion to DCM and vice-versa.
    """
    s = np.array(vector[1:])/(1 + vector[0])
    mrp_1 = MRP(*s)
    if np.isclose(np.trace(mrp_1.dcm), -1.): # Principal Rotation is close to be 180º
        with pytest.raises(ValueError):
            MRP.from_dcm(mrp_1.dcm)
    else:
        mrp_2 = MRP.from_dcm(mrp_1.dcm)
        assert np.isclose(np.transpose(mrp_1.dcm) @ mrp_2.dcm, np.eye(3)).all()


@pytest.mark.parametrize('vector', non_singular_params)
def test_quaternion_conversion(vector):

    q_1 = Quaternion(*vector)
    mrp1 = q_1.as_mrp()
    q_2 = mrp1.as_quaternion()

    assert np.isclose(q_1.vector, q_2.vector).all()


@pytest.mark.parametrize('vector', non_singular_params)
def test_rotation_matrices(vector):
    s = np.array(vector[1:])/(1 + vector[0])
    mrp = MRP(*s)
    mrp.as_short_rotation()
    # if the determinant is close to 1.0 the test passes
    assert np.isclose(mrp.inverse_rotation_matrix @ mrp.rotation_matrix,
                      np.eye(3)).all()
