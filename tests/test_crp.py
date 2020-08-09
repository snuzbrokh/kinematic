import numpy as np

import pytest
from kin.crp import CRP
from kin.helpers import round_radians
from kin.principal_rotation import PrincipalRotation
from kin.quaternion import Quaternion

from .cfg import ALL_ANGLES

non_singular_params = [([np.sin(a)*np.cos(b), np.sin(a)*np.sin(b), np.cos(a)], angle)
                       for a in ALL_ANGLES
                       for b in ALL_ANGLES
                       for angle in ALL_ANGLES
                       if not np.isclose(np.abs(round_radians(angle)), np.pi) and \
                       round_radians(angle)]

singular_params = [([np.sin(a)*np.cos(b), np.sin(a)*np.sin(b), np.cos(a)], angle)
                   for a in ALL_ANGLES
                   for b in ALL_ANGLES
                   for angle in ALL_ANGLES
                   if np.isclose(np.abs(round_radians(angle)), np.pi) or \
                   not round_radians(angle)]


@pytest.mark.parametrize('vector,angle', non_singular_params)
def test_quaternion_conversion(vector, angle):
    """
    Test the conversion from CRP to DCM and vice-versa.
    """
    crp_vector = np.array(vector) * np.tan(angle/2)
    crp_1 = CRP(*crp_vector)
    q = crp_1.as_quaternion()
    crp_2 = q.as_crp()

    assert np.isclose(np.transpose(crp_1.dcm) @ crp_2.dcm, np.eye(3)).all()
    assert np.isclose(np.transpose(crp_1.dcm) @ q.dcm, np.eye(3)).all()
    assert np.isclose(crp_1.vector, crp_2.vector).all()


@pytest.mark.parametrize('vector,angle', non_singular_params)
def test_principal_rotation_conversion(vector, angle):
    """
    Test the conversion from CRP to DCM and vice-versa.
    """
    crp_vector = np.array(vector) * np.tan(angle/2)
    crp_1 = CRP(*crp_vector)
    pr = crp_1.as_principal_rotation()
    crp_2 = pr.as_crp()

    assert np.isclose(np.transpose(crp_1.dcm) @ crp_2.dcm, np.eye(3)).all()
    assert np.isclose(np.transpose(crp_1.dcm) @ pr.dcm, np.eye(3)).all()
    assert np.isclose(crp_1.vector, crp_2.vector).all()


@pytest.mark.parametrize('vector,angle', singular_params)
def test_singular_conversions(vector, angle):
    pr = PrincipalRotation(np.array(vector), angle)
    q = pr.as_quaternion()

    with pytest.raises(ValueError):
        pr.as_crp()

    with pytest.raises(ValueError):
        q.as_crp()


@pytest.mark.parametrize('vector,angle', non_singular_params)
def test_rotation_matrices(vector, angle):
    crp_vector = np.array(vector) * np.tan(angle/2)
    crp = CRP(*crp_vector)
    # if the determinant is close to 1.0 the test passes
    assert np.isclose(crp.inverse_rotation_matrix @ crp.rotation_matrix,
                      np.eye(3)).all()
