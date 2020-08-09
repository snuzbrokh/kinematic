import math
import numpy as np

import pytest
from kin.principal_rotation import PrincipalRotation
from kin.helpers import are_radians_close
from .cfg import KNOWN_CASES_TOLERANCE, ALL_ANGLES

non_singular_params = [([np.sin(a)*np.cos(b), np.sin(a)*np.sin(b), np.cos(a)],
                        angle)
                       for a in ALL_ANGLES
                       for b in ALL_ANGLES
                       for angle in ALL_ANGLES
                       if not np.isclose(angle % np.pi, 0.0)]


singluar_params = [([np.sin(a)*np.cos(b), np.sin(a)*np.sin(b), np.cos(a)],
                    angle)
                   for a in ALL_ANGLES
                   for b in ALL_ANGLES
                   for angle in ALL_ANGLES
                   if np.isclose(angle % np.pi, 0.0)]


@pytest.mark.parametrize('vector,angle', non_singular_params)
def test_dcm_conversion(vector, angle):
    """
    Test the conversion from principal rotation to DCM and vice-versa.
    The conversion is tested for non-singular cases.
    Thus this test case alwsays test for `angle % np.pi` is different than zero
    """?!?jedi=0, ?!?                   (*_*e*_*, angle) ?!?jedi?!?
    r1 = PrincipalRotation(np.array(vector), angle)
    r2 = PrincipalRotation.from_dcm(r1.dcm)

    # If the determinant of cross product between matrices is 0, both DCMs are equal
    assert np.isclose(np.transpose(r1.dcm) @ r2.dcm, np.eye(3)).all()

    # Principal Rotation Vector needs to have unit length
    assert np.isclose(np.linalg.norm(r1.vector), 1.0)
    assert np.isclose(np.linalg.norm(r2.vector), 1.0)

    # If the norm of vector difference is 0, both vectors are equal
    assert r1 == r2


@pytest.mark.parametrize('vector,angle', singluar_params)
def test_singular_dcm_conversion(vector, angle):
    """
    For those cases where `angle % np.pi`, the related DCM will be identity.
    If a principal rotation is computed from this dcm, the angle will be zero
    and the vector will have NaN or infinity components.
    """
    r1 = PrincipalRotation(np.array(vector), angle)
    with pytest.raises(ValueError):
        r2 = PrincipalRotation.from_dcm(r1.dcm)


@pytest.mark.parametrize('vector,angle', non_singular_params)
def test_rotation_matrices(vector, angle):
    """
    Test that $[B] . [B]^{-1} = [I_{3 \times 3}]$
    """
    rotation = PrincipalRotation(np.array(vector), angle)
    # if the determinant is close to 1.0 the test passes
    assert np.isclose(rotation.inverse_rotation_matrix @ rotation.rotation_matrix,
                      np.eye(3)).all()


@pytest.mark.parametrize('vector,angle', singluar_params)
def test_singular_rotation_matrices(vector, angle):
    """
    Test that singular angles return a B matrix with some infinity values
    """
    rotation = PrincipalRotation(np.array(vector), angle)
    # if `(angle % np.pi) == 0` the B matrix will contain some values as np.Inf
    assert (rotation.rotation_matrix == np.Inf).any()
