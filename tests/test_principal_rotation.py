import math
import numpy as np

import pytest
from kin.principal_rotation import PrincipalRotation
from kin.helpers import are_radians_close
from .cfg import KNOWN_CASES_TOLERANCE, ALL_ANGLES

non_singular_params = [(np.array([np.sin(a)*np.cos(b), np.sin(a)*np.sin(b), np.cos(a)]),
                        angle)
                       for a in ALL_ANGLES
                       for b in ALL_ANGLES
                       for angle in ALL_ANGLES
                       if not np.isclose(angle % np.pi, 0.0)]


singluar_params = [(np.array([np.sin(a)*np.cos(b), np.sin(a)*np.sin(b), np.cos(a)]),
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
    """
    r1 = PrincipalRotation(vector, angle)
    r1.compute_dcm()
    r2 = PrincipalRotation.from_dcm(r1.dcm)
    r2.compute_dcm()

    # If the determinant of cross product between matrices is 0, both DCMs are equal
    assert np.isclose(np.transpose(r1.dcm) @ r2.dcm, np.eye(3)).all()

    # Principal Rotation Vector needs to have unit length
    assert np.isclose(np.linalg.norm(r1.vector), 1.0)
    assert np.isclose(np.linalg.norm(r2.vector), 1.0)

    # If the norm of vector difference is 0, both vectors are equal
    assert np.isclose(np.abs(r1.vector) - np.abs(r2.vector), 0.0).all()
    assert are_radians_close(r1.angle, r2.angle)


@pytest.mark.parametrize('vector,angle', singluar_params)
def test_singular_dcm_conversion(vector, angle):
    """
    For those cases where `angle % np.pi`, the related DCM will be identity.
    If a principal rotation is computed from this dcm, the angle will be zero
    and the vector will have NaN or infinity components.
    """
    r1 = PrincipalRotation(vector, angle)
    r1.compute_dcm()
    with pytest.raises(ValueError):
        r2 = PrincipalRotation.from_dcm(r1.dcm)


@pytest.mark.parametrize('vector,angle', non_singular_params)
def test_B_matrix_and_inverse(vector, angle):
    """
    Test that $[B] . [B]^{-1} = [I_{3 \times 3}]$
    """
    rotation = PrincipalRotation(vector, angle)
    rotation.compute_B_matrix()
    # if the determinant is close to 1.0 the test passes
    assert np.isclose(rotation.invB @ rotation.B, np.eye(3)).all()


@pytest.mark.parametrize('vector,angle', singluar_params)
def test_singular_B_matrix(vector, angle):
    """
    Test that singular angles return a B matrix with some infinity values
    """
    rotation = PrincipalRotation(vector, angle)
    rotation.compute_B_matrix()
    # if `(angle % np.pi) == 0` the B matrix will contain some values as np.Inf
    assert (rotation.B == np.Inf).any()
