import numpy as np

import pytest
from kin.principal_rotation import PrincipalRotation
from kin.quaternion import Quaternion

from .cfg import ALL_ANGLES

# We use the four dimensions, the generalization of spherical coordinates in four
# dimensions from https://mathworld.wolfram.com/Hypersphere.html
non_singular_params = [[
    np.sin(psi)*np.sin(phi)*np.cos(theta),
    np.sin(psi)*np.sin(phi)*np.sin(theta),
    np.sin(psi)*np.cos(phi),
    np.cos(psi)]
                       for psi in ALL_ANGLES
                       for phi in ALL_ANGLES
                       for theta in ALL_ANGLES]


@pytest.mark.parametrize('vector', non_singular_params)
def test_dcm_conversion(vector):
    """
    Test the conversion from quaternion to DCM and vice-versa.
    """
    q1 = Quaternion(*vector)
    q2 = Quaternion.from_dcm(q1.dcm)

    # Quaternion components must be a four dimensional unit-radious sphere
    assert np.isclose(np.linalg.norm(q1.vector), 1.0)
    assert np.isclose(np.linalg.norm(q2.vector), 1.0)

    pr1 = q1.as_principal_rotation()
    pr2 = q2.as_principal_rotation()

    # Principal rotation and quaternion DCMs must be the same
    if np.isclose(np.trace(q1.dcm), -1):
        assert np.isclose(q1.dcm @ q2.dcm, np.eye(3)).all()
    else:
        assert np.isclose(np.transpose(q1.dcm) @ q2.dcm, np.eye(3)).all()
        assert pr1 == pr2


    # Quaternion's short and long rotations must be the same
    # TODO: This test is not working.
    # It is possible that there is redundancy on quaternions?
    # example:
    # input = [-1.22464680e-16, -1.22464680e-16, -7.07106781e-01,  7.07106781e-01
    # q1_vector = [ 1.22464680e-16, -1.22464680e-16, -7.07106781e-01,  7.07106781e-01]
    # q2_vector = [ 1.22464680e-16,  1.22464680e-16,  7.07106781e-01, -7.07106781e-01]
    # assert np.isclose(q1.vector, q2.vector).all()
    # assert np.isclose(q1.lvector, q2.lvector).all()

    # if np.isclose(np.transpose(q1.dcm) @ q2.dcm, np.eye(3)).all():
        # assert True
    # elif len(np.unique(np.round(np.abs(q1.vector), 5))) < 4 and \
        # (np.isclose(q1.vector, 0.0).any() or np.isclose(np.abs(q1.vector), 0.5).any() ):
        # assert np.isclose(q1.dcm @ q2.dcm, np.eye(3)).all()
    # else:
        # assert False


@pytest.mark.parametrize('vector', non_singular_params)
def test_quaternion_to_principal_rotation_conversion(vector):
    """
    Test the conversion from quaternion to principal rotation and vice-versa.
    """

    q1 = Quaternion(*vector)

    if np.isclose(q1.vector[0], 1.0):
        # If the quaternion represent a 180º rotation, we can not know the principal
        # rotation's vector direction
        with pytest.raises(ValueError):
            q1.as_principal_rotation()
    else:
        pr1 = q1.as_principal_rotation()
        q2 = pr1.as_quaternion()
        pr2 = q2.as_principal_rotation()

        assert np.isclose(q1.vector, q2.vector).all()
        assert np.isclose(pr1.vector, pr2.vector).all()
        assert np.isclose(pr1.angle, pr2.angle)


@pytest.mark.parametrize('vector', non_singular_params)
def test_rotation_matrices(vector):
    """
    Test that $[B] . [B]^{-1} = [I_{3 \times 3}]$
    """
    q = Quaternion(*vector)
    # if the determinant is close to 1.0 the test passes
    assert np.isclose(q.inverse_rotation_matrix*2 @ q.rotation_matrix*2, np.eye(4)).all()


@pytest.mark.parametrize('vector1,vector2',
                         zip(non_singular_params[:int(len(non_singular_params)/2)],
                             non_singular_params[:int(len(non_singular_params)/2)-1:-1]))
def test_addition_subtraction(vector1, vector2):
    q1 = Quaternion(*vector1)
    q2 = Quaternion(*vector2)
    q3 = q1.add(q2)
    q4 = q3.subtract(q2)

    assert np.isclose(np.transpose(q1.addition_matrix) @ q1.vector,
                      np.array([1, 0, 0, 0])).all()
    assert np.isclose(np.transpose(q2.addition_matrix) @ q2.vector,
                      np.array([1, 0, 0, 0])).all()
    assert np.isclose(np.transpose(q3.addition_matrix) @ q3.vector,
                      np.array([1, 0, 0, 0])).all()
    assert np.isclose(np.transpose(q4.addition_matrix) @ q4.vector,
                      np.array([1, 0, 0, 0])).all()

    #Addition matrix must be orthogonal

    assert np.isclose(np.transpose(q1.addition_matrix) @ \
                      q1.addition_matrix, np.eye(4)).all()
    assert np.isclose(np.transpose(q2.addition_matrix) @ \
                      q2.addition_matrix, np.eye(4)).all()
    assert np.isclose(np.transpose(q3.addition_matrix) @ \
                      q3.addition_matrix, np.eye(4)).all()
    assert np.isclose(np.transpose(q4.addition_matrix) @ \
                      q4.addition_matrix, np.eye(4)).all()

    # Principal rotation and quaternion DCMs must be the same
    if np.isclose(np.transpose(q1.dcm) @ q4.dcm, np.eye(3)).all():
        pr1 = q1.as_principal_rotation()
        pr2 = q4.as_principal_rotation()
        assert pr1 == pr2
    else:
        assert np.isclose(q1.dcm @ q4.dcm, np.eye(3)).all()


