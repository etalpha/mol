import numpy as np
from numpy import outer, cross, dot, arccos, sin, cos
from numpy.linalg import inv, norm, solve

def normalized(vector):
    return vector / norm(vector)


def angle(v1, v2):
    return arccos(dot(v1, v2) / (norm(v1) * norm(v2)))


def pseudo_inv(mat):
    assert mat.ndim == 2
    return solve(mat.T.dot(mat), mat.T)


def homogeneous(coordinates):
    n, d = coordinates.shape
    assert d == 3, f"coordinates.shape is {coordinate.shape}"
    return np.hstack([coordinates, np.ones(n).reshape(n,1)])


def heterogeneous(coordinates):
    assert coordinates.shape[1] == 4
    return coordinates[:, :3]


def rodrigues(v, homogeneous=False):
    theta = norm(v)
    vx, vy, vz = v
    R = np.array([[0.0, -vz,  vy],
                  [ vz, 0.0, -vx],
                  [-vy,  vx, 0.0]])
    rot = np.eye(3) + sin(theta) * R / theta + (1 - cos(theta)) * dot(R, R) / (theta * theta)
    if homogeneous:
        ret = np.eye(4, dtype=np.float64)
        ret[:3, :3] = rot
        return ret.T
    else:
        return rot.T


def translation(v):
    tran = np.eye(4, dtype=np.float64)
    tran[3, :3] = v
    return tran

