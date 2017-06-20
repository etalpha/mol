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

def rodrigues(v):
    theta = norm(v)
    vx, vy, vz = v
    R = np.array([[0.0, -vz,  vy],
                  [ vz, 0.0, -vx],
                  [-vy,  vx, 0.0]])
    return np.eye(3) + sin(theta) * R / theta + (1 - cos(theta)) * dot(R, R) / (theta * theta)
