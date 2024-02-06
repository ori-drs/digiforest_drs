import numpy as np


def efficient_inv(T):
    assert T.shape == (4, 4), "T must be a 4x4 matrix"
    T_inv = np.eye(4)
    T_inv[:3, :3] = T[:3, :3].T
    T_inv[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return T_inv
