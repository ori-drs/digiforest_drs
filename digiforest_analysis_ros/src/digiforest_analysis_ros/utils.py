import numpy as np
from scipy.spatial.transform import Rotation


def pose2T(orientation, position):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(
        np.array(
            [
                orientation.x,
                orientation.y,
                orientation.z,
                orientation.w,
            ]
        )
    ).as_matrix()
    T[:3, 3] = np.array([position.x, position.y, position.z])
    return T


def transform_clusters(clusters, T_new2old, time_stamp=None):
    for i in range(len(clusters)):
        clusters[i]["cloud"].transform(efficient_inv(T_new2old))
        clusters[i]["info"]["axis"]["transform"] = (
            efficient_inv(T_new2old) @ clusters[i]["info"]["axis"]["transform"]
        )
        clusters[i]["info"]["sensor_transform"] = T_new2old
        if time_stamp:
            clusters[i]["info"]["time_stamp"] = time_stamp
    return clusters


def efficient_inv(T):
    assert T.shape == (4, 4), "T must be a 4x4 matrix"
    T_inv = np.eye(4)
    T_inv[:3, :3] = T[:3, :3].T
    T_inv[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return T_inv
