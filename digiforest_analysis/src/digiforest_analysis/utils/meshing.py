import numpy as np


def meshgrid_to_mesh(mgrid: np.ndarray):
    verts = mgrid.reshape(-1, 3)
    M, N, _ = mgrid.shape
    base_tri_1 = np.array([1, 0, N])
    base_tri_2 = np.array([1, N, N + 1])
    # accumulate column tiles
    tris = [(base_tri_1 + i, base_tri_2 + i) for i in range(N - 1)]
    tris = np.array(tris).reshape(-1, 3)
    # accumulate rows
    tris = np.vstack([tris + N * j for j in range(M - 1)])
    return verts, tris
