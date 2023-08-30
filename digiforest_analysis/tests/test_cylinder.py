from digiforest_analysis.utils import cylinder, plotting
from scipy.spatial.transform import Rotation as R

import open3d as o3d
import numpy as np
import random


def test_lsq_fit(noise=0.0, noise_perc=0.0):
    r = 0.32
    h = 2.0
    pos = np.array([0.25, -0.3, 0.1]).reshape(3, 1)
    eul = np.array([0.0, 10, 7])
    noise_perc = 0.2

    # Create cylinder
    cloud = cylinder.generate_test_cloud(
        radius=r,
        height=h,
        num_points=100,
        position=pos.reshape(3, 1),
        rotation=R.from_euler("zyx", eul, degrees=True).as_matrix(),
        noise_std=noise,
        noise_perc=noise_perc,
    )

    # Fit cylinder
    X = cloud.point.positions.numpy()
    N = cloud.point.normals.numpy()

    model = cylinder.fit(X, method="lsq", N=N)

    c = model["position"]
    w = model["axis"]
    r = model["radius"]

    # Colorize by residual
    res = cylinder.cylinder_equation(X, c, w, r)

    for i in range(X.shape[0]):
        res = np.clip(np.abs(cylinder.cylinder_equation(X[i][None], c, w, r)), 0, 1)[0]
        color = plotting.mpl_div_cmap(res)
        cloud.point.colors[i] = o3d.core.Tensor([color[0], color[1], color[2]])

    if model["success"]:
        print(f"Inliers: {model['inliers'].shape[0]} / {X.shape[0]}")
        print(f"height: {model['height']:.2f}")
        print(f"radius: {model['radius']:.2f}")

        # View cylinder
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame()
        cylinder_mesh = cylinder.to_mesh(model)
        o3d.visualization.draw_geometries(
            [origin, cloud.to_legacy(), cylinder_mesh],
            window_name=f"lsq, noise: {noise:.1f} ({100*noise_perc:.1f} %)",
            front=[0.72, 0.60, 0.33],
            lookat=[0.01, 0.17, 0.96],
            up=[-0.26, -0.19, 0.94],
            zoom=0.9,
        )
    # Assert
    assert model["success"]
    # assert model["inliers"].shape[0] == X.shape[0]
    # assert model["height"] == pytest.approx(h)
    # assert model["radius"] == pytest.approx(r)


if __name__ == "__main__":
    random.seed(42)
    for n in np.arange(0, 0.4, 0.1):
        for p in np.arange(0, 0.5, 0.1):
            print(f"lsq, noise: {n:.1f} ({100*p:.1f} %)")
            test_lsq_fit(noise=n, noise_perc=p)
            print()
