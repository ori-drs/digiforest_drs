import random
import open3d as o3d
import numpy as np

from digiforest_analysis.utils import loss
from scipy.spatial.transform import Rotation as R
from scipy.optimize import Bounds, minimize


def cylinder_equation(X, c, w, r):
    X = X[:, :, None]
    c = c.reshape(3, 1)
    w = w.reshape(3, 1)
    return (
        (c - X).transpose([0, 2, 1]) @ (np.eye(3) - w @ w.T) @ (c - X) - r**2
    ).flatten()


def fit(X, method="lsq", **kwargs):
    # Get arguments
    min_inliers = kwargs.get("min_inliers", 10)

    # Fit
    if method == "lsq":
        inliers, params = fit_ls(X, **kwargs)

    elif method == "pcl_ransac":
        inliers, params = fit_pcl(X, **kwargs)

    else:
        raise ValueError(f"Method [{method}] not valid")

    # Get parameters
    cx, cy, cz, wx, wy, wz, r = params

    # Get position
    c = np.array([cx, cy, cz])

    # Get rotation
    w = np.array([wx, wy, wz])
    w = w / np.linalg.norm(w)
    ref_axis = np.array([0, 0, 1])
    angle = np.arccos(np.dot(ref_axis, w))
    axis_angle = np.cross(ref_axis, w)
    axis_angle /= np.linalg.norm(axis_angle)
    axis_angle *= angle
    rotation = R.from_rotvec(axis_angle).as_matrix()

    # Compute height
    height = 0.0
    if inliers.shape[0] > min_inliers:
        # Rotate points to canonical frame
        # import matplotlib.pyplot as plt
        # ax = plt.figure().add_subplot(projection='3d')
        # ax.scatter(X[:,0], X[:,1], X[:,2])
        X2 = (rotation.T @ (X[inliers] - c).T).T
        # ax.scatter(X2[:, 0], X2[:, 1], X2[:, 2])
        # plt.show()
        lowest_point = X2[:, 2].min()
        highest_point = X2[:, 2].max()
        height = highest_point - lowest_point
        # height_shift = np.array([0.0, 0.0, -lowest_point]).reshape(3, 1)
        # # Adjust center
        # c = c - rotation @ height_shift

    # Parameters
    return {
        "success": inliers.shape[0] > min_inliers,
        "position": c,
        "rotation": rotation,
        "axis": w,
        "radius": r,
        "inliers": inliers,
        "height": height,
    }


def fit_ls(X, **kwargs):
    """
    https://www.geometrictools.com/Documentation/LeastSquaresFitting.pdf
    """
    # Get arguments
    N = kwargs.get("N", None)
    cx = kwargs.get("cx", 0.0)
    cy = kwargs.get("cy", 0.0)
    cz = kwargs.get("cz", 0.0)
    wx = kwargs.get("wx", 0.0)
    wy = kwargs.get("wy", 0.0)
    wz = kwargs.get("wz", 1.0)
    r = kwargs.get("r", 10)

    weight_n = kwargs.get("weight_n", 0.0)  # Normals cost weight
    weight_g = kwargs.get("weight_g", 0.0)  # Gravit cost weight

    outlier_thr = kwargs.get("outlier_thr", 0.01)

    # Prepare input data
    X = X[:, :, None].copy()
    X_mean = X.mean(axis=0)
    X -= X_mean

    # Set initial parameters
    p = np.array([cx, cy, cz, wx, wy, wz, r])

    if N is not None:
        N = N[:, :, None].copy()

    def residual_cylinder(p):
        c = np.array([p[0], p[1], p[2]]).reshape(3, 1)
        w = np.array([p[3], p[4], p[5]]).reshape(3, 1)
        r = p[6]

        res = (
            (c - X).transpose([0, 2, 1]) @ (np.eye(3) - w @ w.T) @ (c - X) - r**2
        ).flatten()

        return res

    def residual_normals(p):
        w = np.array([p[3], p[4], p[5]]).reshape(3, 1)
        res = np.dot(N.reshape(-1, 1, 3), w).flatten()
        return res

    def residual_gravity(p):
        w = np.array([0.0, 0.0, 1.0]).reshape(3, 1)
        res = 1.0 - np.dot(N.reshape(-1, 1, 3), w).flatten()
        return res

    def residual(p):
        return np.vstack(
            (
                residual_cylinder(p),
                weight_n * residual_normals(p),
                weight_g * residual_gravity(p),
            )
        ).reshape(-1, 3)

    def chi2(p):
        return (residual(p) ** 2).sum(1)

    def cost(p):
        return (
            loss.smooth_l1(residual_cylinder(p))
            + weight_n * loss.l2(residual_normals(p))
            + weight_g * loss.l2(residual_gravity(p))
        ).sum()

    # Prepare bounds
    bounds = Bounds(
        lb=[-np.inf, -np.inf, -np.inf, -1, -1, -1, 0.1],
        ub=[np.inf, np.inf, np.inf, 1, 1, 1, np.inf],
        keep_feasible=False,
    )

    # Optimize
    # result = least_squares(residual_cylinder, p, loss="soft_l1", bounds=bounds)
    result = minimize(cost, p, bounds=bounds, options={"gtol": 1e-6})

    # Get inliers
    # inliers = np.where(np.abs(result["fun"]) < outlier_thr)[0]
    inliers = np.where(chi2(result["x"]) < outlier_thr)[0]

    # adjust params
    params = result["x"]
    params[0] += X_mean[0].item()
    params[1] += X_mean[1].item()
    params[2] += X_mean[2].item()

    # Output
    return inliers, params


def fit_pcl(X, **kwargs):
    try:
        import pcl
    except Exception:
        raise ImportError(
            "PCL is not installed. Please install 'pip install python-pcl'"
        )
    # Get arguments
    max_tree_diameter = kwargs.get("max_tree_diameter", 2.0)
    outlier_thr = kwargs.get("outlier_thr", 0.01)

    # Prepare data
    X = X.copy()
    cloud = pcl.PointCloud()
    cloud.from_array(X.astype(np.float32))

    # Estimate normals
    ne = cloud.make_NormalEstimation()
    tree = cloud.make_kdtree()
    ne.set_SearchMethod(tree)
    ne.set_KSearch(20)

    # Fit
    seg = cloud.make_segmenter_normals(ksearch=20)
    seg.set_optimize_coefficients(True)
    seg.set_model_type(pcl.SACMODEL_CYLINDER)
    seg.set_normal_distance_weight(1.0)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_max_iterations(1000)
    seg.set_distance_threshold(outlier_thr)
    seg.set_radius_limits(0, 0.5 * max_tree_diameter)
    try:
        [inliers, params] = seg.segment()
    except Exception:
        return 0, np.zeros(7)

    inliers = np.asarray(inliers)
    # Output
    return inliers, params


def to_mesh(model):
    assert "position" in model
    assert "rotation" in model
    assert "radius" in model

    mesh = o3d.geometry.TriangleMesh()
    mesh = mesh.create_cylinder(radius=model["radius"], height=model["height"])
    mesh.rotate(model["rotation"])
    mesh.translate(model["position"])

    return mesh


def generate_test_cloud(
    radius,
    height,
    num_points,
    position=np.array([0, 0, 0]),
    rotation=np.eye(3),
    noise_std=0.0,
    noise_perc=0.3,
):
    theta = np.linspace(0, 2 * np.pi, num_points)
    z = np.linspace(0, height, num_points)
    theta, z = np.meshgrid(theta, z)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    points = np.stack((x.flatten(), y.flatten(), z.flatten())).transpose()

    # Add noise
    n_points = points.shape[0]
    idxs = np.arange(n_points)
    np.random.shuffle(idxs)
    n_noisy = int(n_points * noise_perc)
    idx_noisy = idxs[0:n_noisy]
    points[idx_noisy, :] += np.random.normal(0, noise_std, (n_noisy, 3))

    # Transform
    points = rotation @ points.reshape(-1, 3, 1)
    points = points + position
    points = points.reshape(-1, 3)

    # Create cloud
    cloud = o3d.t.geometry.PointCloud(points)
    cloud.estimate_normals()
    cloud.paint_uniform_color([0.0, 0.0, 0.0])

    return cloud


if __name__ == "__main__":
    """Minimal example"""
    random.seed(42)

    # Create cylinder
    cloud = generate_test_cloud(
        radius=0.3,
        height=1.0,
        num_points=100,
        position=np.array([0, 1, 0]).reshape(3, 1),
        rotation=R.from_euler("zyx", [0, 0, 0], degrees=True).as_matrix(),
        noise_std=0.0,
        noise_perc=0.2,
    )

    origin = o3d.geometry.TriangleMesh.create_coordinate_frame()

    # Fit cylinder
    X = cloud.point.positions.numpy()
    N = cloud.point.normals.numpy()

    model = fit(X, method="lsq", N=N)
    if model["success"]:
        print(f"Inliers: {model['inliers'].shape[0]} / {X.shape[0]}")
        print(f"height: {model['height']}")
        print(f"radius: {model['radius']}")
        # View cylinder
        cylinder_mesh = to_mesh(model)
        o3d.visualization.draw_geometries(
            [origin, cloud.to_legacy(), cylinder_mesh], window_name="lsq"
        )

    model = fit(X, method="pcl_ransac")
    if model["success"]:
        print(f"Inliers: {model['inliers'].shape[0]} / {X.shape[0]}")
        print(f"height: {model['height']}")
        print(f"radius: {model['radius']}")
        # View cylinder
        cylinder_mesh = to_mesh(model)
        o3d.visualization.draw_geometries(
            [origin, cloud.to_legacy(), cylinder_mesh], window_name="pcl_ransac"
        )
