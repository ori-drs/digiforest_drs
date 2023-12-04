from functools import partial
from multiprocessing import Pool
import multiprocessing
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from sklearn.decomposition import PCA
import open3d as o3d

from digiforest_analysis.tasks.tree_reconstruction import Circle


def cluster(cloud, method="dbscan_open3d", **kwargs):
    if method == "dbscan_open3d":
        return dbscan_open3d(cloud, **kwargs)

    elif method == "dbscan_sklearn":
        return dbscan_sklearn(cloud, **kwargs)

    elif method == "hdbscan_sklearn":
        return hdbscan_sklearn(cloud, **kwargs)

    elif method == "hdbscan":
        return hdbscan(cloud, **kwargs)

    elif method == "kmeans_sklearn":
        return kmeans_sklearn(cloud, **kwargs)

    elif method == "euclidean_pcl":
        return euclidean_pcl(cloud, **kwargs)

    elif method == "voronoi":
        return voronoi(cloud, **kwargs)

    else:
        raise ValueError(f"Clustering method [{method}] not supported")


def dbscan_open3d(cloud, **kwargs):
    eps = 0.8
    min_cluster_size = 20
    labels = cloud.cluster_dbscan(
        eps=eps, min_points=min_cluster_size, print_progress=False
    ).numpy()

    return labels


def dbscan_sklearn(cloud, **kwargs):
    cluster_2d = kwargs.get("cluster_2d", False)
    if cluster_2d:
        points = cloud.point.positions.numpy()[:, :2]
    else:
        points = cloud.point.positions.numpy()

    eps = 0.3
    min_cluster_size = 20
    from sklearn.cluster import DBSCAN

    db = DBSCAN(eps=eps, min_samples=min_cluster_size).fit(points)

    return db.labels_


def hdbscan_sklearn(cloud, **kwargs):
    cluster_2d = kwargs.get("cluster_2d", False)
    if cluster_2d:
        points = cloud.point.positions.numpy()[:, :2]
    else:
        points = cloud.point.positions.numpy()

    min_cluster_size = 20

    from sklearn.cluster import HDBSCAN

    db = HDBSCAN(min_samples=min_cluster_size).fit(points)

    return db.labels_


def hdbscan(cloud, **kwargs):
    cluster_2d = kwargs.get("cluster_2d", False)
    if cluster_2d:
        points = cloud.point.positions.numpy()[:, :2]
    else:
        points = cloud.point.positions.numpy()

    import hdbscan

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=20, algorithm="best", core_dist_n_jobs=1
    )
    labels = clusterer.fit_predict(points)

    return labels


def kmeans_sklearn(cloud, **kwargs):
    cluster_2d = kwargs.get("cluster_2d", False)
    if cluster_2d:
        points = cloud.point.positions.numpy()[:, :2]
    else:
        points = cloud.point.positions.numpy()

    num_clusters = 50
    from sklearn.cluster import KMeans

    labels = KMeans(n_clusters=num_clusters, n_init="auto").fit_predict(points)

    return labels


def euclidean_pcl(cloud, **kwargs):
    cluster_tolerance = kwargs.get("cluster_tolerance", 0.10)
    min_cluster_size = kwargs.get("min_cluster_size", 100)
    max_cluster_size = kwargs.get("max_cluster_size", 1000000)

    try:
        import pcl
    except Exception:
        raise ImportError(
            "PCL is not installed. Please install 'pip install python-pcl'"
        )

    points = cloud.point.positions.numpy()
    cloud = pcl.PointCloud()
    cloud.from_array(points.astype(np.float32))

    # creating the kdtree object for the searching
    tree = cloud.make_kdtree()

    # perform Euclidean Clustering
    ec = cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(cluster_tolerance)
    ec.set_MinClusterSize(min_cluster_size)
    ec.set_MaxClusterSize(max_cluster_size)
    ec.set_SearchMethod(tree)

    cluster_indices = ec.Extract()
    labels = -np.ones(cloud.size, dtype=np.int32)
    for i, ind in enumerate(cluster_indices):
        for j in ind:
            labels[j] = i

    return labels


def pnts_to_axes_sq_dist(
    pnts: np.ndarray, axes: np.ndarray, apply_sqrt: bool = False
) -> np.ndarray:
    """Calculate the distance of all point to all axes. For efficiency, two planes are
    constructed fore every axis, which is the intersection of them.
    The distance of a point to the axis is then the L2 norm of the individual distances
    to both planes. This is ~5 times faster than using the cross product.

    Args:
        pnt (np.ndarray[Nx3]): point in 3D space
        axis (np.ndarray[Mx6]): axis in 3D space (direction vector, point on axis)
        sqrt (bool, optional): whether to return the sqrt of the squared distance.
            Defaults to False.

    Returns:
        np.ndarray[NxM]: (squared) distance of all points to all axes
    """
    print("Calculating distances on thread", multiprocessing.current_process().name)
    axis_dirs = axes[:, :3]
    axis_dirs /= np.linalg.norm(axis_dirs, axis=1, keepdims=True)
    axis_pnts = axes[:, 3:]
    # TODO handle case where axis direction is in x-y-plane
    # (extremely unlikely for digiforest)
    normals_a = np.vstack(
        [np.zeros_like(axis_dirs[:, 0]), axis_dirs[:, 2], -axis_dirs[:, 1]]
    ).T
    normals_a /= np.linalg.norm(normals_a, axis=1)[:, None]
    normals_b = np.cross(axis_dirs, normals_a)

    # hesse normal form in einstein notation
    print("axis_pnts_to_origin_a on thread", multiprocessing.current_process().name)
    axis_pnts_to_origin_a = np.einsum("ij,ij->i", axis_pnts, normals_a)
    print("axis_pnts_to_origin_b on thread", multiprocessing.current_process().name)
    axis_pnts_to_origin_b = np.einsum("ij,ij->i", axis_pnts, normals_b)
    print("signed_dist_a on thread", multiprocessing.current_process().name)
    signed_dist_a = np.einsum("ij,kj->ik", pnts, normals_a) - axis_pnts_to_origin_a
    print("signed_dist_b on thread", multiprocessing.current_process().name)
    signed_dist_b = np.einsum("ij,kj->ik", pnts, normals_b) - axis_pnts_to_origin_b
    # this is much faster than np.power and np.sum ?! ^^
    print("squaring on thread", multiprocessing.current_process().name)
    sq_dists = signed_dist_a * signed_dist_a + signed_dist_b * signed_dist_b

    print("Done calculating distances on", multiprocessing.current_process().name)
    return np.sqrt(sq_dists) if apply_sqrt else sq_dists


def voronoi(cloud, filter_radius: float = 0.1, **kwargs):
    labels = -np.ones(cloud.point.positions.shape[0], dtype=np.int32)

    # 1. Normalize heights
    cloth = kwargs.get("cloth", None)
    if cloth is not None:
        interpolator = RegularGridInterpolator(
            points=(cloth[:, 0, 1], cloth[0, :, 0]),
            values=cloth[:, :, 2],
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        heights = interpolator(cloud.point.positions.numpy()[:, :2])
        cloud.point.positions[:, 2] -= heights.astype(np.float32)

    # 2. Crop point cloud between cluster_strip_min and cluster_strip_max
    cluster_strip_min = kwargs.get("cluster_strip_min", 5.0)
    cluster_strip_max = kwargs.get("cluster_strip_max", 8.0)
    points_numpy = cloud.point.positions.numpy()
    cluster_strip_mask = np.logical_and(
        points_numpy[:, 2] > cluster_strip_min, points_numpy[:, 2] < cluster_strip_max
    )
    cluster_strip = cloud.select_by_mask(cluster_strip_mask.astype(bool))

    # 3. Perform db scan clustering after removing outliers
    _, ind = cluster_strip.to_legacy().remove_statistical_outlier(
        nb_neighbors=20, std_ratio=2.0
    )

    cluster_strip = cluster_strip.select_by_index(ind)

    labels_pre = cluster_strip.cluster_dbscan(
        eps=0.7, min_points=20, print_progress=False
    ).numpy()

    # 4. Clean up non-stem points using hough transform
    max_label = np.max(labels_pre)
    axes = []
    for i in range(max_label):
        cluster_points = cluster_strip.select_by_mask(labels_pre == i)
        cluster_points = cluster_points.point.positions.numpy()
        # 4.1. Remove clusters with fewer than 100 points
        if cluster_points.shape[0] < 100:
            if kwargs.get("debug_level", 0) > 0:
                print("Too few points")
            continue

        # 4.2. remove clsters not extending between cluster_strip_min and cluster_strip_max
        if (
            np.min(cluster_points[:, 2]) > 1.05 * cluster_strip_min
            or np.max(cluster_points[:, 2]) < 0.95 * cluster_strip_max
        ):
            if kwargs.get("debug_level", 0) > 0:
                print(
                    "Cluster not extending between cluster_strip_min and cluster_strip_max"
                )
            continue
        # 4.2. Find circles in projection of cloud onto x-y plane
        circ, _, votes = Circle.from_cloud_hough(
            points=cluster_points,
            grid_res=0.02,
            min_radius=0.05,
            max_radius=0.5,
            return_pixels_and_votes=True,
        )
        if votes.max() < 0.1:
            if kwargs.get("debug_level", 0) > 0:
                print("max_vote < 0.1")
            continue

        if kwargs.get("debug_level", 0) > 0:
            print(
                f"Center coordinates of hough circle: ({circ.x}, {circ.y}), Radius: {circ.radius}, Maximum vote: {votes.max()}"
            )

        # 4.3. Remove points that are not close to the circle
        dist = circ.get_distance(cluster_points)
        cluster_points = cluster_points[dist < filter_radius]

        # 5. Fit tree axes to clusters using PCA
        pca = PCA(n_components=3)
        pca.fit(cluster_points)
        tree_direction = pca.components_[0]

        # convert cluster_points into open3d point cloud and give a random color
        cluster_points = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(cluster_points)
        )
        cluster_points.paint_uniform_color(np.random.rand(3))
        axis_dict = {
            "direction": tree_direction,
            "center": np.array([circ.x, circ.y, cluster_strip_min]),
            "radius": circ.radius,
            "rot_mat": np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]) @ pca.components_,
        }
        if kwargs.get("debug_level", 0) > 1:
            axis_dict["cloud"] = cluster_points
        axes.append(axis_dict)

    if kwargs.get("debug_level", 0) > 1:
        cylinders = []
        for axis in axes:
            cylinder_height = cluster_strip_max - cluster_strip_min
            cylinder = o3d.geometry.TriangleMesh.create_cylinder(
                radius=axis["radius"], height=cylinder_height
            )
            # shift up or down depending on third component of pca. Thus the
            # cylinder allways covers the pc
            cylinder.vertices = o3d.utility.Vector3dVector(
                np.array(cylinder.vertices)
                + np.array([0, 0, cylinder_height / 2]) * np.sign(axis["rot_mat"][2, 2])
            )
            cylinder.paint_uniform_color([0.8, 0.8, 1])
            cylinder.vertices = o3d.utility.Vector3dVector(
                (axis["rot_mat"].T @ np.array(cylinder.vertices).T).T + axis["center"]
            )
            cylinders.append(cylinder)
        o3d.visualization.draw_geometries([c["cloud"] for c in axes] + cylinders)

    # 6. Perform voronoi tesselation of point cloud without floor
    # calculate distance to each axis
    axes_np = np.array([np.hstack((a["axis"], a["center"])) for a in axes])
    n_threads = kwargs.get("n_threads_clustering", 8)
    print(f"Clustering with {n_threads} threads")
    if n_threads == 1:
        dists = pnts_to_axes_sq_dist(points_numpy, axes_np)
    else:
        with Pool() as pool:
            points_grouped = np.array_split(points_numpy, n_threads, axis=0)
            dists = pool.map(
                partial(pnts_to_axes_sq_dist, axes=axes_np), points_grouped
            )
            dists = np.vstack(dists)
    print("Clustering done")
    # dists = pnts_to_axes_sq_dist(points_numpy, axes_np)

    labels = np.argmin(dists, axis=1)
    dist_max = kwargs.get("cluster_dist", np.inf)  # m
    if dist_max != np.inf:
        labels[dists[np.arange(dists.shape[0]), labels] > dist_max**2] = -1

    # remove clusters with fewer than 50 points
    filtered_axes = []
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        if count < 50:
            labels[labels == label] = -1
    # make sure the label index is continuous
    unique_labels = np.sort(np.unique(labels))
    for i, label in enumerate(unique_labels[1:]):
        labels[labels == label] = i
        filtered_axes.append(axes[label])

    return labels, filtered_axes
