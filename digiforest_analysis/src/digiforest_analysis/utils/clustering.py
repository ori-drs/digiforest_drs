from functools import partial
from multiprocessing import Pool
import multiprocessing
from digiforest_analysis.utils.timing import Timer
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from sklearn.decomposition import PCA
import open3d as o3d
from scipy.spatial import cKDTree

from digiforest_analysis.tasks.tree_reconstruction import Circle

timer = Timer()


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
    points: np.ndarray,
    axes: np.ndarray,
    apply_sqrt: bool = False,
    debug_level: int = 0,
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
        point_fraction (float, optional): fraction of points to use for
            calculating the distance. The rest is determined by point to point distance
            calculation. Defaults to 0.1.
        debug_level (int, optional): verbosity level. Defaults to 0.

    Returns:
        np.ndarray[NxM]: (squared) distance of all points to all axes
    """
    if debug_level > 0:
        print("Start calculating distances on ", multiprocessing.current_process().name)
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
    if debug_level > 0:
        print("axis_pnts_to_origin_a on thread", multiprocessing.current_process().name)
    axis_pnts_to_origin_a = np.einsum("ij,ij->i", axis_pnts, normals_a)
    if debug_level > 0:
        print("axis_pnts_to_origin_b on thread", multiprocessing.current_process().name)
    axis_pnts_to_origin_b = np.einsum("ij,ij->i", axis_pnts, normals_b)
    if debug_level > 0:
        print("signed_dist_a on thread", multiprocessing.current_process().name)
    signed_dist_a = np.einsum("ij,kj->ik", points, normals_a) - axis_pnts_to_origin_a
    if debug_level > 0:
        print("signed_dist_b on thread", multiprocessing.current_process().name)
    signed_dist_b = np.einsum("ij,kj->ik", points, normals_b) - axis_pnts_to_origin_b
    # this is much faster than np.power and np.sum ?! ^^
    if debug_level > 0:
        print("squaring on thread", multiprocessing.current_process().name)
    sq_dists = signed_dist_a * signed_dist_a + signed_dist_b * signed_dist_b

    if debug_level > 0:
        print("Done calculating distances on", multiprocessing.current_process().name)
    return np.sqrt(sq_dists) if apply_sqrt else sq_dists


def voronoi(  # noqa: C901
    cloud,
    cloth: np.ndarray = None,
    hough_filter_radius: float = 0.1,
    crop_lower_bound: float = 5.0,
    crop_upper_bound: float = 8.0,
    max_cluster_radius: float = np.inf,
    n_threads: int = 1,
    point_fraction: float = 0.1,
    debug_level: int = 0,
    cluster_2d: bool = False,
):
    if cluster_2d:
        # TODO: implement 2D clustering
        raise NotImplementedError("2D clustering not implemented for voronoi")

    labels = -np.ones(cloud.point.positions.shape[0], dtype=np.int32)

    # 1. Normalize heights
    with timer("normalizing heights"):
        if cloth is not None:
            height_interpolator = RegularGridInterpolator(
                points=(cloth[:, 0, 1], cloth[0, :, 0]),
                values=cloth[:, :, 2],
                method="linear",
                bounds_error=False,
                fill_value=0.0,
            )
            heights = height_interpolator(cloud.point.positions.numpy()[:, :2])
            cloud.point.positions[:, 2] -= heights.astype(np.float32)

    # 2. Crop point cloud between cluster_strip_min and cluster_strip_max
    with timer("cropping"):
        points_numpy = cloud.point.positions.numpy()
        cluster_strip_mask = np.logical_and(
            points_numpy[:, 2] > crop_lower_bound, points_numpy[:, 2] < crop_upper_bound
        )
        cluster_strip = cloud.select_by_mask(cluster_strip_mask.astype(bool))
        if cloth is not None:
            cloud.point.positions[:, 2] += heights.astype(np.float32)

    if debug_level > 1:
        o3d.visualization.draw_geometries([cluster_strip.to_legacy()])

    # 3. Perform db scan clustering after removing outliers
    with timer("dbscan clustering of crops"):
        _, ind = cluster_strip.to_legacy().remove_statistical_outlier(
            nb_neighbors=20, std_ratio=2.0
        )

        cluster_strip = cluster_strip.select_by_index(ind)
        labels_pre = cluster_strip.cluster_dbscan(
            eps=0.7, min_points=20, print_progress=False
        ).numpy()

    if debug_level > 1:
        max_label = np.max(labels_pre)
        dbscan_clusters = []
        for i in range(max_label):
            cluster_points = cluster_strip.select_by_mask(labels_pre == i)
            cluster_points.paint_uniform_color(np.random.rand(3))
            dbscan_clusters.append(cluster_points.to_legacy())
        o3d.visualization.draw_geometries(dbscan_clusters)

    # 4. Clean up non-stem points using hough transform
    with timer("hough"):
        max_label = np.max(labels_pre)
        axes = []
        for i in range(max_label):
            cluster_points = cluster_strip.select_by_mask(labels_pre == i)
            cluster_points = cluster_points.point.positions.numpy()
            # 4.1. Remove clusters with fewer than 100 points
            with timer("hough->cluster size check"):
                if cluster_points.shape[0] < 100:
                    if debug_level > 0:
                        print("Too few points")
                    continue

            # 4.2. remove clsters not extending between cluster_strip_min and cluster_strip_max
            with timer("hough->extension check"):
                if (
                    np.min(cluster_points[:, 2]) > 1.05 * crop_lower_bound
                    or np.max(cluster_points[:, 2]) < 0.95 * crop_upper_bound
                ):
                    if debug_level > 0:
                        print(
                            "Cluster not extending between cluster_strip_min and cluster_strip_max"
                        )
                    continue
            # 4.2. Find circles in projection of cloud onto x-y plane
            with timer("hough->hough circle"):
                # slice crop at middle
                middle = (crop_lower_bound + crop_upper_bound) / 2
                slice_height = 0.2  # m
                slice = cluster_points[
                    np.logical_and(
                        cluster_points[:, 2] > middle - slice_height / 2,
                        cluster_points[:, 2] < middle + slice_height / 2,
                    )
                ]
                circ, _, votes, _ = Circle.from_cloud_hough(
                    points=slice,
                    grid_res=0.02,
                    min_radius=0.05,
                    max_radius=0.5,
                    return_pixels_and_votes=True,
                )
                if votes is None:
                    if debug_level > 0:
                        print("No votes")
                    continue

                if votes.max() < 0.1:
                    if debug_level > 0:
                        print("max_vote < 0.1")
                    continue

                if debug_level > 0:
                    print(
                        f"Center coordinates of hough circle: ({circ.x}, {circ.y}), Radius: {circ.radius}, Maximum vote: {votes.max()}"
                    )

            # 4.3. Remove points that are not close to the circle
            with timer("hough->filter"):
                dist = circ.get_distance(cluster_points)
                cluster_points = cluster_points[dist < hough_filter_radius]
                if cluster_points.shape[0] < 10:
                    if debug_level > 0:
                        print("Too few points after filtering")
                    continue

            # 5. Fit tree axes to clusters using PCA
            with timer("hough->PCA fitting"):
                pca = PCA(n_components=3)
                pca.fit(cluster_points)
                rot_mat = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]) @ pca.components_
                if rot_mat[2, 2] < 0:
                    rot_mat[:, 1:] *= -1  # make sure the z component is positive
                rot_mat = rot_mat.T

            # convert cluster_points into open3d point cloud and give a random color
            with timer("hough->points to open3d"):
                cluster_points = o3d.geometry.PointCloud(
                    o3d.utility.Vector3dVector(cluster_points)
                )
                cluster_points.paint_uniform_color(np.random.rand(3))
                T = np.eye(4)
                T[:3, :3] = rot_mat
                T[:3, 3] = np.array([circ.x, circ.y, crop_lower_bound])
                axis_dict = {
                    "transform": T,
                    "radius": circ.radius,
                }
                if debug_level > 1:
                    axis_dict["cloud"] = o3d.t.geometry.PointCloud.from_legacy(
                        cluster_points
                    )
                axes.append(axis_dict)

    if debug_level > 1:
        cylinders = []
        for axis in axes:
            cylinder_height = crop_upper_bound - crop_lower_bound
            cylinder = o3d.geometry.TriangleMesh.create_cylinder(
                radius=axis["radius"], height=cylinder_height
            )
            # shift up or down depending on third component of pca. Thus the
            # cylinder allways covers the pc
            cylinder.vertices = o3d.utility.Vector3dVector(
                np.array(cylinder.vertices) + np.array([0, 0, cylinder_height / 2])
            )
            cylinder.paint_uniform_color([0.8, 0.8, 1])
            cylinder.vertices = o3d.utility.Vector3dVector(
                (np.array(cylinder.vertices) @ axis["transform"][:3, :3].T)
                + axis["transform"][:3, 3]
            )
            cylinders.append(cylinder)
        o3d.visualization.draw_geometries(
            [c["cloud"].to_legacy() for c in axes] + cylinders
        )

    # 6. Perform voronoi tesselation of point cloud without floor
    # calculate distance to each axis
    with timer("voronoi"):
        axes_np = np.array(
            [np.hstack((a["transform"][:3, 2], a["transform"][:3, 3])) for a in axes]
        )
        if point_fraction < 1.0:
            precise_mask = np.random.rand(points_numpy.shape[0]) < point_fraction
            precise_query_points = points_numpy[precise_mask]
        else:
            precise_query_points = points_numpy

        print(f"Clustering with {n_threads} threads")
        if n_threads == 1:
            precise_dists = pnts_to_axes_sq_dist(
                points=precise_query_points,
                axes=axes_np,
                debug_level=debug_level,
            )
        else:
            with Pool() as pool:
                points_grouped = np.array_split(precise_query_points, n_threads, axis=0)
                dists = pool.map(
                    partial(
                        pnts_to_axes_sq_dist, axes=axes_np, debug_level=debug_level
                    ),
                    points_grouped,
                )
                precise_dists = np.vstack(dists)

        precise_labels = np.argmin(precise_dists, axis=1)
        precise_min_dists = precise_dists[
            np.arange(precise_dists.shape[0]), precise_labels
        ]
        if point_fraction < 1.0:
            labels = np.empty((points_numpy.shape[0]), dtype=np.int32)
            min_dists = np.empty((points_numpy.shape[0]))
            # fill precise values
            labels[precise_mask] = precise_labels
            min_dists[precise_mask] = precise_min_dists
            # find other values using cKD tree
            kd_tree = cKDTree(points_numpy[precise_mask])
            _, idcs = kd_tree.query(points_numpy[~precise_mask], k=1, workers=-1)
            labels[~precise_mask] = precise_labels[idcs]
            min_dists[~precise_mask] = precise_min_dists[idcs]
        else:
            dists = precise_dists
            labels = precise_labels
            min_dists = precise_min_dists

        if debug_level > 0:
            print("Clustering done")
        # dists = pnts_to_axes_sq_dist(points_numpy, axes_np)

        if max_cluster_radius != np.inf:
            labels[min_dists > max_cluster_radius**2] = -1

    with timer("data grooming"):
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
        # denormalize heights in clusters
        if cloth is not None:
            tree_centers = np.array([a["transform"][:2, 3] for a in filtered_axes])
            terrain_heights = height_interpolator(tree_centers)
            for i, axis in enumerate(filtered_axes):
                axis["transform"][2, 3] += terrain_heights[i]
    if debug_level > 0:
        print(timer)

    return labels, filtered_axes
