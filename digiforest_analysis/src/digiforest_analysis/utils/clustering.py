from colorsys import hls_to_rgb
from functools import partial
from multiprocessing import Pool
import multiprocessing
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import open3d as o3d
from scipy.spatial import cKDTree

from digiforest_analysis.tasks.tree_reconstruction import Circle
from digiforest_analysis.utils.timing import Timer
from digiforest_analysis.utils.meshing import meshgrid_to_mesh

timer = Timer()
current_point_cloud = 0
current_view = None


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
    crop_bounds: list = [[0.5, 1.5], [2, 3], [4, 5]],
    max_cluster_radius: float = np.inf,
    n_threads: int = 1,
    point_fraction: float = 0.1,
    debug_level: int = 0,
    cluster_2d: bool = False,
    **kwargs,
):
    if cluster_2d:
        # TODO: implement 2D clustering
        raise NotImplementedError("2D clustering not implemented for voronoi")

    final_labels = -np.ones(cloud.point.positions.shape[0], dtype=np.int32)

    # 1. Normalize heights
    with timer("normalizing heights"):
        if cloth is not None:
            height_interpolator = RegularGridInterpolator(
                points=(cloth[:, 0, 0], cloth[0, :, 1]),
                values=cloth[:, :, 2],
                method="linear",
                bounds_error=False,
                fill_value=0.0,
            )
            heights = height_interpolator(cloud.point.positions.numpy()[:, :2])
            index = (12, 12)
            p1 = np.array([cloth[index[0], 0, 0], cloth[0, index[1], 1]])
            v_true = cloth[index[0], index[1], 2]
            v_interp = height_interpolator(p1)
            print(v_true, v_interp)
            cloud_orig = cloud.clone()
            cloud.point.positions[:, 2] -= heights.astype(np.float32)

    if debug_level > 1:
        print("VIZ: Height-normalized Cloud")

        def toggle_point_cloud(vis):
            global current_point_cloud
            current_view = vis.get_view_control().convert_to_pinhole_camera_parameters()
            verts, tris = meshgrid_to_mesh(cloth)
            verts_vec = o3d.utility.Vector3dVector(verts)
            tris_vec = o3d.utility.Vector3iVector(
                np.concatenate((tris, np.flip(tris, axis=1)), axis=0)
            )
            terrain_mesh = o3d.geometry.TriangleMesh(verts_vec, tris_vec)
            if current_point_cloud == 1:
                vis.clear_geometries()
                vis.add_geometry(terrain_mesh)
                vis.add_geometry(cloud_orig.to_legacy())
                current_point_cloud = 2
                print("Showing Unnormalized cloud")
            else:
                vis.clear_geometries()
                vis.add_geometry(cloud.to_legacy())
                print("Showing Normalized cloud")
                current_point_cloud = 1
            vis.get_view_control().convert_from_pinhole_camera_parameters(
                current_view, True
            )

        global current_point_cloud
        current_point_cloud = 1
        visualizer = o3d.visualization.VisualizerWithKeyCallback()
        visualizer.create_window()
        visualizer.add_geometry(cloud.to_legacy())
        visualizer.register_key_callback(ord("T"), toggle_point_cloud)
        visualizer.run()
        visualizer.destroy_window()

    # 2. Crop point cloud between cluster_strip_min and cluster_strip_max
    with timer("cropping"):
        points_numpy = cloud.point.positions.numpy()
        crops = []
        for bounds in crop_bounds:
            mask = np.logical_and(
                points_numpy[:, 2] > bounds[0], points_numpy[:, 2] < bounds[1]
            )
            crop = cloud.select_by_mask(mask.astype(bool))
            crops.append(crop)

    if debug_level > 1:
        print("VIZ: Cropped cloud for clustering")
        o3d.visualization.draw_geometries([c.to_legacy() for c in crops])

    # 3. Perform db scan clustering after removing outliers
    with timer("dbscan clustering of crops"):
        crop_labels = []
        for i in range(len(crops)):
            crop = crops[i]
            _, ind = crop.to_legacy().remove_statistical_outlier(
                nb_neighbors=20, std_ratio=2.0
            )
            crops[i] = crop.select_by_index(ind)
            crop_labels.append(
                crops[i]
                .cluster_dbscan(eps=0.7, min_points=20, print_progress=False)
                .numpy()
            )

    # if debug_level > 1:
    #     print("VIZ: DBSCAN Clustering")
    #     clusters = []
    #     for c, l in zip(crops, crop_labels):
    #         max_label = np.max(l)
    #         for i in range(max_label):
    #             cluster_points = c.select_by_mask(l == i)
    #             cluster_points.paint_uniform_color(np.random.rand(3))
    #             clusters.append(cluster_points.to_legacy())
    #     o3d.visualization.draw_geometries(clusters)

    # 4. Clean up non-stem points using hough transform
    with timer("hough"):
        axes = []
        for c, label, bounds in zip(crops, crop_labels, crop_bounds):
            max_label = np.max(label)
            for i_label in range(max_label):
                cluster_points = c.select_by_mask(label == i_label)
                cluster_points = cluster_points.point.positions.numpy()
                if cluster_points.shape[0] < 50:
                    if debug_level > 0:
                        print(
                            f"Cluster {i_label} has only {cluster_points.shape[0]} points. Skipping"
                        )
                    continue

                # Remove clusters not extending between bounds
                if (
                    np.min(cluster_points[:, 2]) > 1.05 * bounds[0]
                    or np.max(cluster_points[:, 2]) < 0.95 * bounds[1]
                ):
                    if debug_level > 0:
                        print(
                            f"Cluster {i_label} does not extend between bounds. Skipping"
                        )
                    continue
                # Find circles in slice of crop at middle
                slice_height = 0.2  # m
                slice_lower = cluster_points[
                    np.logical_and(
                        cluster_points[:, 2] > bounds[0],
                        cluster_points[:, 2] < bounds[0] + slice_height,
                    )
                ]
                slice_upper = cluster_points[
                    np.logical_and(
                        cluster_points[:, 2] < bounds[1],
                        cluster_points[:, 2] > bounds[1] - slice_height,
                    )
                ]
                hough_kwargs = dict(
                    grid_res=0.02,
                    min_radius=0.05,
                    max_radius=0.5,
                    return_pixels_and_votes=True,
                )
                # circ_lower, _, votes_lower, _ = Circle.from_cloud_hough(
                #     points=slice_lower, **hough_kwargs, circle_height=bounds[0]
                # )
                # circ_upper, _, votes_upper, _ = Circle.from_cloud_hough(
                #     points=slice_upper, **hough_kwargs, circle_height=bounds[1]
                # )

                # if votes_lower is None or votes_upper is None:
                #     if debug_level > 0:
                #         print(f"No votes for cluster {i_label}. Skipping")
                #     continue

                # if votes_lower.max() < 0.1 or votes_upper.max() < 0.1:
                #     if debug_level > 0:
                #         print(f"Maximum vote for cluster {i_label} is too low. Skipping")
                #     continue

                circ_lower = Circle.from_cloud_ransahc(
                    points=slice_lower,
                    **hough_kwargs,
                    circle_height=bounds[0],
                    max_points=50,
                )
                circ_upper = Circle.from_cloud_ransahc(
                    points=slice_upper,
                    **hough_kwargs,
                    circle_height=bounds[1],
                    max_points=50,
                )
                if circ_lower is None or circ_upper is None:
                    if debug_level > 0:
                        print(f"No votes for cluster {i_label}. Skipping")
                    continue

                if debug_level > 0:
                    print("found two hough circles")

                cylinder_radius = (circ_lower.radius + circ_upper.radius) / 2
                T = np.eye(4)
                tree_axis = circ_upper.center - circ_lower.center
                tree_axis /= np.linalg.norm(tree_axis)

                # large trees are expected to be upright!
                if cylinder_radius > 0.8 * hough_kwargs["max_radius"]:
                    max_angle = 10
                else:
                    max_angle = 20
                if np.rad2deg(np.arccos(tree_axis[2])) > max_angle:
                    if debug_level > 0:
                        print(f"Cluster {i_label} is not vertical enough. Skipping")
                    continue

                # Compute Score of circle fit: points insde / points up to r away from circle
                dists = (
                    pnts_to_axes_sq_dist(
                        cluster_points,
                        np.concatenate((tree_axis, circ_lower.center))[None, :],
                        apply_sqrt=True,
                    )
                    - cylinder_radius
                )
                score = (np.abs(dists) < 0.1 * cylinder_radius).sum() / (
                    dists < 0.1 * cylinder_radius
                ).sum()

                if score < 0.25:
                    if debug_level > 0:
                        print(f"Cluster {i_label} has a bad score ({score}). Skipping")
                    continue

                axis_normal = np.array([tree_axis[1], -tree_axis[0], 0])
                axis_normal /= np.linalg.norm(axis_normal)
                T[:3, :3] = np.stack(
                    (axis_normal, np.cross(tree_axis, axis_normal), tree_axis), axis=1
                )
                T[:3, 3] = np.array([circ_lower.x, circ_lower.y, bounds[0]])
                axis_dict = {
                    "transform": T,
                    "radius": cylinder_radius,
                    "height": bounds[1] - bounds[0],
                    "score": score,
                }
                axes.append(axis_dict)

    # Do Non-maximum suppression
    with timer("non-maximum suppression"):
        nms_radius = 1.0
        kdtree = cKDTree(np.array([a["transform"][:2, 3] for a in axes]))
        # find all clusters of radius nms_radius
        indices = kdtree.query_ball_tree(kdtree, nms_radius)
        if max(len(i) for i in indices) > len(crops):
            print("WARNING: NMS radius too large. Some clusters are over-suppressed")

        unique_sets = set([tuple(sorted(i)) for i in indices])
        nms_result = [max(us, key=lambda i: axes[i]["score"]) for us in unique_sets]
        axes_nms = [axes[i] for i in nms_result]

    if debug_level > 1:
        print("VIZ: DBSCAN Clustering and Fitted Axes")
        dbscan_clusters = []
        for c, label in zip(crops, crop_labels):
            max_label = np.max(label)
            for i in range(max_label):
                cluster_points = c.select_by_mask(label == i)
                cluster_points.paint_uniform_color(np.random.rand(3))
                dbscan_clusters.append(cluster_points.to_legacy())

        cylinders = []
        cylinders_nms = []
        for axis in axes:
            cylinder = o3d.geometry.TriangleMesh.create_cylinder(
                radius=axis["radius"], height=axis["height"]
            )
            # shift up or down depending on third component of pca. Thus the
            # cylinder allways covers the pc
            cylinder.vertices = o3d.utility.Vector3dVector(
                np.array(cylinder.vertices) + np.array([0, 0, axis["height"] / 2])
            )
            if axis["score"] > 0.25:
                cylinder.paint_uniform_color([axis["score"], axis["score"] / 2, 0])
            else:
                cylinder.paint_uniform_color([1, 0, 0])
            cylinder.vertices = o3d.utility.Vector3dVector(
                (np.array(cylinder.vertices) @ axis["transform"][:3, :3].T)
                + axis["transform"][:3, 3]
            )
            cylinders.append(cylinder)
        for axis in axes_nms:
            cylinder = o3d.geometry.TriangleMesh.create_cylinder(
                radius=axis["radius"], height=axis["height"] * 1.2
            )
            # shift up or down depending on third component of pca. Thus the
            # cylinder allways covers the pc
            cylinder.vertices = o3d.utility.Vector3dVector(
                np.array(cylinder.vertices) + np.array([0, 0, axis["height"] / 2])
            )
            cylinder.paint_uniform_color([0, 1, 0])
            cylinder.vertices = o3d.utility.Vector3dVector(
                (np.array(cylinder.vertices) @ axis["transform"][:3, :3].T)
                + axis["transform"][:3, 3]
            )
            cylinders_nms.append(cylinder)

        o3d.visualization.draw_geometries(
            [cloud.to_legacy()] + cylinders + cylinders_nms
        )
    axes = axes_nms

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
            final_labels = np.empty((points_numpy.shape[0]), dtype=np.int32)
            min_dists = np.empty((points_numpy.shape[0]))
            # fill precise values
            final_labels[precise_mask] = precise_labels
            min_dists[precise_mask] = precise_min_dists
            # find other values using cKD tree
            kd_tree = cKDTree(points_numpy[precise_mask])
            _, idcs = kd_tree.query(points_numpy[~precise_mask], k=1, workers=-1)
            final_labels[~precise_mask] = precise_labels[idcs]
            min_dists[~precise_mask] = precise_min_dists[idcs]
        else:
            dists = precise_dists
            final_labels = precise_labels
            min_dists = precise_min_dists

        if debug_level > 0:
            print("Clustering done")
        # dists = pnts_to_axes_sq_dist(points_numpy, axes_np)

        if max_cluster_radius != np.inf:
            final_labels[min_dists > max_cluster_radius**2] = -1

    with timer("data grooming"):
        # remove clusters with fewer than 50 points
        filtered_axes = []
        unique_labels, counts = np.unique(final_labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            if count < 50:
                final_labels[final_labels == label] = -1
        # make sure the label index is continuous. This works because new labels are
        # sure to be smaller than the labels with gaps in them
        unique_labels = np.sort(np.unique(final_labels))
        for i, label in enumerate(unique_labels[1:]):
            final_labels[final_labels == label] = i
            filtered_axes.append(axes[label])
        # denormalize heights in clusters
        if cloth is not None:
            tree_centers = np.array([a["transform"][:2, 3] for a in filtered_axes])
            terrain_heights = height_interpolator(tree_centers)
            for i, axis in enumerate(filtered_axes):
                axis["transform"][2, 3] += terrain_heights[i]

    if debug_level > 1:
        print("VIZ: Voronoi Clustering")
        clusters = []
        for label in np.unique(final_labels):
            if label == -1:
                continue
            cluster_points = cloud.select_by_mask(final_labels == label)
            cluster_points.paint_uniform_color(hls_to_rgb(np.random.rand(), 0.6, 1.0))
            clusters.append(cluster_points.to_legacy())
        o3d.visualization.draw_geometries(clusters + cylinders_nms + cylinders)
    if debug_level > 0:
        print(timer)

    # renormalize heights
    if cloth is not None:
        cloud.point.positions[:, 2] += heights.astype(np.float32)

    return final_labels, filtered_axes
