import numpy as np
from scipy.interpolate import RegularGridInterpolator
from digiforest_analysis.utils.timing import Timer
from skimage.transform import hough_circle
from sklearn.decomposition import PCA
import open3d as o3d

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

    num_clusters = 350
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


def voronoi(cloud, **kwargs):
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
        x_c, y_c, r, max_vote = fit_circle_hough(
            points=cluster_points, grid_res=0.02, min_radius=0.05, max_radius=0.5
        )
        if max_vote < 0.1:
            if kwargs.get("debug_level", 0) > 0:
                print("max_vote < 0.1")
            continue

        if kwargs.get("debug_level", 0) > 0:
            print(
                f"Center coordinates of hough circle: ({x_c}, {y_c}), Radius: {r}, Maximum vote: {max_vote}"
            )

        # 4.3. Remove points that are not close to the circle
        filter_radius = 0.1
        dist = np.linalg.norm(cluster_points[:, :2] - np.array([x_c, y_c]), axis=1)
        dist -= r
        cluster_points = cluster_points[dist < filter_radius]

        # 5. Fit tree axes to clusters using PCA
        pca = PCA(n_components=3)
        pca.fit(cluster_points)
        tree_axis = pca.components_[0]

        # convert cluster_points into open3d point cloud and give a random color
        cluster_points = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(cluster_points)
        )
        cluster_points.paint_uniform_color(np.random.rand(3))
        axes.append(
            {
                "axis": tree_axis,
                "center": np.array([x_c, y_c, cluster_strip_min]),
                "radius": r,
                "rot_mat": np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
                @ pca.components_,
                "cloud": cluster_points,
            }
        )

        # # visualization
        # viz_pointcloud = o3d.geometry.PointCloud()
        # viz_pointcloud.points = o3d.utility.Vector3dVector(cluster_points)
        # cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=r, height=3.5)
        # cylinder.vertices = o3d.utility.Vector3dVector(np.array(cylinder.vertices) + np.array([x_c, y_c, 3.5]))

    if kwargs.get("debug_level", 0) > 1:
        cylinders = []
        for axis in axes:
            cylinder_height = cluster_strip_max - cluster_strip_min
            cylinder = o3d.geometry.TriangleMesh.create_cylinder(
                radius=axis["radius"], height=cylinder_height * 2
            )
            cylinder.paint_uniform_color([0, 0, 1])
            cylinder.vertices = o3d.utility.Vector3dVector(
                (axis["rot_mat"].T @ np.array(cylinder.vertices).T).T + axis["center"]
            )
            cylinders.append(cylinder)
        o3d.visualization.draw_geometries([c["cloud"] for c in axes] + cylinders)

    # 6. Perform voronoi tesselation of point cloud without floor
    # calculate distance to each axis
    # calculate distances as euclidean distance to two perpendicular planes
    # meeting at the axis. This is 5 times faster than using the cross product.
    axis_dirs = np.array([axis["axis"] for axis in axes])
    axis_pnts = np.array([axis["center"] for axis in axes])
    normals_a = np.vstack(
        [np.zeros_like(axis_dirs[:, 0]), axis_dirs[:, 2], -axis_dirs[:, 1]]
    ).T
    normals_a /= np.linalg.norm(normals_a, axis=1)[:, None]
    normals_b = np.cross(axis_dirs, normals_a)
    axis_pnts_to_pc = points_numpy[:, None] - axis_pnts
    signed_dist_a = np.einsum("ijk,jk->ij", axis_pnts_to_pc, normals_a)
    signed_dist_b = np.einsum("ijk,jk->ij", axis_pnts_to_pc, normals_b)
    dists = np.sqrt(np.power(signed_dist_a, 2) + np.power(signed_dist_b, 2))

    labels = np.argmin(dists, axis=1)
    dist_max = kwargs.get("cluster_dist", np.inf)  # m
    if dist_max != np.inf:
        labels[dists[np.arange(dists.shape[0]), labels] > dist_max] = -1

    # remove clusters with fewer than 50 points
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        if count < 50:
            labels[labels == label] = -1
    # make sure the label index is continuous
    unique_labels = np.sort(np.unique(labels))
    for i, label in enumerate(unique_labels[1:]):
        labels[labels == label] = i

    return labels


def fit_circle_hough(
    points: np.ndarray,
    grid_res: float,
    min_radius: float,
    max_radius: float,
    point_ratio: float = 0.2,
    entropy_weighting: float = 10.0,
    **kwargs,
) -> tuple:
    """This function fits circles to the points in a slice using the hough
    transform. The z axis is ignored If both previous_center and search_radius
    are given, the search for the circle center is constrained to a circle
    around the previous center. If previous_radius is given, the search for the
    circle radius is constrained to be smaller than this radius.

    Adapted from: https://www.sciencedirect.com/science/article/pii/S0168169917301114?ref=pdf_download&fr=RR-2&rr=81fe181ccd14779b

    Args:
        points (np.ndarray): N x 2 array of points in the slice
        grid_res (float): length of pixel for hough map in m
        min_radius (float): minimum radius of circle in m
        max_radius (float): maximum radius of circle in m
        point_ratio (float, optional): ratio of points in a pixel wrt. number in
            most populated pixel to be counted valid. Defaults to 1.0.
        entropy_weighing (float, optional): weight to weigh the hough votes by
            the entropy of the top 10 votes for that radius. This helps to cope
            with noise at small radii. Defaults to 10.0.

    Returns:
        tuple: center x-y-coordinates, radius, and max vote indicating fitting
            quality
    """
    # construct 2D grid with pixel length of grid_res
    # bounding the point cloud of the current slice
    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
    min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
    # make sure pixels are square
    grid_center = np.array([(max_x + min_x) / 2, (max_y + min_y) / 2])
    grid_width = 1.5 * max(max_x - min_x, max_y - min_y)
    min_x, max_x = grid_center[0] - grid_width / 2, grid_center[0] + grid_width / 2
    min_y, max_y = grid_center[1] - grid_width / 2, grid_center[1] + grid_width / 2
    n_cells = int(grid_width / grid_res)
    # construct the grid
    grid_x, grid_y = np.meshgrid(
        np.linspace(min_x, max_x, n_cells + 1), np.linspace(min_y, max_y, n_cells + 1)
    )
    # reshape and remove last row and column
    grid_x = grid_x[None, :-1, :-1]
    grid_y = grid_y[None, :-1, :-1]

    # count how many points are in every cell (numpy version consumes too much mem)
    pixels = np.zeros((n_cells, n_cells))
    for point in points:
        i = int((point[1] - min_y) / grid_res)
        j = int((point[0] - min_x) / grid_res)
        pixels[i, j] += 1
    pixels[pixels < point_ratio * np.max(pixels)] = 0

    # crop image to only include the pixels containing points with a 50% margin
    filled_pixels = np.argwhere(pixels)
    min_x, max_x = np.min(filled_pixels[:, 0]), np.max(filled_pixels[:, 0])
    min_y, max_y = np.min(filled_pixels[:, 1]), np.max(filled_pixels[:, 1])
    min_x = max(0, min_x - int(0.5 + 0.5 * (max_x - min_x)))
    max_x = min(n_cells, max_x + int(0.5 + 0.5 * (max_x - min_x)))
    min_y = max(0, min_y - int(0.5 + 0.5 * (max_y - min_y)))
    max_y = min(n_cells, max_y + int(0.5 + 0.5 * (max_y - min_y)))
    pixels = pixels[min_x:max_x, min_y:max_y]
    grid_x = grid_x[:, min_x:max_x, min_y:max_y]
    grid_y = grid_y[:, min_x:max_x, min_y:max_y]

    # fit circles to the points in every cell using the hough transform
    min_radius_px = int(0.5 + min_radius / grid_res)
    max_radius_px = int(0.5 + max_radius / grid_res)
    # assume that at least a quarter of the points are seen. Then the max radius
    # is the number of pixels in one direction
    max_radius_px = min(max_radius_px, n_cells)
    # TODO unequally space tryradii for efficiency
    try_radii = np.arange(min_radius_px, max_radius_px)
    if try_radii.shape[0] == 0:
        return 0, 0, 0, pixels
    hough_res = hough_circle(pixels, try_radii)

    # weigh each radius by the entropy of the top 10 hough votes for that radius
    # for small radii there are many circles in a small area leading to many
    # high values. Which might be higher than for higher radii, which we are
    # looking for more. To avoid this, we weigh each radius by the entropy of
    # the top 10 hough votes for that radius, to reward radii that have a clear
    # vote.
    penalty = np.ones_like(try_radii)
    if entropy_weighting != 0.0:
        # To avoid artefacts where large radii have cropped voting-rings that
        # lead to artificially high entropy, exclude radii where both the
        # following applies:
        #    1. the radius is bigger than 0.5 * nc_cells. Thus, the voting rings
        #       are cropped.
        #    2. compared to the radius with max number of voted pixels, this
        #       radius has less than 50% of the votes.

        vote_fraction = np.count_nonzero(hough_res, axis=(1, 2)) / n_cells**2
        radius_mask = try_radii > 0.5 * n_cells
        votes_mask = vote_fraction < 0.50 * np.max(vote_fraction)
        mask = ~np.logical_and(radius_mask, votes_mask)
        if not np.any(mask):
            # return if there's no radii left
            return 0, 0, 0, None
        hough_res = hough_res[mask]
        try_radii = try_radii[mask]

        hough_flattened = hough_res.reshape(hough_res.shape[0], -1)
        top_10 = np.partition(hough_flattened, -10, axis=1)[:, -10:]
        # discard radii where there's fewer than 10 candidates
        # i.e. where top 10 contains 0
        discard_mask = top_10.min(axis=1) < 1e-3
        top_10_normalized = top_10 / (np.sum(top_10, axis=1, keepdims=True) + 1e-12)
        top_10_entropy = -np.sum(
            top_10_normalized * np.log(top_10_normalized + 1e-12), axis=1
        )
        # replace NaNs with max value
        top_10_entropy[np.isnan(top_10_entropy)] = -1
        top_10_entropy[top_10_entropy < 0] = top_10_entropy.max()
        top_10_entropy[discard_mask] = top_10_entropy.max()
        # normalize entropy to be between 0.1 and 1 given max reward of 10x
        penalty = 1 / entropy_weighting + (1 - entropy_weighting) * (
            top_10_entropy - np.min(top_10_entropy)
        ) / (np.max(top_10_entropy) - np.min(top_10_entropy))
        hough_res /= penalty[:, None, None]

    # find the circle with the most votes
    i_rad, x_px, y_px = np.unravel_index(np.argmax(hough_res), hough_res.shape)
    max_vote = hough_res[i_rad, x_px, y_px] * penalty[i_rad]

    # transform pixel coordinates back to world coordinates
    x_c = grid_x[0, x_px, y_px] + grid_res / 2
    y_c = grid_y[0, x_px, y_px] + grid_res / 2
    r = try_radii[i_rad] * grid_res

    return x_c, y_c, r, max_vote
