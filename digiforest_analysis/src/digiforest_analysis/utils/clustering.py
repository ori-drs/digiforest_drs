import numpy as np
from scipy.interpolate import RegularGridInterpolator
from digiforest_analysis.utils.timing import Timer

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
    forest_points = cloud.point.positions.numpy()

    # 1. Normalize heights
    cloth = kwargs.get("cloth", None)
    if cloth is not None:
        with timer("Normalize Heights"):  # TODO remove: 25 ms
            interpolator = RegularGridInterpolator(
                points=(cloth[:, 0, 1], cloth[0, :, 0]),
                values=cloth[:, :, 2],
                method="linear",
                bounds_error=False,
                fill_value=0.0,
            )
            heights = interpolator(forest_points[:, :2])
            forest_points[:, 2] -= heights
        print(timer)

    # 2. Crop point cloud between cluster_stri_min and cluster_stri_max

    # 3. Perform db scan clustering

    # 4. Clean up non-stem points

    # 5. Fit tree axes to clusters

    # 6. Perform voronoi tesselation of point cloud without floor

    return labels
