from digiforest_analysis.tasks import BaseTask

import open3d as o3d
import numpy as np


class TreeSegmentation(BaseTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._normal_thr = kwargs.get("normal_thr", 0.5)
        self._voxel_size = kwargs.get("voxel_size", 0.05)
        self._clustering_method = kwargs.get("clustering_method", "dbscan_sk")

        self._cluster_tolerance = kwargs.get("cluster_tolerance", 0.10)
        self._min_cluster_size = kwargs.get("min_cluster_size", 100)
        self._max_cluster_size = kwargs.get("max_cluster_size", 10000)
        self._min_tree_height = kwargs.get("min_tree_height", 2.0)
        self._max_tree_diameter = kwargs.get("max_tree_diameter", 2.0)
        self._min_gravity_alignment_score = kwargs.get(
            "min_gravity_alignment_score", 0.7
        )
        self._max_trunk_height = kwargs.get("max_trunk_height", 5.0)
        self._debug = kwargs.get("debug", False)

    def _process(self, **kwargs):
        """ "
        Processes the forest cloud with ground removed and outputs a list of tree clouds

        Returns:
            _type_: _description_
        """
        cloud = kwargs.get("cloud")
        assert len(cloud.point.normals) > 0

        # Prefiltering
        cloud = self.prefiltering(cloud)

        # Extract clusters
        clusters = self.coarse_clustering(cloud, self._clustering_method)
        if self._debug:
            print("Extracted " + str(len(clusters)) + " initial clusters.")

        # Filter out implausible clusters
        tree_clouds = self.filter_tree_clusters(clusters)
        trees_info = []
        for tree in tree_clouds:
            cluster_info = self.compute_cluster_info(tree)
            trees_info.append(cluster_info)

        if self._debug:
            num_filtered_clusters = len(clusters) - len(tree_clouds)
            print("Filtered out " + str(num_filtered_clusters) + " clusters.")
        return tree_clouds, trees_info

    def prefiltering(self, cloud, **kwargs):
        # Filter by Z-normals
        mask = (cloud.point.normals[:, 2] >= -self._normal_thr) & (
            cloud.point.normals[:, 2] <= self._normal_thr
        )
        new_cloud = o3d.t.geometry.PointCloud(cloud.select_by_mask(mask))

        # Downsample
        new_cloud = new_cloud.voxel_down_sample(voxel_size=self._voxel_size)

        return new_cloud

    def coarse_clustering(self, cloud, method="dbscan_o3d"):
        if method == "dbscan_o3d":
            eps = 0.8
            min_cluster_size = 20
            labels = np.array(
                cloud.cluster_dbscan(
                    eps=eps, min_points=min_cluster_size, print_progress=False
                )
            )
        elif method == "dbscan_sk":
            from sklearn.cluster import DBSCAN

            eps = 0.3
            min_cluster_size = 20
            X = cloud.point.positions.numpy()[:, :2]
            db = DBSCAN(eps=eps, min_samples=min_cluster_size).fit(X)
            labels = db.labels_
        elif method == "kmeans":
            from sklearn.cluster import KMeans

            num_clusters = 350
            X = cloud.point.positions.numpy()[:, :2]
            labels = KMeans(n_clusters=num_clusters, n_init="auto").fit_predict(X)
        else:
            raise NotImplementedError(f"Method [{method}] not available")

        # Get max number of labels
        num_labels = labels.max() + 1

        # Prepare output clouds
        clusters = []
        for i in range(num_labels):
            mask = labels == i
            seg_cloud = o3d.t.geometry.PointCloud(cloud.select_by_mask(mask))
            clusters.append(seg_cloud)

        return clusters

    def filter_tree_clusters(self, clusters):
        tree_clouds = []
        for cluster in clusters:
            is_invalid_cluster = (
                (not self.is_cluster_alignment_straight_enough(cluster))
                or self.is_cluster_too_low(cluster)
                or self.is_cluster_radius_too_big(cluster)
            )

            if is_invalid_cluster:
                continue
            else:
                tree_clouds.append(cluster)
        return tree_clouds

    def is_cluster_too_low(self, cluster):
        cluster_dim = self.get_cluster_dimensions(cluster)
        if cluster_dim["dim_z"] < self._min_tree_height:
            return True
        return False

    def is_cluster_radius_too_big(self, cluster):
        cluster_dim = self.get_cluster_dimensions(cluster)
        if (
            cluster_dim["dim_x"] > self._max_tree_diameter
            or cluster_dim["dim_y"] > self._max_tree_diameter
        ):
            return True
        return False

    def is_cluster_alignment_straight_enough(self, cluster):
        cluster_pca = self.compute_pca(cluster)
        cluster_principal_axis = cluster_pca[:, 0]
        gravity_axis = np.array([0, 0, 1])
        alignment_score = np.abs(np.dot(cluster_principal_axis, gravity_axis))
        if alignment_score < self._min_gravity_alignment_score:
            return False
        return True

    def get_cluster_dimensions(self, cluster):
        cluster_np = cluster.point.positions.numpy()
        cluster_np_dim = np.max(cluster_np, axis=0) - np.min(cluster_np, axis=0)
        cluster_min_z = float(np.min(cluster_np, axis=0)[2])
        cluster_np_dim = cluster_np_dim.tolist()
        cluster_dim = {
            "dim_x": cluster_np_dim[0],
            "dim_y": cluster_np_dim[1],
            "dim_z": cluster_np_dim[2],
            "min_z": cluster_min_z,
        }
        return cluster_dim

    def compute_pca(self, cluster):
        cluster_np = cluster.point.positions.numpy()
        # Center the data by subtracting the mean
        mean = np.mean(cluster_np, axis=0)
        centered_data = cluster_np - mean

        # Calculate the covariance matrix
        covariance_matrix = np.cov(centered_data, rowvar=False)

        # Calculate eigenvectors and eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        # Sort eigenvectors by eigenvalues
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        # Choose the number of principal components (eigenvectors) you want to keep
        num_components = 3
        principal_components = sorted_eigenvectors[:, :num_components]
        return principal_components

    def compute_dbh(self, cloud):
        # passthrough filter to get the trunk points
        cluster_dim = self.get_cluster_dimensions(cloud)

        # Get bounding box and constrain the H size
        bbox = cloud.get_axis_aligned_bounding_box()
        min_bound = bbox.get_center() - bbox.get_half_extent()
        max_bound = bbox.get_center() + bbox.get_half_extent()
        max_bound[2] = cluster_dim["min_z"] + self._max_trunk_height
        bbox = o3d.t.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

        cloud_filtered = cloud.crop(bbox)

        if self._debug:
            print(
                f"DBH: Passthrough filter let {len(cloud_filtered.point.positions)} of {len(cloud.point.positions)} points."
            )
        # cloud_filtered = cloud
        if cloud_filtered.size < 10:
            dbh = 0
            if self._debug:
                print("DBH Insufficient points for fitting a cylinder")
            return dbh

        # compute cloud normals
        ne = cloud_filtered.make_NormalEstimation()
        tree = cloud_filtered.make_kdtree()
        ne.set_SearchMethod(tree)
        ne.set_KSearch(20)

        # fit cylinder
        seg = cloud_filtered.make_segmenter_normals(ksearch=20)
        seg.set_optimize_coefficients(True)
        # seg.set_model_type(pcl.SACMODEL_CYLINDER)
        seg.set_normal_distance_weight(0.1)
        # seg.set_method_type(pcl.SAC_RANSAC)
        seg.set_max_iterations(1000)
        seg.set_distance_threshold(0.10)
        seg.set_radius_limits(0, 0.5 * self._max_tree_diameter)
        [inliers_cylinder, coefficients_cylinder] = seg.segment()
        radius_cylinder = coefficients_cylinder[6]
        dbh = 2 * radius_cylinder
        return dbh

    def compute_cluster_info(self, cluster):
        # compute cluster mean
        cluster_np = cluster.point.positions.numpy()
        cluster_mean = np.mean(cluster_np, axis=0)
        cluster_mean = cluster_mean.tolist()
        # compute cluster dimensions
        cluster_dim = self.get_cluster_dimensions(cluster)
        # compute DBH
        dbh = self.compute_dbh(cluster)

        cluster_info = {
            "mean": cluster_mean,
            "dim": cluster_dim,
            "dbh": dbh,
        }
        return cluster_info


if __name__ == "__main__":
    """Minimal example"""
    import os
    import sys
    import json
    import matplotlib.pyplot as plt
    from digiforest_analysis.utils import pcd

    print("Tree segmentation")
    if len(sys.argv) != 3:
        print("Usage : ./script input_cloud output_folder")
    else:
        filename, extension = os.path.splitext(sys.argv[1])

        if extension != ".pcd":
            sys.exit("Input file must be a pcd file")

    cloud, header = pcd.load(sys.argv[1], binary=True)
    assert len(cloud.point.normals) > 0

    print("Processing", sys.argv[1])

    app = TreeSegmentation(
        voxel_size=0.05,
        cluster_tolerance=1.0,
        min_cluster_size=100,
        max_cluster_size=25000,
        min_tree_height=1.0,
        max_trunk_height=5.0,
        max_tree_diameter=2.5,
        min_gravity_alignment_score=0.0,
        debug=True,
    )
    tree_clouds, trees_info = app.process(cloud=cloud)

    print("Found " + str(len(tree_clouds)) + " trees")

    # Plot tree locations and DBH as scatter plot
    trees_loc_x = []
    trees_loc_y = []
    trees_dbh = []
    for tree_info in trees_info:
        trees_loc_x.append(tree_info["mean"][0])
        trees_loc_y.append(tree_info["mean"][1])
        trees_dbh.append(tree_info["dbh"])
    trees_area = [10**3 * 3.14 * (dbh / 2) ** 2 for dbh in trees_dbh]

    plt.scatter(
        trees_loc_x, trees_loc_y, s=trees_area, color="blue", marker="o", alpha=0.5
    )
    # Add index labels to each point
    for i, (xi, yi) in enumerate(zip(trees_loc_x, trees_loc_y)):
        plt.text(xi + 0.1, yi + 0.1, str(i), fontsize=10, ha="center", va="bottom")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Tree Locations")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Write output clusters to disk
    for j, tree_cloud in enumerate(tree_clouds):
        tree_name = "tree_cloud_" + str(j) + ".pcd"
        tree_cloud_filename = os.path.join(sys.argv[2], tree_name)
        tree_cloud.to_file(str.encode(tree_cloud_filename))

    # write tree info
    data = {"name": "John", "age": 30, "city": "New York"}
    for j, tree_info in enumerate(trees_info):
        tree_name = "tree_info_" + str(j) + ".json"
        tree_info_filename = os.path.join(sys.argv[2], tree_name)
        with open(tree_info_filename, "w") as json_file:
            json.dump(tree_info, json_file, indent=4)
