from digiforest_analysis.tasks import BaseTask

import pcl
import numpy as np


class TreeSegmentation(BaseTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._voxel_size = kwargs.get("voxel_size", 0.05)
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

        # extract clusters
        clusters = self.extract_clusters(cloud)
        if self._debug:
            print("Extracted " + str(len(clusters)) + " initial clusters.")

        # filter out implausible clusters
        tree_clouds = self.filter_tree_clusters(clusters)
        trees_info = []
        for tree in tree_clouds:
            cluster_info = self.compute_cluster_info(tree)
            trees_info.append(cluster_info)

        if self._debug:
            num_filtered_clusters = len(clusters) - len(tree_clouds)
            print("Filtered out " + str(num_filtered_clusters) + " clusters.")
        return tree_clouds, trees_info

    def extract_clusters(self, cloud):
        # downsample the input cloud
        vg = cloud.make_voxel_grid_filter()
        vg.set_leaf_size(self._voxel_size, self._voxel_size, self._voxel_size)
        cloud_ds = vg.filter()

        # creating the kdtree object for the searching
        tree = cloud_ds.make_kdtree()

        # perform Euclidean Clustering
        ec = cloud_ds.make_EuclideanClusterExtraction()
        ec.set_ClusterTolerance(self._cluster_tolerance)
        ec.set_MinClusterSize(self._min_cluster_size)
        ec.set_MaxClusterSize(self._max_cluster_size)
        ec.set_SearchMethod(tree)
        cluster_indices = ec.Extract()

        if not cluster_indices:
            print("No clusters found in the pointcloud.")
            return

        # get cluster points
        clusters = []
        for j, indices in enumerate(cluster_indices):
            cluster = pcl.PointCloud()
            points = np.zeros((len(indices), 3), dtype=np.float32)
            for i, indice in enumerate(indices):
                points[i][0] = cloud_ds[indice][0]
                points[i][1] = cloud_ds[indice][1]
                points[i][2] = cloud_ds[indice][2]
            cluster.from_array(points)
            clusters.append(cluster)
        return clusters

    def filter_tree_clusters(self, clusters):
        tree_clouds = []
        for cluster in clusters:
            isInvalidCluster = (
                (not self.is_cluster_alignment_straight_enough(cluster))
                or self.is_cluster_too_low(cluster)
                or self.is_cluster_radius_too_big(cluster)
            )

            if isInvalidCluster:
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
        cluster_np = cluster.to_array()
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
        cluster_np = cluster.to_array()
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
        passthrough = cloud.make_passthrough_filter()
        passthrough.set_filter_field_name("z")
        passthrough.set_filter_limits(
            cluster_dim["min_z"], cluster_dim["min_z"] + self._max_trunk_height
        )
        cloud_filtered = passthrough.filter()
        if self._debug:
            print(
                "DBH: Passthrough filter let "
                + str(cloud_filtered.size)
                + " of "
                + str(cloud.size)
                + "points."
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
        seg.set_model_type(pcl.SACMODEL_CYLINDER)
        seg.set_normal_distance_weight(0.1)
        seg.set_method_type(pcl.SAC_RANSAC)
        seg.set_max_iterations(1000)
        seg.set_distance_threshold(0.10)
        seg.set_radius_limits(0, 0.5 * self._max_tree_diameter)
        [inliers_cylinder, coefficients_cylinder] = seg.segment()
        radius_cylinder = coefficients_cylinder[6]
        dbh = 2 * radius_cylinder
        return dbh

    def compute_cluster_info(self, cluster):
        # compute cluster mean
        cluster_np = cluster.to_array()
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

    print("Tree segmentation")
    if len(sys.argv) != 3:
        print("Usage : ./script input_cloud output_folder")
    else:
        filename, extension = os.path.splitext(sys.argv[1])

        if extension != ".pcd":
            sys.exit("Input file must be a pcd file")

        print("Processing", sys.argv[1])

        cloud = pcl.PointCloud()
        cloud._from_pcd_file(sys.argv[1].encode("utf-8"))
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
