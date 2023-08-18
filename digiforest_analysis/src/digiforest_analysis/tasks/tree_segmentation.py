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
        if self._debug:
            num_filtered_clusters = len(clusters) - len(tree_clouds)
            print("Filtered out " + str(num_filtered_clusters) + " clusters.")
        return tree_clouds

    def extract_clusters(self, cloud):
        # downsample the input cloud
        vg = cloud.make_voxel_grid_filter()
        vg.set_leaf_size(self._voxel_size, self._voxel_size, self._voxel_size)
        cloud_ds = vg.filter()

        # Creating the kdtree object for the searching
        tree = cloud_ds.make_kdtree()
        # tree = cloud_ds.make_kdtree_flann()

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
        cluster_dim = {
            "dim_x": cluster_np_dim[0],
            "dim_y": cluster_np_dim[1],
            "dim_z": cluster_np_dim[2],
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


if __name__ == "__main__":
    """Minimal example"""
    import os
    import sys

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
            max_tree_diameter=2.0,
            min_gravity_alignment_score=0.0,
            debug=True,
        )
        tree_clouds = app.process(cloud=cloud)

        print("Found " + str(len(tree_clouds)) + " trees")

        # Write output clusters to disk
        for j, tree_cloud in enumerate(tree_clouds):
            tree_name = "tree_cloud_" + str(j) + ".pcd"
            tree_cloud_filename = os.path.join(sys.argv[2], tree_name)
            tree_cloud.to_file(str.encode(tree_cloud_filename))
