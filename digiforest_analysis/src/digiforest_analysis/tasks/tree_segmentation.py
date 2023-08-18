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
        print("Extracted " + str(len(clusters)) + " clusters")

        # filter out implausible clusters
        tree_clouds = self.filter_tree_clusters(clusters)
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
                not self.isClusterAlignmentStraightEnough(cluster)
                or self.isClusterTooLow(cluster)
                or self.isClusterRadiusTooBig(cluster)
            )

            if not isInvalidCluster:
                tree_clouds.append(cluster)
        return tree_clouds

    def isClusterTooLow(self, cluster):
        return False

    def isClusterRadiusTooBig(self, cluster):
        return False

    def isClusterAlignmentStraightEnough(self, cluster):
        return True


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
            cluster_tolerance=0.10,
            min_cluster_size=100,
            max_cluster_size=25000,
        )
        tree_clouds = app.process(cloud=cloud)

        # Write output clusters to disk
        for j, tree_cloud in enumerate(tree_clouds):
            tree_name = "tree_cloud_" + str(j) + ".pcd"
            tree_cloud_filename = os.path.join(sys.argv[2], tree_name)
            tree_cloud.to_file(str.encode(tree_cloud_filename))
