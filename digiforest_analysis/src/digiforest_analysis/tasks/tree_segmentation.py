from digiforest_analysis.tasks import BaseTask

import open3d as o3d
import numpy as np


class TreeSegmentation(BaseTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._normal_thr = kwargs.get("normal_thr", 0.5)
        self._voxel_size = kwargs.get("voxel_size", 0.05)
        self._clustering_method = kwargs.get("clustering_method", "dbscan_sk")

        # Filtering parameters
        self._filter_clusters = kwargs.get("filter_clusters", True)
        self._min_tree_height = kwargs.get("min_tree_height", 0.6)
        self._max_tree_diameter = kwargs.get("max_tree_diameter", 3.0)
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
        assert len(cloud.point.normals) > 0

        # Prefiltering
        cloud = self.prefiltering(cloud)

        # Extract clusters
        clusters = self.coarse_clustering(cloud, self._clustering_method)
        if self._debug:
            print("Extracted " + str(len(clusters)) + " initial clusters.")

        # Compute cluster attributes
        clusters = self.compute_clusters_info(clusters)

        # Filter out implausible clusters
        if self._filter_clusters:
            filtered_clusters = self.filter_tree_clusters(clusters)
        else:
            filtered_clusters = clusters

        if self._debug:
            num_filtered_clusters = len(clusters) - len(filtered_clusters)
            print(f"Filtered out {num_filtered_clusters} clusters.")
        return filtered_clusters

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
            clusters.append({"cloud": seg_cloud, "info": {}})

        return clusters

    def filter_tree_clusters(self, clusters):
        filtered_clusters = []
        for i, cluster in enumerate(clusters):
            if self._debug:
                print(f"-- Cluster {i} --")
            valid_alignment = self.check_cluster_alignment(cluster)
            valid_height = self.check_cluster_height(cluster)
            valid_size = self.check_cluster_size(cluster)

            if valid_alignment and valid_height and valid_size:
                filtered_clusters.append(cluster)

        return filtered_clusters

    def check_cluster_alignment(self, cluster):
        cluster_principal_axis = cluster["info"]["principal_axis"]
        gravity_axis = np.array([0, 0, 1])
        alignment_score = np.abs(np.dot(cluster_principal_axis, gravity_axis))
        if alignment_score > self._min_gravity_alignment_score:
            return True
        if self._debug:
            print(
                f"Alignment check: {alignment_score} < {self._min_gravity_alignment_score}"
            )
        return False

    def check_cluster_height(self, cluster):
        if cluster["info"]["dim_z"] > self._min_tree_height:
            return True
        if self._debug:
            print(
                f"Height check: {cluster['info']['dim_z'] } < {self._min_tree_height} "
            )
        return False

    def check_cluster_size(self, cluster):
        if (
            cluster["info"]["dim_x"] < self._max_tree_diameter
            or cluster["info"]["dim_y"] < self._max_tree_diameter
        ):
            return True
        if self._debug:
            print(
                f"Size check: {cluster['info']['dim_x'] } > {self._max_tree_diameter} or {cluster['info']['dim_y'] } > {self._max_tree_diameter}"
            )
        return False

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

    def compute_clusters_info(self, clusters):
        for i, c in enumerate(clusters):
            # compute cluster center
            cluster_np = c["cloud"].point.positions.numpy()
            cluster_mean = np.mean(cluster_np, axis=0)
            cluster_mean = cluster_mean.tolist()
            clusters[i]["info"]["mean"] = cluster_mean

            # Compute cluster dimensions
            bbox = c["cloud"].get_axis_aligned_bounding_box()
            min_bound = bbox.get_center() - bbox.get_half_extent()
            max_bound = bbox.get_center() + bbox.get_half_extent()

            clusters[i]["info"]["dim_x"] = (max_bound[0] - min_bound[0]).item()
            clusters[i]["info"]["dim_y"] = (max_bound[1] - min_bound[1]).item()
            clusters[i]["info"]["dim_z"] = (max_bound[2] - min_bound[2]).item()
            clusters[i]["info"]["min_z"] = (min_bound[2]).item()

            # Main axis
            cluster_pca = self.compute_pca(c["cloud"])
            cluster_principal_axis = cluster_pca[:, 0]
            clusters[i]["info"]["principal_axis"] = cluster_principal_axis.tolist()

            # # Compute DBH
            # dbh = self.compute_dbh(c["cloud"])
            clusters[i]["info"]["dbh"] = 0.5

        return clusters


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
        min_tree_height=0.6,
        max_trunk_height=2.0,
        max_tree_diameter=2.5,
        min_gravity_alignment_score=0.7,
        debug=True,
        filter_clusters=False,
    )
    trees = app.process(cloud=cloud)
    print("Found " + str(len(trees)) + " trees")

    # Visualize clouds
    n_points = len(cloud.point.positions)
    cloud.paint_uniform_color([0.9, 0.9, 0.9])

    viz_clouds = []
    viz_clouds.append(cloud.to_legacy())

    n_trees = len(trees)
    cmap = plt.get_cmap("prism")
    for i, t in enumerate(trees):
        tree_cloud = t["cloud"]
        tree_cloud.paint_uniform_color(cmap(i)[:3])
        viz_clouds.append(tree_cloud.to_legacy())

    o3d.visualization.draw_geometries(
        viz_clouds,
        zoom=0.5,
        front=[0.79, 0.02, 0.60],
        lookat=[2.61, 2.04, 1.53],
        up=[-0.60, -0.012, 0.79],
    )

    # Plot tree locations and DBH as scatter plot
    trees_loc_x = []
    trees_loc_y = []
    trees_dbh = []
    for tree in trees:
        trees_loc_x.append(tree["info"]["mean"][0])
        trees_loc_y.append(tree["info"]["mean"][1])
        trees_dbh.append(tree["info"]["dbh"])
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

    # Write output
    for i, tree in enumerate(trees):
        # Write clouds
        tree_name = f"tree_cloud_{i}.pcd"
        tree_cloud_filename = os.path.join(sys.argv[2], tree_name)
        header_fix = {"VIEWPOINT": header["VIEWPOINT"]}
        pcd.write(tree["cloud"], header_fix, tree_cloud_filename)

        # write tree info
        tree_name = f"tree_info_{i}.json"
        tree_info_filename = os.path.join(sys.argv[2], tree_name)
        with open(tree_info_filename, "w") as json_file:
            json.dump(tree["info"], json_file, indent=4)
