from digiforest_analysis.tasks import BaseTask

import open3d as o3d
import numpy as np


class TreeSegmentation(BaseTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._rejected_clusters = []

        self._normal_thr = kwargs.get("normal_thr", 0.5)
        self._voxel_size = kwargs.get("voxel_size", 0.1)
        self._cluster_2d = kwargs.get("cluster_2d", False)
        self._clustering_method = kwargs.get("clustering_method", "hdbscan")

        # Filtering parameters
        self._min_tree_height = kwargs.get("min_tree_height", 1.5)
        self._max_tree_diameter = kwargs.get("max_tree_diameter", 10.0)
        self._min_tree_diameter = kwargs.get("min_tree_diameter", 0.1)
        self._min_gravity_alignment_score = kwargs.get(
            "min_gravity_alignment_score", 0.1
        )

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
        filtered_clusters = self.filter_tree_clusters(clusters)

        if self._debug:
            num_filtered_clusters = len(clusters) - len(filtered_clusters)
            print(f"Filtered out {num_filtered_clusters} clusters.")

        return filtered_clusters

    @property
    def rejected_clusters(self):
        return self._rejected_clusters

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
        if self._cluster_2d:
            points = cloud.point.positions.numpy()[:, :2]
        else:
            points = cloud.point.positions.numpy()

        if method == "dbscan_o3d":
            eps = 0.8
            min_cluster_size = 20
            labels = cloud.cluster_dbscan(
                eps=eps, min_points=min_cluster_size, print_progress=False
            ).numpy()

        elif method == "dbscan_sk":
            from sklearn.cluster import DBSCAN

            eps = 0.3
            min_cluster_size = 20
            db = DBSCAN(eps=eps, min_samples=min_cluster_size).fit(points)
            labels = db.labels_

        elif method == "hdbscan_sk":
            from sklearn.cluster import HDBSCAN

            min_cluster_size = 20
            db = HDBSCAN(min_samples=min_cluster_size).fit(points)
            labels = db.labels_

        elif method == "hdbscan":
            import hdbscan

            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=20, algorithm="best", core_dist_n_jobs=1
            )
            labels = clusterer.fit_predict(points)

        elif method == "kmeans":
            from sklearn.cluster import KMeans

            num_clusters = 350
            labels = KMeans(n_clusters=num_clusters, n_init="auto").fit_predict(points)

        else:
            raise NotImplementedError(f"Method [{method}] not available")

        # Get max number of labels
        num_labels = labels.max() + 1

        # Prepare output clouds
        clusters = []
        for i in range(num_labels):
            mask = labels == i
            seg_cloud = o3d.t.geometry.PointCloud(cloud.select_by_mask(mask))
            clusters.append({"cloud": seg_cloud, "info": {"id": i}})

        return clusters

    def filter_tree_clusters(self, clusters):
        filtered_clusters = []
        self._rejected_clusters = []
        for i, cluster in enumerate(clusters):
            valid_alignment = self.check_cluster_alignment(cluster)
            valid_height = self.check_cluster_height(cluster)
            valid_size = self.check_cluster_size(cluster)

            if valid_alignment and valid_height and valid_size:
                filtered_clusters.append(cluster)
            else:
                if not valid_alignment:
                    cluster["cloud"].paint_uniform_color([0.0, 0.0, 0.0])
                elif not valid_height:
                    cluster["cloud"].paint_uniform_color([1.0, 0.0, 0.0])
                elif not valid_size:
                    cluster["cloud"].paint_uniform_color([0.0, 0.0, 1.0])
                self._rejected_clusters.append(cluster)

        return filtered_clusters

    def check_cluster_alignment(self, cluster):
        cluster_principal_axis = cluster["info"]["principal_axis"]
        gravity_axis = np.array([0, 0, 1])
        alignment_score = np.abs(np.dot(cluster_principal_axis, gravity_axis))
        if alignment_score >= self._min_gravity_alignment_score:
            return True
        if self._debug:
            print(
                f"Cluster {cluster['id']}. Invalid alignment: {alignment_score:.2f} < {self._min_gravity_alignment_score:.2f}"
            )
        return False

    def check_cluster_height(self, cluster):
        if cluster["info"]["size_z"] >= self._min_tree_height:
            return True
        if self._debug:
            print(
                f"Cluster {cluster['id']}. Invalid height: {cluster['info']['size_z'] } < {self._min_tree_height} "
            )
        return False

    def check_cluster_size(self, cluster):
        x_max_valid = cluster["info"]["size_x"] < self._max_tree_diameter
        y_max_valid = cluster["info"]["size_y"] < self._max_tree_diameter

        x_min_valid = cluster["info"]["size_x"] > self._min_tree_diameter
        y_min_valid = cluster["info"]["size_y"] > self._min_tree_diameter

        if self._debug:
            if not x_max_valid:
                print(
                    f"Cluster {cluster['id']}. Invalid x_size: {cluster['info']['size_x']:.2f} > {self._max_tree_diameter:.2f}"
                )
            if not y_max_valid:
                print(
                    f"Cluster {cluster['id']}. Invalid y_size: {cluster['info']['size_y']:.2f} > {self._max_tree_diameter:.2f}"
                )
            if not x_min_valid:
                print(
                    f"Cluster {cluster['id']}. Invalid x_size: {cluster['info']['size_x']:.2f} < {self._min_tree_diameter:.2f}"
                )
            if not y_min_valid:
                print(
                    f"Cluster {cluster['id']}. Invalid y_size: {cluster['info']['size_y']:.2f} < {self._min_tree_diameter:.2f}"
                )

        return x_max_valid and y_max_valid and x_min_valid and y_min_valid

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
            extent = bbox.get_extent()

            clusters[i]["info"]["size_x"] = extent[0].item()
            clusters[i]["info"]["size_y"] = extent[1].item()
            clusters[i]["info"]["size_z"] = extent[2].item()

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
    import yaml
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
        debug=False,
    )
    trees = app.process(cloud=cloud)

    # Visualize clouds
    n_points = len(cloud.point.positions)
    cloud.paint_uniform_color([0.9, 0.9, 0.9])

    viz_clouds = []
    # viz_clouds.append(cloud.to_legacy())

    n_trees = len(trees)
    cmap = plt.get_cmap("tab20")
    for i, t in enumerate(trees):
        color = cmap(i % 20)[:3]

        tree_cloud = t["cloud"]
        tree_cloud.paint_uniform_color(color)
        viz_clouds.append(tree_cloud.to_legacy())

        bbox = tree_cloud.get_axis_aligned_bounding_box()
        bbox.set_color(color)
        viz_clouds.append(bbox.to_legacy())

    for i, t in enumerate(app.rejected_clusters):
        color = [0.1, 0.1, 0.1]
        tree_cloud = t["cloud"]
        viz_clouds.append(tree_cloud.to_legacy())

        bbox = tree_cloud.get_axis_aligned_bounding_box()
        bbox.set_color(color)
        viz_clouds.append(bbox.to_legacy())

    # viewer = o3d.visualization.Visualizer()
    # viewer.create_window()
    # for g in viz_clouds:
    #     viewer.add_geometry(g)
    # opt = viewer.get_render_option()
    # opt.show_coordinate_frame = True
    # opt.background_color = np.asarray([1.0, 1.0, 1.0])
    # viewer.run()
    # viewer.destroy_window()

    o3d.visualization.draw_geometries(
        viz_clouds,
        zoom=0.5,
        front=[0.79, 0.02, 0.60],
        lookat=[2.61, 2.04, 1.53],
        up=[-0.60, -0.012, 0.79],
    )

    # Write output
    for tree in trees:
        # Write clouds
        i = tree["info"]["id"]

        # Write cloud
        tree_name = f"tree_cloud_{i:04}.pcd"
        tree_cloud_filename = os.path.join(sys.argv[2], tree_name)
        header_fix = {"VIEWPOINT": header["VIEWPOINT"]}
        pcd.write(tree["cloud"], header_fix, tree_cloud_filename)

        # Write tree info
        tree_name = f"tree_info_{i:04}.yaml"
        tree_info_filename = os.path.join(sys.argv[2], tree_name)
        with open(tree_info_filename, "w") as yaml_file:
            yaml.dump(tree["info"], yaml_file, indent=4)
