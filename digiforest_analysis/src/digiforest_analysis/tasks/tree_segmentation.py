from digiforest_analysis.tasks import BaseTask
from digiforest_analysis.utils import clustering
from digiforest_analysis.utils import plotting

import open3d as o3d
import numpy as np


class TreeSegmentation(BaseTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._rejected_clusters = []

        self._normal_thr = kwargs.get("normal_thr", 0.5)
        self._voxel_size = kwargs.get("voxel_size", 0.05)
        self._cluster_2d = kwargs.get("cluster_2d", False)
        self._clustering_method = kwargs.get("clustering_method", "hdbscan")

        # Filtering parameters
        self._min_tree_height = kwargs.get("min_tree_height", 1.5)
        self._max_tree_diameter = kwargs.get("max_tree_diameter", 10.0)
        self._min_tree_diameter = kwargs.get("min_tree_diameter", 0.1)
        self._min_gravity_alignment_score = kwargs.get(
            "min_gravity_alignment_score", 0.1
        )
        self._max_cluster_size = kwargs.get("max_cluster_size", 2.0)

        # Colormap parameters
        self._cmap = plotting.color_palette
        self._ncolors = plotting.n_colors

    def _process(self, **kwargs):
        """ "
        Processes the forest cloud with ground removed and outputs a list of tree clouds

        Returns:
            _type_: _description_
        """
        cloud = kwargs.get("cloud")
        assert len(cloud.point.normals) > 0
        del kwargs["cloud"]  # so **kwargs doesn't lead to duplicate cloud arg

        # Prefiltering
        cloud = self.prefiltering(cloud)

        # Extract clusters
        clusters = self.clustering(cloud, **kwargs)
        if self._debug_level > 0:
            print("Extracted " + str(len(clusters)) + " initial clusters.")

        # Compute cluster attributes
        clusters = self.compute_clusters_info(clusters)

        # Filter out implausible clusters
        filtered_clusters = self.filter_tree_clusters(clusters)

        if self._debug_level > 0:
            num_filtered_clusters = len(clusters) - len(filtered_clusters)
            print(f"Filtered out {num_filtered_clusters} clusters.")

        # Debug visualizations
        if self._debug_level > 1:
            self.debug_visualizations(cloud, filtered_clusters)

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
        # new_cloud = new_cloud.voxel_down_sample(voxel_size=self._voxel_size) # faster way
        new_cloud, _, _ = new_cloud.to_legacy().voxel_down_sample_and_trace(
            self._voxel_size,
            new_cloud.get_min_bound().numpy().astype(np.float64),
            new_cloud.get_max_bound().numpy().astype(np.float64),
        )
        new_cloud = o3d.t.geometry.PointCloud.from_legacy(new_cloud)

        return new_cloud

    def clustering(self, cloud, cloth=None, recluster_flag=True, **kwargs):
        # Run clustering
        labels = clustering.cluster(
            cloud,
            cloth=cloth,
            method=self._clustering_method,
            cluster_2d=self._cluster_2d,
            debug_level=self._debug_level,
            **kwargs,
        )

        # Get max number of labels
        num_labels = labels.max() + 1

        # Prepare output clouds
        clusters = []
        for i in range(num_labels):
            mask = labels == i
            seg_cloud = o3d.t.geometry.PointCloud(cloud.select_by_mask(mask))
            clusters.append({"cloud": seg_cloud, "info": {"id": i}})

        refined_clusters = []
        idx = 0
        for cluster in clusters:
            extent = (
                cluster["cloud"].get_axis_aligned_bounding_box().get_extent().numpy()
            )
            if recluster_flag and (
                extent[0] > self._max_cluster_size or extent[1] > self._max_cluster_size
            ):
                labels = clustering.cluster(
                    cluster["cloud"],
                    method=self._clustering_method,
                    cluster_2d=self._cluster_2d,
                    debug_level=self._debug_level**kwargs,
                )

                # Get max number of labels
                num_labels = labels.max() + 1
                for i in range(num_labels):
                    mask = labels == i
                    seg_cloud = o3d.t.geometry.PointCloud(
                        cluster["cloud"].select_by_mask(mask)
                    )
                    color = self._cmap[idx % self._ncolors]
                    refined_clusters.append(
                        {"cloud": seg_cloud, "info": {"id": idx, "color": color}}
                    )
                    idx += 1
            else:
                color = self._cmap[idx % self._ncolors]
                refined_clusters.append(
                    {"cloud": cluster["cloud"], "info": {"id": idx, "color": color}}
                )
                idx += 1

        return refined_clusters

    def filter_tree_clusters(self, clusters):
        filtered_clusters = []
        self._rejected_clusters = []
        for cluster in clusters:
            valid_alignment = self.check_cluster_alignment(cluster)
            valid_height = self.check_cluster_height(cluster)
            valid_size = self.check_cluster_size(cluster)

            if valid_height and valid_size:
                filtered_clusters.append(cluster)
            else:
                if not valid_alignment:
                    cluster["cloud"].paint_uniform_color([0.0, 0.0, 0.0])
                elif not valid_height:
                    cluster["cloud"].paint_uniform_color([1.0, 0.0, 0.0])
                elif not valid_size:
                    cluster["cloud"].paint_uniform_color([0.0, 0.0, 1.0])
                self._rejected_clusters.append(cluster)

            if self._debug_level > 2:
                bbox = cluster["cloud"].get_axis_aligned_bounding_box()
                bbox.set_color([0.0, 0.0, 0.0])
                o3d.visualization.draw_geometries(
                    [
                        cluster["cloud"].to_legacy(),
                        bbox.to_legacy(),
                    ],
                    window_name=f"Cluster {cluster['info']['id'] }",
                )

        return filtered_clusters

    def check_cluster_alignment(self, cluster):
        cluster_principal_axis = cluster["info"]["principal_axis"]
        gravity_axis = np.array([0, 0, 1])
        alignment_score = np.abs(np.dot(cluster_principal_axis, gravity_axis))
        if alignment_score >= self._min_gravity_alignment_score:
            return True
        if self._debug_level > 0:
            print(
                f"Cluster {cluster['info']['id']}. Invalid alignment: {alignment_score:.2f} < {self._min_gravity_alignment_score:.2f}"
            )
        return False

    def check_cluster_height(self, cluster):
        if cluster["info"]["size_z"] >= self._min_tree_height:
            return True
        if self._debug_level > 0:
            print(
                f"Cluster {cluster['info']['id']}. Invalid height: {cluster['info']['size_z']:.2f} < {self._min_tree_height} "
            )
        return False

    def check_cluster_size(self, cluster):
        x_max_valid = cluster["info"]["size_x"] < self._max_tree_diameter
        y_max_valid = cluster["info"]["size_y"] < self._max_tree_diameter

        x_min_valid = cluster["info"]["size_x"] > self._min_tree_diameter
        y_min_valid = cluster["info"]["size_y"] > self._min_tree_diameter

        if self._debug_level > 0:
            if not x_max_valid:
                print(
                    f"Cluster {cluster['info']['id']}. Invalid x_size: {cluster['info']['size_x']:.2f} > {self._max_tree_diameter:.2f}"
                )
            if not y_max_valid:
                print(
                    f"Cluster {cluster['info']['id']}. Invalid y_size: {cluster['info']['size_y']:.2f} > {self._max_tree_diameter:.2f}"
                )
            if not x_min_valid:
                print(
                    f"Cluster {cluster['info']['id']}. Invalid x_size: {cluster['info']['size_x']:.2f} < {self._min_tree_diameter:.2f}"
                )
            if not y_min_valid:
                print(
                    f"Cluster {cluster['info']['id']}. Invalid y_size: {cluster['info']['size_y']:.2f} < {self._min_tree_diameter:.2f}"
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

    def debug_visualizations(self, cloud, clusters):
        from digiforest_analysis.utils import visualization as viz

        # Visualize clouds
        viz_clouds = []
        # origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        # viz_clouds.append(origin)

        # cloud_copy = cloud.clone()
        # cloud_copy.paint_uniform_color([0.7, 0.7, 0.7])
        # viz_clouds.append(cloud_copy.to_legacy())

        for c in clusters:
            color = c["info"]["color"]

            tree_cloud = c["cloud"].clone()
            tree_cloud.paint_uniform_color(color)
            viz_clouds.append(tree_cloud.to_legacy())

            bbox = tree_cloud.get_axis_aligned_bounding_box()
            bbox.set_color(color)
            viz_clouds.append(bbox.to_legacy())

            mesh = viz.bbox_to_mesh(bbox, depth=0.1, offset=[0, 0, -0.1], color=color)
            viz_clouds.append(mesh)

        # for i, t in enumerate(self.rejected_clusters):
        #     color = [0.1, 0.1, 0.1]
        #     tree_cloud = t["cloud"].clone()
        #     tree_cloud.paint_uniform_color(color)
        #     viz_clouds.append(tree_cloud.to_legacy())

        #     bbox = tree_cloud.get_axis_aligned_bounding_box()
        #     bbox.set_color(color)
        #     viz_clouds.append(bbox.to_legacy())

        #     mesh = viz.bbox_to_mesh(bbox, depth=0.1, offset=[0, 0, -0.1], color=color)
        #     viz_clouds.append(mesh)

        o3d.visualization.draw_geometries(
            viz_clouds,
            zoom=self.viz_zoom,
            front=[0.79, 0.2, 0.60],
            lookat=self.viz_center,
            up=[-0.55, -0.15, 0.8],
            window_name="tree_segmentation",
        )


if __name__ == "__main__":
    """Minimal example"""
    import os
    import sys
    import yaml
    from digiforest_analysis.utils import io

    print("Tree segmentation")
    if len(sys.argv) != 3:
        print("Usage : ./script input_cloud output_folder")
    else:
        filename, extension = os.path.splitext(sys.argv[1])

        if extension != ".pcd":
            sys.exit("Input file must be a pcd file")

    cloud, header = io.load(sys.argv[1], binary=True)
    assert len(cloud.point.normals) > 0

    print("Processing", sys.argv[1])

    app = TreeSegmentation(debug_level=2, cluster_2d=True, clustering_method="hdbscan")
    trees = app.process(cloud=cloud)

    # Write output
    for tree in trees:
        # Write clouds
        i = tree["info"]["id"]

        # Tree height normalize
        tree_cloud = tree["cloud"]

        # shift to zero (debugging)
        # z_shift = cloud.point.positions[:, 2].min()
        # cloud.point.positions[:, 2] = cloud.point.positions[:, 2] - z_shift

        # Write cloud
        tree_name = f"tree_cloud_{i:04}.pcd"
        tree_cloud_filename = os.path.join(sys.argv[2], tree_name)
        header_fix = {"VIEWPOINT": header["VIEWPOINT"]}
        io.write(tree_cloud, header_fix, tree_cloud_filename)

        # Write tree info
        tree_name = f"tree_info_{i:04}.yaml"
        tree_info_filename = os.path.join(sys.argv[2], tree_name)
        with open(tree_info_filename, "w") as yaml_file:
            yaml.dump(tree["info"], yaml_file, indent=4)
