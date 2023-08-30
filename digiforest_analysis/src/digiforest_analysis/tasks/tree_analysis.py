from digiforest_analysis.tasks import BaseTask
from digiforest_analysis.utils import cylinder

import open3d as o3d


class TreeAnalysis(BaseTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._max_dist_to_ground = kwargs.get("max_dist_to_ground", 1.0)
        self._fitting_method = kwargs.get("fitting_method", "pcl_ransac")
        self._breast_height = kwargs.get("breast_height", 1.3)
        self._breast_height_range = kwargs.get("breast_height_range", 0.5)
        self._max_valid_radius = kwargs.get("max_valid_radius", 0.5)
        self._min_inliers = kwargs.get("min_inliers", 4)
        self._outlier_thr = kwargs.get("outlier_thr", 0.01)
        self._loss_scale = kwargs.get("loss_scale", 0.1)

    def _process(self, **kwargs):
        trees = kwargs.get("trees")
        ground_cloud = kwargs.get("ground_cloud")

        filtered_trees = []
        for tree in trees:
            if self._debug_level > 0:
                print(f"Processing tree {tree['info']['id']}...")

            valid_dbh, model = self.compute_dbh_model(tree, ground_cloud)

            if valid_dbh:
                tree["info"]["dbh"] = model["radius"] * 2
                tree["info"]["dbh_model"] = model
                filtered_trees.append(tree)
            elif self._debug_level > 0:
                print("Skipped")

        # Debug visualizations
        if self._debug_level > 1:
            self.debug_visualizations(trees, filtered_trees)

        return filtered_trees

    def compute_dbh_model(self, tree, ground_cloud):
        cloud = tree["cloud"]
        info = tree["info"]
        n_points = len(cloud.point.positions)

        # # Get bounding box
        bbox = cloud.get_axis_aligned_bounding_box()
        min_bound = bbox.get_center() - bbox.get_half_extent()
        max_bound = bbox.get_center() + bbox.get_half_extent()

        # Check if the bounding box is close to the ground
        bounding_box = o3d.geometry.PointCloud()
        bounding_box.points.append(min_bound.numpy()[:, None])
        dist = bounding_box.compute_point_cloud_distance(ground_cloud.to_legacy())[0]

        if dist > self._max_dist_to_ground:
            if self._debug_level > 0:
                print(
                    f"Tree {info['id']} discarded: Too far from ground {dist:.2f}>{self._max_dist_to_ground}"
                )
            return False, {}

        # Constraint points to compute DBH
        min_bound[2] = min_bound[2] + self._breast_height - self._breast_height_range
        max_bound[2] = min_bound[2] + 2 * self._breast_height_range
        bbox_filtered = o3d.t.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

        cloud_filtered = cloud.crop(bbox_filtered)
        n_points = len(cloud_filtered.point.positions)

        # RANSAC
        if n_points < self._min_inliers:
            if self._debug_level > 0:
                print(
                    f"Tree {info['id']} discarded: Not enough points {n_points}/{self._min_inliers}"
                )
            return False, {}

        # Fit
        points = cloud_filtered.point.positions.numpy()
        normals = cloud_filtered.point.normals.numpy()
        model = cylinder.fit(
            points,
            method=self._fitting_method,
            N=normals,
            min_inliers=self._min_inliers,
            outlier_thr=self._outlier_thr,
            loss_scale=self._loss_scale,
        )

        success_check = model["success"]
        radius_check = (
            model["radius"] >= 0.01 and model["radius"] < self._max_valid_radius
        )

        # Debug
        if self._debug_level > 2:
            cylinder_mesh = cylinder.to_mesh(model)
            cloud.paint_uniform_color([0.0, 0.0, 0.0])
            cloud_filtered.paint_uniform_color([0.0, 0.0, 1.0])

            if not success_check:
                cylinder_mesh.paint_uniform_color([1.0, 0.0, 0.0])  # red
                window_msg = f"Tree {info['id']}: Invalid inliers: {model['inliers'].shape[0]}/{self._min_inliers}"
            elif not radius_check:
                cylinder_mesh.paint_uniform_color([1.0, 0.5, 0.0])  # orange
                window_msg = f"Tree {info['id']}: Invalid radius: {model['radius']:.3f}"
            else:
                cylinder_mesh.paint_uniform_color([0.0, 1.0, 0.0])  # green
                window_msg = f"Tree {info['id']}: Success"

            o3d.visualization.draw_geometries(
                [cloud.to_legacy(), cloud_filtered.to_legacy(), cylinder_mesh],
                window_name=window_msg,
                front=[0.18, -0.9, 0.39],
                lookat=cloud.point.positions.mean(dim=0).numpy(),
                up=[0.14, 0.42, 0.89],
                zoom=0.7,
            )

        if not success_check:
            if self._debug_level > 0:
                print(
                    f"Tree {info['id']} discarded. Not enough inliers {model['inliers'].shape[0]}/{self._min_inliers}"
                )
            return False, {}

        if not radius_check:
            if self._debug_level > 0:
                print(
                    f"Tree {info['id']} discarded. Invalid radius {model['radius']:.3f}]"
                )
            return False, {}

        if self._debug_level > 0:
            n_inliers = len(model["inliers"])
            print(
                f"Tree {info['id']} DBH: {n_inliers}/{n_points} points. radius: {model['radius']:.3f}"
            )

        return True, model

    def debug_visualizations(self, trees, filtered_trees):
        import matplotlib.pyplot as plt
        import numpy as np
        from digiforest_analysis.utils import visualization as viz

        cmap = plt.get_cmap("tab20b")

        # Visualize clouds
        viz_clouds = []

        for t in trees:
            tree_cloud = t["cloud"].clone()
            i = t["info"]["id"]
            # tree_cloud.paint_uniform_color(cmap(i % 20)[:3])
            tree_cloud.paint_uniform_color([0.7, 0.7, 0.7])
            viz_clouds.append(tree_cloud.to_legacy())

        trees_center = np.zeros(3, dtype=np.float32)
        for t in filtered_trees:
            tree_cloud = t["cloud"].clone()
            trees_center += tree_cloud.get_center().numpy()

            i = t["info"]["id"]
            color = cmap(i % 20)[:3]
            tree_cloud.paint_uniform_color(color)
            viz_clouds.append(tree_cloud.to_legacy())

            model = t["info"]["dbh_model"]
            cylinder_mesh = cylinder.to_mesh(model)
            cylinder_mesh.paint_uniform_color(color)
            viz_clouds.append(cylinder_mesh)

            bbox = tree_cloud.get_axis_aligned_bounding_box()
            mesh = viz.bbox_to_mesh(bbox, depth=0.1, offset=[0, 0, -0.1], color=color)
            viz_clouds.append(mesh)

        n_trees = len(filtered_trees)
        trees_center /= n_trees

        o3d.visualization.draw_geometries(
            viz_clouds,
            zoom=0.7,
            front=[0.79, 0.02, 0.60],
            lookat=trees_center,
            up=[-0.60, -0.012, 0.79],
            window_name="tree_analysis",
        )

        from digiforest_analysis.utils import marteloscope

        fig, ax = plt.subplots()
        marteloscope.plot(filtered_trees, ax, cmap="tab20b")

        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_title("Tree Locations")
        ax.autoscale_view()
        ax.set_aspect("equal")
        # ax.legend()
        ax.grid(True)
        plt.show()


if __name__ == "__main__":
    """Minimal example"""
    import os
    import sys
    import yaml
    from digiforest_analysis.utils import io
    from pathlib import Path

    print("Tree analysis")
    if len(sys.argv) != 3:
        print("Usage : ./script input_folder output_folder")
    else:
        base_path = Path(sys.argv[1])
        cloud_files = list(base_path.glob("tree_cloud*.pcd"))
        info_files = list(base_path.glob("tree_info*.yaml"))
        cloud_files.sort()
        info_files.sort()

    trees = []
    for c, i in zip(cloud_files, info_files):
        cloud, header = io.load(str(c), binary=True)
        info = yaml.safe_load(i.read_text())
        trees.append({"cloud": cloud, "info": info})

    app = TreeAnalysis(
        debug_level=1,
    )
    filtered_trees = app.process(trees=trees)

    # Write cloud
    for tree in filtered_trees:
        # Write clouds
        i = tree["info"]["id"]
        cloud = tree["cloud"]

        tree_name = f"tree_cloud_{i:04}.pcd"
        tree_cloud_filename = os.path.join(sys.argv[2], tree_name)
        header_fix = {"VIEWPOINT": header["VIEWPOINT"]}
        io.write(cloud, header_fix, tree_cloud_filename)

        # Write tree info
        tree_name = f"tree_info_{i:04}.yaml"
        tree_info_filename = os.path.join(sys.argv[2], tree_name)
        with open(tree_info_filename, "w") as yaml_file:
            yaml.dump(tree["info"], yaml_file, indent=4)
