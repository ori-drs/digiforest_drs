from digiforest_analysis.tasks import BaseTask
from digiforest_analysis.utils import cylinder

import open3d as o3d


class TreeAnalysis(BaseTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._fitting_method = kwargs.get("fitting_method", "pcl_ransac")
        self._breast_height = kwargs.get("breast_height", 1.3)
        self._breast_height_range = kwargs.get("breast_height_range", 0.5)
        self._max_valid_radius = kwargs.get("max_valid_radius", 0.5)
        self._min_inliers = kwargs.get("min_inliers", 4)

    def _process(self, **kwargs):
        trees = kwargs.get("trees")

        filtered_trees = []
        for tree in trees:
            if self._debug_level > 0:
                print(f"Processing tree {tree['info']['id']}...")

            valid_dbh, model = self.compute_dbh_model(tree)

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

    def compute_dbh_model(self, tree):
        cloud = tree["cloud"]
        info = tree["info"]
        n_points = len(cloud.point.positions)

        # # Get bounding box and constrain the H size
        bbox = cloud.get_axis_aligned_bounding_box()
        min_bound = bbox.get_center() - bbox.get_half_extent()
        max_bound = bbox.get_center() + bbox.get_half_extent()

        min_bound[2] = min_bound[2] + self._breast_height - self._breast_height_range
        max_bound[2] = min_bound[2] + 2 * self._breast_height_range
        bbox_filtered = o3d.t.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

        cloud_filtered = cloud.crop(bbox_filtered)
        n_points = len(cloud_filtered.point.positions)

        # RANSAC
        if n_points < self._min_inliers:
            if self._debug_level > 0:
                print(
                    f"Tree {info['id']} discarded because too few points {n_points}/ min {self._min_inliers}"
                )
            return False, {}

        # Fit
        points = cloud_filtered.point.positions.numpy()
        normals = cloud_filtered.point.normals.numpy()
        model = cylinder.fit(
            points,
            method=self._fitting_method,
            N=normals,
            outlier_thr=0.1,
            min_inliers=self._min_inliers,
        )

        # Debug
        if self._debug_level > 2:
            cylinder_mesh = cylinder.to_mesh(model)
            cloud.paint_uniform_color([1.0, 0.0, 0.0])
            cloud_filtered.paint_uniform_color([0.0, 0.0, 1.0])

            o3d.visualization.draw_geometries(
                [cloud.to_legacy(), cloud_filtered.to_legacy(), cylinder_mesh],
                window_name=f"Tree {info['id']}",
            )

        if not model["success"]:
            if self._debug_level > 0:
                print(f"Tree {info['id']} discarded because fitting failed")
            return False, {}

        radius = model["radius"]
        if radius > self._max_valid_radius or radius <= 0.01:
            if self._debug_level > 0:
                print(
                    f"Tree {info['id']} discarded because radius [{radius:.3f}] is invalid"
                )
            return False, {}

        if self._debug_level > 0:
            n_inliers = len(model["inliers"])
            print(
                f"Tree {info['id']} DBH: {n_inliers}/{n_points} points. radius: {radius:.3f}"
            )

        return True, model

    def debug_visualizations(self, trees, filtered_trees):
        import matplotlib.pyplot as plt

        cmap = plt.get_cmap("tab20")

        # Visualize clouds
        viz_clouds = []

        for t in trees:
            tree_cloud = t["cloud"].clone()
            i = t["info"]["id"]
            # tree_cloud.paint_uniform_color(cmap(i % 20)[:3])
            tree_cloud.paint_uniform_color([0.7, 0.7, 0.7])
            viz_clouds.append(tree_cloud.to_legacy())

        for t in filtered_trees:
            tree_cloud = t["cloud"].clone()
            i = t["info"]["id"]
            color = cmap(i % 20)[:3]
            tree_cloud.paint_uniform_color(color)
            viz_clouds.append(tree_cloud.to_legacy())

            model = t["info"]["dbh_model"]
            cylinder_mesh = cylinder.to_mesh(model)
            cylinder_mesh.paint_uniform_color(color)
            viz_clouds.append(cylinder_mesh)

        o3d.visualization.draw_geometries(
            viz_clouds,
            zoom=0.5,
            front=[0.79, 0.02, 0.60],
            lookat=[2.61, 2.04, 1.53],
            up=[-0.60, -0.012, 0.79],
            window_name="tree_analysis",
        )

        from digiforest_analysis.utils import marteloscope

        fig, ax = plt.subplots()
        marteloscope.plot(filtered_trees, ax, cmap="tab20")

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
