from digiforest_analysis.tasks import BaseTask

import open3d as o3d
import pyransac3d as pyrsc


class TreeAnalysis(BaseTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._breast_height = kwargs.get("breast_height", 1.3)
        self._breast_height_range = kwargs.get("breast_height_range", 1.0)
        self._max_valid_radius = kwargs.get("max_valid_radius", 1.0)

    def _process(self, **kwargs):
        trees = kwargs.get("trees")

        filtered_trees = []
        for i, tree in enumerate(trees):
            dbh, valid_dbh = self.compute_dbh(tree)

            if valid_dbh:
                tree["info"]["dbh"] = dbh
                filtered_trees.append(tree)

        return filtered_trees

    def compute_dbh(self, tree):
        cloud = tree["cloud"]
        info = tree["info"]

        # Get bounding box and constrain the H size
        bbox = cloud.get_axis_aligned_bounding_box()
        min_bound = bbox.get_center() - bbox.get_half_extent()
        max_bound = bbox.get_center() + bbox.get_half_extent()
        min_bound[2] = info["size_z"] + self._breast_height - self._breast_height_range
        max_bound[2] = info["size_z"] + self._breast_height + self._breast_height_range
        bbox = o3d.t.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

        cloud_filtered = cloud.crop(bbox)
        n_points = len(cloud_filtered.point.positions)

        # RANSAC
        if n_points < 10:
            return -1, False

        circle_model = pyrsc.Circle()
        points = cloud_filtered.point.positions.numpy()
        points[:, 2] = 0.0
        center, axis, radius, inliers = circle_model.fit(points, 0.01)
        n_inliers = len(inliers)

        if radius > self._max_valid_radius or n_inliers < 3:
            return -1, False

        if self._debug:
            print(
                f"Tree {info['id']} DBH: RANSAC let {n_inliers}/{n_points} points. radius: {radius}"
            )
            # o3d.visualization.draw_geometries([cloud_filtered.to_legacy()])

        dbh = 2 * radius
        return dbh, True


if __name__ == "__main__":
    """Minimal example"""
    import sys
    import yaml
    import matplotlib.pyplot as plt
    from digiforest_analysis.utils import pcd
    from pathlib import Path

    print("Tree segmentation")
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
        cloud, header = pcd.load(str(c), binary=True)
        info = yaml.safe_load(i.read_text())
        trees.append({"cloud": cloud, "info": info})

    app = TreeAnalysis(
        debug=True,
    )
    trees = app.process(trees=trees)

    # Visualize clouds
    n_points = len(cloud.point.positions)
    cloud.paint_uniform_color([0.9, 0.9, 0.9])

    viz_clouds = []
    # viz_clouds.append(cloud.to_legacy())

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
        plt.text(
            xi + 0.1,
            yi + 0.1,
            tree["info"]["id"],
            fontsize=10,
            ha="center",
            va="bottom",
        )
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Tree Locations")
    plt.legend()
    plt.grid(True)
    plt.show()

    # # Write output
    # for i, tree in enumerate(trees):
    #     # Write clouds
    #     tree_name = f"tree_cloud_{i}.pcd"
    #     tree_cloud_filename = os.path.join(sys.argv[2], tree_name)
    #     header_fix = {"VIEWPOINT": header["VIEWPOINT"]}
    #     pcd.write(tree["cloud"], header_fix, tree_cloud_filename)

    #     # write tree info
    #     tree_name = f"tree_info_{i}.json"
    #     tree_info_filename = os.path.join(sys.argv[2], tree_name)
    #     with open(tree_info_filename, "w") as json_file:
    #         json.dump(tree["info"], json_file, indent=4)
