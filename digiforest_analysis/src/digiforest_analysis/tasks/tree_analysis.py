from digiforest_analysis.tasks import BaseTask

import open3d as o3d


class TreeAnalysis(BaseTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._breast_height = kwargs.get("breast_height", 0.5)
        self._breast_height_range = kwargs.get("breast_height_range", 0.2)
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

    def cylinder_fit(self, points, xm=0, ym=0, xr=0, yr=0, r=1):
        from scipy.optimize import least_squares
        import numpy as np

        """
        # Source: https://github.com/philwilkes/TLS2trees/blob/main/tls2trees/fit_cylinders.py
        https://stackoverflow.com/a/44164662/1414831

        This is a fitting for a vertical cylinder fitting
        Reference:
        http://www.int-arch-photogramm-remote-sens-spatial-inf-sci.net/XXXIX-B5/169/2012/isprsarchives-XXXIX-B5-169-2012.pdf
        xyz is a matrix contain at least 5 rows, and each row stores x y z of a cylindrical surface
        p is initial values of the parameter;
        p[0] = Xc, x coordinate of the cylinder centre
        P[1] = Yc, y coordinate of the cylinder centre
        P[2] = alpha, rotation angle (radian) about the x-axis
        P[3] = beta, rotation angle (radian) about the y-axis
        P[4] = r, radius of the cylinder
        th, threshold for the convergence of the least squares
        """
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        p = np.array([xm, ym, xr, yr, r])

        def fitfunc(p, x, y, z):
            return (
                -np.cos(p[3]) * (p[0] - x)
                - z * np.cos(p[2]) * np.sin(p[3])
                - np.sin(p[2]) * np.sin(p[3]) * (p[1] - y)
            ) ** 2 + (
                z * np.sin(p[2]) - np.cos(p[2]) * (p[1] - y)
            ) ** 2  # fit function

        def errfunc(p, x, y, z):
            return fitfunc(p, x, y, z) - p[4] ** 2  # error function

        result = least_squares(errfunc, p, args=(x, y, z), loss="soft_l1")

        # Parameters
        params = result["x"]
        return {
            "x": params[0],
            "y": params[1],
            "rot_x": params[2],
            "rot_y": params[3],
            "radius": params[4],
        }

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
        if n_points < 5:
            print(f"Tree {info['id']} discarded because too few points {n_points}")
            return -1, False

        points = cloud_filtered.point.positions.numpy()

        # import pyransac3d as pyrsc
        # circle_model = pyrsc.Circle()
        # points[:, 2] = 0.0
        # center, axis, radius, inliers = circle_model.fit(points, 0.01)
        # n_inliers = len(inliers)
        params = self.cylinder_fit(points)
        radius = params["radius"]
        n_inliers = 4

        if radius > self._max_valid_radius or radius <= 0 or n_inliers < 3:
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

    # # Debug tree heights
    # n_trees = len(trees)
    # cmap = plt.get_cmap("prism")
    # for i, t in enumerate(trees):
    #     tree_cloud = t["cloud"]
    #     tree_cloud.paint_uniform_color(cmap(i)[:3])
    #     viz_clouds.append(tree_cloud.to_legacy())

    # o3d.visualization.draw_geometries(
    #     viz_clouds,
    #     zoom=0.5,
    #     front=[0.79, 0.02, 0.60],
    #     lookat=[2.61, 2.04, 1.53],
    #     up=[-0.60, -0.012, 0.79],
    # )

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

    fig, ax = plt.subplots()
    for tree in trees:
        x = tree["info"]["mean"][0]
        y = tree["info"]["mean"][1]
        r = tree["info"]["dbh"] / 2
        circle = plt.Circle((x, y), r, color="b")
        ax.add_artist(
            circle,
        )

        # ax.text(
        #     x + 0.1,
        #     y + 0.1,
        #     tree["info"]["id"],
        #     fontsize=10,
        #     ha="center",
        #     va="bottom",
        # )

    #     trees_loc_x.append(x)
    #     trees_loc_y.append(y)
    #     trees_dbh.append(r)
    # trees_area = [10**3 * 3.14 * (dbh / 2) ** 2 for dbh in trees_dbh]

    # plt.scatter(
    #     trees_loc_x, trees_loc_y, s=trees_area, color="blue", marker="o", alpha=0.5
    # )
    # Add index labels to each point
    # for i, (xi, yi) in enumerate(zip(trees_loc_x, trees_loc_y)):
    #     plt.text(
    #         xi + 0.1,
    #         yi + 0.1,
    #         tree["info"]["id"],
    #         fontsize=10,
    #         ha="center",
    #         va="bottom",
    #     )

    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_title("Tree Locations")
    # ax.legend()
    ax.grid(True)
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
