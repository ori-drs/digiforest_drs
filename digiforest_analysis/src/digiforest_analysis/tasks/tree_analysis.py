from digiforest_analysis.tasks import BaseTask

import open3d as o3d


class TreeAnalysis(BaseTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._max_trunk_height = kwargs.get("max_trunk_height", 5.0)

    def _process(self, **kwargs):
        trees = kwargs.get("trees")

        # Implement your code here
        #

        return trees

    def compute_dbh(self, cloud):
        # passthrough filter to get the trunk points
        cluster_dim = self.get_cluster_dimensions(cloud)

        # Get bounding box and constrain the H size
        bbox = cloud.get_axis_aligned_bounding_box()
        min_bound = bbox.get_center() - bbox.get_half_extent()
        max_bound = bbox.get_center() + bbox.get_half_extent()
        max_bound[2] = cluster_dim["min_z"] + self._max_trunk_height
        bbox = o3d.t.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

        cloud_filtered = cloud.crop(bbox)

        if self._debug:
            num_filtered = len(cloud_filtered.point.positions)
            num_points = len(cloud.point.positions)
            print(f"DBH: Passthrough filter let {num_filtered} of {num_points} points.")
        # cloud_filtered = cloud
        if len(cloud_filtered.point.positions) < 10:
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
        # seg.set_model_type(pcl.SACMODEL_CYLINDER)
        seg.set_normal_distance_weight(0.1)
        # seg.set_method_type(pcl.SAC_RANSAC)
        seg.set_max_iterations(1000)
        seg.set_distance_threshold(0.10)
        seg.set_radius_limits(0, 0.5 * self._max_tree_diameter)
        [inliers_cylinder, coefficients_cylinder] = seg.segment()
        radius_cylinder = coefficients_cylinder[6]
        dbh = 2 * radius_cylinder
        return dbh


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

    app = TreeAnalysis(
        debug=True,
    )
    trees = app.process(cloud=cloud)
    print("Found " + str(len(trees)) + " trees")

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
