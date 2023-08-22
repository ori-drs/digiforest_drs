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
