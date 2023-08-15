from digiforest_analysis.tasks import BaseTask

import pcl
import numpy as np
import open3d as o3d
from numpy.typing import NDArray
from numpy import float64


class GroundSegmentation(BaseTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._max_distance_to_plane = kwargs.get("max_distance_to_plane", 0.5)
        self._cell_size = kwargs.get("cell_size", 4.0)
        self._normal_thr = kwargs.get("normal_thr", 0.95)
        self._cloud_boxsize = kwargs.get("box_size", 80)
        self._debug = kwargs.get("debug", False)

    def _process(self, **kwargs):
        """ "
        Processes the cloud and outputs the ground and forest clouds

        Returns:
            _type_: _description_
        """
        cloud = kwargs.get("cloud")

        # Filter by normal
        mask = cloud.point.normals[:, 2] > self._normal_thr
        ground_cloud = cloud.select_by_mask(mask)
        forest_cloud = cloud.select_by_mask(mask, invert=True)

        # Get the terain by plane fitting
        ground_cloud = self.compute_ground_cloud(ground_cloud)

        return ground_cloud, forest_cloud

    def compute_ground_cloud(self, cloud):
        """
        filter the points of the input cloud by fitting planes on each cell of a fixed-size grid
        """

        # Create KD-tree
        cloud_copy = o3d.geometry.PointCloud(cloud.to_legacy())
        kd_tree = o3d.geometry.KDTreeFlann(cloud_copy)

        # number of cells is (self._cloud_boxsize/cell_size) squared
        mid_point = cloud.point.positions.mean(dim=0)

        d_x = np.arange(
            mid_point[0].item() - self._cloud_boxsize / 2,
            mid_point[0].item() + self._cloud_boxsize / 2,
            self._cell_size,
        )
        d_y = np.arange(
            mid_point[1].item() - self._cloud_boxsize / 2,
            mid_point[1].item() + self._cloud_boxsize / 2,
            self._cell_size,
        )

        cloud_inliers = []
        for xx in d_x:
            for yy in d_y:
                cell_midpoint = np.array([xx, yy, 0]).reshape((3, 1))
                sub_cloud_indices = kd_tree.search_radius_vector_3d(
                    query=cell_midpoint, radius=self._cell_size
                )

                # Extract indices from kd-tree output
                indices = np.asarray(sub_cloud_indices[1], dtype=np.int64)

                # Check if there are enough points
                if sub_cloud_indices[0] > 100:
                    # Mask the cloud
                    sub_cloud = cloud.select_by_index(indices=indices)

                    # Run plane fitting
                    plane_model, inliers = sub_cloud.to_legacy().segment_plane(
                        distance_threshold=self._max_distance_to_plane,
                        ransac_n=3,
                        num_iterations=1000,
                    )
                    [a, b, c, d] = plane_model

                    if len(inliers) > 20:
                        # This is ugly and requires to flatten the vector afterwards
                        cloud_inliers.extend(indices[inliers])

        cloud_inliers = np.asarray(cloud_inliers, dtype=np.int64).flatten()
        return cloud.select_by_index(cloud_inliers)


class GroundSegmentationPcl(BaseTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._max_distance_to_plane = kwargs.get("max_distance_to_plane", 0.5)
        self._cell_size = kwargs.get("cell_size", 4.0)
        self._normal_thr = kwargs.get("normal_thr", 0.95)
        self._cloud_boxsize = kwargs.get("box_size", 80)
        self._debug = kwargs.get("debug", False)

    def _process(self, **kwargs):
        """
        Processes the cloud and outputs the ground and forest clouds

        Returns:
            _type_: _description_
        """
        cloud = kwargs.get("cloud")

        # remove non-up points
        ground_cloud = self.filter_up_normal(cloud, self._normal_thr)

        # # drop from xyznormal to xyz
        ground_cloud = self.remove_normals(ground_cloud)

        # get the terrain height
        ground_array = self.compute_ground_cloud(ground_cloud)
        ground_cloud = pcl.PointCloud()
        ground_cloud.from_list(ground_array)

        # get forest cloud
        # remove points of cloud that are in ground_cloud
        cloud = self.remove_normals(cloud)
        cloud_array = cloud.to_array()
        cloud_pts = cloud_array.view([("", cloud_array.dtype)] * cloud_array.shape[1])
        ground_pts = ground_array.view(
            [("", ground_array.dtype)] * ground_array.shape[1]
        )
        forest_array = (
            np.setdiff1d(cloud_pts, ground_pts)
            .view(cloud_array.dtype)
            .reshape(-1, cloud_array.shape[1])
        )

        forest_cloud = pcl.PointCloud()
        forest_cloud.from_array(forest_array)

        return ground_cloud, forest_cloud

    def crop_box(self, cloud: pcl.PointCloud, midpoint, boxsize) -> pcl.PointCloud:
        clipper = pcl.PointCloud(cloud.to_array()[:, :3]).make_cropbox()
        outcloud = pcl.PointCloud()
        tx = 0
        ty = 0
        tz = 0
        clipper.set_Translation(tx, ty, tz)
        rx = 0
        ry = 0
        rz = 0
        clipper.set_Rotation(rx, ry, rz)
        minx = midpoint[0] - boxsize / 2
        miny = midpoint[1] - boxsize / 2
        minz = -1000000000
        mins = 0
        maxx = midpoint[0] + boxsize / 2
        maxy = midpoint[1] + boxsize / 2
        maxz = 10000000000
        maxs = 0
        clipper.set_MinMax(minx, miny, minz, mins, maxx, maxy, maxz, maxs)
        outcloud = clipper.filter()
        return outcloud

    def filter_up_normal(
        self, x: pcl.PointCloud_PointNormal, upthreshold: float, keep_up=True
    ) -> pcl.PointCloud_PointNormal:
        # filter out not-up points from PCLXYZNormal
        cloud_filtered = pcl.PointCloud_PointNormal()
        xy_dat = x.to_array()
        if keep_up:
            x_displayed = xy_dat[(xy_dat[:, 5] > upthreshold)]
        else:
            x_displayed = xy_dat[(xy_dat[:, 5] <= upthreshold)]
        cloud_filtered.from_array(x_displayed)
        return cloud_filtered

    def remove_normals(self, cloud: pcl.PointCloud_PointNormal) -> pcl.PointCloud:
        """
        Removes the normal fields from a cloud
        """
        array_xyz = cloud.to_array()[:, 0:3]
        cloud = pcl.PointCloud()
        cloud.from_array(array_xyz)
        return cloud

    def point_plane_distance(self, a, b, c, d, array: NDArray[float64]):
        """
        Calculate the distance between a point and a plane
        ax + by + cz + d = 0
        """
        normal = np.tile(np.array([a, b, c]), (array.shape[0], 1))
        d_array = np.tile(d, (array.shape[0], 1))

        # Calculate the distance using the formula
        # numerator = np.abs(np.dot(pt, normal) + d_array)
        dot_product = np.sum(array * normal, axis=1)
        dot_product = dot_product[:, np.newaxis]  # make it a column vector
        numerator = np.abs(dot_product + d_array)
        denominator = np.linalg.norm(np.array([a, b, c]))

        distance = numerator / denominator
        return distance

    def compute_ground_cloud(self, cloud: pcl.PointCloud) -> NDArray[float64]:
        """
        filter the points of the input cloud and return the points that are
        on the ground
        output is an np.array of points [x,y,z]
        """

        np_cloud = cloud.to_array()

        # number of cells is (self._cloud_boxsize/cell_size) squared
        cloud_midpoint_round = (
            np.round(np.mean(np_cloud, 0) / self._cell_size) * self._cell_size
        )

        d_x = np.arange(
            cloud_midpoint_round[0] - self._cloud_boxsize / 2,
            cloud_midpoint_round[0] + self._cloud_boxsize / 2,
            self._cell_size,
        )
        d_y = np.arange(
            cloud_midpoint_round[1] - self._cloud_boxsize / 2,
            cloud_midpoint_round[1] + self._cloud_boxsize / 2,
            self._cell_size,
        )
        X = np.empty(shape=np_cloud.shape, dtype=np.float32)

        for xx in d_x:
            for yy in d_y:
                cell_midpoint = np.array([xx, yy, 0])
                pBox = self.crop_box(cloud, cell_midpoint, self._cell_size)
                if pBox.size > 100:
                    # plane fitting
                    seg = pBox.make_segmenter_normals(ksearch=50)
                    seg.set_optimize_coefficients(True)
                    seg.set_model_type(pcl.SACMODEL_NORMAL_PLANE)
                    seg.set_method_type(pcl.SAC_RANSAC)
                    seg.set_distance_threshold(0.01)
                    seg.set_normal_distance_weight(0.01)
                    seg.set_max_iterations(100)

                    indices, coefficients = seg.segment()
                    if len(coefficients) == 4:
                        # plane fitting worked
                        # keep point that are close to the plane
                        points = pBox.to_array()
                        if self._debug:
                            print("plane found", len(indices))
                        dist = self.point_plane_distance(
                            coefficients[0],
                            coefficients[1],
                            coefficients[2],
                            coefficients[3],
                            points,
                        )
                        idx = dist < self._max_distance_to_plane
                        idx = idx.flatten()  # make it a row vector
                        X = np.append(X, points[idx], axis=0)

        return X


if __name__ == "__main__":
    """Minimal example"""
    import pcl
    import os
    import sys
    from digiforest_analysis.utils import pcd

    print("PCL implementation")
    if len(sys.argv) != 3:
        print("Usage : ./script input_cloud output_folder")
    else:
        filename, extension = os.path.splitext(sys.argv[1])

        if extension != ".pcd":
            sys.exit("Input file must be a pcd file")

        # Read header
        header = pcd.load_header(sys.argv[1], binary=True)

    # PCL implementation
    cloud = pcl.PointCloud_PointNormal()
    cloud._from_pcd_file(sys.argv[1].encode("utf-8"))
    app = GroundSegmentationPcl(
        max_distance_to_plane=0.5,
        cell_size=4.0,
        normal_thr=0.92,
        box_size=80,
    )
    ground_cloud, forest_cloud = app.process(cloud=cloud)
    header_fix = {"VIEWPOINT": header["VIEWPOINT"]}
    pcd.write_pcl(
        ground_cloud, header_fix, os.path.join(sys.argv[2], "pcl_ground_cloud.pcd")
    )
    pcd.write_pcl(
        forest_cloud, header_fix, os.path.join(sys.argv[2], "pcl_forest_cloud.pcd")
    )

    print("Open3D implementation")
    cloud = o3d.t.io.read_point_cloud(sys.argv[1])
    assert len(cloud.point.normals) > 0

    # Open3D implementation
    app = GroundSegmentation(
        max_distance_to_plane=0.5,
        cell_size=4.0,
        normal_thr=0.92,
        box_size=80,
    )
    ground_cloud, forest_cloud = app.process(cloud=cloud)
    header_fix = {"VIEWPOINT": header["VIEWPOINT"]}
    pcd.write_open3d(
        ground_cloud, header_fix, os.path.join(sys.argv[2], "o3d_ground_cloud.pcd")
    )
    pcd.write_open3d(
        forest_cloud, header_fix, os.path.join(sys.argv[2], "o3d_forest_cloud.pcd")
    )
