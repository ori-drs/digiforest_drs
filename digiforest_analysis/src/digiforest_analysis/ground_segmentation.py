import pcl
import numpy as np
from numpy.typing import NDArray
from numpy import float64
import time


class GroundSegmentation:
    def __init__(
        self,
        max_distance_to_plane: float = 0.5,
        cell_size: float = 4.0,
        normal_thr: float = 0.95,
        box_size: float = 80,
        *args,
        **kwargs,
    ):

        self._max_distance_to_plane = max_distance_to_plane
        self._cell_size = cell_size
        self._normal_thr = normal_thr
        self._cloud_boxsize = box_size

        self._debug = kwargs.get("debug", False)

    def process(self, cloud: pcl.PointCloud_PointNormal):
        """
        Processes the cloud and outputs the ground and forest clouds
        """

        # remove non-up points
        ground_cloud = self.filter_up_normal(cloud, self._normal_thr)

        # drop from xyznormal to xyz
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
        clipper = cloud.make_cropbox()
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

    def compute_ground_cloud(self, p: pcl.PointCloud) -> NDArray[float64]:
        """
        filter the points of the input cloud and return the points that are
        on the ground
        output is an np.array of points [x,y,z]
        """

        # number of cells is (self._cloud_boxsize/cell_size) squared
        cloud_midpoint_round = (
            np.round(np.mean(p, 0) / self._cell_size) * self._cell_size
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
        X = np.empty(shape=[0, 3], dtype=np.float32)

        for xx in d_x:
            for yy in d_y:
                cell_midpoint = np.array([xx, yy, 0])
                pBox = self.crop_box(p, cell_midpoint, self._cell_size)
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
                        idx = dist < self.max_distance_to_plane
                        idx = idx.flatten()  # make it a row vector
                        X = np.append(X, points[idx], axis=0)

        return X
