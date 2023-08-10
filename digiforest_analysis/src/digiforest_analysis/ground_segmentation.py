import digiforest_analysis.terrain_mapping as df
import pcl
import numpy as np
from typing import Tuple
from numpy.typing import NDArray
from numpy import float64


class GroundSegmentation:
    def __init__(self, cloud_filename: str):
        self.cloud_filename = cloud_filename

    def remove_normals(self, cloud: pcl.PointCloud_PointNormal) -> pcl.PointCloud:
        array_xyz = cloud.to_array()[:, 0:3]
        cloud = pcl.PointCloud()
        cloud.from_array(array_xyz)
        return cloud

    def point_plane_distance(self, a, b, c, d, point) -> float:
        """
        Calculate the distance between a point and a plane
        ax + by + cz + d = 0
        """
        normal = np.array([a, b, c])

        # Calculate the distance using the formula
        numerator = abs(np.dot(point, normal) + d)
        denominator = np.linalg.norm(normal)

        distance = numerator / denominator
        return distance

    def getTerrainCloud(self, p: pcl.PointCloud) -> NDArray[float64]:
        """
        filter the points of the input cloud and return the points that are
        on the ground
        output is an np.array of points [x,y,z]
        """

        # number of cells is (cloud_boxsize/cell_size) squared
        cloud_boxsize = 80
        cell_size = 4.0
        cloud_midpoint_round = np.round(np.mean(p, 0) / cell_size) * cell_size

        d_x = np.arange(
            cloud_midpoint_round[0] - cloud_boxsize / 2,
            cloud_midpoint_round[0] + cloud_boxsize / 2,
            cell_size,
        )
        d_y = np.arange(
            cloud_midpoint_round[1] - cloud_boxsize / 2,
            cloud_midpoint_round[1] + cloud_boxsize / 2,
            cell_size,
        )
        X = np.empty(shape=[0, 3], dtype=np.float32)

        for xx in d_x:
            for yy in d_y:
                cell_midpoint = np.array([xx, yy, 0])
                pBox = df.cropBox(p, cell_midpoint, cell_size)
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
                        print("plane found", len(indices))
                        for point in points:
                            dist = self.point_plane_distance(
                                coefficients[0],
                                coefficients[1],
                                coefficients[2],
                                coefficients[3],
                                point,
                            )
                            if dist < 0.5:
                                X = np.append(X, [point], axis=0)

        return X

    def generate_height_map(self) -> Tuple[pcl.PointCloud, pcl.PointCloud]:
        cloud = pcl.PointCloud_PointNormal()
        cloud._from_pcd_file(self.cloud_filename.encode("utf-8"))

        # remove non-up points
        ground_cloud = df.filterUpNormal(cloud, 0.95)

        # drop from xyznormal to xyz
        ground_cloud = self.remove_normals(ground_cloud)

        # get the terrain height
        ground_array = self.getTerrainCloud(ground_cloud)
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
