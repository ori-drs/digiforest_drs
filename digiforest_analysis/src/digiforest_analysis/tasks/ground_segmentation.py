from digiforest_analysis.tasks import BaseTask

import numpy as np
import open3d as o3d


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

    def crop_box(self, cloud, midpoint, boxsize):
        minx = midpoint[0] - boxsize / 2
        miny = midpoint[1] - boxsize / 2
        minz = -1000000000
        maxx = midpoint[0] + boxsize / 2
        maxy = midpoint[1] + boxsize / 2
        maxz = 10000000000

        cropping_box = o3d.t.geometry.AxisAlignedBoundingBox(
            np.array([minx, miny, minz]),
            np.array([maxx, maxy, maxz]),
        )

        # Crop the point cloud
        cropped_point_cloud = cloud.crop(cropping_box)
        return cropped_point_cloud

    def compute_ground_cloud(self, cloud):
        """
        filter the points of the input cloud by fitting planes on each cell of a fixed-size grid
        """

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

        merged_cloud = o3d.t.geometry.PointCloud()
        for xx in d_x:
            for yy in d_y:
                # Crop the cloud
                cropped_cloud = self.crop_box(cloud, (xx, yy), self._cell_size)

                # Check if there are enough points
                # if sub_cloud_indices[0] > 100:
                if len(cropped_cloud.point.positions) > 100:
                    # Run plane fitting

                    plane_model, inliers = cropped_cloud.to_legacy().segment_plane(
                        distance_threshold=self._max_distance_to_plane,
                        ransac_n=3,
                        num_iterations=1000,
                    )
                    if len(inliers) > 20:
                        if not ("positions" in merged_cloud.point):
                            merged_cloud = cropped_cloud.select_by_index(
                                indices=inliers
                            )
                        else:
                            merged_cloud += cropped_cloud.select_by_index(
                                indices=inliers
                            )

        return merged_cloud


if __name__ == "__main__":
    """Minimal example"""
    import os
    import sys
    from digiforest_analysis.utils import pcd

    if len(sys.argv) != 3:
        print("Usage : ./script input_cloud output_folder")
    else:
        filename, extension = os.path.splitext(sys.argv[1])

        if extension != ".pcd":
            sys.exit("Input file must be a pcd file")

    cloud, header = pcd.load(sys.argv[1], binary=True)
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
    pcd.write(
        ground_cloud, header_fix, os.path.join(sys.argv[2], "o3d_ground_cloud.pcd")
    )
    pcd.write(
        forest_cloud, header_fix, os.path.join(sys.argv[2], "o3d_forest_cloud.pcd")
    )
