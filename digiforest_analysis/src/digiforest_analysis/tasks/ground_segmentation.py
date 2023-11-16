from digiforest_analysis.tasks import BaseTask
from digiforest_analysis.utils.timing import Timer

import numpy as np
import open3d as o3d

timer = Timer()


class GroundSegmentation(BaseTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._method = kwargs.get("method", "default")
        self._voxel_filter_size = kwargs.get("voxel_filter_size", 0.05)
        self._max_distance_to_plane = kwargs.get("max_distance_to_plane", 0.5)
        self._cell_size = kwargs.get("cell_size", 4.0)
        self._normal_thr = kwargs.get("normal_thr", 0.9)
        self._cloud_boxsize = kwargs.get("box_size", 80)

    def _process(self, **kwargs):
        """ "
        Processes the cloud and outputs the ground and forest clouds

        Returns:
            _type_: _description_
        """
        cloud = kwargs.get("cloud")

        if self._method == "default":
            ground_cloud, forest_cloud = self.run_default(cloud)
        elif self._method == "indexing":
            ground_cloud, forest_cloud = self.run_indexing(cloud)
        elif self._method == "csf":
            export_cloth = kwargs.get("export_cloth", False)
            if export_cloth:
                ground_cloud, forest_cloud, cloth = self.run_csf(cloud, export_cloth)
            else:
                ground_cloud, forest_cloud = self.run_csf(cloud, export_cloth)

        # Debug visualizations
        if self._debug_level > 1:
            self.debug_visualizations(ground_cloud, forest_cloud)

        if self._method == "csf" and export_cloth:
            return ground_cloud, forest_cloud, cloth
        else:
            return ground_cloud, forest_cloud

    def run_default(self, cloud):
        # Filter by normal
        mask = cloud.point.normals[:, 2] > self._normal_thr
        coarse_ground_cloud = cloud.select_by_mask(mask)
        forest_cloud = cloud.select_by_mask(mask, invert=True)

        if self._voxel_filter_size > 0.0:
            coarse_ground_cloud = coarse_ground_cloud.voxel_down_sample(
                voxel_size=self._voxel_filter_size
            )

        # Fit plane per cell
        mid_point = coarse_ground_cloud.point.positions.mean(dim=0)

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

        def crop_box(cloud, midpoint, boxsize):
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

        refined_ground_cloud = o3d.t.geometry.PointCloud()
        for xx in d_x:
            for yy in d_y:
                # Crop the cloud
                cropped_cloud = crop_box(coarse_ground_cloud, (xx, yy), self._cell_size)

                # Check if there are enough points
                if len(cropped_cloud.point.positions) > 100:
                    # Run plane fitting
                    plane_model, inliers = cropped_cloud.to_legacy().segment_plane(
                        distance_threshold=self._max_distance_to_plane,
                        ransac_n=3,
                        num_iterations=1000,
                    )
                    if len(inliers) > 20:
                        if not ("positions" in refined_ground_cloud.point):
                            refined_ground_cloud = cropped_cloud.select_by_index(
                                indices=inliers
                            )
                        else:
                            refined_ground_cloud += cropped_cloud.select_by_index(
                                indices=inliers
                            )

        return refined_ground_cloud, forest_cloud

    def run_indexing(self, cloud):
        # Filter by normal
        mask = cloud.point.normals[:, 2] > self._normal_thr
        coarse_ground_cloud = cloud.select_by_mask(mask)
        forest_cloud = cloud.select_by_mask(mask, invert=True)

        if self._voxel_filter_size > 0.0:
            coarse_ground_cloud = coarse_ground_cloud.voxel_down_sample(
                voxel_size=self._voxel_filter_size
            )

        # Fit plane per cell
        # number of cells is (self._cloud_boxsize/cell_size) squared
        mid_point = coarse_ground_cloud.point.positions.mean(dim=0)

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

        def get_cell_mask(cloud, midpoint, boxsize):
            minx = midpoint[0] - boxsize / 2
            miny = midpoint[1] - boxsize / 2
            maxx = midpoint[0] + boxsize / 2
            maxy = midpoint[1] + boxsize / 2
            mask = (
                (cloud.point.positions[:, 0] >= minx)
                & (cloud.point.positions[:, 0] < maxx)
                & (cloud.point.positions[:, 1] >= miny)
                & (cloud.point.positions[:, 0] < maxy)
            )
            return mask

        cloud_inliers = []
        for xx in d_x:
            for yy in d_y:
                cell_mask = get_cell_mask(
                    coarse_ground_cloud, (xx, yy), self._cell_size
                )

                # Check if there are enough points
                if cell_mask.numpy().sum() > 100:
                    # Mask the cloud
                    sub_cloud = coarse_ground_cloud.select_by_mask(cell_mask)

                    # Run plane fitting
                    plane_model, inliers = sub_cloud.to_legacy().segment_plane(
                        distance_threshold=self._max_distance_to_plane,
                        ransac_n=3,
                        num_iterations=1000,
                    )

                    if len(inliers) > 20:
                        # This is ugly and requires to flatten the vector afterwards
                        cloud_inliers.extend(inliers)

        cloud_inliers = np.asarray(cloud_inliers, dtype=np.int64).flatten()
        refined_ground_cloud = coarse_ground_cloud.select_by_index(cloud_inliers)

        return refined_ground_cloud, forest_cloud

    def run_csf(self, cloud, export_cloth=False):
        # This requires to "pip install cloth-simulation-filter"
        import CSF

        csf = CSF.CSF()
        csf.params.bSloopSmooth = False
        csf.params.cloth_resolution = self._cell_size
        points = cloud.point.positions.numpy().tolist()

        csf.setPointCloud(points)
        ground_indices = CSF.VecInt()
        forest_indices = CSF.VecInt()
        csf.do_filtering(ground_indices, forest_indices)
        ground_indices = np.array(ground_indices, dtype=int)
        forest_indices = np.array(forest_indices, dtype=int)

        ground_cloud = cloud.select_by_index(ground_indices)
        forest_cloud = cloud.select_by_index(forest_indices)

        if export_cloth:
            verts = np.array(csf.do_cloth_export()).reshape((-1, 3))
            # transform this height representation into a MxNx3 tensor with
            # X, Y, Z coordinates.
            # Round X, Y to 10th of a milimeter to use it as a unique key
            with timer("creating cloth"):
                verts[:, :2] = verts[:, :2].round(decimals=4)
                x_cos = np.sort(np.unique(verts[:, 0]))
                y_cos = np.sort(np.unique(verts[:, 1]))
                X, Y = np.meshgrid(x_cos, y_cos)
                Z = np.zeros_like(X)

                # when developing, this ordering was the one used in CSF
                x_id, y_id = np.meshgrid(
                    np.arange(x_cos.shape[0]), np.arange(y_cos.shape[0])
                )
                x_id, y_id = x_id.reshape(-1), y_id.reshape(-1)

                # # If for whatever reason the sorting of the cloth elevations is
                # # different, this is a way to get the correct indices (~4x slower)
                # x_cos_dict = {x: i for i, x in enumerate(x_cos)}
                # y_cos_dict = {y: i for i, y in enumerate(y_cos)}
                # x_id = np.vectorize(x_cos_dict.get)(verts[:, 0])
                # y_id = np.vectorize(y_cos_dict.get)(verts[:, 1])

                Z[y_id, x_id] = verts[:, 2]
                cloth = np.stack((X, Y, Z), axis=-1)
            print(timer)
            return ground_cloud, forest_cloud, cloth
        else:
            return ground_cloud, forest_cloud

    def debug_visualizations(self, ground_cloud, forest_cloud):
        # Visualize clouds
        ground = ground_cloud.clone()
        ground.paint_uniform_color([0.0, 0.0, 1.0])

        forest = forest_cloud.clone()
        forest.paint_uniform_color([1.0, 0.0, 0.0])

        # origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries(
            [
                # origin,
                ground.to_legacy(),
                forest.to_legacy(),
            ],
            zoom=self.viz_zoom,
            front=[0.79, 0.2, 0.60],
            lookat=self.viz_center,
            up=[-0.55, -0.15, 0.8],
            window_name="ground_segmentation",
        )


if __name__ == "__main__":
    """Minimal example"""
    import os
    import sys
    from digiforest_analysis.utils import io

    if len(sys.argv) != 3:
        print("Usage : ./script input_cloud output_folder")
    else:
        filename, extension = os.path.splitext(sys.argv[1])

        if extension != ".pcd":
            sys.exit("Input file must be a pcd file")

    cloud, header = io.load(sys.argv[1], binary=True)
    assert len(cloud.point.normals) > 0

    app = GroundSegmentation(
        debug_level=0,
    )

    ground_cloud, forest_cloud = app.process(cloud=cloud)

    # Write clouds
    header_fix = {"VIEWPOINT": header["VIEWPOINT"]}
    io.write(ground_cloud, header_fix, os.path.join(sys.argv[2], "ground_cloud.pcd"))
    io.write(forest_cloud, header_fix, os.path.join(sys.argv[2], "forest_cloud.pcd"))
