from digiforest_analysis.tasks import BaseTask

import open3d as o3d


class Preprocessing(BaseTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._crop_x = kwargs.get("crop_x", -1)
        self._crop_y = kwargs.get("crop_y", -1)
        self._crop_z = kwargs.get("crop_z", -1)

        self._noise_filter_points = kwargs.get("noise_filter_points", 20)
        self._noise_filter_radius = kwargs.get("noise_filter_radius", 0.2)

        self._intensity_thr = kwargs.get("intensity_thr", 20)

    def _process(self, **kwargs):
        cloud = kwargs.get("cloud").clone()

        # Crop input cloud
        crop_box = self.get_crop_box_from_bottom(cloud)
        processed_cloud = cloud.crop(crop_box)

        # remove outliers
        # self._cropped_cloud, inliers = self._cropped_cloud.remove_statistical_outliers(nb_neighbors=10, std_ratio=0.2)
        processed_cloud, inliers = processed_cloud.remove_radius_outliers(
            nb_points=self._noise_filter_points, search_radius=self._noise_filter_radius
        )

        # Filter by intensity
        if "intensity" in processed_cloud.point:
            processed_cloud = processed_cloud.select_by_mask(
                (processed_cloud.point.intensity.numpy() > self._intensity_thr)[:, 0]
            )

        # Debug visualizations
        if self._debug_level > 1:
            self.debug_visualizations(cloud, processed_cloud)

        return processed_cloud

    def get_crop_box_from_bottom(self, cloud):
        # Get bounding box of original cloud
        bbox = cloud.get_axis_aligned_bounding_box()
        # Get half extend of the bounding box
        bbox_half_extent = bbox.get_half_extent().numpy()

        # Get half-extents for cropping if they are valid
        crop_half_x = self._crop_x / 2 if self._crop_x > 0 else bbox_half_extent[0]
        crop_half_y = self._crop_y / 2 if self._crop_y > 0 else bbox_half_extent[1]
        crop_z = self._crop_z if self._crop_z > 0 else (2 * bbox_half_extent[2])

        # Prepare the cropping bbox
        # We first get a translation along XY
        crop_half_extent = o3d.core.Tensor(
            [crop_half_x, crop_half_y, 0], dtype=o3d.core.Dtype.Float32
        )
        # The min and max bounds are given by shifting the center by the half extent
        min_bound = bbox.get_center() - crop_half_extent
        max_bound = bbox.get_center() + crop_half_extent

        # To crop in z, we want to crop from the bottom, not the center
        # So we first shift the min_bound by half the extent
        min_bound[2] -= bbox_half_extent[2]
        # And finally we add the full extent along z for cropping
        max_bound[2] = min_bound[2] + crop_z

        # We make the bounding box and return
        crop_box = o3d.t.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        return crop_box

    def debug_visualizations(self, cloud, cropped_cloud):
        # Visualize clouds
        original = cloud.clone()
        original.paint_uniform_color([0.0, 0.0, 1.0])
        bbox_original = cloud.get_axis_aligned_bounding_box()
        bbox_original.set_color([0.0, 0.0, 1.0])

        cropped = cropped_cloud.clone()
        cropped.paint_uniform_color([1.0, 0.0, 0.0])
        bbox_cropped = cropped_cloud.get_axis_aligned_bounding_box()
        bbox_cropped.set_color([1.0, 0.0, 0.0])

        # origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries(
            [
                # origin,
                original.to_legacy(),
                bbox_original.to_legacy(),
                cropped.to_legacy(),
                bbox_cropped.to_legacy(),
            ],
            zoom=self.viz_zoom,
            front=[0.79, 0.2, 0.60],
            lookat=bbox_cropped.get_center().numpy(),
            up=[-0.55, -0.15, 0.8],
            window_name="preprocessing",
        )
