from digiforest_analysis.utils import plotting
from digiforest_analysis.utils.clustering import voronoi

import numpy as np
import open3d as o3d

from digiforest_analysis.tasks import BaseTask


class TreeSegmentationVoronoi(BaseTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._rejected_clusters = []

        self._normal_thr = kwargs.get("normal_thr", 0.5)
        self._voxel_size = kwargs.get("voxel_size", 0.05)

        # Colormap parameters
        self._cmap = plotting.color_palette
        self._ncolors = plotting.n_colors

        self._debug_level = kwargs.get("debug_level", 0)

    def _process(self, **kwargs):
        """ "
        Processes the forest cloud with ground removed and outputs a list of tree clouds

        Returns:
            dict: cluster dicts
        """
        cloud = kwargs.get("cloud")
        assert len(cloud.point.normals) > 0

        # Prefiltering
        cloud = self.prefiltering(cloud)

        # Extract clusters
        labels, axes = voronoi(
            cloud,
            cloth=kwargs.get("cloth", None),
            debug_level=self._debug_level,
        )

        clusters = []
        num_labels = labels.max() + 1
        for i in range(num_labels):
            mask = labels == i
            seg_cloud = cloud.select_by_mask(mask)
            color = np.array(self._cmap[i % self._ncolors]).astype(np.float32)
            cluster = {
                "cloud": seg_cloud,
                "info": {"id": i, "color": color, "axis": axes[i]},
            }
            clusters.append(cluster)

        if self._debug_level > 0:
            print("Extracted " + str(len(clusters)) + " initial clusters.")

        return clusters

    def prefiltering(self, cloud, **kwargs):
        # Filter by Z-normals
        mask = (cloud.point.normals[:, 2] >= -self._normal_thr) & (
            cloud.point.normals[:, 2] <= self._normal_thr
        )
        new_cloud = cloud.select_by_mask(mask)

        # Downsample
        # new_cloud = new_cloud.voxel_down_sample(voxel_size=self._voxel_size) # faster way
        new_cloud, _, _ = new_cloud.to_legacy().voxel_down_sample_and_trace(
            self._voxel_size,
            new_cloud.get_min_bound().numpy().astype(np.float64),
            new_cloud.get_max_bound().numpy().astype(np.float64),
        )
        new_cloud = o3d.t.geometry.PointCloud.from_legacy(new_cloud)

        return new_cloud
