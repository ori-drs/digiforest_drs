import numpy as np

from digiforest_analysis.tasks import BaseTask


class TerrainFitting(BaseTask):
    def __init__(
        self, sloop_smooth: bool = False, cloth_cell_size: float = 0.1, **kwargs
    ):
        import CSF

        super().__init__(**kwargs)
        self.csf = CSF.CSF()
        self.csf.params.bSloopSmooth = sloop_smooth
        self.csf.params.cloth_resolution = cloth_cell_size

    def _process(self, cloud, **kwargs):
        cloud = cloud.voxel_down_sample(voxel_size=self.csf.params.cloth_resolution / 4)

        self.csf.setPointCloud(cloud.point.positions.numpy().tolist())
        csf_mesh = self.csf.do_cloth_export()
        verts = np.array(csf_mesh).reshape((-1, 3))

        verts[:, :2] = verts[:, :2].round(decimals=3)
        num_x_cos = np.unique(verts[:, 0]).shape[0]
        num_y_cos = np.unique(verts[:, 1]).shape[0]
        cloth = verts.reshape((num_y_cos, num_x_cos, 3)).transpose((1, 0, 2))

        return cloth
