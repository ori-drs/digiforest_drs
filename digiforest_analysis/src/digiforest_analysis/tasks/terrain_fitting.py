import numpy as np
import CSF

from digiforest_analysis.tasks import BaseTask


class TerrainFitting(BaseTask):
    def __init__(
        self, sloop_smooth: bool = False, cloth_cell_size: float = 0.1, **kwargs
    ):
        super().__init__(**kwargs)
        self.csf = CSF.CSF()
        self.csf.params.bSloopSmooth = sloop_smooth
        self.csf.params.cloth_resolution = cloth_cell_size

    def _process(self, cloud, **kwargs):
        print(f"Cloud has {len(cloud.point.positions)} points")
        cloud = cloud.voxel_down_sample(voxel_size=self.csf.params.cloth_resolution / 4)
        print(f"Cloud now has {len(cloud.point.positions)} points")

        self.csf.setPointCloud(cloud.point.positions.numpy().tolist())
        csf_mesh = self.csf.do_cloth_export()
        verts = np.array(csf_mesh).reshape((-1, 3))

        # round to mm to make sure there are no duplicates
        verts[:, :2] = verts[:, :2].round(decimals=3)
        x_cos = np.sort(np.unique(verts[:, 0]))
        y_cos = np.sort(np.unique(verts[:, 1]))
        X, Y = np.meshgrid(x_cos, y_cos)
        Z = np.zeros_like(X)
        x_id, y_id = np.meshgrid(np.arange(x_cos.shape[0]), np.arange(y_cos.shape[0]))
        x_id, y_id = x_id.reshape(-1), y_id.reshape(-1)
        Z[y_id, x_id] = verts[:, 2]

        cloth = np.stack((X, Y, Z), axis=-1)

        return cloth

    def meshgrid_to_mesh(self, mgrid: np.ndarray):
        verts = mgrid.reshape(-1, 3)
        M, N, _ = mgrid.shape
        base_tri_1 = np.array([1, 0, N])
        base_tri_2 = np.array([1, N, N + 1])
        # accumulate column tiles
        tris = [(base_tri_1 + i, base_tri_2 + i) for i in range(N - 1)]
        tris = np.array(tris).reshape(-1, 3)
        # accumulate rows
        tris = np.vstack([tris + N * j for j in range(M - 1)])
        return verts, tris
