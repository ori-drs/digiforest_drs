import digiforest_analysis.tasks.tree_segmentation as ts
from digiforest_analysis.utils.io import load

import numpy as np
import open3d as o3d

if __name__ == "__main__":
    pcd_file = "/home/ori/logs/logs_evo_finland/exp01/2023-05-01-14-01-05-exp01/payload_clouds/cloud_1682946124_761436000.pcd"
    cloud, header = load(pcd_file, binary=True)
    if "VIEWPOINT" in header:
        header_data = [float(x) for x in header["VIEWPOINT"]]
        location = np.array(header_data[:3])
        rotation = np.array(header_data[3:])
        # apply transformation to point cloud
        R = o3d.geometry.TriangleMesh.get_rotation_matrix_from_quaternion(rotation)
        cloud.rotate(R, center=location)
    segmenter = ts.TreeSegmentation(
        cloud=cloud,
        debug_level=2,
        normal_thr=0.5,
        voxel_size=0.05,
        cluster_2d=False,
        clustering_method="voronoi",
        min_tree_height=1.5,
        max_tree_diameter=10.0,
        min_tree_diameter=0.1,
        min_gravity_alignment_score=0.1,
    )
    segmenter.process(cloud=cloud)
