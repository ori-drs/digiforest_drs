import digiforest_analysis.tasks.tree_segmentation as ts
import digiforest_analysis.tasks.ground_segmentation as gs
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

    ground_seg = gs.GroundSegmentation(debug_level=2, method="default")
    ground_cloud, forest_cloud = ground_seg.process(cloud=cloud)
    tree_seg = ts.TreeSegmentation(
        debug_level=2,
        clustering_method="voronoi",
        normal_thr=0.5,
        voxel_size=0.05,
        cluster_2d=False,
        min_tree_height=1.5,
        max_tree_diameter=10.0,
        min_tree_diameter=0.1,
        min_gravity_alignment_score=0.1,
    )
    tree_seg.process(cloud=cloud, ground_cloud=ground_cloud)
