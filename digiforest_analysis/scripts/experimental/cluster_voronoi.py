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

    ground_seg = gs.GroundSegmentation(debug_level=0, method="csf", cell_size=2)
    # _, forest_cloud, cloth = ground_seg.process(cloud=cloud, export_cloth=True)
    tree_seg = ts.TreeSegmentation(debug_level=0, clustering_method="voronoi")
    tree_seg.process(cloud=cloud, recluster_flag=False, cluster_dist=2)
