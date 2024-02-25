from typing import List
from digiforest_analysis.tasks.terrain_fitting import TerrainFitting
from digiforest_analysis.tasks.tree_reconstruction import Tree

from digiforest_analysis.tasks.tree_segmentation_voronoi import TreeSegmentationVoronoi
from digiforest_analysis.utils.io import load
from digiforest_analysis.utils.timing import Timer
from digiforest_analysis.utils.meshing import meshgrid_to_mesh

import numpy as np
import open3d as o3d
import trimesh


timer = Timer()

if __name__ == "__main__":
    # pcd_file = "/home/ori/logs/logs_evo_finland/exp16/combined_cloud.pcd"
    # pcd_file = "/home/ori/git/digiforest_drs/payload_clouds/pc021.pcd"
    # pcd_file = "/home/ori/logs/logs_evo_finland/exp01/2023-05-01-14-01-05-exp01/payload_clouds/cloud_1682946124_761436000.pcd"
    pcd_file = "/home/ori/logs/2024-02-19-anymal-dolly-forest-of-dean/slam_outputs/2024-02-19-14-41-35-prior-map/combined_cloud.pcd"
    debug_level = 0
    cloud, header = load(pcd_file, binary=True)
    if "VIEWPOINT" in header:
        header_data = [float(x) for x in header["VIEWPOINT"]]
        location = np.array(header_data[:3])
        rotation = np.array(header_data[3:])
        # apply transformation to point cloud
        R = o3d.geometry.TriangleMesh.get_rotation_matrix_from_quaternion(rotation)
        cloud.rotate(R, center=location)

    # FIT TERRAIN
    terrain_fitter = TerrainFitting(sloop_smooth=True, cloth_cell_size=1.0)
    terrain = terrain_fitter.process(cloud=cloud)
    verts, tris = meshgrid_to_mesh(terrain)
    tm = trimesh.Trimesh(vertices=verts, faces=tris)

    import trimesh

    terrain_mesh = trimesh.Trimesh(vertices=verts, faces=tris)
    terrain_mesh.fix_normals()
    terrain_mesh.export(
        "/home/ori/git/digiforest_drs/digiforest_analysis_ros/output/trees/logs/raw/trees/terrain_mesh.obj"
    )

    # SEGMENT TREES
    tree_seg = TreeSegmentationVoronoi(
        debug_level=debug_level, clustering_method="voronoi"
    )
    clusters = tree_seg.process(
        cloud=cloud,
        cloth=terrain,
        max_cluster_radius=2,
        n_threads=1,
        point_fraction=0.1,
    )

    # FIT TREES
    trees: List[Tree] = []
    for cluster in clusters:
        tree = Tree(id=cluster["info"]["id"])
        cluster["info"]["T_sensor2map"] = np.eye(4)
        tree.add_cluster(cluster)
        tree.reconstruct()
        trees.append(tree)

    # # SAVE CLUSTER TO DISK
    # dir = "/home/ori/git/realtime-trees/single_trees/clustering_map"
    # for file in os.listdir(dir):
    #     if os.path.isfile(os.path.join(dir, file)):
    #         os.remove(os.path.join(dir, file))
    # for i, cluster in enumerate(clusters):
    #     path = os.path.join(dir, f"tree{str(i).zfill(3)}")
    #     # save pointcloud to disk as las
    #     o3d.io.write_point_cloud(
    #         path + ".pcd",
    #         cluster["cloud"].to_legacy(),
    #     )
    #     with open(path + ".pkl", "wb") as file:
    #         pickle.dump(cluster, file)

    # # PLOT TREE MESHES
    # viz_objects = []
    # for tree in trees:
    #     verts, tris = tree.generate_mesh()
    #     if verts is None or tris is None:
    #         continue

    #     mesh = o3d.geometry.TriangleMesh(
    #         o3d.utility.Vector3dVector(verts),
    #         o3d.utility.Vector3iVector(tris)
    #     )
    #     mesh.compute_vertex_normals()
    #     viz_objects.append(mesh)

    #     viz_objects.append(tree.clusters[0]["cloud"].to_legacy())
    # o3d.visualization.draw_geometries(viz_objects)

    # tree_manager = TreeManager()
    # tree_manager.trees = trees
    # with open("/home/ori/git/digiforest_drs/digiforest_analysis_ros/output/trees/logs/raw/tree_manager_aggregate.pkl", "wb") as file:
    #     pickle.dump(tree_manager, file)
    # print("Done!")
