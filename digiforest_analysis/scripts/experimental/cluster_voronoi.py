import digiforest_analysis.tasks.tree_segmentation as ts
from digiforest_analysis.tasks.tree_reconstruction import Tree
from digiforest_analysis.utils.io import load

import numpy as np
import open3d as o3d
from multiprocessing import Pool
from matplotlib import pyplot as plt


def plot_mesh(vertices, triangles, points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Extract x, y, z coordinates from vertices
    x, y, z = zip(*vertices)

    # Plot vertices
    ax.scatter(x, y, z, c="r", marker="o")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c="b", marker="o", s=0.1)

    # Plot triangles
    for triangle in triangles:
        ax.plot(
            [vertices[triangle[0]][0], vertices[triangle[1]][0]],
            [vertices[triangle[0]][1], vertices[triangle[1]][1]],
            [vertices[triangle[0]][2], vertices[triangle[1]][2]],
            "b-",
        )

        ax.plot(
            [vertices[triangle[1]][0], vertices[triangle[2]][0]],
            [vertices[triangle[1]][1], vertices[triangle[2]][1]],
            [vertices[triangle[1]][2], vertices[triangle[2]][2]],
            "b-",
        )

        ax.plot(
            [vertices[triangle[2]][0], vertices[triangle[0]][0]],
            [vertices[triangle[2]][1], vertices[triangle[0]][1]],
            [vertices[triangle[2]][2], vertices[triangle[0]][2]],
            "b-",
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()
    ax.set_box_aspect((xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]))
    plt.show()


if __name__ == "__main__":
    # pcd_file = "/home/ori/logs/logs_evo_finland/exp16/combined_cloud.pcd"
    pcd_file = "/home/ori/logs/logs_evo_finland/exp01/2023-05-01-14-01-05-exp01/payload_clouds/cloud_1682946124_761436000.pcd"
    cloud, header = load(pcd_file, binary=True)
    if "VIEWPOINT" in header:
        header_data = [float(x) for x in header["VIEWPOINT"]]
        location = np.array(header_data[:3])
        rotation = np.array(header_data[3:])
        # apply transformation to point cloud
        R = o3d.geometry.TriangleMesh.get_rotation_matrix_from_quaternion(rotation)
        cloud.rotate(R, center=location)

    # ground_seg = gs.GroundSegmentation(debug_level=0, method="csf", cell_size=2)
    # _, forest_cloud, cloth = ground_seg.process(cloud=cloud, export_cloth=True)
    tree_seg = ts.TreeSegmentation(debug_level=0, clustering_method="voronoi")
    clusters = tree_seg.process(cloud=cloud, cluster_dist=2)

    # import pickle
    # for i, cluster in enumerate(clusters):
    #     path = f"/home/ori/git/realtime-trees/single_trees/clustering_2/tree{str(i).zfill(3)}"
    #     # save pointcloud to disk as las
    #     o3d.io.write_point_cloud(
    #         path + ".pcd",
    #         cluster["cloud"].to_legacy(),
    #     )
    #     with open(path + ".pkl", 'wb') as file:
    #         pickle.dump(cluster, file)

    with Pool() as pool:
        trees = pool.map(Tree.from_cluster, clusters)

    print(np.array([1 if len(t.circles) else 0 for t in trees]).sum())
    # for tree, points in zip(trees, [clusters["cloud"].point.positions.numpy() for clusters in clusters]):
    #     verts, tris = tree.generate_mesh()
    #     if len(verts) == 0 or len(tris) == 0:
    #         continue
    #     plot_mesh(verts, tris, points)
    #     print(
    #         "Tree mesh has {} vertices and {} triangles".format(len(verts), len(tris))
    #     )

    print("Done!")
