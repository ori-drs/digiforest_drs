#!/usr/bin/env python3
import digiforest_analysis.ground_segmentation as gs
import pcl
import os
import sys
import time

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage : ./script input_cloud output_folder")
    else:
        filename, extension = os.path.splitext(sys.argv[1])

        if extension != ".pcd":
            sys.exit("Input file must be a pcd file")

        print("Processing", sys.argv[1])

        cloud = pcl.PointCloud_PointNormal()
        tic = time.perf_counter()
        cloud._from_pcd_file(sys.argv[1].encode("utf-8"))
        toc = time.perf_counter()
        print(f"Load cloud:    {(toc - tic):0.4f} s")

        for cell in [2, 4, 8, 16]:
            for box in [80, 20]:
                tic = time.perf_counter()
                app = gs.GroundSegmentation(
                    max_distance_to_plane=0.5,
                    cell_size=cell,
                    normal_thr=0.92,
                    box_size=box,
                )
                ground_cloud, forest_cloud = app.process(cloud)
                toc = time.perf_counter()
                print(
                    f"Process cloud (PCL) cell_size[{cell}], box_size[{box}]: {(toc - tic)*1000:0.4f} ms"
                )

                # saving two point clouds to pcd files
                tic = time.perf_counter()
                gound_cloud_filename = os.path.join(
                    sys.argv[2], f"cell{cell}_box{box}_ground_cloud.pcd"
                )
                ground_cloud.to_file(str.encode(gound_cloud_filename))
                forest_cloud_filename = os.path.join(
                    sys.argv[2], f"cell{cell}_box{box}_forest_cloud.pcd"
                )
                forest_cloud.to_file(str.encode(forest_cloud_filename))
                toc = time.perf_counter()
                print(f"Write clouds:   {(toc - tic):0.4f} s")

        # Open3D
        import open3d as o3d

        ply_point_cloud = o3d.data.PLYPointCloud()
        pcd = o3d.io.read_point_cloud(sys.argv[1])
        assert pcd.has_normals()

        tic = time.perf_counter()
        # using all defaults
        oboxes = pcd.detect_planar_patches(
            normal_variance_threshold_deg=60,
            coplanarity_deg=75,
            outlier_ratio=0.75,
            min_plane_edge_length=0,
            min_num_points=0,
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30),
        )
        toc = time.perf_counter()
        print(f"Process cloud (Open3D): {(toc - tic):0.4f} s, {len(oboxes)} patches")

        # Visualize
        geometries = []
        for obox in oboxes:
            mesh = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(
                obox, scale=[1, 1, 0.0001]
            )
            mesh.paint_uniform_color(obox.color)
            geometries.append(mesh)
            geometries.append(obox)
        geometries.append(pcd)

        o3d.visualization.draw_geometries(
            geometries,
            zoom=0.62,
            front=[0.4361, -0.2632, -0.8605],
            lookat=[2.4947, 1.7728, 1.5541],
            up=[-0.1726, -0.9630, 0.2071],
        )
