#!/usr/bin/env python3
import digiforest_analysis.ground_segmentation as gs
import sys
import os

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage : ./script input_cloud output_folder")
    else:
        filename, extension = os.path.splitext(sys.argv[1])

        if extension != ".pcd":
            sys.exit("Input file must be a pcd file")

        print("Processing", sys.argv[1])
        app = gs.GroundSegmentation(sys.argv[1])
        ground_cloud, forest_cloud = app.generate_height_map()

        # saving two point clouds to pcd files
        gound_cloud_filename = os.path.join(sys.argv[2], 'ground_cloud.pcd')
        ground_cloud.to_file(str.encode(gound_cloud_filename))
        forest_cloud_filename = os.path.join(sys.argv[2], 'forest_cloud.pcd')
        forest_cloud.to_file(str.encode(forest_cloud_filename))
