#!/usr/bin/env python3
from digiforest_analysis.terrain_mapping import generate_height_maps
import sys

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage : ./script input_cloud_directory output_height_maps_directory")
    else:
        generate_height_maps(sys.argv[1], sys.argv[2])
