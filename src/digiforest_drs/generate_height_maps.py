#!/usr/bin/env python3
import digiforest_drs as df
import sys

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage : ./script input_cloud_directory output_height_maps_directory")
    else:
        df.generate_height_maps(sys.argv[1], sys.argv[2])

