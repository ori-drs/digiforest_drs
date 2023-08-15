#!/usr/bin/env python3
from digiforest_analysis.pipeline import Pipeline
from digiforest_analysis.utils import pcd
from pathlib import Path

import argparse
import open3d as o3d

import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="offline_pipeline",
        description="Processes a point cloud and extracts forest inventory attributes",
        epilog="Text at the bottom of help",
    )
    parser.add_argument("filename")  # positional argument
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    # Check validity of input
    filename = Path(args.filename)
    if not filename.exists():
        raise ValueError(f"Input file [{filename}] does not exist")

    out_folder = args.out if args.out is not None else "/tmp"

    # Read cloud
    cloud = o3d.t.io.read_point_cloud(str(filename))
    header = pcd.load_header(str(filename))
    assert len(cloud.point.normals) > 0

    # Configure pipeline
    pipeline = Pipeline()

    # Process cloud
    report = pipeline.process(cloud=cloud)

    # Extract report
    print(report)

    # Prepare output folders
    if args.out is not None:
        out_dir = Path(args.out)
    else:
        out_dir = filename.parent
    out_dir.mkdir(exist_ok=True)

    # Prepare header fix
    header_fix = {"VIEWPOINT": header["VIEWPOINT"]}

    # Save ground
    ground = pipeline.ground
    pcd.write_open3d(ground, header_fix, os.path.join(out_folder, "ground_cloud.pcd"))

    # Save forest cloud
    forest = pipeline.forest
    pcd.write_open3d(forest, header_fix, os.path.join(out_folder, "forest_cloud.pcd"))

    # Get trees
    trees = pipeline.trees
    for t in trees:
        pcd.write_open3d(
            t["cloud"], header_fix, os.path.join(out_folder, f"tree_{t['id']}.pcd")
        )
