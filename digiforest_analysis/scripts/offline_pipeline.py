#!/usr/bin/env python3
from digiforest_analysis.pipeline import Pipeline
from digiforest_analysis.utils import pcd
from pathlib import Path

import argparse
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

    # Read cloud
    cloud, header = pcd.load(str(filename), binary=True)
    assert len(cloud.point.normals) > 0

    # Configure pipeline
    pipeline = Pipeline()

    # Process cloud
    report = pipeline.process(cloud=cloud)

    # Extract report
    print(f"\nPipeline report: \n{report}")

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
    pcd.write(ground, header_fix, os.path.join(out_dir, "ground_cloud.pcd"))

    # Save forest cloud
    forest = pipeline.forest
    pcd.write(forest, header_fix, os.path.join(out_dir, "forest_cloud.pcd"))

    # Get trees
    trees = pipeline.trees
    for t in trees:
        pcd.write_open3d(
            t["cloud"], header_fix, os.path.join(out_dir, f"tree_{t['id']}.pcd")
        )
