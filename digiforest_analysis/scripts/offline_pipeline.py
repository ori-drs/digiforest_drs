#!/usr/bin/env python3
from digiforest_analysis import Pipeline
from pathlib import Path

import argparse
import open3d as o3d

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
    cloud = o3d.io.read_point_cloud(filename)

    assert cloud.has_normals()

    # Configure pipeline
    pipeline = Pipeline()

    # Process cloud
    report = pipeline.process(cloud)

    # Extract report
    print(report)

    # Prepare output folders
    if args.out is not None:
        out_dir = Path(args.out)
    else:
        out_dir = filename.parent
    out_dir.mkdir(exist_ok=True)

    # Save ground
    ground = pipeline.ground

    # Save forest cloud
    forest = pipeline.forest

    # Get trees
    trees_cloud = pipeline.tree_clouds

    # Get tree attributes
    trees_att = pipeline._tree_attributes
