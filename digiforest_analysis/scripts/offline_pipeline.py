#!/usr/bin/env python3
from digiforest_analysis.pipeline import Pipeline
from pathlib import Path

import argparse
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="offline_pipeline",
        description="Processes a point cloud and extracts forest inventory attributes",
        epilog="Text at the bottom of help",
    )
    parser.add_argument("filename")  # positional argument
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    # Default config
    params = {
        "ground_segmentation": {
            "method": "default",  # default, indexing, csf
            "debug_level": 2,  # 0: none, 1: messages, 2: 3d visualizations, 3: per-tree visualizations
            "voxel_filter_size": 0.05,
            "max_distance_to_plane": 0.5,
            "cell_size": 4.0,
            "normal_thr": 0.9,
            "box_size": 80,
        },
        "tree_segmentation": {
            "debug_level": 2,  # 0: none, 1: messages, 2: 3d visualizations, 3: per-tree visualizations
            "normal_thr": 0.5,
            "voxel_size": 0.05,
            "cluster_2d": False,
            "clustering_method": "hdbscan",  # many options, check clustering.py
            "min_tree_height": 1.5,
            "max_tree_diameter": 10.0,
            "min_tree_diameter": 0.1,
            "min_gravity_alignment_score": 0.1,
        },
        "tree_analysis": {
            "debug_level": 2,  # 0: none, 1: messages, 2: 3d visualizations, 3: per-tree visualizations
            "max_dist_to_ground": 1e6,
            "fitting_method": "lsq",  # pcl_ransac, lsq
            "breast_height": 1.3,
            "breast_height_range": 4.5,
            "max_valid_radius": 0.8,
            "min_inliers": 100,
            "outlier_thr": 0.01,
            "loss_scale": 0.001,
        },
    }

    # Configure pipeline
    pipeline = Pipeline(file=args.filename, out_dir=args.out, **params)

    # Process cloud
    print("Processing...")
    report = pipeline.process()

    # Plot marteloscope
    from digiforest_analysis.utils import marteloscope

    fig, ax = plt.subplots()
    marteloscope.plot(pipeline.trees, ax)
    marteloscope_path = Path(args.out, "marteloscope.pdf")
    fig.set_tight_layout(True)
    fig.savefig(str(marteloscope_path), dpi=150)

    # Extract report
    print("\nPipeline report:")
    for k, v in report.items():
        print(f"{k:<20}: {v}")
