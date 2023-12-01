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
    parser.add_argument("--viz_zoom", default=0.7, type=float)
    args = parser.parse_args()

    # Default config
    params = {
        "preprocessing": {
            "debug_level": 2,  # 0: none, 1: messages, 2: 3d visualizations, 3: per-tree
            "viz_zoom": args.viz_zoom,
            "crop_x": 80.0,  # Full extent along x
            "crop_y": 80.0,  # Full extent along y
            "crop_z": 20.0,  # Full extent along z
            "noise_filter_points": 20,  # For radius filtering
            "noise_filter_radius": 0.2,  # For radius filtering
            "intensity_thr": 20,  # For intensity filtering
        },
        "ground_segmentation": {
            "debug_level": 2,  # 0: none, 1: messages, 2: 3d visualizations, 3: per-tree visualizations
            "viz_zoom": args.viz_zoom,
            "method": "default",  # default, indexing, csf
            "voxel_filter_size": 0.05,
            "max_distance_to_plane": 0.5,
            "cell_size": 4.0,
            "normal_thr": 0.9,
            "box_size": 80,
            "num_plane_support_points": 100,
            "num_plane_support_inliers": 20,
        },
        "tree_segmentation": {
            "debug_level": 2,  # 0: none, 1: messages, 2: 3d visualizations, 3: per-tree visualizations
            "viz_zoom": args.viz_zoom,
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
            "viz_zoom": args.viz_zoom,
            "max_dist_to_ground": 1e6,
            "fitting_method": "lsq",  # lsq
            "breast_height": 1.5,
            "breast_height_range": 1.0,
            "max_valid_radius": 0.8,
            "min_inliers": 100,
            "outlier_thr": 0.01,
            "loss_scale": 1.0,
            "weight_n": 0.0,
            "weight_g": 0.0,
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
