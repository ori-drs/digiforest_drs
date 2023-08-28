#!/usr/bin/env python3
from digiforest_registration.tasks.vertical_alignment import VerticalRegistration
from digiforest_registration.utils import CloudLoader
from pathlib import Path

import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="cloud_registration",
        description="Registers a frontier cloud to a reference UAV cloud",
        epilog="Text at the bottom of help",
    )
    parser.add_argument("uav_cloud")
    parser.add_argument("frontier_cloud")
    args = parser.parse_args()

    # Check validity of inputs
    uav_cloud_filename = Path(args.uav_cloud)
    if not uav_cloud_filename.exists():
        raise ValueError(f"Input file [{uav_cloud_filename}] does not exist")

    frontier_cloud_filename = Path(args.frontier_cloud)
    if not frontier_cloud_filename.exists():
        raise ValueError(f"Input file [{frontier_cloud_filename}] does not exist")

    loader = CloudLoader()

    uav_cloud = loader.load_cloud(str(uav_cloud_filename))
    frontier_cloud = loader.load_cloud(str(frontier_cloud_filename))

    vertical_registration = VerticalRegistration(uav_cloud, frontier_cloud)
    vertical_registration.process()
