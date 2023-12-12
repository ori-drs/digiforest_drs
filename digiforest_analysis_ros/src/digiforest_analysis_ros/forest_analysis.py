#!/usr/bin/env python
# author: Leonard Freissmuth

from functools import partial
import os
import pickle
import time
from typing import List, Tuple
import numpy as np
import open3d as o3d
from multiprocessing import Pool
from scipy.spatial import cKDTree

import rospy
import tf2_ros
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from nav_msgs.msg import Path
import message_filters

from digiforest_analysis.tasks.tree_segmentation import TreeSegmentation
from digiforest_analysis.tasks.ground_segmentation import GroundSegmentation
from digiforest_analysis.tasks.tree_reconstruction import Tree
from digiforest_analysis.utils.timing import Timer


class ForestAnalysis:
    def __init__(self) -> None:
        self.read_params()
        self.setup_ros()
        self.ts = TreeSegmentation(
            clustering_method=self._clustering_method,
            debug_level=self._debug_level,
        )
        self.last_pc_header = None

        rospy.on_shutdown(self.shutdown_routine)

    def read_params(self):
        # Subscribers
        self._payload_cloud_topic = rospy.get_param(
            "~payload_cloud_topic", "/local_mapping/payload_in_local"
        )
        self._payload_path_topic = rospy.get_param(
            "~payload_path_topic", "/local_mapping/path"
        )

        # Publishers
        self._tree_meshes_topic = rospy.get_param("~tree_meshes_topic", "~/tree_meshes")
        self._debug_clusters_topic = rospy.get_param(
            "~debug_clusters_topic", "~/debug/clusters"
        )
        self._debug_cluster_labels_topic = rospy.get_param(
            "~debug_cluster_labels_topic", "~/debug/cluster_labels"
        )

        self._debug_level = rospy.get_param("~debug_level", 0)
        self._base_frame_id = rospy.get_param("~base_frame_id", "base")

        # Ground Segmentation
        self._cloth_enabled = rospy.get_param("~cloth_enabled", False)
        self._cloth_cell_size = rospy.get_param("~cloth_cell_size", 2)

        # Clustering
        self._clustering_method = rospy.get_param("~clustering_method", "voronoi")
        self._clustering_hough_filter_radius = rospy.get_param(
            "~clustering_hough_filter_radius", 0.1
        )
        self._clustering_crop_lower_bound = rospy.get_param(
            "~clustering_crop_lower_bound", 5.0
        )
        self._clustering_crop_upper_bound = rospy.get_param(
            "~clustering_crop_upper_bound", 8.0
        )
        self._clustering_max_cluster_radius = rospy.get_param(
            "~clustering_max_cluster_radius", 3.0
        )
        self._clustering_n_threads = rospy.get_param("~clustering_n_threads", 8)
        self._clustering_cluster_2d = rospy.get_param("~clustering_cluster_2d", False)
        self._clustering_distance_calc_point_fraction = rospy.get_param(
            "~clustering_distance_calc_point_fraction", 0.1
        )

        # Fitting
        self._fitting_slice_heights = rospy.get_param("~fitting_slice_heights", 0.5)
        self._fitting_slice_thickness = rospy.get_param("~fitting_slice_thickness", 0.3)
        self._fitting_outlier_radius = rospy.get_param("~fitting_outlier_radius", 0.02)
        self._fitting_max_center_deviation = rospy.get_param(
            "~fitting_max_center_deviation", 0.05
        )
        self._fitting_max_radius_deviation = rospy.get_param(
            "~fitting_max_radius_deviation", 0.05
        )
        self._fitting_filter_min_points = rospy.get_param(
            "~fitting_filter_min_points", 10
        )
        self._fitting_min_hough_vote = rospy.get_param("~fitting_min_hough_vote", 0.1)
        self._fitting_grid_res = rospy.get_param("~fitting_grid_res", 0.01)
        self._fitting_point_ratio = rospy.get_param("~fitting_point_ratio", 0.2)
        self._fitting_entropy_weighting = rospy.get_param(
            "~fitting_entropy_weighting", 10.0
        )
        self._fitting_max_consecutive_fails = rospy.get_param(
            "~fitting_max_consecutive_fails", 3
        )
        self._fitting_max_height = rospy.get_param("~fitting_max_height", 10.0)
        self._fitting_save_points = rospy.get_param("~fitting_save_points", True)
        self._fitting_save_debug_results = rospy.get_param(
            "~fitting_save_debug_results", False
        )
        self._fitting_n_threads = rospy.get_param("~fitting_n_threads", 1)

        # Internals
        self._tf_buffer = None
        self._tf_listener = None

    def setup_ros(self):
        # listeners for transforms
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)

        # # Subscribers
        # self._sub_payload_cloud = rospy.Subscriber(
        #     self._payload_cloud_topic, PointCloud2, self.payload_cloud_callback
        # )

        self._sub_payload_cloud = message_filters.Subscriber(
            self._payload_cloud_topic, PointCloud2
        )
        self._sub_payload_info = message_filters.Subscriber(
            self._payload_path_topic, Path
        )
        self._ts = message_filters.TimeSynchronizer(
            [self._sub_payload_cloud, self._sub_payload_info], 10
        )
        self._ts.registerCallback(self.payload_with_path_callback)

        # Publishers
        self._pub_tree_meshes = rospy.Publisher(
            self._tree_meshes_topic, MarkerArray, queue_size=1, latch=True
        )

        self._pub_debug_clusters = rospy.Publisher(
            self._debug_clusters_topic, PointCloud2, queue_size=1, latch=True
        )

        self._pub_cluster_labels = rospy.Publisher(
            self._debug_cluster_labels_topic, MarkerArray, queue_size=1, latch=True
        )

    def pc2_to_o3d(self, cloud: PointCloud2):
        cloud_numpy = np.frombuffer(cloud.data, dtype=np.float32).reshape(-1, 12)
        # clear points with at least one nan
        cloud_numpy = cloud_numpy[~np.isnan(cloud_numpy).any(axis=1)]

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(cloud_numpy[:, :3])
        cloud.normals = o3d.utility.Vector3dVector(cloud_numpy[:, 4:7])
        cloud = o3d.t.geometry.PointCloud.from_legacy(cloud)

        return cloud

    def genereate_mesh_msg(
        self,
        vertices: np.ndarray,
        triangles: np.ndarray,
        id: int = None,
    ) -> Marker:
        if vertices.shape[0] == 0:
            return None

        mesh_msg = Marker()

        mesh_msg.header.frame_id = self._base_frame_id
        mesh_msg.header.stamp = self.last_pc_header.stamp  # rospy.Time.now()
        mesh_msg.ns = "realtime_trees"
        mesh_msg.id = id if id is not None else np.random.randint(0, 100000)
        mesh_msg.type = Marker.TRIANGLE_LIST
        mesh_msg.action = Marker.ADD

        mesh_msg.pose.orientation.w = 1.0
        mesh_msg.scale.x = 1.0
        mesh_msg.scale.y = 1.0
        mesh_msg.scale.z = 1.0
        mesh_msg.color.a = 0.9
        mesh_msg.color.r = 140 / 255.0
        mesh_msg.color.g = 102 / 255.0
        mesh_msg.color.b = 87 / 255.0

        mesh_msg.points = [
            Point(x, y, z) for x, y, z in vertices[triangles].reshape(-1, 3).tolist()
        ]

        mesh_msg.colors = [mesh_msg.color for _ in range(len(mesh_msg.points))]

        return mesh_msg

    def publish_colored_pointclouds(self, clouds: list, colors: list):
        # convert colors to single float32
        for i in range(len(colors)):
            color = colors[i]
            color = np.floor(np.array([color[2], color[1], color[0], 0.0]) * 255)
            color = np.frombuffer(color.astype(np.uint8).tobytes(), dtype=np.float32)
            colors[i] = color

        # Convert numpy arrays to pointcloud2 data
        header = rospy.Header()
        header.frame_id = self.last_pc_header.frame_id
        header.stamp = self.last_pc_header.stamp

        fields = [
            PointField("x", 0, PointField.FLOAT32, 1),
            PointField("y", 4, PointField.FLOAT32, 1),
            PointField("z", 8, PointField.FLOAT32, 1),
            PointField("rgb", 12, PointField.FLOAT32, 1),
        ]

        points = np.concatenate(
            [
                np.hstack([cloud, np.repeat(color, cloud.shape[0])[:, None]]).astype(
                    np.float32
                )
                for cloud, color in zip(clouds, colors)
            ]
        )

        # Publish the pointcloud
        cloud = point_cloud2.create_cloud(header, fields, points)
        self._pub_debug_clusters.publish(cloud)
        if self._debug_level > 0:
            print(
                f"Published {len(clouds)} colored pointclouds on the topic {self._debug_clusters_topic}"
            )

    def publish_cluster_labels(self, labels: List[str], locations: List[np.ndarray]):
        marker_array = MarkerArray()
        for i, (label, location) in enumerate(zip(labels, locations)):
            marker = Marker()
            marker.header.frame_id = self.last_pc_header.frame_id
            marker.header.stamp = self.last_pc_header.stamp
            marker.ns = "realtime_trees"
            marker.id = i
            marker.type = Marker.TEXT_VIEW_FACING
            marker.action = Marker.ADD
            marker.pose.position.x = location[0]
            marker.pose.position.y = location[1]
            marker.pose.position.z = location[2]
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.5
            marker.scale.y = 0.5
            marker.scale.z = 0.5
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 1.0
            marker.text = label
            marker_array.markers.append(marker)

        self._pub_cluster_labels.publish(marker_array)
        if self._debug_level > 0:
            print(
                f"Published {len(labels)} cluster labels on the topic {self._debug_cluster_labels_topic}"
            )

    def payload_with_path_callback(self, point_cloud: PointCloud2, path: Path):
        rospy.loginfo("Received payload cloud and path")
        self.last_pc_header = point_cloud.header

        try:
            odom2sensor = self._tf_buffer.lookup_transform(
                self._base_frame_id,
                self.last_pc_header.frame_id,
                self.last_pc_header.stamp,
            )
            translation = odom2sensor.transform.translation
            translation = np.array([translation.x, translation.y, translation.z])
            rotation = odom2sensor.transform.rotation
            rotation = np.array([rotation.x, rotation.y, rotation.z, rotation.w])
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn("Could not get transform from odom to sensor", e)
            return

        timer = Timer()
        with timer("all"):
            with timer("conversion"):
                cloud = self.pc2_to_o3d(point_cloud)

            if self._cloth_enabled:
                with timer("Cloth"):
                    ground_seg = GroundSegmentation(
                        debug_level=self._debug_level,
                        method="csf",
                        cell_size=self._cloth_cell_size,
                    )
                    _, _, cloth = ground_seg.process(cloud=cloud, export_cloth=True)
            else:
                cloth = None
            TIME = time.time()
            with timer("Clusterting"):
                clusters = self.ts.process(
                    cloud=cloud,
                    cloth=cloth,
                    hough_filter_radius=self._clustering_hough_filter_radius,
                    crop_lower_bound=self._clustering_crop_lower_bound,
                    crop_upper_bound=self._clustering_crop_upper_bound,
                    max_cluster_radius=self._clustering_max_cluster_radius,
                    n_threads=self._clustering_n_threads,
                    point_fraction=self._clustering_distance_calc_point_fraction,
                )

                # send clusters to rviz
                clouds = [c["cloud"].point.positions.numpy() for c in clusters]
                colors = [c["info"]["color"] for c in clusters]
                # colors = [np.random.rand(3) for c in clusters]
                self.publish_colored_pointclouds(clouds, colors)
                self.publish_cluster_labels(
                    labels=[f"tree{str(c['info']['id']).zfill(3)}" for c in clusters],
                    locations=[c["info"]["axis"]["center"] for c in clusters],
                )
            print(f"Clustering took {time.time() - TIME} seconds")
            with timer("Fitting"):
                tree_from_cluster = partial(
                    Tree.reconstruct,
                    slice_heights=self._fitting_slice_heights,
                    slice_thickness=self._fitting_slice_thickness,
                    outlier_radius=self._fitting_outlier_radius,
                    max_center_deviation=self._fitting_max_center_deviation,
                    max_radius_deviation=self._fitting_max_radius_deviation,
                    filter_min_points=self._fitting_filter_min_points,
                    min_hough_vote=self._fitting_min_hough_vote,
                    grid_res=self._fitting_grid_res,
                    point_ratio=self._fitting_point_ratio,
                    entropy_weighting=self._fitting_entropy_weighting,
                    max_consecutive_fails=self._fitting_max_consecutive_fails,
                    max_height=self._fitting_max_height,
                    save_points=self._fitting_save_points,
                    save_debug_results=self._fitting_save_debug_results,
                    debug_level=self._debug_level,
                )
                trees = []
                if self._fitting_n_threads == 1:
                    for cluster in clusters:
                        trees.append(tree_from_cluster(cluster))
                else:
                    with Pool(processes=self._fitting_n_threads) as pool:
                        trees = pool.map(tree_from_cluster, clusters)

            if self._debug_level > 1:
                with timer("saving to disk"):
                    directory = os.path.join("trees", str(self.last_pc_header.stamp))
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    else:
                        for f in os.listdir(directory):
                            os.remove(os.path.join(directory, f))
                    for cluster, tree in zip(clusters, trees):
                        cluster["tree"] = tree
                    for cluster in clusters:
                        with open(
                            os.path.join(
                                directory,
                                f"tree{str(cluster['info']['id']).zfill(3)}.pkl",
                            ),
                            "wb",
                        ) as file:
                            pickle.dump(cluster, file)

            with timer("meshing"):
                trees = [t for t in trees if len(t.circles)]
                mesh_array = MarkerArray()
                for tree in trees:
                    tree.apply_transform(translation, rotation)
                    verts, tris = tree.generate_mesh()
                    mesh_msg = self.genereate_mesh_msg(verts, tris)
                    if mesh_msg is not None:
                        mesh_array.markers.append(mesh_msg)
                self._pub_tree_meshes.publish(mesh_array)

        rospy.loginfo(timer)
        rospy.loginfo(f"Done: Found {len(trees)} trees")

    def shutdown_routine(self, *args):
        """Executes the operations before killing the mission analysis procedures"""
        rospy.loginfo("Digiforest Analysis node stopped!")


class TreeManager:
    def __init__(
        self,
        distance_threshold: float = 0.1,
        reco_min_angle_coverage: float = 1.5 * np.pi,
        reco_min_distance: float = 4.0,
    ) -> None:
        """constructor of the TreeManager class

        Args:
            distance_threshold (float, optional): Distance in m closer than which the
                center points of two tree axes, the axes are considered to be identical.
                Defaults to 0.1.
            reco_min_angle_coverage (float, optional): Minimum arc length in rad the
                rays from sensor to tree axis of all paths have to have swept for the
                tree to be reconstructed. Defaults to 1.5*np.pi.
            reco_min_distance (float, optional): Distance in m closer than which the
                sensor must have visited the tree, so it is reconstructed.
                Defaults to 4.0.
        """
        self.distance_threshold = distance_threshold
        self.reco_min_angle_coverage = reco_min_angle_coverage
        self.reco_min_distance = reco_min_distance

        self.tree_locations: List[np.ndarray] = []  # for KD tree lookup
        self.tree_reco_flags: List[List[bool]] = []
        self.trees: List[Tree] = []
        self.tree_kd_tree: cKDTree = None

        self._tree_id_counter = 0

    def _new_tree_from_cluster(self, cluster: dict):
        """adds a new tree given a cluster. THIS DOES NOT UPDATE THE KD TREE!
        Call self.update_kd_tree() after adding all clusters.

        Args:
            cluster (dict): Dict as in the list returned by TreeSegmentation.process()
        """
        new_tree = Tree(self._tree_id_counter)
        new_tree.add_cluster(cluster)
        self.tree_locations.append(cluster["info"]["axis"]["center"])
        self.trees.append(new_tree)
        self.tree_reco_flags.append([False, False])  # angle converage and distance
        self._tree_id_counter += 1

    def update_kd_tree(self):
        """reinitializes KD tree. Has to be called after adding new clusters."""
        self.tree_kd_tree = cKDTree(self.tree_locations)

    def add_clusters(self, clusters: List[dict]):
        """This function checks every cluster. If a tree close to the detected cluster
        already exists, the cluster is added to the tree. If no tree is close enough, a
        new tree is created. This function updates the KD tree.

        Args:
            clusters (List[dict]): List of clusters as returned by
                TreeSegmentation.process()
        """
        if len(self.trees) == 0:
            # create new trees for all clusters and add them to the list
            for cluster in clusters:
                self._new_tree_from_cluster(cluster)
        else:
            # for all clusters check if tree at this coordinate already exists
            cluster_centers = [c["info"]["axis"]["center"] for c in clusters]
            distances, indices = self.tree_kd_tree.query(cluster_centers)
            new_clusters_mask = distances > self.distance_threshold

            new_clusters = np.array(clusters)[new_clusters_mask]
            existing_clusters = np.array(clusters)[~new_clusters_mask]
            cluster_correspondences = indices[~new_clusters_mask]

            # add clusters to existing trees
            for cluster, tree_index in zip(existing_clusters, cluster_correspondences):
                self.trees[tree_index].add_cluster(cluster)

            # create new trees for non-existing clusters and add them to the list
            for cluster in new_clusters:
                self._new_tree_from_cluster(cluster)

        self.update_kd_tree()
        self.try_reconstructions()

    def add_clusters_with_path(self, clusters: List[dict], path: np.ndarray):
        """This function checks every cluster and performs the same as add_clusters().
        In addition, it calculates the covered angle and covered distance of the sensor
        to the tree axis for every cluster and adds this information to the
        dict cluster["info"].

        Args:
            clusters (List[dict]): List of clusters as returned by
                TreeSegmentation.process()
            path (np.ndarray): Nx7 array describing consecutive 7D poses of the sensor.
                The first three columns are the x, y, z coordinates of the position. The
                last four columns are the x, y, z, w quaternions describing orientation.
        """

        for cluster in clusters:
            angle_from, angle_to, d_min, d_max = self.calculate_coverage(cluster, path)
            cluster["info"]["coverage"] = {
                "angle_from": angle_from,
                "angle_to": angle_to,
                "distance_min": d_min,
                "distance_max": d_max,
            }

        self.add_clusters(clusters)

    def calculate_coverage(self, cluster: dict, path: np.ndarray) -> Tuple[float]:
        """Calculates the covered angle of the sensor to the tree axis for a given
        cluster. The covered angle is given by two values defining the global extent
        of the arc around the z-axis in the x-y-plane bounding all rays from the
        center of the tree axis to all sensor poses. The parametrization is given by a
        start angle and end angle.
        The angles are given wrt. to the global x-axis and are in [0, 2*pi].

        Args:
            cluster (dict): Dict as in the list returned by TreeSegmentation.process()
            path (np.ndarray): Nx7 array describing consecutive 7D poses of the sensor.
                The first three columns are the x, y, z coordinates of the position. The
                last four columns are the x, y, z, w quaternions describing orientation.

        Returns:
            float: start angle of arc wrt to global x-axis
            float: end angle of arc wrt to global x-axis
            float: min distance from sensor pose to tree axis
            float: max distance from sensor pose to tree axis
        """
        min_distance = np.inf
        max_distance = -np.inf
        angles = []
        for pose in path:
            # calculate connecting vector between sensor pose and tree axis center
            ray_vector = pose[:2] - cluster["info"]["axis"]["center"][:2]
            ray_length = np.linalg.norm(ray_vector)

            # calculate angle of ray vector wrt. global x-axis
            ray_vector /= ray_length
            angle = np.arctan2(-ray_vector[1], -ray_vector[0]) + np.pi  # range 0 to 2pi
            angles.append(angle)

            # update min and max distance and angle
            min_distance = min(min_distance, ray_length)
            max_distance = max(max_distance, ray_length)

        angles = np.unwrap(np.asarray(angles))

        coverage = np.abs(angles.max() - angles.min())
        if np.isclose(coverage, 2 * np.pi) or coverage > 2 * np.pi:
            angle_from, angle_to = 0, 2 * np.pi
        else:
            angle_from, angle_to = angles.min() % (2 * np.pi), angles.max() % (
                2 * np.pi
            )

        return angle_from, angle_to, min_distance, max_distance

    def check_angle_coverage(self, angles: List[Tuple[float]]) -> bool:
        """Checks if the union of angle intervals given by the list of tuples is greater
        than the reco_min_angle_coverage parameter.

        Args:
            angles (List[Tuple[float]]): List of tuples of angles. Each tuple contains
                two angles defining the start and end angle of an arc wrt. the global
                x-axis.

        Returns:
            bool: True if the union of intervals is large enough, False otherwise.
        """
        angle_accumulator = np.zeros(360, dtype=bool)
        for angle_from, angle_to in angles:
            angle_from = int(np.around(np.rad2deg(angle_from)))
            angle_to = int(np.around(np.rad2deg(angle_to)))
            angle_accumulator[angle_from:angle_to] = True

        return angle_accumulator.sum() >= np.rad2deg(self.reco_min_angle_coverage)

    def check_distance_coverage(self, distances: List[Tuple[float]]) -> bool:
        """Checks if the distances given by the list of tuples are covered by the
        distance_threshold parameter.

        Args:
            distances (List[Tuple[float]]): List of tuples of distances. Each tuple
                contains two distances defining the min and max distance of a sensor
                pose to the tree axis.

        Returns:
            bool: True if all distances are covered, False otherwise.
        """
        distances = np.array(distances)
        d_min = np.min(distances[:, 0])
        return d_min <= self.reco_min_distance

    def try_reconstructions(self) -> bool:
        """Checks all trees in the data base if they satisfy the conditions to be
        reconstructed. If so, they are reconstrcuted.

        Returns:
            bool: True if at least one tree was reconstructed, False otherwise.
        """
        reco_happened = False
        for i, tree in enumerate(self.trees):
            coverages = [
                (
                    cluster["info"]["coverage"]["angle_from"],
                    cluster["info"]["coverage"]["angle_to"],
                    cluster["info"]["coverage"]["distance_min"],
                    cluster["info"]["coverage"]["distance_max"],
                )
                for cluster in tree.clusters
            ]

            if not self.check_angle_coverage([c[:2] for c in coverages]):
                continue
            self.tree_reco_flags[i][0] = True

            if not self.check_distance_coverage([c[2:] for c in coverages]):
                continue
            self.tree_reco_flags[i][1] = True

            reco_happened |= tree.reconstruct()

        return reco_happened
