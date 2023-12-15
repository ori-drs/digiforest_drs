#!/usr/bin/env python
# author: Leonard Freissmuth

import os
import pickle
import numpy as np
import open3d as o3d
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from typing import List, Tuple

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
from digiforest_analysis.tasks.tree_reconstruction import Circle, Tree
from digiforest_analysis.utils.timing import Timer

timer = Timer()


class ForestAnalysis:
    def __init__(self) -> None:
        self.read_params()
        self.setup_ros()

        self._tree_segmenter = TreeSegmentation(
            clustering_method=self._clustering_method,
            debug_level=self._debug_level,
        )

        self._tree_manager = TreeManager(
            self._distance_threshold,
            self._reco_min_angle_coverage,
            self._reco_min_distance,
            self._clustering_crop_upper_bound,
            self._clustering_crop_lower_bound,
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
        self._tree_meshes_topic = rospy.get_param(
            "~tree_meshes_topic", "digiforest_forest_analysis/tree_meshes"
        )
        self._debug_clusters_topic = rospy.get_param(
            "~debug_clusters_topic", "digiforest_forest_analysis/debug/cluster_clouds"
        )
        self._debug_cluster_labels_topic = rospy.get_param(
            "~debug_cluster_labels_topic",
            "digiforest_forest_analysis/debug/cluster_labels",
        )

        self._debug_level = rospy.get_param("~debug_level", 0)
        self._base_frame_id = rospy.get_param("~base_frame_id", "base")
        self._odom_frame_id = rospy.get_param("~odom_frame_id", "odom_vilens")

        # Tree Manager
        self._distance_threshold = rospy.get_param(
            "~tree_manager/distance_threshold", 0.1
        )
        self._reco_min_angle_coverage = np.deg2rad(
            rospy.get_param("~tree_manager/reco_min_angle_coverage", 180)
        )
        self._reco_min_distance = rospy.get_param(
            "~tree_manager/reco_min_distance", 4.0
        )

        # Ground Segmentation
        self._cloth_enabled = rospy.get_param("~cloth/enabled", False)
        self._cloth_cell_size = rospy.get_param("~cloth/cell_size", 2)

        # Clustering
        self._clustering_method = rospy.get_param("~clustering/method", "voronoi")
        self._clustering_hough_filter_radius = rospy.get_param(
            "~clustering/hough_filter_radius", 0.1
        )
        self._clustering_crop_lower_bound = rospy.get_param(
            "~clustering/crop_lower_bound", 5.0
        )
        self._clustering_crop_upper_bound = rospy.get_param(
            "~clustering/crop_upper_bound", 8.0
        )
        self._clustering_max_cluster_radius = rospy.get_param(
            "~clustering/max_cluster_radius", 3.0
        )
        self._clustering_n_threads = rospy.get_param("~clustering/n_threads", 8)
        self._clustering_cluster_2d = rospy.get_param("~clustering/cluster_2d", False)
        self._clustering_distance_calc_point_fraction = rospy.get_param(
            "~clustering/distance_calc_point_fraction", 0.1
        )

        # Fitting
        self._fitting_slice_heights = rospy.get_param("~fitting/slice_heights", 0.5)
        self._fitting_slice_thickness = rospy.get_param("~fitting/slice_thickness", 0.3)
        self._fitting_outlier_radius = rospy.get_param("~fitting/outlier_radius", 0.02)
        self._fitting_max_center_deviation = rospy.get_param(
            "~fitting/max_center_deviation", 0.05
        )
        self._fitting_max_radius_deviation = rospy.get_param(
            "~fitting/max_radius_deviation", 0.05
        )
        self._fitting_filter_min_points = rospy.get_param(
            "~fitting/filter_min_points", 10
        )
        self._fitting_min_hough_vote = rospy.get_param("~fitting/min_hough_vote", 0.1)
        self._fitting_grid_res = rospy.get_param("~fitting/grid_res", 0.01)
        self._fitting_point_ratio = rospy.get_param("~fitting/point_ratio", 0.2)
        self._fitting_entropy_weighting = rospy.get_param(
            "~fitting/entropy_weighting", 10.0
        )
        self._fitting_max_consecutive_fails = rospy.get_param(
            "~fitting/max_consecutive_fails", 3
        )
        self._fitting_max_height = rospy.get_param("~fitting/max_height", 10.0)
        self._fitting_save_debug_results = rospy.get_param(
            "~fitting/save_debug_results", False
        )
        self._fitting_n_threads = rospy.get_param("~fitting/n_threads", 1)

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
        self._tree_segmenter = message_filters.TimeSynchronizer(
            [self._sub_payload_cloud, self._sub_payload_info], 10
        )
        self._tree_segmenter.registerCallback(self.payload_with_path_callback)

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
        frame_id: str = None,
    ) -> Marker:
        if vertices.shape[0] == 0:
            return None

        mesh_msg = Marker()
        mesh_msg.header.frame_id = frame_id if frame_id else self._base_frame_id
        mesh_msg.header.stamp = self.last_pc_header.stamp  # rospy.Time.now()
        mesh_msg.ns = "realtime_trees/meshes"
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
            marker.ns = "realtime_trees/markers"
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
            with timer("Clusterting"):
                clusters = self._tree_segmenter.process(
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
            with timer("Tree Manager"):
                path = np.array(
                    [
                        [p.pose.position.x, p.pose.position.y, p.pose.position.z]
                        for p in path.poses
                    ]
                )
                self._tree_manager.add_clusters_with_path(
                    clusters, path, time_stamp=self.last_pc_header.stamp
                )

            with timer("Publishing"):
                self.publish_tree_manager_state(rotation, translation)
                self._tree_manager.write_results(
                    "/home/ori/git/digiforest_drs/trees/logs"
                )

            if self._debug_level > 1:
                with timer("saving to disk"):
                    directory = os.path.join("trees", str(self.last_pc_header.stamp))
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    else:
                        for f in os.listdir(directory):
                            os.remove(os.path.join(directory, f))
                    for cluster in clusters:
                        with open(
                            os.path.join(
                                directory,
                                f"tree{str(cluster['info']['id']).zfill(3)}.pkl",
                            ),
                            "wb",
                        ) as file:
                            pickle.dump(cluster, file)

        rospy.loginfo("Timing results:\n" + str(timer))

    def publish_tree_manager_state(
        self, rotation: np.ndarray = None, translation: np.ndarray = None
    ):
        label_texts = []
        label_positions = []
        mesh_messages = MarkerArray()
        for tree, reco_flags, coverage_angle in zip(
            self._tree_manager.trees,
            self._tree_manager.tree_reco_flags,
            self._tree_manager.tree_coverage_angles,
        ):
            label_texts.append(
                f"##### tree{str(tree.id).zfill(3)} #####\n"
                + f"angle:     {np.rad2deg(coverage_angle):.0f} deg\n"
                + f"angle_flag:    {reco_flags['angle_flag']}\n"
                + f"distance_flag:  {reco_flags['distance_flag']}\n"
            )
            label_positions.append(tree.axis["center"])

            if tree.reconstructed:
                verts, tris = tree.generate_mesh()
                if rotation is not None and translation is not None:
                    verts = self.apply_transform(
                        verts, translation, rotation, inverse=False
                    )
                mesh_messages.markers.append(
                    self.genereate_mesh_msg(
                        verts, tris, id=tree.id, frame_id=self._base_frame_id
                    )
                )
            else:
                verts, tris = self._tree_manager.generate_tree_placeholder(tree.id)
                if rotation is not None and translation is not None:
                    verts = self.apply_transform(
                        verts, translation, rotation, inverse=False
                    )
                mesh_messages.markers.append(
                    self.genereate_mesh_msg(
                        verts, tris, id=tree.id, frame_id=self._base_frame_id
                    )
                )

        self._pub_tree_meshes.publish(mesh_messages)
        self.publish_cluster_labels(label_texts, label_positions)

    def apply_transform(
        self,
        points: np.ndarray,
        translation: np.ndarray,
        rotation: np.ndarray,
        inverse: bool = False,
    ) -> np.ndarray:
        """Transforms the given points by the given translation and rotation.
        Optionally performs the inverse transformation.

        Args:
            points (np.ndarray): Nx3 array of points to be transformed
            translation (np.ndarray): 3x1 array of translation
            rotation (np.ndarray): 3x3 rotation matrix or 4x1 quaternion
            inverse (bool, optional): Flag to calculate inverse. Defaults to False.

        Returns:
            np.ndarray: Nx3 array of transformed points
        """
        if rotation.shape[0] == 4:
            rot_mat = Rotation.from_quat(rotation).as_matrix()
        elif rotation.shape == (3, 3):
            rot_mat = rotation
        else:
            raise ValueError("rotation must be given as 3x3 matrix or quaternion")
        if inverse:
            points = (points - translation) @ rot_mat
        else:
            points = points @ rot_mat.T + translation

        return points

    def shutdown_routine(self, *args):
        """Executes the operations before killing the mission analysis procedures"""
        rospy.loginfo("Digiforest Analysis node stopped!")


class TreeManager:
    def __init__(
        self,
        distance_threshold: float = 0.1,
        reco_min_angle_coverage: float = 1.5 * np.pi,
        reco_min_distance: float = 4.0,
        crop_upper_bound: float = 1.0,
        crop_lower_bound: float = 3.0,
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
            crop_lower_bound (float, optional): Lower bounding height in m used for
                cropping during clustering.
            crop_upper_bound (float, optional): Upper bounding height in m used for
                cropping during clustering.
        """
        self.distance_threshold = distance_threshold
        self.reco_min_angle_coverage = reco_min_angle_coverage
        self.reco_min_distance = reco_min_distance
        self.crop_upper_bound = crop_upper_bound
        self.crop_lower_bound = crop_lower_bound

        self.tree_reco_flags: List[List[bool]] = []
        self.tree_coverage_angles: List[float] = []
        self.trees: List[Tree] = []
        self._kd_tree = None

        self._tree_id_counter = 0
        self._last_cluster_time = None

    def _update_kd_tree(self):
        """Updates the KD tree with the current tree centers"""
        centers = [tree.axis["center"] for tree in self.trees]
        self._kd_tree = cKDTree(centers)

    def _new_tree_from_cluster(self, cluster: dict):
        """adds a new tree given a cluster.

        Args:
            cluster (dict): Dict as in the list returned by TreeSegmentation.process()
        """
        new_tree = Tree(self._tree_id_counter)
        new_tree.add_cluster(cluster)
        self.trees.append(new_tree)
        self.tree_reco_flags.append({"angle_flag": False, "distance_flag": False})
        self.tree_coverage_angles.append(0.0)
        self._tree_id_counter += 1

    def distance_line_to_line(self, line_1: dict, line_2: dict) -> float:
        """Calculates the minum distance between two axes. If the keyword "axis_length"
        is present in the dicts, the closest points are bounded to be between the basis
        poit of the axis and not further away from the basis point in the z direction
        of the given rot_mat than the axis_length. Otherwise, the closest points can be
        anywhere on the axis.

        Args:
            line1 (dict): Dict with the keys "center", "rot_mat" and optionally
                "axis_length" describing the first axis.
            line2 (dict): Dict with the keys "center", "rot_mat" and optionally
                "axis_length" describing the second axis.

        Returns:
            float: minimum distance between axes
        """
        axis_pnt_1 = line_1["center"]
        axis_pnt_2 = line_2["center"]
        axis_dir_1 = line_1["rot_mat"][2, :]
        axis_dir_2 = line_2["rot_mat"][2, :]
        normal = np.cross(axis_dir_1, axis_dir_2)
        normal_length = np.linalg.norm(normal)
        if np.isclose(normal_length, 0.0):
            meetin_point_1 = axis_pnt_1
            # Gram Schmidt
            meeting_point_2 = axis_pnt_1 - axis_dir_1 * (
                (axis_pnt_2 - axis_pnt_1) @ axis_dir_1
            )
        else:
            normal /= np.linalg.norm(normal_length)
            v_normal = np.cross(axis_dir_1, normal)
            v_normal /= np.linalg.norm(v_normal)
            w_normal = np.cross(axis_dir_2, normal)
            w_normal /= np.linalg.norm(w_normal)
            s = w_normal @ (axis_pnt_2 - axis_pnt_1) / (w_normal @ axis_dir_1)
            t = v_normal @ (axis_pnt_1 - axis_pnt_2) / (v_normal @ axis_dir_2)

            meetin_point_1 = axis_pnt_1 + s * axis_dir_1
            meeting_point_2 = axis_pnt_2 + t * axis_dir_2

        if "axis_length" in line_1.keys() and "axis_length" in line_2.keys():
            axis_len_1 = line_1["axis_length"]
            axis_len_2 = line_2["axis_length"]
            pos_1_normalized = (meetin_point_1 - axis_pnt_1) @ axis_dir_1 / axis_len_1
            pos_2_normalized = (meeting_point_2 - axis_pnt_2) @ axis_dir_2 / axis_len_2

            if pos_1_normalized < 0:
                meetin_point_1 = axis_pnt_1
            if pos_1_normalized > 1:
                meetin_point_1 = axis_pnt_1 + axis_dir_1 * axis_len_1
            if pos_2_normalized < 0:
                meeting_point_2 = axis_pnt_2
            if pos_2_normalized > 1:
                meeting_point_2 = axis_pnt_2 + axis_dir_2 * axis_len_2

        return np.linalg.norm(meetin_point_1 - meeting_point_2)

    def add_clusters(self, clusters: List[dict], time_stamp: rospy.Time = None):
        """This function checks every cluster. If a tree close to the detected cluster
        already exists, the cluster is added to the tree. If no tree is close enough, a
        new tree is created. This function updates the KD tree.

        Args:
            clusters (List[dict]): List of clusters as returned by
                TreeSegmentation.process()
            time_stamp (rospy.Time, optional): Time stamp of the point cloud. Defaults
                to None.
        """
        if time_stamp is not None:
            for cluster in clusters:
                cluster["time_stamp"] = time_stamp

        if len(self.trees) == 0:
            # create new trees for all clusters and add them to the list
            for cluster in clusters:
                self._new_tree_from_cluster(cluster)
        else:
            # for all clusters check if tree at this coordinate already exists
            with timer("Finding Correspondences"):
                num_existing, num_new = 0, 0
                candiate_centers = [c["info"]["axis"]["center"] for c in clusters]
                _, existing_indices = self._kd_tree.query(candiate_centers)
                for i_candidate, i_existing in enumerate(existing_indices):
                    axis_length = self.crop_upper_bound - self.crop_lower_bound
                    candidate_axis = clusters[i_candidate]["info"]["axis"]
                    candidate_axis["axis_length"] = axis_length
                    existing_axis = self.trees[i_existing].axis
                    existing_axis["axis_length"] = axis_length

                    distance = self.distance_line_to_line(candidate_axis, existing_axis)
                    if distance < self.distance_threshold:
                        self.trees[i_existing].add_cluster(clusters[i_candidate])
                        num_existing += 1
                    else:
                        self._new_tree_from_cluster(clusters[i_candidate])
                        num_new += 1

            rospy.loginfo(f"Found {num_existing} existing and {num_new} new clusters")

        self._update_kd_tree()
        self.try_reconstructions()

    def add_clusters_with_path(
        self, clusters: List[dict], path: np.ndarray, time_stamp: rospy.Time = None
    ):
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
            time_stampe (rospy.Time, optional): Time stamp of the point cloud. Defaults
                to None.
        """
        if time_stamp is not None:
            self._last_cluster_time = time_stamp

        for cluster in clusters:
            angle_from, angle_to, d_min, d_max = self.calculate_coverage(cluster, path)
            cluster["info"]["coverage"] = {
                "angle_from": angle_from,
                "angle_to": angle_to,
                "distance_min": d_min,
                "distance_max": d_max,
            }

        self.add_clusters(clusters, time_stamp)

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

    def compute_angle_coverage(self, intervals: List[Tuple[float]]) -> float:
        """Calculates the coverd angle of the union of the given angle intervals.

        Args:
            intervals (List[Tuple[float]]): Tuple of angles. Each tuple contains two
                angles defining the start and end angle of an arc wrt. the global
                x-axis.


        Returns:
            float: coverage angle in rad
        """
        angle_accumulator = np.zeros(360, dtype=bool)
        for angle_from, angle_to in intervals:
            angle_from = int(np.around(np.rad2deg(angle_from)))
            angle_to = int(np.around(np.rad2deg(angle_to)))
            angle_accumulator[angle_from:angle_to] = True
        return np.deg2rad(angle_accumulator.sum())

    def check_angle_coverage(self, intervals: List[Tuple[float]], tree_id: int) -> bool:
        """Checks if the union of angle intervals given by the list of tuples is greater
        than the reco_min_angle_coverage parameter.

        Args:
            intervals (List[Tuple[float]]): List of tuples of angles. Each tuple
                contains  two angles defining the start and end angle of an arc wrt.
                the global x-axis.
            tree_id (int): ID of the tree the angle intervals belong to.

        Returns:
            bool: True if the union of intervals is large enough, False otherwise.
        """
        coverage_angle = self.compute_angle_coverage(intervals)
        self.tree_coverage_angles[tree_id] = coverage_angle

        return coverage_angle >= self.reco_min_angle_coverage

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

    def generate_tree_placeholder(
        self, tree_id: int, cylinder_height: float = 5.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generates a mesh for a tree placeholder represented as a list of vertices and
        a list of triangle indices.
        The placeholder is a cylinder represents the tree axis as a cylinder with the
        radius that has been determined with hough fitting during clustering and
        alignment accordint to the axis determined during fitting.

        Args:
            tree_id (int): unique id of the tree
            cylinder_height (float, optional): height of the cylinder. Defaults to 5.0.

        Returns:
            np.ndarray: Nx3 array of vertices
            np.ndarray: Mx3 array of triangle indices
        """
        axis = self.trees[tree_id].axis
        lower_center = axis["center"]
        cylinder_radius = axis["radius"]
        cylinder_axis = axis["rot_mat"][2, :]
        cylinder_height = self.crop_upper_bound - self.crop_lower_bound
        upper_center = lower_center + cylinder_axis * cylinder_height

        bottom_circle = Circle(lower_center, cylinder_radius, cylinder_axis)
        top_circle = Circle(upper_center, cylinder_radius, cylinder_axis)

        return bottom_circle.genereate_cone_frustum_mesh(top_circle)

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

            if not self.check_angle_coverage([c[:2] for c in coverages], tree.id):
                continue
            self.tree_reco_flags[i]["angle_flag"] = True

            if not self.check_distance_coverage([c[2:] for c in coverages]):
                continue
            self.tree_reco_flags[i]["distance_flag"] = True

            rospy.loginfo(f"Reconstructing tree {tree.id}")
            reco_happened |= tree.reconstruct()

        return reco_happened

    def write_results(self, path: str):
        """Writes the tree data base to a csv file

        Args: path (str, optional): Path to the directory where the csv and xlsx file is
            saved.
        """
        if not os.path.exists(os.path.join(path, "csv")):
            os.makedirs(os.path.join(path, "csv"))
        if not os.path.exists(os.path.join(path, "xlsx")):
            os.makedirs(os.path.join(path, "xlsx"))
        file_name = (
            "TreeManagerState_"
            + f"{self._last_cluster_time.secs}_{self._last_cluster_time.nsecs}"
        )
        file_name_csv = file_name + ".csv"
        file_name_xlsx = file_name + ".xlsx"

        # write header
        columns = [
            "tree_id",
            "location_x",
            "location_y",
            "number_clusters",
            "coverage_angle",
            "DBH",
            "number_bends",
            "clear_wood",
        ]
        # write tree data
        data = []
        for tree in self.trees:
            # TODO replace with accurate calculation of DBH
            tree.DBH = tree.axis["radius"] * 2
            data.append(
                [
                    tree.id,
                    tree.axis["center"][0],
                    tree.axis["center"][1],
                    len(tree.clusters),
                    np.rad2deg(self.tree_coverage_angles[tree.id]),
                    tree.DBH,
                    tree.number_bends,
                    tree.clear_wood,
                ]
            )
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(
            os.path.join(path, "csv", file_name_csv), float_format="%.3f", index=False
        )
        df.to_excel(os.path.join(path, "xlsx", file_name_xlsx), index=False)
