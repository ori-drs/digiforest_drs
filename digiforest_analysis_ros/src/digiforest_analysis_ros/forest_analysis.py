#!/usr/bin/env python
# author: Leonard Freissmuth

from copy import deepcopy
import os
import pickle
import numpy as np
import open3d as o3d
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from typing import List, Tuple
from threading import Lock

import rospy
import tf2_ros
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from nav_msgs.msg import Path
import message_filters
from vilens_slam_msgs.msg import PoseGraph
from geometry_msgs.msg import PoseStamped

from digiforest_analysis.tasks.tree_segmentation import TreeSegmentation
from digiforest_analysis.tasks.terrain_fitting import TerrainFitting
from digiforest_analysis.tasks.tree_reconstruction import Tree
from digiforest_analysis.utils.timing import Timer
from digiforest_analysis_ros.utils import pose2T, transform_clusters, efficient_inv

timer = Timer()


class ForestAnalysis:
    def __init__(self) -> None:
        self.read_params()
        self.setup_ros()

        self._terrain_fitter = TerrainFitting(
            sloop_smooth=self._terrain_smoothing,
            cloth_cell_size=self._terrain_cloth_cell_size,
        )

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
        self.tree_manager_lock = Lock()
        self.pose_graph_stamps_lock = Lock()

        rospy.on_shutdown(self.shutdown_routine)

    def read_params(self):
        self._debug_level = rospy.get_param("~debug_level", 0)

        # IDs
        self._base_frame_id = rospy.get_param("~id/base_frame", "base")
        self._odom_frame_id = rospy.get_param("~id/odom_frame", "odom_vilens")
        self._map_frame_id = rospy.get_param("~id/map_frame_", "map")

        # Subscribers
        self._payload_cloud_topic = rospy.get_param(
            "~topic/payload_cloud", "/local_mapping/payload_in_local"
        )
        self._payload_path_topic = rospy.get_param(
            "~topic/payload_path", "/local_mapping/path"
        )
        self._posegraph_update_topic = rospy.get_param(
            "~topic/posegraph_update", "/vilens_slam/pose_graph"
        )

        # Publishers
        self._tree_meshes_topic = rospy.get_param(
            "~tree_meshes_topic", "digiforest_forest_analysis/tree_meshes"
        )
        self._debug_clusters_topic = rospy.get_param(
            "~topic/debug_clusters", "digiforest_forest_analysis/debug/cluster_clouds"
        )
        self._debug_cluster_labels_topic = rospy.get_param(
            "~topic/debug_cluster_labels",
            "digiforest_forest_analysis/debug/cluster_labels",
        )

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

        # Terrain Fitting
        self._terrain_enabled = rospy.get_param("~terrain/enabled", False)
        self._terrain_cloth_cell_size = rospy.get_param("~terrain/cell_size", 2)
        self._terrain_smoothing = rospy.get_param("~terrain_fitting/smoothing", False)

        # Clustering
        self._clustering_crop_radius = rospy.get_param("~clustering/crop_radius", 15.0)
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

        self._sub_posegraph_update = rospy.Subscriber(
            self._posegraph_update_topic,
            PoseGraph,
            self.posegraph_changed_callback,
            queue_size=10,
        )

        self._sub_payload_cloud = message_filters.Subscriber(
            self._payload_cloud_topic, PointCloud2
        )
        self._sub_payload_info = message_filters.Subscriber(
            self._payload_path_topic, Path
        )
        self._path_cloud_synchronizer = message_filters.TimeSynchronizer(
            [self._sub_payload_cloud, self._sub_payload_info], 10
        )
        self._path_cloud_synchronizer.registerCallback(self.payload_with_path_callback)

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
        self._pub_cropped_pc = rospy.Publisher(
            "digiforest_forest_analysis/debug/cropped_pc",
            PointCloud2,
            queue_size=1,
            latch=True,
        )
        self._pub_terrain_model = rospy.Publisher(
            "digiforest_forest_analysis/debug/terrain_model",
            Marker,
            queue_size=1,
            latch=True,
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

    def crop_pc(
        self,
        cloud: o3d.t.geometry.PointCloud,
        sensor_pose: np.ndarray,
        radius: float,
    ):
        """Crops a point cloud using a max distance from the sensor. The sensor pose

        Args:
            cloud (o3d.t.geometry.PointCloud): Point cloud to be cropped
            sensor_pose (np.ndarray): 4x4 transformation matrix from sensor to odom
            radius (float): maximum distance from sensor in m
        """
        # Transform the point cloud to sensor frame
        cloud = cloud.transform(efficient_inv(sensor_pose))
        num_points_before = len(cloud.point.positions)
        # Calculate the distance from the sensor
        distances = np.linalg.norm(cloud.point.positions.numpy()[:, :2], axis=1)
        mask = distances <= radius
        cloud = cloud.select_by_mask(mask)
        cloud = cloud.transform(sensor_pose)
        if self._debug_level > 1:
            print(
                f"cropped cloud to {100.*mask.sum()/num_points_before:.3f} % -> {len(cloud.point.positions)} points"
            )

        return cloud

    def genereate_mesh_msg(
        self,
        vertices: np.ndarray,
        triangles: np.ndarray,
        id: int = None,
        frame_id: str = None,
        time_stamp=None,
        color=[140 / 255.0, 102 / 255.0, 87 / 255.0],
    ) -> Marker:
        if vertices.shape[0] == 0:
            return None

        mesh_msg = Marker()
        mesh_msg.header.frame_id = frame_id if frame_id else self._base_frame_id
        mesh_msg.header.stamp = time_stamp if time_stamp else self.last_pc_header.stamp
        mesh_msg.ns = "realtime_trees/meshes"
        mesh_msg.id = id if id is not None else np.random.randint(0, 100000)
        mesh_msg.type = Marker.TRIANGLE_LIST
        mesh_msg.action = Marker.ADD

        mesh_msg.pose.orientation.w = 1.0
        mesh_msg.scale.x = 1.0
        mesh_msg.scale.y = 1.0
        mesh_msg.scale.z = 1.0
        mesh_msg.color.a = 0.9
        mesh_msg.color.r = color[0]
        mesh_msg.color.g = color[1]
        mesh_msg.color.b = color[2]

        mesh_msg.points = [
            Point(x, y, z) for x, y, z in vertices[triangles].reshape(-1, 3).tolist()
        ]

        mesh_msg.colors = [
            mesh_msg.color,
        ] * len(mesh_msg.points)

        return mesh_msg

    def publish_pointclouds(
        self,
        pub: rospy.Publisher,
        clouds: list,
        colors: list = None,
        frame_id=None,
        time_stamp=None,
    ):
        # convert colors to single float32
        if not colors:
            colors = [np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)] * len(clouds)
        for i in range(len(colors)):
            color = colors[i]
            color = np.floor(np.array([color[2], color[1], color[0], 0.0]) * 255)
            color = np.frombuffer(color.astype(np.uint8).tobytes(), dtype=np.float32)
            colors[i] = color

        # Convert numpy arrays to pointcloud2 data
        header = rospy.Header()
        header.frame_id = frame_id if frame_id else self.last_pc_header.frame_id
        header.stamp = time_stamp if time_stamp else self.last_pc_header.stamp

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
        pub.publish(cloud)
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

    def payload_with_path_callback(self, point_cloud: PointCloud2, path_odom: Path):
        rospy.loginfo("Received payload cloud and path")
        self.last_pc_header = point_cloud.header

        # find index closest to center of path that also is in posegraph
        center_index = len(path_odom.poses) // 2
        with self.pose_graph_stamps_lock:
            center_index_found = False
            for i, step_size in enumerate(range(len(path_odom.poses))):
                center_index += step_size * (-1) ** i
                if path_odom.poses[center_index].header.stamp in self.pose_graph_stamps:
                    center_index_found = True
                    break
            if not center_index_found:
                rospy.logerr("Could not find any cloud's stamp in posegraph")
                return

        center_pose = path_odom.poses[center_index].pose
        center_stamp = path_odom.poses[center_index].header.stamp
        T_sensor2odom = pose2T(center_pose.orientation, center_pose.position)

        with timer("all"):
            with timer("conversion"):
                cloud = self.pc2_to_o3d(point_cloud)
                cloud = self.crop_pc(
                    cloud,
                    sensor_pose=T_sensor2odom,
                    radius=self._clustering_crop_radius,
                )

            if self._terrain_enabled:
                with timer("Terrain"):
                    terrain = self._terrain_fitter.process(cloud=cloud)
                    verts, tris = self._terrain_fitter.meshgrid_to_mesh(terrain)
                    terrain_marker = self.genereate_mesh_msg(
                        verts,
                        tris,
                        id=0,
                        color=[0.5, 0.5, 0.5],
                        frame_id=self._odom_frame_id,
                        time_stamp=rospy.Time.now(),
                    )
                    self._pub_terrain_model.publish(terrain_marker)
            else:
                terrain = None
            with timer("Clusterting"):
                clusters = self._tree_segmenter.process(
                    cloud=cloud,
                    cloth=terrain,
                    hough_filter_radius=self._clustering_hough_filter_radius,
                    crop_lower_bound=self._clustering_crop_lower_bound,
                    crop_upper_bound=self._clustering_crop_upper_bound,
                    max_cluster_radius=self._clustering_max_cluster_radius,
                    n_threads=self._clustering_n_threads,
                    point_fraction=self._clustering_distance_calc_point_fraction,
                )

                # send clusters to rviz
                self.publish_pointclouds(
                    self._pub_debug_clusters,
                    clouds=[c["cloud"].point.positions.numpy() for c in clusters],
                    colors=[c["info"]["color"] for c in clusters]
                    # colors = [np.random.rand(3) for c in clusters]
                )
            with timer("Tree Manager"):
                path_odom = np.array(
                    [
                        [p.pose.position.x, p.pose.position.y, p.pose.position.z]
                        for p in path_odom.poses
                    ]
                )
                # convert clusters into stamped sensor frame
                clusters = transform_clusters(
                    clusters, T_new2old=T_sensor2odom, time_stamp=center_stamp
                )
                print(clusters[0]["info"]["time_stamp"])
                with self.tree_manager_lock:
                    self._tree_manager.add_clusters_with_path(clusters, path_odom)

            with timer("Publishing"):
                self.publish_tree_manager_state()
                self._tree_manager.write_results(
                    "/home/ori/git/digiforest_drs/trees/logs"
                )

            if self._debug_level > 1:
                with timer("dumping clusters to disk"):
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
        rospy.loginfo("Finished payload cloud and path")

    def posegraph_changed_callback(self, posegraph_msg: PoseGraph):
        with self.pose_graph_stamps_lock:
            self.pose_graph_stamps = [
                pose.header.stamp for pose in posegraph_msg.path.poses
            ]
        with self.tree_manager_lock:
            self._tree_manager.update_poses(
                posegraph_msg.path.poses,
                self._tf_buffer,
                self._map_frame_id,
                self._odom_frame_id,
            )

    def publish_tree_manager_state(self):
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
            label_positions.append(tree.axis["transform"][:3, 3])

            verts, tris = tree.generate_mesh()
            mesh_messages.markers.append(
                self.genereate_mesh_msg(
                    verts, tris, id=tree.id, frame_id=self._odom_frame_id
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
        self._kd_tree: cKDTree = None

        self.num_trees = 0
        self._last_cluster_time = None

        self.capture_Ts_with_stamps: List[dict] = []

    def _update_kd_tree(self):
        """Updates the KD tree with the current tree centers"""
        centers = [tree.axis["transform"][:2, 3] for tree in self.trees]
        self._kd_tree = cKDTree(centers)

    def _new_tree_from_cluster(self, cluster: dict):
        """adds a new tree given a cluster.

        Args:
            cluster (dict): Dict as in the list returned by TreeSegmentation.process()
        """
        place_holder_height = self.crop_upper_bound - self.crop_lower_bound
        new_tree = Tree(self.num_trees, place_holder_height)
        new_tree.add_cluster(cluster)
        self.trees.append(new_tree)
        self.tree_reco_flags.append({"angle_flag": False, "distance_flag": False})
        self.tree_coverage_angles.append(0.0)
        self.num_trees += 1

    def distance_line_to_line(
        self, line_1: dict, line_2: dict, margin: float = 0
    ) -> float:
        """Calculates the minum distance between two axes. If the keyword "axis_length"
        is present in the dicts, the closest points are bounded to be between the basis
        poit of the axis and not further away from the basis point in the z direction
        of the given rot_mat than the axis_length. Otherwise, the closest points can be
        anywhere on the axis.
        If the key "axis_length" is not present in the dicts, the distance is not bound
            by constraining the closest points to be on the axis.

        Args:
            line1 (dict): Dict with the keys "transform" and optionally
                "axis_length" describing the first axis.
            line2 (dict): Dict with the keys "transform" and optionally
                "axis_length" describing the second axis.
            margin (float, optional): Margin in m added to the axis length on both
                sides. This increases the length so axes at different heights can be
                matched. Defaults to 0.

        Returns:
            float: minimum distance between axes
        """
        axis_pnt_1 = line_1["transform"][:3, 3]
        axis_pnt_2 = line_2["transform"][:3, 3]
        axis_pnt_1 -= margin * line_1["transform"][:3, 2]
        axis_pnt_2 -= margin * line_2["transform"][:3, 2]
        axis_dir_1 = line_1["transform"][:3, 2]
        axis_dir_2 = line_2["transform"][:3, 2]
        normal = np.cross(axis_dir_1, axis_dir_2)
        normal_length = np.linalg.norm(normal)
        if np.isclose(normal_length, 0.0):
            meetin_point_1 = axis_pnt_1
            # Part of Gram Schmidt
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
            axis_len_1 = line_1["axis_length"] + 2 * margin
            axis_len_2 = line_2["axis_length"] + 2 * margin
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

    def add_clusters(self, clusters_base: List[dict]):
        """This function checks every cluster (in BASE FRAME). If a tree close to the
        detected cluster already exists, the cluster is added to the tree. If no tree is
        close enough, a new tree is created. This function updates the KD tree.

        Args:
            clusters (List[dict]): List of clusters as returned by
                TreeSegmentation.process()
        """
        # time stamp and sensor transform are same for all clusters, so just take first
        self.capture_Ts_with_stamps.append(
            {
                "stamp": clusters_base[0]["info"]["time_stamp"],
                "pose": clusters_base[0]["info"]["sensor_transform"],
            }
        )

        self._last_cluster_time = clusters_base[0]["info"]["time_stamp"]

        if len(self.trees) == 0:
            # create new trees for all clusters and add them to the list
            for cluster in clusters_base:
                self._new_tree_from_cluster(cluster)
        else:
            # for all clusters check if tree at this coordinate already exists
            with timer("Finding Correspondences"):
                num_existing, num_new = 0, 0
                candidate_transforms_odom = [
                    c["info"]["sensor_transform"] @ c["info"]["axis"]["transform"]
                    for c in clusters_base
                ]
                candidate_centers = [t[:2, 3] for t in candidate_transforms_odom]
                _, existing_indices = self._kd_tree.query(candidate_centers)
                for i_candidate, i_existing in enumerate(existing_indices):
                    candidate_axis = deepcopy(
                        clusters_base[i_candidate]["info"]["axis"]
                    )
                    # transform candidate axis to odom frame
                    candidate_axis["transform"] = candidate_transforms_odom[i_candidate]
                    # candidate_axis["axis_length"] = axis_length
                    existing_axis = self.trees[i_existing].axis
                    # existing_axis["axis_length"] = axis_length

                    distance = self.distance_line_to_line(
                        candidate_axis, existing_axis, margin=5
                    )
                    if distance < self.distance_threshold:
                        self.trees[i_existing].add_cluster(clusters_base[i_candidate])
                        num_existing += 1
                    else:
                        self._new_tree_from_cluster(clusters_base[i_candidate])
                        num_new += 1

            rospy.loginfo(f"Found {num_existing} existing and {num_new} new clusters")

        self._update_kd_tree()
        self.try_reconstructions()

    def add_clusters_with_path(
        self,
        clusters_base: List[dict],
        path_odom: np.ndarray,
    ):
        """This function checks every cluster and performs the same as add_clusters().
        In addition, it calculates the covered angle and covered distance of the sensor
        to the tree axis for every cluster and adds this information to the
        dict cluster["info"].

        Args:
            clusters (List[dict]): List of clusters as returned by
                TreeSegmentation.process()
            path (np.ndarray): Nx7 array describing consecutive 7D poses of the sensor
                in the same coordinate frame as the clusters.
                The first three columns are the x, y, z coordinates of the position. The
                last four columns are the x, y, z, w quaternions describing orientation.
        """

        for cluster_base in clusters_base:
            angle_from, angle_to, d_min, d_max = self.calculate_coverage(
                cluster_base, path_odom
            )
            cluster_base["info"]["coverage"] = {
                "angle_from": angle_from,
                "angle_to": angle_to,
                "distance_min": d_min,
                "distance_max": d_max,
            }

        self.add_clusters(clusters_base)

    def update_poses(
        self,
        new_posegraph: List[PoseStamped],
        tf_buffer: tf2_ros.Buffer,
        map_frame_id: str = "map",
        odom_frame_id: str = "odom_vilens",
    ):
        """detects changes in the posegraph and updates the coordinate systems of all
        clusters in all trees.

        Args:
            posegraph_poses (List[PoseStamped]): List of of all poses of the posegraph
                where the timestamp is used as the pose's unique ID.
        """
        # find all poses that have changed since capture
        new_stamps = [pose.header.stamp for pose in new_posegraph]
        changed_poses = []
        for i, T_with_stamp in enumerate(self.capture_Ts_with_stamps):
            try:
                i_new = new_stamps.index(T_with_stamp["stamp"])
                new_stamped_pose_map = new_posegraph[i_new]
                map2odom = tf_buffer.lookup_transform(
                    target_frame=odom_frame_id,
                    source_frame=map_frame_id,
                    time=new_posegraph[-1].header.stamp,
                    timeout=rospy.Duration(0.1),
                )
                T_map2odom = pose2T(
                    map2odom.transform.rotation,
                    map2odom.transform.translation,
                )
                new_T_odom = T_map2odom @ pose2T(
                    new_stamped_pose_map.pose.orientation,
                    new_stamped_pose_map.pose.position,
                )
                if not np.allclose(new_T_odom, T_with_stamp["pose"]):
                    changed_poses.append(
                        {
                            "stamp": T_with_stamp["stamp"],
                            "pose": new_T_odom,
                        }
                    )
                    self.capture_Ts_with_stamps[i]["pose"] = new_T_odom
            except ValueError:
                print("Timestamp not found")
                print(self.capture_Ts_with_stamps)
                continue

        for changed_pose in changed_poses:
            for tree in self.trees:
                for cluster in tree.clusters:
                    if changed_pose["stamp"] == cluster["info"]["time_stamp"]:
                        cluster["info"]["sensor_transform"] = changed_pose["pose"]

    def calculate_coverage(
        self, cluster_base: dict, path_odom: np.ndarray
    ) -> Tuple[float]:
        """Calculates the covered angle of the sensor to the tree axis for a given
        cluster. The covered angle is given by two values defining the global extent
        of the arc around the z-axis in the x-y-plane bounding all rays from the
        center of the tree axis to all sensor poses. The parametrization is given by a
        start angle and end angle.
        The angles are given wrt. to the global x-axis and are in [0, 2*pi].

        Args:
            cluster_base (dict): Dict as in the list returned by TreeSegmentation.process().
                Mind that clusters must be in SENSOR FRAME.
            path_odom (np.ndarray): Nx7 array describing consecutive 7D poses of the
                sensor in the ODOM FRAME.
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
        for pose in path_odom:
            # calculate connecting vector between sensor pose and tree axis center
            cluster_transform_odom = (
                cluster_base["info"]["sensor_transform"]
                @ cluster_base["info"]["axis"]["transform"]
            )
            ray_vector = pose[:2] - cluster_transform_odom[:2, 3]
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
            if angle_to < angle_from:
                angle_accumulator[angle_from:] = True
                angle_accumulator[:angle_to] = True
            else:
                angle_accumulator[angle_from:angle_to] = True
        return np.deg2rad(angle_accumulator.sum())

    def try_reconstructions(self) -> bool:
        """Checks all trees in the data base if they satisfy the conditions to be
        reconstructed. If so, they are reconstrcuted.

        Returns:
            bool: True if at least one tree was newly reconstructed, False otherwise.
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

            coverage_angle = self.compute_angle_coverage([c[:2] for c in coverages])
            self.tree_coverage_angles[tree.id] = coverage_angle
            angle_flag = coverage_angle > self.reco_min_angle_coverage
            self.tree_reco_flags[i]["angle_flag"] = angle_flag

            distances = np.array([c[2:] for c in coverages])
            d_min = np.min(distances[:, 0])
            distance_flag = d_min < self.reco_min_distance
            self.tree_reco_flags[i]["distance_flag"] = distance_flag

            if (
                len(tree.clusters) == tree.num_clusters_after_last_reco
                and not tree.cosys_changed_after_last_reco
            ):
                continue

            if not angle_flag or not distance_flag:
                continue

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
                    tree.axis["transform"][0, 3],
                    tree.axis["transform"][1, 3],
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
