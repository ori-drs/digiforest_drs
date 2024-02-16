#!/usr/bin/env python
# author: Leonard Freissmuth

import colorsys
from copy import deepcopy
from functools import partial
import os
import pickle
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.stats import multivariate_normal
from typing import List, Tuple
from threading import Lock
from multiprocessing.pool import ThreadPool

import rospkg
import rospy
import tf2_ros
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from nav_msgs.msg import Path
import message_filters
from geometry_msgs.msg import PoseStamped
import trimesh

from digiforest_analysis.tasks.tree_reconstruction import Tree
from digiforest_analysis.utils.timing import Timer
from digiforest_analysis_ros.utils import (
    pose2T,
    clustering_worker_fun,
    apply_transform,
)
from digiforest_analysis.utils.meshing import meshgrid_to_mesh
from digiforest_analysis.utils.distances import distance_line_to_line

timer = Timer()
# ignore divide by zero warnings, code actively works with nans
np.seterr(divide="ignore", invalid="ignore")

# cd ~/catkin_ws/src/vilens_config/vc_stein_am_rhein/config/procman && rosrun procman_ros sheriff -l frontier.pmd
# cd ~/logs/logs_evo_finland/exp01 && rosbag play frontier_2023-05-01-14-* --clock --pause --topics /hesai/pandar /alphasense_driver_ros/imu  -r 1
# cd ~/logs/logs_stein_am_rhein/2023-08-08-16-14-48-exp03 && rosbag play *.bag --clock --pause --topics /hesai/pandar /alphasense_driver_ros/imu  -r 1


class ForestAnalysis:
    def __init__(self) -> None:
        # Initialize path for outputs
        package_path = rospkg.RosPack().get_path("digiforest_analysis_ros")
        self.base_output_path = os.path.join(package_path, "output")
        os.makedirs(self.base_output_path, exist_ok=True)

        self.read_params()
        self.setup_ros()

        self._clustering_pool = ThreadPool(processes=self._clustering_n_threads)

        self._tree_manager = TreeManager(
            self._distance_threshold,
            self._reco_min_angle_coverage,
            self._reco_min_distance,
            self._clustering_crop_upper_bound,
            self._clustering_crop_lower_bound,
            self._terrain_confidence_stds,
            self._terrain_confidence_sensor_weight,
            self._terrain_use_embree,
            self._generate_canopy_mesh,
            output_path=self.base_output_path,
        )

        self.last_pc_header = None
        self.pc_counter = 0
        self.tree_manager_lock = Lock()

        rospy.on_shutdown(self.shutdown_routine)

    def read_params(self):
        self._debug_level = rospy.get_param("~debug_level", 0)

        # IDs
        self._base_frame_id = rospy.get_param("~id/base_frame", "base_vilens_optimized")
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
        self._stem_meshes_topic = rospy.get_param(
            "~topic/stem_meshes", "/digiforest_forest_analysis/stem_meshes"
        )
        self._canopy_meshes_topic = rospy.get_param(
            "~topic/canopy_meshes", "/digiforest_forest_analysis/canopy_meshes"
        )
        self._debug_clusters_topic = rospy.get_param(
            "~topic/debug_clusters", "/digiforest_forest_analysis/debug/cluster_clouds"
        )
        self._terrain_topic = rospy.get_param(
            "~topic/terrain_model", "/digiforest_forest_analysis/terrain"
        )
        self._tree_point_clouds_topic = rospy.get_param(
            "~topic/tree_point_clouds", "digiforest_forest_analysis/tree_point_clouds"
        )

        # Tree Manager
        self._distance_threshold = rospy.get_param(
            "~tree_manager/distance_threshold", 0.5
        )
        self._reco_min_angle_coverage = np.deg2rad(
            rospy.get_param("~tree_manager/reco_min_angle_coverage", 180)
        )
        self._reco_min_distance = rospy.get_param(
            "~tree_manager/reco_min_distance", 4.0
        )
        self._terrain_confidence_stds = rospy.get_param(
            "~tree_manager/confidence_stds", [3, 5, 5]
        )
        self._terrain_confidence_sensor_weight = rospy.get_param(
            "~tree_manager/confidence_sensor_weight", 0.9999
        )
        self._terrain_use_embree = rospy.get_param("~tree_manager/use_embree", True)

        # Terrain Fitting
        self._terrain_enabled = rospy.get_param("~terrain/enabled", True)
        self._terrain_smoothing = rospy.get_param("~terrain_fitting/smoothing", False)
        self._terrain_cloth_cell_size = rospy.get_param("~terrain/cell_size", 1)

        # Clustering
        self._clustering_crop_radius = rospy.get_param("~clustering/crop_radius", 40.0)
        self._clustering_hough_filter_radius = rospy.get_param(
            "~clustering/hough_filter_radius", 0.1
        )
        self._clustering_crop_lower_bound = rospy.get_param(
            "~clustering/crop_lower_bound", 4.0
        )
        self._clustering_crop_upper_bound = rospy.get_param(
            "~clustering/crop_upper_bound", 8.0
        )
        self._clustering_max_cluster_radius = rospy.get_param(
            "~clustering/max_cluster_radius", 5.0
        )
        self._clustering_n_threads = rospy.get_param("~clustering/n_threads", 4)
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
        self._fitting_max_height = rospy.get_param("~fitting/max_height", 15.0)
        self._fitting_save_debug_results = rospy.get_param(
            "~fitting/save_debug_results", False
        )
        self._fitting_n_threads = rospy.get_param("~fitting/n_threads", 1)
        self._generate_canopy_mesh = rospy.get_param("~fitting/generate_canopy", True)

        # Internals
        self._tf_buffer = None
        self._tf_listener = None

    def setup_ros(self):
        # imported here to avoid conflicts with the multiprocessing....
        from vilens_slam_msgs.msg import PoseGraph

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
        self._pub_stem_meshes = rospy.Publisher(
            self._stem_meshes_topic, MarkerArray, queue_size=1, latch=True
        )
        self._pub_debug_clusters = rospy.Publisher(
            self._debug_clusters_topic, PointCloud2, queue_size=1, latch=True
        )
        self._pub_terrain_model = rospy.Publisher(
            self._terrain_topic, Marker, queue_size=1, latch=True
        )
        self._pub_canopy_meshes = rospy.Publisher(
            self._canopy_meshes_topic, MarkerArray, queue_size=1, latch=True
        )
        self._pub_tree_clusters = rospy.Publisher(
            self._tree_point_clouds_topic, PointCloud2, queue_size=1, latch=True
        )
        self._pub_cluster_labels = rospy.Publisher(
            "/digiforest_forest_analysis/debug/cluster_labels",
            MarkerArray,
            queue_size=1,
            latch=True,
        )
        self._pub_cropped_pc = rospy.Publisher(
            "/digiforest_forest_analysis/debug/cropped_pc",
            PointCloud2,
            queue_size=1,
            latch=True,
        )

    def generate_mesh_msg(
        self,
        vertices: np.ndarray,
        triangles: np.ndarray,
        id: int = None,
        frame_id: str = None,
        time_stamp=None,
        color=[140 / 255.0, 102 / 255.0, 87 / 255.0],
        alpha=1.0,
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
        mesh_msg.color.a = alpha
        mesh_msg.color.r = color[0]
        mesh_msg.color.g = color[1]
        mesh_msg.color.b = color[2]

        mesh_msg.points = [
            Point(x, y, z) for x, y, z in vertices[triangles].reshape(-1, 3).tolist()
        ]

        # mesh_msg.colors = [
        #     mesh_msg.color,
        # ] * len(mesh_msg.points)

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

    def publish_cluster_labels(
        self, labels: List[str], locations: List[np.ndarray], frame_id=None
    ):
        marker_array = MarkerArray()
        for i, (label, location) in enumerate(zip(labels, locations)):
            marker = Marker()
            marker.header.frame_id = (
                frame_id if frame_id else self.last_pc_header.frame_id
            )
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

    def _clustering_worker_finished_callback(
        self, result, path_odom: np.ndarray, pc_counter: int
    ):
        print("clustering worker finished callback")
        clusters, terrain = result
        with timer("cwc"):
            with timer("cwc/publishing_clustering"):
                clouds = [
                    c["cloud"]
                    .clone()
                    .transform(c["info"]["T_sensor2map"])
                    .point.positions.numpy()
                    for c in clusters
                ]
                self.publish_pointclouds(
                    self._pub_debug_clusters,
                    clouds=clouds,
                    colors=[c["info"]["color"] for c in clusters],
                    frame_id=self._map_frame_id,
                )
            with timer("cwc/tree_manager"):
                path_odom_pos = np.array(
                    [
                        [p.pose.position.x, p.pose.position.y, p.pose.position.z]
                        for p in path_odom.poses
                    ]
                )
                path_odom_ori = [
                    Rotation.from_quat(
                        [
                            p.pose.orientation.x,
                            p.pose.orientation.y,
                            p.pose.orientation.z,
                            p.pose.orientation.w,
                        ]
                    )
                    for p in path_odom.poses
                ]
                pose_odom2map = self._tf_buffer.lookup_transform(
                    self._map_frame_id,
                    self._odom_frame_id,
                    rospy.Time(0),
                    rospy.Duration(1),
                )
                quat_odom2map = [
                    pose_odom2map.transform.rotation.x,
                    pose_odom2map.transform.rotation.y,
                    pose_odom2map.transform.rotation.z,
                    pose_odom2map.transform.rotation.w,
                ]
                R_odom2map = Rotation.from_quat(quat_odom2map)
                t_odom2map = np.array(
                    [
                        pose_odom2map.transform.translation.x,
                        pose_odom2map.transform.translation.y,
                        pose_odom2map.transform.translation.z,
                    ]
                )
                R_map2sensor = Rotation.from_matrix(
                    clusters[0]["info"]["T_sensor2map"][:3, :3].T
                )
                t_map2sensor = (
                    -clusters[0]["info"]["T_sensor2map"][:3, :3].T
                    @ clusters[0]["info"]["T_sensor2map"][:3, 3]
                )
                path_map = np.stack(
                    [
                        np.concatenate(
                            (
                                R_odom2map.apply(p) + t_odom2map,
                                (R_odom2map * o).as_quat(),
                            )
                        )
                        for p, o in zip(path_odom_pos, path_odom_ori)
                    ],
                    axis=0,
                )
                path_sensor = np.stack(
                    [
                        np.concatenate(
                            (
                                R_map2sensor.apply(p) + t_map2sensor,
                                (R_map2sensor * Rotation.from_quat(o)).as_quat(),
                            )
                        )
                        for p, o in zip(path_map[:, :3], path_map[:, 3:])
                    ]
                )

                with self.tree_manager_lock:
                    self._tree_manager.add_clusters_with_path(clusters, path_sensor)
                    verts_odom, tris = meshgrid_to_mesh(terrain)
                    verts_map = R_odom2map.apply(verts_odom) + t_odom2map
                    self._tree_manager.add_terrain(
                        verts_map,
                        tris,
                        clusters[0]["info"]["time_stamp"],
                        clusters[0]["info"]["T_sensor2map"],
                        path_map,
                        self._terrain_cloth_cell_size,
                    )

            with timer("cwc/publishing_tree_manager"):
                self.publish_tree_manager_state()
                self._tree_manager.write_results(
                    f"{self.base_output_path}/trees/logs"
                )
            if self._debug_level > 1:
                with timer("cwc/dumping_clusters"):
                    directory = os.path.join(f"{self.base_output_path}/trees", str(self.last_pc_header.stamp))
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

        rospy.loginfo(f"Finished processing payload cloud {pc_counter}:\n{timer}")

    def payload_with_path_callback(self, cloud_odom: PointCloud2, path_odom: Path):
        self.pc_counter += 1
        rospy.loginfo(f"Received payload cloud {self.pc_counter} and path")
        self.last_pc_header = cloud_odom.header

        self._clustering_pool.apply_async(
            clustering_worker_fun,
            kwds={
                "tf_buffer": self._tf_buffer,
                "cloud_msg": cloud_odom,
                "path_odom": path_odom,
                "pose_graph_stamps": self.pose_graph_stamps,
                "terrain_enabled": self._terrain_enabled,
                "terrain_smoothing": self._terrain_smoothing,
                "terrain_cloth_cell_size": self._terrain_cloth_cell_size,
                "clustering_crop_radius": self._clustering_crop_radius,
                "hough_filter_radius": self._clustering_hough_filter_radius,
                "crop_lower_bound": self._clustering_crop_lower_bound,
                "crop_upper_bound": self._clustering_crop_upper_bound,
                "max_cluster_radius": self._clustering_max_cluster_radius,
                "point_fraction": self._clustering_distance_calc_point_fraction,
                "debug_level": self._debug_level,
                "map_frame_id": self._map_frame_id,
                "odom_frame_id": self._odom_frame_id,
            },
            callback=partial(
                self._clustering_worker_finished_callback,
                path_odom=path_odom,
                pc_counter=self.pc_counter,
            ),
        )
        print(f"set everything up for payload cloud {self.pc_counter}")

    def posegraph_changed_callback(self, posegraph_msg):
        self.pose_graph_stamps = [
            pose.header.stamp for pose in posegraph_msg.path.poses
        ]
        with self.tree_manager_lock:
            self._tree_manager.update_poses(posegraph_msg.path.poses)

    def publish_tree_manager_state(self):
        label_texts = []
        label_positions = []
        mesh_messages = MarkerArray()
        canopy_messages = MarkerArray()
        print("Publishing Tree Manager State")

        # trees with labels
        tree_clouds = []
        tree_colors = []
        for tree, reco_flags, coverage_angle in zip(
            self._tree_manager.trees,
            self._tree_manager.tree_reco_flags,
            self._tree_manager.tree_coverage_angles,
        ):
            if len(tree.clusters) < 3:
                continue
            label_texts.append(
                f"##### tree{str(tree.id).zfill(3)} #####\n"
                + f"angle:     {np.rad2deg(coverage_angle):.0f} deg\n"
                + f"angle_flag:    {reco_flags['angle_flag']}\n"
                + f"distance_flag:  {reco_flags['distance_flag']}\n"
            )
            label_positions.append(tree.axis["transform"][:3, 3])

            verts, tris = tree.generate_mesh()
            mesh_messages.markers.append(
                self.generate_mesh_msg(
                    verts, tris, id=tree.id, frame_id=self._map_frame_id
                )
            )

            # sample random hue and for all clusters random brighntess between 0.5 and 1
            hue = np.random.rand()
            lightness = [
                np.random.rand() * 0.8 + 0.1 for _ in range(len(tree.clusters))
            ]
            tree_colors.extend([colorsys.hls_to_rgb(hue, l, 1.0) for l in lightness])
            # add voxel downsampled pcs to tree_clouds
            tree_clouds.extend(
                [
                    cluster["cloud"]
                    .clone()
                    .transform(cluster["info"]["T_sensor2map"])
                    .point.positions.numpy()
                    for cluster in tree.clusters
                ]
            )
            # tree_clouds.append(tree.points)
            # tree_colors.append(colorsys.hls_to_rgb(hue, 0.6, 1.0))
            if tree.canopy_mesh is not None:
                message = self.generate_mesh_msg(
                    tree.canopy_mesh["verts"],
                    tree.canopy_mesh["tris"],
                    id=tree.id,
                    frame_id=self._map_frame_id,
                    color=[150 / 255, 217 / 255, 121 / 255],
                    alpha=0.5,
                )
                canopy_messages.markers.append(message)

        self._pub_stem_meshes.publish(mesh_messages)
        self._pub_canopy_meshes.publish(canopy_messages)
        self.publish_cluster_labels(label_texts, label_positions, self._map_frame_id)
        if len(tree_clouds) > 0:
            self.publish_pointclouds(
                self._pub_tree_clusters, tree_clouds, tree_colors, self._map_frame_id
            )

        # Terrain
        verts, tris = self._tree_manager.get_terrain()
        self._pub_terrain_model.publish(
            self.generate_mesh_msg(
                verts,
                tris,
                frame_id=self._map_frame_id,
                time_stamp=self.last_pc_header.stamp,
                color=[0.5, 0.5, 0.5],
                alpha=1.0,
                id=0,
            )
        )

    def shutdown_routine(self, *args):
        """Executes the operations before killing the mission analysis procedures"""       
        path = f"{self.base_output_path}/trees/logs/raw/"
        os.makedirs(path, exist_ok=True)

        # for tree in self._tree_manager.trees:
        #     # write tree as pickle
        #     with open(path + f"tree{str(tree.id).zfill(3)}.pkl", "wb") as file:
        #         pickle.dump(tree, file)
        # save terrain model as pickle
        with open(path + "tree_manager.pkl", "wb") as file:
            pickle.dump(self._tree_manager, file)
        self._clustering_pool.close()
        rospy.loginfo("Digiforest Analysis node stopped!")


class TreeManager:
    def __init__(
        self,
        distance_threshold: float = 0.1,
        reco_min_angle_coverage: float = 1.5 * np.pi,
        reco_min_distance: float = 4.0,
        crop_upper_bound: float = 1.0,
        crop_lower_bound: float = 3.0,
        terrain_confidence_stds: list = [3, 10, 10],
        terrain_confidence_sensor_weight: float = 0.9999,
        terrain_use_embree: bool = True,
        generate_canopy_mesh: bool = True,
        output_path: str = "/tmp"
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
            terrain_confidence_stds (float, optional): Variances of confidence model.
                Moving the sensor along its x axis to the ground, this variances along
                the sensor axis is used to evaluate the extent of the confidence region.
            terrain_confidence_sensor_weight  (list, optional): weighs the path-based
                confidence model wrt just the distance of the ground mesh from the path.
            use_embree (bool, optional): Flag to use Embree for ray tracing (faster but
                inferior quality). Defaults to True.
            generate_canopy_mesh (bool, optional): Flag to generate the canopy mesh.
                Defaults to True.
        """
        self.distance_threshold = distance_threshold
        self.reco_min_angle_coverage = reco_min_angle_coverage
        self.reco_min_distance = reco_min_distance
        self.crop_upper_bound = crop_upper_bound
        self.crop_lower_bound = crop_lower_bound
        self.terrain_confidence_stds = terrain_confidence_stds
        self.terrain_confidence_sensor_weight = terrain_confidence_sensor_weight
        self.use_embree = terrain_use_embree
        self.generate_canopy_mesh = generate_canopy_mesh
        self.base_output_path = output_path

        self.tree_reco_flags: List[List[bool]] = []
        self.tree_coverage_angles: List[float] = []
        self.trees: List[Tree] = []
        self._kd_tree: cKDTree = None

        self.num_trees = 0
        self._last_cluster_time = None

        self.terrains = []

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
                "pose": clusters_base[0]["info"]["T_sensor2map"],
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
                    c["info"]["T_sensor2map"] @ c["info"]["axis"]["transform"]
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
                    existing_axis = self.trees[i_existing].axis

                    distance = distance_line_to_line(
                        candidate_axis, existing_axis, clip_heights=[0, 10]
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
        path_sensor: np.ndarray,
    ):
        """This function checks every cluster and performs the same as add_clusters().
        In addition, it calculates the covered angle and covered distance of the sensor
        to the tree axis for every cluster and adds this information to the
        dict cluster["info"].

        Args:
            clusters (List[dict]): List of clusters as returned by
                TreeSegmentation.process()
            path (np.ndarray): Nx3 array describing consecutive 7D poses of the sensor
                in the same coordinate frame as the clusters.
                The columns are the x, y, z coordinates of the position.
        """

        for cluster_base in clusters_base:
            angle_from, angle_to, d_min, d_max = self.calculate_coverage(
                cluster_base, path_sensor
            )
            cluster_base["info"]["coverage"] = {
                "angle_from": angle_from,
                "angle_to": angle_to,
                "distance_min": d_min,
                "distance_max": d_max,
            }

        self.add_clusters(clusters_base)

    def get_terrain(self) -> np.ndarray:
        """Returns the terrain of the entire map in a 2.5D meshgrid format.

        Returns:
            np.ndarray: MxNx3 grid where the last axis are for x, y, z and the other two
                axes are the x and y coordinates of the gridpoints.
        """

        # find most recent sensor transforms using time stamp of terrain maps
        sensor_transforms = []
        terrains = []
        pose_graph_stamps = [t["stamp"] for t in self.capture_Ts_with_stamps]
        for terrain in self.terrains:
            if terrain["time_stamp"] not in pose_graph_stamps:
                print(f"terrain time stamp {terrain['time_stamp']} not in posegraph")
                continue
            else:
                terrains.append(terrain)
                T_sensor2map = self.capture_Ts_with_stamps[
                    pose_graph_stamps.index(terrain["time_stamp"])
                ]["pose"]
                sensor_transforms.append(T_sensor2map)

        # aggregate database entries into lists
        meshes_map = [
            deepcopy(terrain["mesh_sensor"]).apply_transform(terrain["T_sensor2map"])
            for terrain in terrains
        ]

        bboxes = np.asarray(
            [
                (mesh.vertices.min(axis=0), mesh.vertices.max(axis=0))
                for mesh in meshes_map
            ]
        )

        # generate a regular grid of query points for the final terrain map
        bbox = [bboxes.reshape(-1, 3).min(axis=0), bboxes.reshape(-1, 3).max(axis=0)]
        (query_X, query_Y,) = np.meshgrid(
            np.arange(bbox[0][0], bbox[1][0], self.terrains[0]["cell_size"]),
            np.arange(bbox[0][1], bbox[1][1], self.terrains[0]["cell_size"]),
        )
        query_positions = np.stack(
            (query_X, query_Y, -100 * np.ones_like(query_X)), axis=2
        )
        query_positions = query_positions.reshape(-1, 3)
        query_rays = np.zeros_like(query_positions)
        query_rays[:, 2] = 1.0

        # aggregate all maps into a single tensor of map heights and weights
        heights = np.ones((*query_X.shape, len(self.terrains)))
        weights = np.zeros((*query_X.shape, len(self.terrains)))
        for i, (terrain, mesh_map) in enumerate(zip(self.terrains, meshes_map)):
            mesh_weights = terrain["vertex_weights"]
            # querying all rays is not slower than preselecting the rays
            # inside the meshe's bounding box.
            # find indices in triangles where rays hit mesh
            tri_inds = mesh_map.ray.intersects_first(query_positions, query_rays)
            verts_mask = tri_inds != -1
            tri_inds = tri_inds[verts_mask]
            # just take the center of the tri as an approximation
            intersects = mesh_map.vertices[mesh_map.faces[tri_inds]].mean(axis=1)
            heights[verts_mask.reshape(query_X.shape), i] = intersects[:, 2]
            weights[verts_mask.reshape(query_X.shape), i] = mesh_weights[
                mesh_map.faces[tri_inds]
            ].mean(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            heights = np.sum(heights * weights, axis=2) / weights.sum(axis=2)

        # convert to vertices and triangles
        mgrid = np.stack((query_X, query_Y, heights), axis=2)
        verts, tris = meshgrid_to_mesh(mgrid)

        # remove verts with nan
        nan_mask = np.isnan(verts[:, 2])
        # remove verts where there are no maps contributing strongly
        weights_mask = weights.max(axis=2) < 0.2 * (
            1 - self.terrain_confidence_sensor_weight
        )

        # # for debugging puproposes, use this to tune the value of the threshold
        # from matplotlib import pyplot as plt
        # # plt.imshow(weights_mask.reshape(query_X.shape))
        # plt.imshow(weights_mask.reshape(query_X.shape)*1 + nan_mask.reshape(query_X.shape)*1)
        # # plt.imshow(weights.max(axis=2))
        # plt.colorbar()
        # plt.show()

        verts_mask = np.logical_or(nan_mask, weights_mask.reshape(-1))
        remove_indices = np.where(verts_mask)[0]
        tri_mask = np.any(np.isin(tris, remove_indices), axis=1)
        tris = tris[~tri_mask]

        # flip normals
        tris = np.flip(tris, axis=1)

        return verts, tris

    def add_terrain(
        self,
        terrain_mesh_verts_map: np.ndarray,
        terrain_mesh_tris: np.ndarray,
        time_stamp: rospy.Time,
        T_sensor2map: np.ndarray,
        path_map: np.ndarray,
        cell_size: float,
    ):
        """Adds the local terrain to the terrain accumulator. Also updates the
        basepoints of all trees.

        Args:
            terrain (np.ndarray): Nx3 array of points describing the terrain
        """

        def measurement_likelihood(verts, path):
            # calculate distances from sensor to all verts
            rot_mats = np.array([Rotation.from_quat(p[3:]).as_matrix() for p in path])
            means = 9.0 * rot_mats[:, :, 0] + path[:, :3]
            variances = (
                rot_mats
                @ np.diag(self.terrain_confidence_stds) ** 2
                @ rot_mats.transpose([0, 2, 1])
            )
            weights_sensor = [
                multivariate_normal.pdf(verts, mean, var)
                for mean, var in zip(means, variances)
            ]
            weights_sensor = np.array(weights_sensor).sum(axis=0)
            weights_sensor /= weights_sensor.max()
            weights_distance = np.linalg.norm(
                verts[:, None, :] - path[:, :3], axis=2
            ).min(axis=1)
            max_distance = weights_distance.max()
            weights_distance = np.sqrt(0.5) * max_distance - weights_distance
            weights_distance /= np.sqrt(0.5) * max_distance
            weights_distance = np.clip(weights_distance, 0, 1)
            weights = (
                (1 - self.terrain_confidence_sensor_weight) * weights_distance
                + self.terrain_confidence_sensor_weight * weights_sensor
            )

            # # for debugging puproposes, use this for visualization
            # from matplotlib import pyplot as plt
            # ax = plt.subplot(projection="3d")
            # ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], c=weights, vmin=0, vmax=1, alpha=1, zorder=-1)
            # # ax.scatter(means[:, 0], means[:, 1], means[:, 2], color="green", s=20)
            # ax.set_box_aspect([1.0, 1.0, 1.0])
            # for m, r in zip(path, rot_mats):
            #     ax.plot(
            #         [m[0], m[0] + r[0, 0]],
            #         [m[1], m[1] + r[1, 0]],
            #         [m[2], m[2] + r[2, 0]],
            #         color="red",
            #         zorder=2,
            #         linewidth=2.
            #     )
            #     ax.plot(
            #         [m[0], m[0] + r[0, 1]],
            #         [m[1], m[1] + r[1, 1]],
            #         [m[2], m[2] + r[2, 1]],
            #         color="green",
            #         zorder=2,
            #         linewidth=2.
            #     )
            #     ax.plot(
            #         [m[0], m[0] + r[0, 2]],
            #         [m[1], m[1] + r[1, 2]],
            #         [m[2], m[2] + r[2, 2]],
            #         color="blue",
            #         zorder=2,
            #         linewidth=2.
            #     )
            # ax.plot(path[:, 0], path[:, 1], path[:, 2], color="red")
            # set_axes_equal(ax)
            # plt.show()

            return weights

        weights = measurement_likelihood(terrain_mesh_verts_map, path_map)
        terrain_mesh_sensor_verts = apply_transform(
            terrain_mesh_verts_map,
            T_sensor2map[:3, 3],
            T_sensor2map[:3, :3],
            inverse=True,
        )

        terrain_mesh_sensor = trimesh.Trimesh(
            terrain_mesh_sensor_verts.astype(np.float64),
            terrain_mesh_tris.astype(np.int64),
            use_embree=self.use_embree,
        )

        retval = {
            "mesh_sensor": terrain_mesh_sensor,
            "time_stamp": time_stamp,
            "T_sensor2map": T_sensor2map,
            "vertex_weights": weights,
            "path": path_map,
            "cell_size": cell_size,
        }

        with open(
            f"{self.base_output_path}/output/terrains/terrain_{time_stamp.to_nsec()}.pkl",
            "wb",
        ) as file:
            pickle.dump(retval, file)
        self.terrains.append(retval)

    def update_poses(self, new_posegraph: List[PoseStamped]):
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
                new_T_map = pose2T(
                    new_stamped_pose_map.pose.orientation,
                    new_stamped_pose_map.pose.position,
                )

                if not np.allclose(new_T_map, T_with_stamp["pose"]):
                    changed_poses.append(
                        {
                            "stamp": T_with_stamp["stamp"],
                            "pose": new_T_map,
                        }
                    )
                    self.capture_Ts_with_stamps[i]["pose"] = new_T_map
            except ValueError:
                print("Timestamp not found")
                print(self.capture_Ts_with_stamps)
                continue

        for changed_pose in changed_poses:
            for tree in self.trees:
                for cluster in tree.clusters:
                    if changed_pose["stamp"] == cluster["info"]["time_stamp"]:
                        cluster["info"]["T_sensor2map"] = changed_pose["pose"]

    def calculate_coverage(
        self,
        cluster_sensor: dict,
        path_sensor: np.ndarray,
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
        for pose in path_sensor:
            # calculate connecting vector between sensor pose and tree axis center
            ray_vector = pose[:2] - cluster_sensor["info"]["axis"]["transform"][:2, 3]
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
            reco_sucess = tree.reconstruct2()
            if reco_sucess:
                tree.num_clusters_after_last_reco = len(tree.clusters)
                tree.cosys_changed_after_last_reco = False
                if self.generate_canopy_mesh:
                    tree.generate_canopy()
            reco_happened |= reco_sucess

        return reco_happened

    def write_results(self, path: str):
        """Writes the tree data base to a csv file

        Args: path (str, optional): Path to the directory where the csv and xlsx file is
            saved.
        """
        if not os.path.exists(os.path.join(path, "csv")):
            os.makedirs(os.path.join(path, "csv"), exist_ok=True)
        if not os.path.exists(os.path.join(path, "xlsx")):
            os.makedirs(os.path.join(path, "xlsx"), exist_ok=True)
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
