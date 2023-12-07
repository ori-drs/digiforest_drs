#!/usr/bin/env python
# author: Leonard Freissmuth

from functools import partial
import os
import pickle
from typing import List
import numpy as np
import open3d as o3d
from multiprocessing import Pool

import rospy
import tf2_ros
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

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
        self.last_header = None

        rospy.on_shutdown(self.shutdown_routine)

    def read_params(self):
        # Subscribers
        self._payload_cloud_topic = rospy.get_param(
            "~payload_cloud_topic", "/local_mapping/payload_in_local"
        )

        # Publishers
        self._tree_meshes_topic = rospy.get_param(
            "~tree_meshes_topic", "/realtime_trees/tree_meshes"
        )
        self._debug_clusters_topic = rospy.get_param(
            "~debug_clusters_topic", "/realtime_trees/debug/clusters"
        )
        self._debug_cluster_labels_topic = rospy.get_param(
            "~debug_cluster_labels_topic", "/realtime_trees/debug/cluster_labels"
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

        # Subscribers
        self._sub_payload_cloud = rospy.Subscriber(
            self._payload_cloud_topic, PointCloud2, self.payload_cloud_callback
        )

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
        mesh_msg.header.stamp = self.last_header.stamp  # rospy.Time.now()
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
        header.frame_id = self.last_header.frame_id
        header.stamp = self.last_header.stamp

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
            marker.header.frame_id = self.last_header.frame_id
            marker.header.stamp = self.last_header.stamp
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

    def payload_cloud_callback(self, data: PointCloud2):
        rospy.loginfo("Received payload cloud")
        self.last_header = data.header

        try:
            odom2sensor = self._tf_buffer.lookup_transform(
                self._base_frame_id, self.last_header.frame_id, self.last_header.stamp
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
                cloud = self.pc2_to_o3d(data)

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
                clusters = self.ts.process(
                    cloud=cloud,
                    cloth=cloth,
                    hough_filter_radius=self._clustering_hough_filter_radius,
                    crop_lower_bound=self._clustering_crop_lower_bound,
                    crop_upper_bound=self._clustering_crop_upper_bound,
                    max_cluster_radius=self._clustering_max_cluster_radius,
                    n_threads=self._clustering_n_threads,
                )
                clouds = [c["cloud"].point.positions.numpy() for c in clusters]
                colors = [c["info"]["color"] for c in clusters]
                # colors = [np.random.rand(3) for c in clusters]
                self.publish_colored_pointclouds(clouds, colors)
                self.publish_cluster_labels(
                    labels=[f"tree{str(c['info']['id']).zfill(3)}" for c in clusters],
                    locations=[c["info"]["axis"]["center"] for c in clusters],
                )

            with timer("Fitting"):
                tree_from_cluster = partial(
                    Tree.from_cluster,
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
                    directory = os.path.join("trees", str(self.last_header.stamp))
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
