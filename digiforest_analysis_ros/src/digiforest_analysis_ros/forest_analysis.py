#!/usr/bin/env python
# author: Leonard Freissmuth

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
from digiforest_analysis.tasks.tree_reconstruction import Tree
from digiforest_analysis.utils.timing import Timer


class ForestAnalysis:
    def __init__(self) -> None:
        self.read_params()
        self.setup_ros()
        self.ts = TreeSegmentation(clustering_method="voronoi")
        self.last_header = None

    def read_params(self):
        # Subscribers
        self._payload_cloud_topic = rospy.get_param(
            "~payload_cloud_topic", "/local_mapping/payload_in_local"
        )

        # Publishers
        self._tree_meshes_topic = rospy.get_param(
            "~tree_meshes_topic", "/realtime_trees/tree_meshes"
        )

        self._base_frame_id = rospy.get_param("~base_frame_id", "base")

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
            self._tree_meshes_topic, MarkerArray, queue_size=1
        )

        self._pub_debug_clusters = rospy.Publisher(
            "/realtime_trees/debug/clusters", PointCloud2, queue_size=1
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

    def payload_cloud_callback(self, data: PointCloud2):
        print("Received payload cloud")
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
            print("Could not get transform from odom to sensor", e)
            return

        timer = Timer()
        with timer("all"):
            with timer("conversion"):
                cloud = self.pc2_to_o3d(data)

            with timer("clusterting"):
                clusters = self.ts.process(cloud=cloud, cluster_dist=2)
                clouds = [c["cloud"].point.positions.numpy() for c in clusters]
                colors = [c["info"]["color"] for c in clusters]
                self.publish_colored_pointclouds(clouds, colors)

            with timer("fitting"):
                trees = []
                with Pool(processes=8) as pool:
                    all_points = [
                        cluster["cloud"].point.positions.numpy() for cluster in clusters
                    ]
                    trees = pool.map(Tree.from_cloud, all_points)

            with timer("meshing"):
                mesh_array = MarkerArray()
                for tree in trees:
                    tree.apply_transform(translation, rotation)
                    verts, tris = tree.generate_mesh()
                    mesh_msg = self.genereate_mesh_msg(verts, tris)
                    if mesh_msg is not None:
                        mesh_array.markers.append(mesh_msg)

            self._pub_tree_meshes.publish(mesh_array)

        print(timer)
