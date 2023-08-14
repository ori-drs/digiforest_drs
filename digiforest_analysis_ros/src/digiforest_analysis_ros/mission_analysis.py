#!/usr/bin/env python
# author: Matias Mattamala

import os
import rospy
import tf2_ros

from datetime import datetime
from dynamic_reconfigure.msg import Config
from geometry_msgs.msg import (
    PoseStamped,
    PoseWithCovarianceStamped,
    TwistStamped,
    TwistWithCovarianceStamped,
)
from nav_msgs.msg import Path
from ros2raw.converters import (
    PathConverter,
    PoseStampedConverter,
    ROSMsgConverter,
    TransformStampedConverter,
    TwistStampedConverter,
)


class MissionAnalysis:
    def __init__(self):
        """Constructor"""
        self.read_params()
        self.setup_ros()
        self.set_internals()

        # Set kill signals
        rospy.on_shutdown(self.shutdown_routine)

    def read_params(self):
        """Read parameters from parameter server"""

        # Subscription topics
        self._slam_graph_topic = rospy.get_param(
            "~slam_graph_topic", "/vilens_slam/slam_poses"
        )

        self._state_twist = rospy.get_param("~twist_topic", "/vilens/twist_optimized")
        self._state_pose = rospy.get_param("~pose_topic", "/vilens/pose_optimized")

        # Optional topics
        self._reference_twist_topic = rospy.get_param(
            "~reference_twist_topic", "/local_guidance_path_follower/twist"
        )

        self._operator_twist_topic = rospy.get_param(
            "~operator_twist_topic", "/motion_reference/command_twist"
        )

        self._local_planner_param_topic = rospy.get_param(
            "~local_planner_param_topic", "/field_local_planner/parameter_updates"
        )

        self._rmp_param_topic = rospy.get_param(
            "~rmp_param_topic", "/field_local_planner/rmp/parameter_updates"
        )

        self._tf_reference_frames = rospy.get_param(
            "~tf_reference_frames", ["base_vilens", "map_vilens"]
        )

        self._tf_query_frames = rospy.get_param(
            "~tf_query_frames", ["LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"]
        )

    def setup_ros(self):
        """Set up all ROS-related stuff"""

        # TF listener
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)

        # Subscribers
        self._sub_slam_graph = rospy.Subscriber(
            self._slam_graph_topic, Path, self.slam_graph_callback
        )
        self._sub_state_pose = rospy.Subscriber(
            self._state_pose, PoseWithCovarianceStamped, self.state_pose_callback
        )
        self._sub_state_twist = rospy.Subscriber(
            self._state_twist, TwistWithCovarianceStamped, self.state_twist_callback
        )

        # Optional
        self._sub_reference_twist = rospy.Subscriber(
            self._reference_twist_topic,
            TwistStamped,
            self.reference_twist_callback,
        )

        self._sub_operator_twist = rospy.Subscriber(
            self._operator_twist_topic,
            TwistStamped,
            self.operator_twist_callback,
        )

        self._sub_local_planner_param = rospy.Subscriber(
            self._local_planner_param_topic,
            Config,
            self.local_planner_param_callback,
        )

        self._sub_rmp_param = rospy.Subscriber(
            self._rmp_param_topic,
            Config,
            self.rmp_param_callback,
        )

        # Set tf subscriber timer
        self._sub_tf_timer = rospy.Timer(
            rospy.Duration(secs=1 / 20), self.tf_frames_callback
        )

    def set_internals(self):
        """Set up internal variables"""
        self._last_slam_graph = None

        # Output folder
        self.make_mission_report_folder()

        # Set converters
        self._slam_graph_converter = PathConverter(
            output_folder=self.output_folder,
            label="states",
            prefix="slam_graph",
            only_keep_last=True,
        )

        self._pose_converter = PoseStampedConverter(
            output_folder=self.output_folder,
            label="states",
            prefix="state_pose",
        )

        self._twist_converter = TwistStampedConverter(
            output_folder=self.output_folder,
            label="states",
            prefix="state_twist",
        )

        self._reference_twist_converter = TwistStampedConverter(
            output_folder=self.output_folder,
            label="states",
            prefix="reference_twist",
        )

        self._operator_twist_converter = TwistStampedConverter(
            output_folder=self.output_folder,
            label="states",
            prefix="operator_twist",
        )

        self._local_planner_param_converter = ROSMsgConverter(
            output_folder=self.output_folder,
            label="states",
            prefix="local_planner_param",
        )

        self._rmp_param_converter = ROSMsgConverter(
            output_folder=self.output_folder,
            label="states",
            prefix="rmp_param",
        )

        self._tf_converter = {}
        for parent in self._tf_reference_frames:
            for child in self._tf_query_frames:
                prefix = f"{parent}_{child}"
                self._tf_converter[prefix] = TransformStampedConverter(
                    output_folder=self.output_folder, label="states", prefix=prefix
                )

    # Callbacks
    def slam_graph_callback(self, msg: Path):
        rospy.loginfo_throttle(60, "Logging SLAM graph...")
        self._last_slam_graph = msg

    def state_pose_callback(self, msg: PoseWithCovarianceStamped):
        rospy.loginfo_throttle(60, "Logging state pose...")
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose = msg.pose.pose
        self._pose_converter.save(pose)

    def state_twist_callback(self, msg: TwistWithCovarianceStamped):
        rospy.loginfo_throttle(60, "Logging state twist...")
        twist = TwistStamped()
        twist.header = msg.header
        twist.twist = msg.twist.twist
        self._twist_converter.save(twist)

    def tf_frames_callback(self, msg):
        rospy.loginfo_throttle(60, "Logging TFs...")
        now = rospy.Time(0)
        for parent in self._tf_reference_frames:
            for child in self._tf_query_frames:
                try:
                    pose = self._tf_buffer.lookup_transform(
                        parent, child, now, timeout=rospy.Duration(0.1)
                    )
                    prefix = f"{parent}_{child}"
                    self._tf_converter[prefix].save(pose)
                except Exception as e:
                    rospy.logwarn(e)

    def reference_twist_callback(self, msg: TwistStamped):
        rospy.loginfo_throttle(60, "Logging reference twist...")
        self._reference_twist_converter.save(msg)

    def operator_twist_callback(self, msg: TwistStamped):
        rospy.loginfo_throttle(60, "Logging operator twist...")
        self._operator_twist_converter.save(msg)

    def local_planner_param_callback(self, msg: Config):
        rospy.loginfo_throttle(60, "Logging local planner param config changes...")
        self._local_planner_param_converter.save(msg, stamp=rospy.Time.now())

    def rmp_param_callback(self, msg: Config):
        rospy.loginfo_throttle(60, "Logging local planner RMP param config changes...")
        self._rmp_param_converter.save(msg, stamp=rospy.Time.now())

    # Other methods
    def make_mission_report_folder(self):
        home_path = os.path.expanduser("~")
        today = datetime.now()
        self.output_folder = os.path.join(
            home_path, "digiforest_mission_data", today.strftime("%Y-%m-%d-%H-%M-%S")
        )

        # Make folder
        print(f"Writing output to {self.output_folder}")
        os.makedirs(self.output_folder)

        # Make symlink to latest
        latest_path = os.path.join(home_path, "digiforest_mission_data/latest")
        if os.path.exists(latest_path):
            os.unlink(latest_path)
        os.symlink(self.output_folder, latest_path)

    def shutdown_routine(self, *args):
        """Executes the operations before killing the mission analysis procedures"""
        rospy.logwarn("Analyzing mission data, please wait...")

        # Save slam graph
        rospy.loginfo("Saving SLAM graph")
        self._slam_graph_converter.save(self._last_slam_graph)

        # Re-read the raw files

        # Compute statistics

        # Generate plots and report

        rospy.loginfo("Done!")
