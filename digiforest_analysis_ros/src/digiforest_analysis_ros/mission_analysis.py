#!/usr/bin/env python
# author: Matias Mattamala

import os
import rospy
import tf2_ros

from datetime import datetime
from geometry_msgs.msg import TwistStamped, TwistWithCovarianceStamped
from nav_msgs.msg import Path
from ros2raw.converters import TwistStampedConverter, PathConverter


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

        # Optional topics
        self._operator_twist_topic = rospy.get_param(
            "~operator_twist_topic", "/motion_reference/command_twist"
        )

        self._tf_frames = rospy.get_param("~tf_frames_topic", ["base"])

    def setup_ros(self):
        """Set up all ROS-related stuff"""

        # TF listener
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)

        # Subscribers
        self._sub_slam_graph = rospy.Subscriber(
            self._slam_graph_topic, Path, self.slam_graph_callback
        )
        self._sub_state_twist = rospy.Subscriber(
            self._state_twist, TwistWithCovarianceStamped, self.state_twist_callback
        )

        # Optional
        self._sub_operator_twist = rospy.Subscriber(
            self._operator_twist_topic,
            TwistStamped,
            self.operator_twist_callback,
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

        self._twist_converter = TwistStampedConverter(
            output_folder=self.output_folder,
            label="states",
            prefix="state_twist",
        )

        self._operator_twist_converter = TwistStampedConverter(
            output_folder=self.output_folder,
            label="states",
            prefix="operator_twist",
        )

    # Callbacks
    def slam_graph_callback(self, msg: Path):
        rospy.loginfo_throttle(10, "Logging SLAM graph...")
        self._last_slam_graph = msg

    def state_twist_callback(self, msg: TwistWithCovarianceStamped):
        rospy.loginfo_throttle(10, "Logging state twist...")
        # Save velocity to file
        twist = TwistStamped()
        twist.header = msg.header
        twist.twist = msg.twist.twist
        self._twist_converter.save(twist)

    def operator_twist_callback(self, msg: TwistStamped):
        rospy.loginfo_throttle(10, "Logging operator twist...")
        # Save velocity to file
        self._operator_twist_converter.save(msg)

    def tf_frames_callback(self, msg):
        # Save requested frames to file
        pass

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
