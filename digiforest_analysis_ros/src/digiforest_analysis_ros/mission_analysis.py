#!/usr/bin/env python
# author: Matias Mattamala

import os
import rospy
import tf2_ros

from datetime import datetime
from geometry_msgs.msg import TwistStamped
from vilens_msgs.msg import State
from vilens_slam_msgs.msg import PoseGraph

from ros2raw.converters import TwistStampedConverter


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
        self._vilens_graph_topic = rospy.get_param(
            "~vilens_graph_topic", "/vilens_slam/pose_graph"
        )

        self._vilens_state_topic = rospy.get_param(
            "~vilens_state_topic", "/vilens/state_optimized"
        )

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
        self._sub_vilens_graph = rospy.Subscriber(
            self._vilens_graph_topic, PoseGraph, self.vilens_slam_graph_callback
        )
        self._sub_vilens_state = rospy.Subscriber(
            self._vilens_state_topic, State, self.vilens_state_callback
        )

        # Optional
        self._sub_operator_twist = rospy.Subscriber(
            self._operator_twist_topic,
            TwistStamped,
            self.operator_twist_callback,
        )

    def set_internals(self):
        """Set up internal variables"""
        self._last_vilens_graph = None

        # Output folder
        self.make_mission_report_folder()

        # Set converters
        self._state_twist_converter = TwistStampedConverter(
            output_folder=self.output_folder,
            label="states",
            prefix="state_twist",
            tf_handler=self._tf_buffer,
            tf=["odom", "map"],
        )

        self._operator_twist_converter = TwistStampedConverter(
            output_folder=self.output_folder,
            label="states",
            prefix="operator_twist",
            tf_handler=self._tf_buffer,
            tf=["odom", "map"],
        )

    # Callbacks
    def vilens_slam_graph_callback(self, msg: PoseGraph):
        self._last_vilens_graph = msg

    def vilens_state_callback(self, msg: State):
        # Save velocity to file
        self._state_twist_converter.save(msg)

    def operator_twist_callback(self, msg: TwistStamped):
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

        # Re-read the raw files

        # Compute statistics

        # Generate plots and report

        rospy.loginfo("Done!")
