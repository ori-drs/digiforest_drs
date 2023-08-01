#!/usr/bin/env python
# author: Matias Mattamala

import rospy

from geometry_msgs.msg import TwistStamped
from vilens_msgs.msg import State
from vilens_slam_msgs.msg import PoseGraph


class MissionAnalysis:
    def __init__(self):
        """Constructor"""
        self.read_params()
        self.set_internals()
        self.setup_ros()

    def read_params(self):
        """Read parameters from parameter server"""

        # Subscription topics
        self._vilens_graph_topic = rospy.get_param(
            "~vilens_graph_topic", "/vilens_slam/pose_graph"
        )

        self._vilens_state_topic = rospy.get_param(
            "~vilens_state_topic", "/vilens/state_propagated"
        )

        # Optional topics
        self._operator_twist_topic = rospy.get_param(
            "~operator_twist_topic", "/motion_reference/command_twist"
        )

        self._tf_frames = rospy.get_param("~tf_frames_topic", ["base"])

    def set_internals(self):
        """Set up internal variables"""
        self._last_vilens_graph = None
        self._last_vilens_state = None
        self._last_operator_twist = None

    def setup_ros(self):
        """Set up all ROS-related stuff"""

        # Subscribers
        self._sub_vilens_graph = rospy.Subscriber(
            self._vilens_graph_topic, PoseGraph, self.vilens_slam_graph_callback
        )
        self._sub_vilens_state = rospy.Subscriber(
            self._vilens_state_topic, State, self.vilens_state_callback
        )

        # Optional
        self._sub_operator_twist = rospy.Subscriber(
            "/motion_reference/command_twist",
            TwistStamped,
            self.operator_twist_callback,
        )

    # Callbacks
    def vilens_slam_graph_callback(self, msg: PoseGraph):
        pass

    def vilens_state_callback(self, msg: State):
        pass

    def operator_twist_callback(self, msg: TwistStamped):
        pass

    def tf_frames_callback(self, msg):
        pass
