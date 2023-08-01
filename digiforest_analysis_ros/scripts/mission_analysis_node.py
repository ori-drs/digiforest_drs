#!/usr/bin/env python
# author: Matias Mattamala

import rospy

from geometry_msgs.msg import TwistStamped
from vilens_msgs.msg import State
from vilens_slam_msgs.msg import PoseGraph


class MissionAnalysisInterface:
    def __init__(self):
        """Constructor"""
        self.read_params()
        self.set_internals()
        self.setup_ros()

    def read_params(self):
        """Read parameters from parameter server"""
        pass

    def set_internals(self):
        """Set up internal variables"""
        pass

    def setup_ros(self):
        """Set up all ROS-related stuff"""

        # Subscribers
        self._sub_vilens_graph = rospy.Subscriber(
            "/vilens_slam/pose_graph", PoseGraph, self.vilens_slam_graph_callback
        )
        self._sub_vilens_state = rospy.Subscriber(
            "/vilens/state_propagated", State, self.vilens_state_callback
        )
        self._sub_hbc_command = rospy.Subscriber(
            "/motion_reference/command_twist", TwistStamped, self.vilens_hbc_callback
        )

    # Callbacks
    def vilens_slam_graph_callback(self, msg: PoseGraph):
        pass

    def vilens_state_callback(self, msg: State):
        pass

    def vilens_hbc_callback(self, msg: TwistStamped):
        pass


if __name__ == "__main__":
    rospy.init_node("digiforest_mission_analysis_node")
    app = MissionAnalysisInterface()
    rospy.loginfo("[digiforest_mission_analysis_node] Ready")
    rospy.spin()
