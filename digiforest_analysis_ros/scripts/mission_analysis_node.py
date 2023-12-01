#!/usr/bin/env python
# author: Matias Mattamala

import rospy
from digiforest_analysis_ros.mission_analysis import MissionAnalysis

if __name__ == "__main__":
    rospy.init_node("digiforest_mission_analysis_node")
    app = MissionAnalysis()

    rospy.loginfo("[digiforest_mission_analysis_node] Ready")
    rospy.spin()
