#!/usr/bin/env python
# author: Matias Mattamala

import rospy
from digiforest_analysis_ros.forest_analysis import ForestAnalysis

if __name__ == "__main__":
    rospy.init_node("digiforest_forest_analysis_node")
    app = ForestAnalysis()

    rospy.loginfo("[digiforest_forest_analysis_node] Ready")
    rospy.spin()
    print("Bye!")
