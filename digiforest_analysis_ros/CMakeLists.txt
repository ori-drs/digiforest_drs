cmake_minimum_required(VERSION 3.0.2)
project(digiforest_analysis_ros)

find_package(catkin REQUIRED COMPONENTS
  pcl_ros
  tf_conversions
)

catkin_python_setup()

catkin_package(
  # INCLUDE_DIRS
  #   include
  CATKIN_DEPENDS 
    pcl_ros
    ros2raw
    tf_conversions
    vilens_msgs
    vilens_slam_msgs
)

###########
## Python #
###########

catkin_install_python(PROGRAMS scripts/mission_analysis_node.py   
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})


#############
## Install ##
#############

install(TARGETS ${TARGETS}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)


# Mark cpp header files for installation
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})



