cmake_minimum_required(VERSION 2.8.3)

project(digiforest_slam_logger)
add_compile_options(-std=c++14 -ggdb)

find_package(catkin REQUIRED COMPONENTS
  pcl_ros
  tf_conversions
)


catkin_package(
  INCLUDE_DIRS
    include
  CATKIN_DEPENDS 
    pcl_ros
    tf_conversions
)

###########
## Build ##
###########

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)


add_executable(slam_logger_node src/slam_logger/slam_logger_node.cpp  src/slam_logger/slam_logger_tools.cpp)
target_link_libraries(slam_logger_node ${catkin_LIBRARIES})


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



