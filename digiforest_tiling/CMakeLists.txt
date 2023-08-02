cmake_minimum_required(VERSION 2.8.3)

project(digiforest_tiling)
set (CMAKE_CXX_STANDARD 14)

list(APPEND CMAKE_MODULE_PATH /usr/share/cmake/geographiclib/)
find_package(GeographicLib REQUIRED)

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
  ${GeographicLib_INCLUDE_DIRS}
)

add_executable(generate_tiles src/generate_tiles.cpp src/gnss_handler.cpp src/slam_mission.cpp)
target_link_libraries(generate_tiles ${catkin_LIBRARIES} ${GeographicLib_LIBRARIES})


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



