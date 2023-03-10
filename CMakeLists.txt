cmake_minimum_required(VERSION 2.8.3)

project(digiforest_drs)
add_compile_options(-std=c++14 -ggdb)

find_package(catkin REQUIRED COMPONENTS
  pcl_ros
)


catkin_package(
  #INCLUDE_DIRS
  #  include
  CATKIN_DEPENDS 
  pcl_ros
)

###########
## Python #
###########
catkin_python_setup()
catkin_install_python(PROGRAMS src/digiforest_drs/generate_height_maps.py         
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

###########
## Build ##
###########

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(generate_mesh src/tools/generate_mesh.cpp)
target_link_libraries(generate_mesh ${catkin_LIBRARIES})

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



