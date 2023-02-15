
cmake_minimum_required(VERSION 2.8.3)
project(lab_palopoli)

## Find catkin and any catkin packages
find_package(catkin REQUIRED COMPONENTS roscpp std_msgs sensor_msgs)
find_package(Eigen3 3.3 REQUIRED)


## Declare a catkin package
catkin_package()

## Build talker and listener
include_directories(include ${catkin_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})

add_executable(motionPlan motionPlan.cpp motionPlan.h kinematics.cpp kinematics.h main.cpp)

target_link_libraries(motionPlan ${catkin_LIBRARIES})
install(TARGETS motionPlan
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
