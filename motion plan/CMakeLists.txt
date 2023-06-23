
cmake_minimum_required(VERSION 2.8.3)
project(motion_plan_pkg)

## Find catkin and any catkin packages
find_package(catkin REQUIRED COMPONENTS roscpp std_msgs sensor_msgs)
find_package(Eigen3 3.3 REQUIRED)


## Declare a catkin package
catkin_package()

## Build talker and listener
include_directories(include ${catkin_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})
include_directories(/home/lucaubu/ros_ws/src/locosim/robot_control/lab_exercises/lab_palopoli/cpp/json/include)
include_directories(/home/lucaubu/ros_ws/src/motion_plan_2)

add_executable(motion_plan motionPlan.cpp motionPlan.h kinematics.cpp kinematics.h main.cpp)

target_link_libraries(motion_plan ${catkin_LIBRARIES})
install(TARGETS motion_plan
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
