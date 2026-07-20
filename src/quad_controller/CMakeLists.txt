cmake_minimum_required(VERSION 3.16)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.8/bin/nvcc)
project(quad_controller LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(CUDAToolkit REQUIRED)

find_path(TRT_INCLUDE NvInfer.h)
find_library(TRT_NVINFER nvinfer)

# --- quad_core: policy logic, no GPU and no ROS -------------------------
# Kept free of CUDA so it builds and tests on any machine.
add_library(quad_core src/observation_builder.cpp)
target_include_directories(quad_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

# --- quad_trt: TensorRT backend -----------------------------------------
add_library(quad_trt src/policy_engine.cpp)
target_include_directories(quad_trt PRIVATE ${TRT_INCLUDE})
target_link_libraries(quad_trt quad_core ${TRT_NVINFER} CUDA::cudart)

# --- ROS 2 nodes ---------------------------------------------------------
add_executable(quad_node src/quad_node.cpp)
target_link_libraries(quad_node quad_trt)
ament_target_dependencies(quad_node
  rclcpp sensor_msgs nav_msgs geometry_msgs std_msgs ament_index_cpp)

add_executable(benchmark src/benchmark.cpp)
target_link_libraries(benchmark quad_trt)

add_executable(waypoint_navigator src/waypoint_navigator.cpp)
ament_target_dependencies(waypoint_navigator rclcpp nav_msgs geometry_msgs)

# --- tests ---------------------------------------------------------------
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_observation_builder test/test_observation_builder.cpp)
  target_link_libraries(test_observation_builder quad_core)
endif()

install(TARGETS quad_core quad_trt
  ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(TARGETS quad_node waypoint_navigator benchmark DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY models DESTINATION share/${PROJECT_NAME})

ament_package()
