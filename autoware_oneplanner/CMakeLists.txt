cmake_minimum_required(VERSION 3.14)
project(autoware_oneplanner)

find_package(autoware_cmake REQUIRED)
autoware_package()

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(OpenSSL REQUIRED)

# ---------------------------------------------------------------------------------------------
# Host library: everything that runs on the CPU (context tensors, map conversion, trajectory
# and detection postprocessing, engine identity). Built everywhere, so its tests also run on
# machines without a GPU.
# ---------------------------------------------------------------------------------------------
ament_auto_add_library(${PROJECT_NAME}_core SHARED
  src/bev_feature/detection_decode.cpp
  src/bev_feature/history_selection.cpp
  src/engine_identity.cpp
  src/geometry.cpp
  src/input_provider_registry.cpp
  src/postprocess/detection_postprocessor.cpp
  src/postprocess/object_postprocess.cpp
  src/postprocess/planning_factor.cpp
  src/postprocess/trajectory_postprocessor.cpp
  src/providers/context_input_provider.cpp
  src/types.cpp
  src/vector_map/lane_segments.cpp
  src/vector_map/map_conversion.cpp
)
target_link_libraries(${PROJECT_NAME}_core
  Eigen3::Eigen
  OpenSSL::Crypto
)

# ---------------------------------------------------------------------------------------------
# GPU part: the TensorRT engines, the BEV feature pipeline and the node. Built only where CUDA,
# TensorRT and spconv are available, like the other CUDA packages of Autoware.
# ---------------------------------------------------------------------------------------------
# FindCUDA (cuda_add_library) is deprecated in CMake but still how Autoware's CUDA packages and
# spconv's headers are built; keep it without the deprecation warning.
if(POLICY CMP0146)
  cmake_policy(SET CMP0146 OLD)
endif()
find_package(CUDA)
find_library(NVINFER nvinfer)
find_library(NVONNXPARSER nvonnxparser)
find_library(NVINFER_PLUGIN nvinfer_plugin)
find_package(cumm QUIET)
find_package(spconv QUIET)

set(GPU_AVAILABLE ON)
if(NOT CUDA_FOUND)
  message(WARNING "CUDA is not found: ${PROJECT_NAME}'s node is not built")
  set(GPU_AVAILABLE OFF)
endif()
if(NOT (NVINFER AND NVONNXPARSER AND NVINFER_PLUGIN))
  message(WARNING "TensorRT is not found: ${PROJECT_NAME}'s node is not built")
  set(GPU_AVAILABLE OFF)
endif()
if(NOT (cumm_FOUND AND spconv_FOUND))
  message(WARNING "spconv is not found: ${PROJECT_NAME}'s node is not built")
  set(GPU_AVAILABLE OFF)
endif()

if(GPU_AVAILABLE)
  # TensorRT headers are noisy with deprecations the node does not use.
  add_compile_options(-Wno-deprecated-declarations)

  list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr -diag-suppress 1675 --extended-lambda")
  list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_75,code=sm_75")
  list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_86,code=sm_86")
  list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_87,code=sm_87")
  list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_89,code=sm_89")
  if(CUDA_VERSION VERSION_GREATER_EQUAL "12.8")
    if(CUDA_VERSION VERSION_LESS "13.0")
      list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_101,code=sm_101")
    else()  # CUDA 13.0 renamed SM101 to SM110
      list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_110,code=sm_110")
    endif()
    list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_120,code=sm_120")
    list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_120,code=compute_120")
  else()
    list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_89,code=compute_89")
  endif()

  include_directories(include)
  cuda_add_library(${PROJECT_NAME}_cuda_lib SHARED
    src/bev_feature/bev_warp_kernel.cu
    src/bev_feature/detection_decode_kernel.cu
    src/bev_feature/voxelizer.cu
  )
  target_compile_definitions(${PROJECT_NAME}_cuda_lib PRIVATE TV_CUDA)
  target_link_libraries(${PROJECT_NAME}_cuda_lib spconv::spconv)

  ament_auto_add_library(${PROJECT_NAME}_component SHARED
    src/bev_feature/temporal_bev_cache.cpp
    src/bev_feature/trt_bev_feature_extractor.cpp
    src/inference_engine.cpp
    src/oneplanner_node.cpp
    src/providers/bev_feature_input_provider.cpp
    src/trt/trt_engine.cpp
  )
  target_include_directories(${PROJECT_NAME}_component SYSTEM PUBLIC ${CUDA_INCLUDE_DIRS})
  target_link_libraries(${PROJECT_NAME}_component
    ${PROJECT_NAME}_core
    ${PROJECT_NAME}_cuda_lib
    ${NVINFER}
    ${NVONNXPARSER}
    ${NVINFER_PLUGIN}
    ${CUDA_LIBRARIES}
    ${CMAKE_DL_LIBS}
  )

  rclcpp_components_register_node(${PROJECT_NAME}_component
    PLUGIN "autoware::oneplanner::OnePlannerNode"
    EXECUTABLE ${PROJECT_NAME}_node
  )

  install(TARGETS ${PROJECT_NAME}_cuda_lib DESTINATION lib)
endif()

# ---------------------------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------------------------
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)

  # Host tests: no GPU needed.
  ament_add_gtest(test_${PROJECT_NAME}_core
    test/bev_warp_test.cpp
    test/detection_postprocessor_test.cpp
    test/helpers_test.cpp
    test/history_selection_test.cpp
    test/map_tensors_test.cpp
    test/trajectory_postprocessor_test.cpp
    test/types_test.cpp
    test/vector_map_test.cpp
  )
  target_link_libraries(test_${PROJECT_NAME}_core ${PROJECT_NAME}_core)

  if(GPU_AVAILABLE)
    find_package(nlohmann_json REQUIRED)
    # GPU tests: each skips itself on a machine without a CUDA device.
    ament_add_gtest(test_${PROJECT_NAME}_gpu
      test/bev_warp_golden_test.cpp
      test/detection_decode_test.cpp
      test/temporal_bev_cache_test.cpp
    )
    target_link_libraries(test_${PROJECT_NAME}_gpu
      ${PROJECT_NAME}_component
      nlohmann_json::nlohmann_json
    )
    target_compile_definitions(test_${PROJECT_NAME}_gpu PRIVATE
      ONEPLANNER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/data")
  endif()

  # Static checks of the launch file: no ROS graph and no GPU.
  ament_add_pytest_test(test_launch_files test/test_launch_files.py)
endif()

ament_auto_package(
  INSTALL_TO_SHARE
    config
    launch
)
