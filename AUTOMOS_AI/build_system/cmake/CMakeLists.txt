# AUTOMOS AI CMake Configuration
# Cross-platform build system for multiple hardware platforms

cmake_minimum_required(VERSION 3.16)
project(AUTOMOS_AI VERSION 1.0.0 LANGUAGES C CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Platform detection
if(NOT DEFINED TARGET_PLATFORM)
    set(TARGET_PLATFORM "x86_64")
endif()

if(NOT DEFINED TARGET_OS)
    set(TARGET_OS "linux")
endif()

# Architecture-specific settings
if(TARGET_ARCH STREQUAL "arm64")
    set(CMAKE_SYSTEM_NAME "Linux")
    set(CMAKE_SYSTEM_PROCESSOR "aarch64")
elseif(TARGET_ARCH STREQUAL "x86_64")
    set(CMAKE_SYSTEM_NAME "Linux")
    set(CMAKE_SYSTEM_PROCESSOR "x86_64")
endif()

# OS-specific settings
if(TARGET_OS STREQUAL "qnx")
    add_definitions(-D_QNX_SOURCE)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lrt")
elseif(TARGET_OS STREQUAL "linux")
    add_definitions(-D_LINUX_SOURCE)
endif()

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

# Compiler flags
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -fPIC")

# Find required packages
find_package(PkgConfig REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# CUDA support (if available)
if(TARGET_PLATFORM MATCHES "jetson.*" OR TARGET_PLATFORM STREQUAL "drive_agx")
    enable_language(CUDA)
    find_package(CUDA REQUIRED)
    set(CUDA_ENABLED TRUE)
else()
    set(CUDA_ENABLED FALSE)
endif()

# AUTOMOS AI components
option(BUILD_CORE_REASONING "Build core reasoning engine" ON)
option(BUILD_PERCEPTION "Build perception pipeline" ON)
option(BUILD_SAFETY "Build safety systems" ON)
option(BUILD_WORLD_MODEL "Build world model" ON)
option(BUILD_INTEGRATION "Build integration framework" ON)
option(BUILD_EDGE_OPTIMIZATION "Build edge optimization" ON)
option(BUILD_MONITORING "Build monitoring systems" ON)
option(BUILD_ALL_FEATURES "Build all features" OFF)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}/perception
    ${CMAKE_CURRENT_SOURCE_DIR}/safety
    ${CMAKE_CURRENT_SOURCE_DIR}/world_model
    ${CMAKE_CURRENT_SOURCE_DIR}/integration
    ${CMAKE_CURRENT_SOURCE_DIR}/edge_optimization
    ${CMAKE_CURRENT_SOURCE_DIR}/monitoring
    ${OpenCV_INCLUDE_DIRS}
)

# Core reasoning engine
if(BUILD_CORE_REASONING OR BUILD_ALL_FEATURES)
    add_subdirectory(core)
    set(CORE_LIBRARIES automos_core)
endif()

# Perception pipeline
if(BUILD_PERCEPTION OR BUILD_ALL_FEATURES)
    add_subdirectory(perception)
    set(PERCEPTION_LIBRARIES automos_perception)
endif()

# Safety systems
if(BUILD_SAFETY OR BUILD_ALL_FEATURES)
    add_subdirectory(safety)
    set(SAFETY_LIBRARIES automos_safety)
endif()

# World model
if(BUILD_WORLD_MODEL OR BUILD_ALL_FEATURES)
    add_subdirectory(world_model)
    set(WORLD_MODEL_LIBRARIES automos_world_model)
endif()

# Integration framework
if(BUILD_INTEGRATION OR BUILD_ALL_FEATURES)
    add_subdirectory(integration)
    set(INTEGRATION_LIBRARIES automos_integration)
endif()

# Edge optimization
if(BUILD_EDGE_OPTIMIZATION OR BUILD_ALL_FEATURES)
    add_subdirectory(edge_optimization)
    set(EDGE_OPT_LIBRARIES automos_edge_opt)
endif()

# Monitoring systems
if(BUILD_MONITORING OR BUILD_ALL_FEATURES)
    add_subdirectory(monitoring)
    set(MONITORING_LIBRARIES automos_monitoring)
endif()

# Main AUTOMOS AI executable
add_executable(automos_ai
    src/main.cpp
    src/system_coordinator.cpp
    src/config_manager.cpp
)

# Link libraries
target_link_libraries(automos_ai
    ${CORE_LIBRARIES}
    ${PERCEPTION_LIBRARIES}
    ${SAFETY_LIBRARIES}
    ${WORLD_MODEL_LIBRARIES}
    ${INTEGRATION_LIBRARIES}
    ${EDGE_OPT_LIBRARIES}
    ${MONITORING_LIBRARIES}
    ${OpenCV_LIBS}
    ${Python3_LIBRARIES}
)

# CUDA libraries (if enabled)
if(CUDA_ENABLED)
    target_link_libraries(automos_ai ${CUDA_LIBRARIES})
endif()

# Platform-specific libraries
if(TARGET_OS STREQUAL "qnx")
    target_link_libraries(automos_ai rt socket)
elseif(TARGET_OS STREQUAL "linux")
    target_link_libraries(automos_ai pthread rt dl)
endif()

# Installation
install(TARGETS automos_ai
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# Configuration files
install(FILES
    config/automos_config.json
    config/platform_config.json
    DESTINATION etc/automos_ai
)

# Systemd service (if Linux)
if(TARGET_OS STREQUAL "linux")
    install(FILES
        scripts/automos_ai.service
        DESTINATION lib/systemd/system
    )
endif()

# Package configuration
set(CPACK_PACKAGE_NAME "AUTOMOS_AI")
set(CPACK_PACKAGE_VERSION "1.0.0")
set(CPACK_PACKAGE_DESCRIPTION "AUTOMOS AI Autonomous Driving System")
set(CPACK_PACKAGE_VENDOR "AUTOMOS AI")
set(CPACK_PACKAGE_CONTACT "support@automos-ai.com")

if(TARGET_PLATFORM STREQUAL "jetson_nano")
    set(CPACK_PACKAGE_FILE_NAME "automos_ai_jetson_nano")
elseif(TARGET_PLATFORM STREQUAL "jetson_xavier")
    set(CPACK_PACKAGE_FILE_NAME "automos_ai_jetson_xavier")
elseif(TARGET_PLATFORM STREQUAL "jetson_orin")
    set(CPACK_PACKAGE_FILE_NAME "automos_ai_jetson_orin")
elseif(TARGET_PLATFORM STREQUAL "drive_agx")
    set(CPACK_PACKAGE_FILE_NAME "automos_ai_drive_agx")
elseif(TARGET_PLATFORM STREQUAL "industrial_pc")
    set(CPACK_PACKAGE_FILE_NAME "automos_ai_industrial_pc")
endif()

include(CPack)
