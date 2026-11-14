cmake_minimum_required(VERSION 3.14)
project(isaac_track_lab_deployment LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(BUILD_EXAMPLES OFF CACHE BOOL "Do not build Unitree SDK examples" FORCE)
add_subdirectory(
  "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/unitree_sdk2"
  "${CMAKE_CURRENT_BINARY_DIR}/unitree_sdk2"
  EXCLUDE_FROM_ALL
)

add_executable(as2_sport_bridge as2_sport_bridge.cpp)
target_link_libraries(as2_sport_bridge PRIVATE unitree_sdk2)
target_compile_options(as2_sport_bridge PRIVATE -Wall -Wextra -Wpedantic -Werror)
