cmake_minimum_required(VERSION 3.14)
project(CMatrix)

set(CMAKE_CXX_STANDARD 17)

# Add core source
add_subdirectory(cmx_core)
add_subdirectory(cmx_include)

# Example app
add_executable(cmx_demo cmx_apps/gesture_demo/main.cpp)
target_link_libraries(cmx_demo cmx_core)

# Include headers
include_directories(
    cmx_include
    cmx_core/include
)

# Optional test
enable_testing()
add_subdirectory(cmx_tests)
