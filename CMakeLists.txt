cmake_minimum_required(VERSION 3.8)
project(sketchmap_py CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-Wpedantic -O3")

find_package(ArrayFire)
find_package(OpenCL)
include_directories(${ArrayFire_INCLUDE_DIRS})
set(EXTRA_LIBS ${CMAKE_THREAD_LIBS_INIT} ${OpenCL_LIBRARIES})

add_subdirectory(third-party/pybind11)
set(SOURCE_FILES src/sketchmap.cpp src/libsketchmap.cpp src/libsketchmap.h src/libmath.h)
pybind11_add_module(sketchmap ${SOURCE_FILES})
target_link_libraries(sketchmap PRIVATE ${ArrayFire_LIBRARIES} ${EXTRA_LIBS})
