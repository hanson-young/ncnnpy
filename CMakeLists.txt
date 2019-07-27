project (ncnnpy)
cmake_minimum_required (VERSION 2.6.0)

# Include package config, boost-python
find_package(PkgConfig REQUIRED)
include(cmake/boost-python.cmake)

find_package(OpenMP)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfma")

if (NOT PYTHON_INCLUDE_DIRS OR NOT PYTHON_LIBRARY)
    SET(PYTHON_INCLUDE_DIRS "/usr/include/python2.7")
    SET(PYTHON_LIBRARY "/usr/lib/python2.7/config-x86_64-linux-gnu/libpython2.7.so")
endif()

# Build np<=>opencv converter library
AUX_SOURCE_DIRECTORY(./src DIR_SRCS)
boost_python_module(ncnnpy ${DIR_SRCS})
set_target_properties(ncnnpy PROPERTIES CXX_STANDARD 11)
target_link_libraries(ncnnpy ncnnavx boost_system boost_python)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/include)
add_library(ncnnavx STATIC IMPORTED) 
set_property(TARGET ncnnavx PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/src/lib/libncnn.a)
target_link_libraries(ncnnpy ncnnavx)
