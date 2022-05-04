cmake_minimum_required(VERSION 3.13)
project(GEMM)
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include
#                 ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/opencv/arm64/include/)
# set(LINK_DIR ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/opencv/arm64/lib)
# include_directories(${INCLUDE_DIR})
# link_directories(${LINK_DIR})
# message(STATUS "INCLUDE_DIR : ${INCLUDE_DIR}")
# message(STATUS "LINK_DIR : ${LINK_DIR}")

# file(GLOB_RECURSE SOURCE_FILES src/*.cpp)
# add_library(integral_meanfilter ${SOURCE_FILES})

set(TEST_FILES quadrbits_gemm.cpp)

add_executable(test_armv8_quadrbits_gemm ${TEST_FILES})
# target_link_libraries(test_integral_meanfilter 
#                         integral_meanfilter
#                         opencv_java4
#                         log
#                         jnigraphics
#                         z
#                         mediandk
#                         c++_shared)
