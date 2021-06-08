cmake_minimum_required(VERSION 2.8)

project(psd_tracking)

add_definitions(-w)

ADD_COMPILE_OPTIONS(-std=c++11 )
ADD_COMPILE_OPTIONS(-std=c++14 )
set(CMAKE_CXX_FLAGS "-std=c++11 -O3" )
set(CMAKE_CXX_FLAGS "-std=c++11 -Wall")
set(CMAKE_CXX_FLAGS_RELEASE  "-std=c++11 -O3 -fopenmp -pthread")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build/lib)

# include(${PROJECT_SOURCE_DIR}/conanbuildinfo.cmake)
# conan_basic_setup()

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Eigen3 REQUIRED)

find_package(nlohmann_json 3.2.0 REQUIRED)

find_package(PCL REQUIRED)


find_package(Ceres REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include
                    ${PROJECT_SOURCE_DIR}
                    ${EIGEN3_INCLUDE_DIR}
                    ${PCL_INCLUDE_DIRS}
                    ${Ceres_INCLUDE_DIRS}
)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


add_library(${PROJECT_NAME} SHARED
    src/earth.cpp
)



add_executable(barrier_detection sample/barrier_detection.cpp)

target_link_libraries(barrier_detection ${OpenCV_LIBS}
                        ${Eigen_LIBRARIES}
                        ${PCL_LIBRARIES}
                        ${CERES_LIBRARIES}
)

add_executable(resampling sample/resampling.cpp)

target_link_libraries(resampling ${OpenCV_LIBS}
                        ${Eigen_LIBRARIES}
                        ${PCL_LIBRARIES}
                        ${CERES_LIBRARIES}
)

add_executable(doffiltering sample/doffiltering.cpp)

target_link_libraries(doffiltering ${OpenCV_LIBS}
                        ${Eigen_LIBRARIES}
                        ${PCL_LIBRARIES}
                        ${CERES_LIBRARIES}
)

add_executable(detatch_line sample/detatch_line.cpp)

target_link_libraries(detatch_line ${OpenCV_LIBS}
                        ${Eigen_LIBRARIES}
                        ${PCL_LIBRARIES}
                        ${CERES_LIBRARIES}
)

add_executable(condition_cluster sample/condition_cluster.cpp)

target_link_libraries(condition_cluster
                        ${PROJECT_NAME}
                        ${OpenCV_LIBS}
                        ${Eigen_LIBRARIES}
                        ${PCL_LIBRARIES}
                        ${CERES_LIBRARIES}
                        nlohmann_json::nlohmann_json
)

add_executable(compute_normal sample/compute_normal.cpp)

target_link_libraries(compute_normal ${OpenCV_LIBS}
                        ${Eigen_LIBRARIES}
                        ${PCL_LIBRARIES}
                        ${CERES_LIBRARIES}
)