cmake_minimum_required(VERSION 2.8)

project(SRA_Challenge)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")

find_package(OpenCV  REQUIRED)
set(EXTRA_LIBS "-larpack" "-lparpack" "-lgfortran")

file ( GLOB SRCS
     "human_detection/src/*.cpp"
     "contour_estimation/src/gPb/*.cpp"
     "contour_estimation/src/seg/*.cpp"
     "contour_estimation/src/sPb/*.cpp"
     "scene_completion/src/*.cpp"
     "enet_segmentation/src/*.cpp"
    )

include_directories (
    "human_detection/include"
    "contour_estimation/include/seg"
    "contour_estimation/include/gPb"
    "contour_estimation/include/sPb"
    "enet_segmentation/include"
    "scene_completion/include"
)

ADD_EXECUTABLE (intelligent_inpainting application/main.cpp ${SRCS})

if (OpenCV_FOUND)
    include_directories( ${OpenCV_INCLUDE_DIRS} )
#   include_directories("${OPENCV_CONFIG_PATH}/include")
    target_link_libraries( intelligent_inpainting ${OpenCV_LIBS} ${EXTRA_LIBS})
else (OpenCV_FOUND)
message("PB->OPENCV = ${OpenCV_INCLUDE_DIRS}")
endif (OpenCV_FOUND)