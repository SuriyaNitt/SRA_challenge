cmake_minimum_required(VERSION 2.8)

project(SRA_Challenge)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")

find_package(OpenCV  REQUIRED)

set(SRCS
    "human_detection/src/human_detection.cpp"
    "human_detection/src/human_detector.cpp"
    "human_detection/src/config_parser.cpp"
    # "contour_estimation/include/*.hpp"
    # "contour_estimation/src/*.cpp"
    # "scene_completion/include/*.hpp"
    # "scene_completion/src/*.cpp"
    )

include_directories (
    "human_detection/include"
    # "contour_estimation/include/"
    # "scene_completion/include"
)

ADD_EXECUTABLE (intelligent_inpainting application/main.cpp ${SRCS})

if (OpenCV_FOUND)
    include_directories( ${OpenCV_INCLUDE_DIRS} )
#   include_directories("${OPENCV_CONFIG_PATH}/include")
    target_link_libraries( intelligent_inpainting ${OpenCV_LIBS} )
else (OpenCV_FOUND)
message("PB->OPENCV = ${OpenCV_INCLUDE_DIRS}")
endif (OpenCV_FOUND)