cmake_minimum_required(VERSION 2.8)

project(SRA_Challenge)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -DCPU_ONLY=1 -O3")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")

find_package(OpenCV  REQUIRED)
set(EXTRA_LIBS "-lboost_system -lglog")

set(ROOT_DIR ${CMAKE_CURRENT_LIST_DIR})
set(CAFFE_LIBS "${ROOT_DIR}/caffe_enet/build/lib/libcaffe.so" "${ROOT_DIR}/caffe_enet/build/lib/libproto.a")

find_package(Boost REQUIRED)

file ( GLOB SRCS
     "human_detection/src/*.cpp"
     # "contour_estimation/src/gPb/*.cpp"
     # "contour_estimation/src/seg/*.cpp"
     # "contour_estimation/src/sPb/*.cpp"
     "scene_completion/src/*.cpp"
     "ENet/src/*.cpp"
    )

include_directories (
    "human_detection/include"
    "contour_estimation/include/seg"
    "contour_estimation/include/gPb"
    "contour_estimation/include/sPb"
    "ENet/include"
    "scene_completion/include"
    "caffe_enet/include"
)

ADD_EXECUTABLE (intelligent_inpainting application/main.cpp ${SRCS})

if (OpenCV_FOUND)
    include_directories( ${OpenCV_INCLUDE_DIRS} )
    target_link_libraries( intelligent_inpainting ${OpenCV_LIBS} ${EXTRA_LIBS} ${CAFFE_LIBS} ${Boost_LIBS})
else (OpenCV_FOUND)
message("PB->OPENCV = ${OpenCV_INCLUDE_DIRS}")
endif (OpenCV_FOUND)