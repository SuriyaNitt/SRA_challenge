#ifndef ENET_SEGMENTATION_H
#define ENET_SEGMENTATION_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

cv::Mat enet_segmentation(char** argv, cv::Mat);

#endif