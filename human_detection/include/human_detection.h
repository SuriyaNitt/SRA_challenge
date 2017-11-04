#ifndef HUMAN_DETECTION_HPP
#define HUMAN_DETECTION_HPP

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

std::vector<cv::Rect> human_detection(cv::Mat &image);

#endif
