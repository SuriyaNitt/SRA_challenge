#include <iostream>
#include <opencv2/core/core.hpp>

#include <human_detector.h>
#include <human_detection.h>
#include <human_detection_private.h>

std::vector<cv::Rect> human_detection(cv::Mat &image) {
    cv::Mat inputImage = image.clone();
    std::cout << "Detecting humans.." << std::endl;

    std::vector<CRect> results;
    std::vector<cv::Rect> detections;
    std::unordered_map<std::string, double> config = parse_config("./config.txt");

    int height = 108;//config["height"];
    int width  = 36;//config["width"];
    int xdiv   = 9;//config["xdiv"];
    int ydiv   = 4;//config["ydiv"];

    Detector detector(height, width, xdiv, ydiv, 256, 0.8);
    loadCascade(detector);
    IntImage<double> inputIntImage;
    inputIntImage.load(inputImage);
    detector.fastScan(inputIntImage, results, 2);
    postProcess(results, 2);
    postProcess(results, 0);
    removeCoveredRectangles(results);

    int results_size = results.size();

    for (int i=0; i<results_size; i++) {
        int left = results[i].left;
        int top  = results[i].top;
        int bottom = results[i].bottom;
        int right  = results[i].right;
        cv::Rect detection(left, top, right - left, bottom - top);
        detections.push_back(detection);
    }

    return detections;
}

