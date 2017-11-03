#include "inpainter.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

cv::Point gClick(-1, -1);

void on_mouse(int event, int x, int y, int flags, void *param) {
    if (event == cv::EVENT_LBUTTONDOWN) {
        gClick = cv::Point(x, y);
    }
}

bool point_lies_inside_rect(cv::Rect rect, cv::Point point) {
    int x = point.x;
    int y = point.y;
    if (x >= rect.x && x <= rect.x + rect.width &&\
        y >= rect.y && y <= rect.y + rect.height) {
        return true;
    }
    else
        return false;
}

cv::Mat extact_human(cv::Mat fullImage, std::vector<cv::Rect> humans) {
    int numHumans = humans.size();
    cv::Rect targetHuman;

    for (int i=0; i<numHumans; i++) {
        if (point_lies_inside_rect(humans[i], gClick)) {
            targetHuman = humans[i];
            break;
        } 
    }

    cv::Mat targetHumanMat = fullImage(targetHuman);
    return targetHumanMat;
}

int main(int argc, char *argv[])
{
    if (argc == 1 || argv[1] == '-h' || argv[1] == '--help' || argv[1] == '-help') {
        std::cout << "Usage information:\n";
        std::cout << "./intelligent_inpainting <image_path>\n";
        std::cout << "Note: Supported Image format: *.jpg\n";
        return -1; 
    }

    char* imagePath = argv[1];

    inputImage = cv::imread(imagePath, CV_LOAD_IMAGE_COLOR);
    if(!inputImage.data){
        std::cout<<std::endl<<"Error unable to open input image"<<std::endl;
        return 0;
    }

    cv::NamedWindow('Intelligent Inpainting', 1);
    cv::imshow("Input Image", inputImage);
    cv::setMouseCallback("Input Image", on_mouse, 0);

    /**********************************************
    * Human detection and extraction from the click
    ***********************************************/

    cv::Mat image1 = inputImage.clone();
    vector<cv::Rect> humans = humans_detection(image1);
    cv::Mat targetHumanImage = extract_human(image1, humans);

    /**********************************************
    * Human Contour detection
    ***********************************************/    

    cv::Mat gPb. gPb_thin;
    vector<cv::Mat> gPb_ori;

    cv::globalPb(img0, gPb, gPb_thin, gPb_ori);

    /**********************************************
    * Image inpainting, exemplar
    ***********************************************/
    cv::Mat image2 = inputImage.clone();
    Inpainter i(image2, mask, 3);
    if (i.checkValidInputs() == i.CHECK_VALID) {
        i.inpaint();
        cv::imwrite("edited_output.jpg", i.result);
        cv::imshow("Edited Output", i.result);
    }
    else {
        std::cout<<std::endl<<"Error : invalid parameters"<<std::endl;
    }

    return 0;
    }