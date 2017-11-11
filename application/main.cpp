/**
* Main application code
*/

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "human_detection.h"
#include "inpainter.h"
#include "enet_segmentation.h"

cv::Point gClick(-1, -1);
int gWaitTime = 30;
cv::Rect gLocalizedHuman(-1, -1, 0, 0);

/**
*   On mouse callback to register mouse click
*/

void on_mouse(int event, int x, int y, int flags, void *param) {
    if (event == cv::EVENT_LBUTTONDOWN) {
        gClick = cv::Point(x, y);
    }
    else if( event == cv::EVENT_MOUSEMOVE && (flags & cv::EVENT_FLAG_LBUTTON) )
    {
        gClick = cv::Point(x,y);
    }
}

/**
*   Function to check if a human is present in the region clicked by the user
*/

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

/**
*   Extract the bounding box of human containing the point clicked by the user
*/

cv::Mat extract_human(cv::Mat fullImage, std::vector<cv::Rect> humans) {
    int numHumans = humans.size();
    cv::Rect targetHuman;

    for (int i=0; i<numHumans; i++) {
        if (point_lies_inside_rect(humans[i], gClick)) {
            targetHuman = humans[i];
            targetHuman.x -= 0.05 * targetHuman.width;
            targetHuman.y -= 0.05 * targetHuman.height;
            targetHuman.width += 0.1 * targetHuman.width;
            targetHuman.height += 0.05 * targetHuman.height;
            gLocalizedHuman = targetHuman;
            break;
        } 
    }

    cv::Mat targetHumanMat = fullImage(targetHuman);
    return targetHumanMat;
}

int main(int argc, char *argv[])
{
    if (argc == 1 || !strcmp(argv[1], "-h") || !strcmp(argv[1], "--help") || !strcmp(argv[1], "-help")) {
        std::cout << "Usage information:\n";
        std::cout << "./intelligent_inpainting <image_path>\n";
        std::cout << "Note: Supported Image format: *.jpg\n";
        return -1; 
    }

    char* imagePath = argv[1];

    cv::Mat inputImage = cv::imread(imagePath, CV_LOAD_IMAGE_COLOR);

    if(!inputImage.data){
        std::cout<<std::endl<<"Error unable to open input image"<<std::endl;
        return 0;
    }

    int newH = inputImage.rows, newW = inputImage.cols;
    if (inputImage.rows > 768)
        newH = 768;
    if (inputImage.cols > 1360)
        newW = 1360;

    cv::resize(inputImage, inputImage, cv::Size(newW, newH));
    cv::namedWindow("IIP", 1);
    cv::imshow("IIP", inputImage);
    while (gClick.x == -1) {
        cv::waitKey(10);
        cv::setMouseCallback("IIP", on_mouse, 0);
    }

    /**********************************************
    * Human detection and extraction from the click
    ***********************************************/

    cv::Mat image1;
    inputImage.copyTo(image1);
    std::vector<cv::Rect> humans = human_detection(image1);
    cv::Mat targetHumanImg = extract_human(image1, humans);
    cv::Mat targetHumanImage;
    targetHumanImg.copyTo(targetHumanImage);

    if (gLocalizedHuman.x == -1) {
        std::cout << "No human detected in the region clicked!\n";
        return 0;
    }

    for(size_t i = 0; i < humans.size(); i++)
    {
        cv::rectangle(image1, cvPoint(humans[i].x,humans[i].y),cvPoint(humans[i].x + humans[i].width, humans[i].y + humans[i].height),cv::Scalar(0,255,0),2 );
    }

    /**********************************************
    * Enet segmentation
    ***********************************************/

    cv::Mat segmentedImage, hsvImage;
    inputImage.copyTo(segmentedImage);
    segmentedImage = enet_segmentation(segmentedImage);
    segmentedImage.copyTo(hsvImage);
    cv::cvtColor(segmentedImage, hsvImage, cv::COLOR_BGR2HSV);
    
    cv::Mat lower_red_hue_range;
    cv::Mat upper_red_hue_range;
    cv::inRange(hsvImage, cv::Scalar(0, 100, 100), cv::Scalar(10, 255, 255), lower_red_hue_range);
    cv::inRange(hsvImage, cv::Scalar(160, 100, 100), cv::Scalar(179, 255, 255), upper_red_hue_range);

    cv::Mat red_hue_image;
    cv::addWeighted(lower_red_hue_range, 1.0, upper_red_hue_range, 1.0, 0.0, red_hue_image);

    cv::resize(red_hue_image, red_hue_image, cv::Size(inputImage.cols, inputImage.rows));
    for(size_t i = 0; i < humans.size(); i++)
    {
        cv::rectangle(red_hue_image, cvPoint(humans[i].x,humans[i].y),cvPoint(humans[i].x + humans[i].width, humans[i].y + humans[i].height),cv::Scalar(0,255,0),2 );
    }

    targetHumanImage = extract_human(red_hue_image, humans);

    cv::Mat mask = cv::Mat::zeros(inputImage.rows, inputImage.cols, CV_32F);
    cv::copyMakeBorder(targetHumanImage, mask, gLocalizedHuman.y, \
                       inputImage.rows - (gLocalizedHuman.y + gLocalizedHuman.height), \
                       gLocalizedHuman.x, \
                       inputImage.cols - (gLocalizedHuman.x + gLocalizedHuman.width), \
                       cv::BORDER_CONSTANT | cv::BORDER_ISOLATED, \
                       0);

    cv::Mat element = getStructuringElement( cv::MORPH_ELLIPSE,
                                       cv::Size(3, 3),
                                       cv::Point(0, 0) );

    cv::dilate(mask, mask, element);
    cv::erode(mask, mask, element);
    cv::dilate(mask, mask, element);

    /**********************************************
    * Image inpainting, exemplar
    ***********************************************/
    cv::Mat image2 = inputImage.clone();
    cv::resize(image2, image2, cv::Size(320, 240));
    cv::resize(mask, mask, cv::Size(320, 240));
    Inpainter i(image2, mask, 3);
    if (i.checkValidInputs() == i.CHECK_VALID) {
        std::cout << "Painting the patch\n";
        i.inpaint();
        cv::imwrite("edited_output.jpg", i.result);
        cv::imshow("Edited Output", i.result);
    }
    else {
        std::cout<<std::endl<<"Error : invalid parameters"<<std::endl;
    }

    cv::waitKey(0);

    return 0;
}