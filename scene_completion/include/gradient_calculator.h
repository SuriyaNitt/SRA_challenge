#ifndef GRADIENTCALCULATOR_H
#define GRADIENTCALCULATOR_H

#include<opencv/core/core.hpp>
#include<opencv/highgui/highgui.hpp>

class GradientCalculator
{
private:
    cv::Mat gradX;   //!< mat to store calculated x direction gradient/isophate
    cv::Mat gradY;   //!< mat to store calculated y direction gradient/isophate
public:
    GradientCalculator();
    void calculateGradient(cv::Mat &src);
    cv::Mat getGradX();
    cv::Mat getGradY();
};

#endif // GRADIENTCALCULATOR_H
