#ifndef HUMAN_DETECTOR_HPP
#define HUMAN_DETECTOR_HPP

/**
* Interface file to human detection helper classes
*/

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

template<class T>
class Array2d {
public:
    int nrow;  /**< Number of rows */
    int ncol;  /**< Number of columns */
    T** p;  /**< 2d array */
    T* buf;  /**< 1d view of p */

    Array2d();
    Array2d(const int nrow,const int ncol);
    Array2d(const Array2d<T>& source); //!< copy constructor.
    virtual ~Array2d();

    Array2d<T>& operator=(const Array2d<T>& source); //!< assignment overloading.
    void create(const int nrowArg,const int ncolArg); //!< initialize p.
    void swap(Array2d<T>& array2); //!< swap this with the given object.
    void zero(const T t = 0); //!< make all of p zero.
    void clear(); //!< delete the 1D and 2D buffers.
}; //!< Array2d class abstracts image representations

template<class T>
class IntImage:public Array2d<T> {
private:
    IntImage(const IntImage<T> &source) { } //!< This is to avoid copy constructor
public:
    IntImage():variance(0.0),label(-1) { } //!< contructor.
    virtual ~IntImage();

    virtual void clear(void); //!< deletes image buffer.
    inline void setSize(const int h, const int w); //!< defines image size.
    bool load(cv::Mat img, char channel='I'); //!< loads the particular image channel into the internal buffer.
    void save(const std::string& fileName) const; //!< saves the image to a file.
    void swap(IntImage<T>& image2); //!< swaps the image with another integral image.
    void calcIntegralImageInPlace(void); //!< forms the integral image from the internal image buffer.
    void resize(IntImage<T>& result, const double ratio) const; //!< resizes the image to the given ratio.
    void resize(IntImage<T>& result, const int height, const int width) const; //!< resizes the image to the given width and height
    IntImage<T>& operator=(const IntImage<T>& source);//!< assignment opertator overloading.
    void Sobel(IntImage<double>& result,const bool useSqrt,const bool normalize); //!< performs sobel filtering.

    using Array2d<T>::nrow; //!< 
    using Array2d<T>::ncol; //!< 
    using Array2d<T>::buf; //!< 
    using Array2d<T>::p; //!< 
    double variance; //!< 
    int label; //!< 

}; //!< IntImage abstracts Integral image representations.
//!< This class inherits Array2d class.

class CRect {
public:
    double left; /**< left rectangle coordinate*/
    double top; /**< top rectangle coordinate*/
    double right; /**< right rectangle coordinate*/
    double bottom; /**< bottom rectangle coordinate*/

    CRect() {
        clear();
    } //!< Initializes the object.
    ~CRect() {
        clear();
    } //!< clears up the internal variables.

    bool empty() const; //!< checks if the rectangle is empty
    void clear(); //!< clears the internal variables
    double size() const; //!< returns the rectangle area
    bool intersect(CRect& result,const CRect& rect2) const; //!< returns
    bool Union(CRect& result,const CRect& rect2) const; //!< returns
}; //!< CRect class defines rectangle representations

class Node {
private:
    double minvalue;
    double maxvalue;

public:
    enum NodeType { CD_LIN, CD_HIK, LINEAR, HISTOGRAM };
    int type; /**< Node type */
    Array2d<double> classifier;
    double thresh;
    int featureLength;
    int upperBound;
    int index;
    std::string fileName;

    Node(const NodeType _type,const int _featurelength,const int _upper_bound,const int _index,const char* _filename);
    ~Node(){}

    void load(const NodeType _type,const int _featurelength,const int _upper_bound,const int _index,const char* _filename);
    bool classify(int* f);
    void setValues(const double v) {
        if(v>maxvalue) maxvalue = v;
        if(v<minvalue) minvalue = v;
    }
};

class Cascade {
public:
    int size, length;
    Node **nodes;

    Cascade();
    ~Cascade();
    void add_node(const Node::NodeType _type,\
                  const int featureLength, \
                  const int upperBound, \
                  std::string fileName);
};

class Detector {
private:
    IntImage<double>* integrals;
    IntImage<double> image, sobelImage;
    IntImage<int> ct;
    Array2d<int> hist;
    IntImage<double> scores;

    void initImage(IntImage<double>& original);
    void initIntegralImages(const int stepsize);
    void resizeImage(); //!< Resize the input image and then re-compute Sobel image etc
public:
    int height,width;
    int xdiv,ydiv;
    int baseflength;
    double ratio;

    Cascade* cascade;

    Detector()
        : height(0), width(0), xdiv(0), ydiv(0), baseflength(0), ratio(0.0), cascade(NULL), integrals(NULL)
    {
    }
    Detector(const int _height,const int _width,const int _xdiv,const int _ydiv,
                     const int _baseflength,const double _ratio)
        :height(_height),width(_width),xdiv(_xdiv),ydiv(_ydiv),
         baseflength(_baseflength),ratio(_ratio),cascade(NULL),integrals(NULL)
    {
    }
    ~Detector()
    {
        delete cascade;
        delete[] integrals;
    }

    void loadDetector(std::vector<Node::NodeType>& types,std::vector<int>& upper_bounds,std::vector<std::string>& filenames);
    int scan(IntImage<double>& original,std::vector<CRect>& results,const int stepsize,const int round,std::ofstream* out,const int upper_bound);
    int fastScan(IntImage<double>& original,std::vector<CRect>& results,const int stepsize);
    //!< The function that does the double detection
    int featureLength() const
    {
        return (xdiv-1)*(ydiv-1)*baseflength;
    }
};

void computeCT(IntImage<double>& original,IntImage<int>& ct);
//!< compute the Sobel image "ct" from "original"

double useSVM_CD_FastEvaluationStructure(const char* modelfile, \
                                         const int m,\
                                         Array2d<double>& result);
//!< Load SVM models -- linear SVM trained using LIBLINEAR

double useSVM_CD_FastEvaluationStructure(const char* modelfile, \
                                         const int m, \
                                         const int upper_bound, \
                                         Array2d<double>& result);
//!< Load SVM models -- Histogram Intersectin Kernel SVM trained by libHIK

void postProcess(std::vector<CRect>& result,const int combine_min);
//!< A simple post-process (NMS, non-maximal suppression)
//!< "result" -- rectangles before merging
//!<          -- after this function it contains rectangles after NMS
//!< "combine_min" -- threshold of how many detection are needed to survive

void removeCoveredRectangles(std::vector<CRect>& result);
//!< If one detection (after NMS) is inside another, remove the inside one

void loadCascade(Detector& ds);
//!< Functions that load the two classifiers

#endif
