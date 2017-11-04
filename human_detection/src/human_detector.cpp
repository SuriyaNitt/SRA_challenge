/**
*   Source file defing the helper classes
*/

#include <human_detector.h>
#include <fstream>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

const int HUMAN_height = 108;
const int HUMAN_width = 36;
const int HUMAN_xdiv = 9;
const int HUMAN_ydiv = 4;
static const int EXT = 1;

/*********************
*   Array2D
**********************/

template<class T>
Array2d<T>::Array2d():nrow(0),ncol(0),p(NULL),buf(NULL) {} //!< default constructor.

template<class T>
Array2d<T>::Array2d(const int nrow,const int ncol):nrow(0),ncol(0),p(NULL),buf(NULL)
{
    create(nrow,ncol);
} //!< constructor.

template<class T>
Array2d<T>::Array2d(const Array2d<T>& source):nrow(0),ncol(0),p(NULL) {
    if(source.buf!=NULL)
    {
        create(source.nrow,source.ncol);
        std::copy(source.buf,source.buf+nrow*ncol,buf);
    }
}

template<class T>
Array2d<T>& Array2d<T>::operator=(const Array2d<T>& source) {
    if(source.buf!=NULL)
    {
        create(source.nrow,source.ncol);
        std::copy(source.buf,source.buf+nrow*ncol,buf);
    }
    else
        clear();
    return *this;
}

template<class T>
Array2d<T>::~Array2d()
{
    clear();
} //!< virtual destructor to avoid mem. leak.

template<class T>
void Array2d<T>::create(const int nrowArg, const int ncolArg) {
    assert(nrowArg>0 && ncolArg>0);
    if(nrow==nrowArg && ncol==ncolArg) return;
    clear();
    nrow = nrowArg;
    ncol = ncolArg;
    buf = new T[nrow*ncol];
    assert(buf!=NULL);
    for(int i=0; i<nrow; i++) p[i] = buf + i * ncol;
}

template<class T>
void Array2d<T>::swap(Array2d<T>& array2) {
    std::swap(nrow, array2.nrow);
    std::swap(ncol, array2.ncol);
    std::swap(p, array2.p);
    std::swap(buf,array2.buf);
}

template<class T>
void Array2d<T>::zero(const T t) {
    if(nrow>0) std::fill(buf,buf+nrow*ncol,t);
}

template<class T>
void Array2d<T>::clear() {
    delete[] buf;
    buf = NULL;
    delete[] p;
    p = NULL;
    nrow = ncol = 0;
}

/*********************
*   IntImage
**********************/

template<class T>
IntImage<T>::~IntImage() {
    clear();
} //!< virtual destructor to avoid memory leaks.

template<class T>
void IntImage<T>::clear(void) {
    Array2d<T>::clear();
    variance = 0.0;
    label = -1;
}

template<class T>
bool IntImage<T>::load(cv::Mat img, char channel) {
    if (img.empty()) return false;
    if (channel == 'R' || channel == 'G' || channel == 'B') {
        int c;
        switch (channel) {
            case 'B':
                c = 0;
                break;
            case 'G':
                c = 1;
                break;
            default:
                c = 2;
                break;
        }
        cv::Mat planes[3];
        split(img, planes);
        img = planes[c];
    }
    else {
        cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    }
    setSize(img.rows, img.cols);
    for (int i=0, ih=img.rows, iw=img.cols; i<ih; i++) {
        T *pdata = p[i];
        unsigned char* pimg = reinterpret_cast<unsigned char*>(img.data + img.step*i);
        for (int j=0; j<iw; j++) {
            pdata[j] = pimg[j];
        }
    }
    return true;
}

template<class T>
void IntImage<T>::save(const std::string& fileName) const {
    IplImage *img;
    img = cvCreateImage(cvSize(ncol, nrow), IPL_DEPTH_8U, 1);
    for(int i=0,ih=img->height,iw=img->width; i<ih; i++)
    {
        T* pdata = p[i];
        unsigned char* pimg = reinterpret_cast<unsigned char*>(img->imageData+img->widthStep*i);
        for(int j=0; j<iw; j++) pimg[j] = (unsigned char)pdata[j];
    }
    cvSaveImage(fileName.c_str(),img);
    cvReleaseImage(&img);
}

template<class T>
void IntImage<T>::setSize(const int h, const int w) {
    if((h == nrow) && (w == ncol)) return;
    clear();
    Array2d<T>::create(h,w);
}

template<class T>
IntImage<T>& IntImage<T>::operator=(const IntImage<T>& source) {
    if(&source==this) return *this;
    setSize(source.nrow,source.ncol);
    std::copy(source.buf,source.buf+nrow*ncol,buf);
    label = source.label;
    variance = source.variance;
    return *this;
}

template<class T>
void IntImage<T>::resize(IntImage<T> &result,const double ratio) const {
    resize(result, int(nrow*ratio), int(ncol*ratio));
}

template<class T>
void IntImage<T>::resize(IntImage<T>& result,const int height,const int width) const {
    assert(height>0 && width>0);
    result.setSize(height, width);
    double ixratio = nrow*1.0/height;
    double iyratio = ncol*1.0/width;

    double *p_y = new double[result.ncol];
    assert(p_y != NULL);
    int *p_y0 = new int[result.ncol];
    assert(p_y0 != NULL);
    for (int i=0; i<width; i++) {
        p_y[i] = i*iyratio;
        p_y0[i] = (int) p_y[i];
        if(p_y0[i]==ncol-1) p_y0[i]--;
        p_y[i] -= p_y0[i];
    }

    for (int i=0; i<height; i++) {
        int x0;
        double x;
        x = i*ixratio;
        x0 = (int) x;
        if (x0==nrow-1) x0--;
        x -= x0;
        T* rp = result.p[i];
        const T* px0 = p[x0];
        const T* px1 = p[x0+1];
        for (int j=0; j<width; j++) {
            int y0 = p_y0[j];
            double y =p_y[j], fx0, fx1;

            fx0 = (double)(px0[y0] + y*(px0[y0+1]-px0[y0]));
            fx1 = (double)(px1[y0] + y*(px1[y0+1]-px1[y0]));

            rp[j] = T(fx0 + x*(fx1-fx0));   
        }
    }

    delete[] p_y;
    delete[] p_y0;
    p_y = NULL;
    p_y0 = NULL;
}

template<class T>
void IntImage<T>::calcIntegralImageInPlace(void) {
    for(int i=1; i<ncol; i++)   // process the first line
    {
        buf[i] += buf[i-1];
    }
    for(int i=1; i<nrow; i++)
    {
        double partialsum = 0;
        T* curp = p[i];
        T* prep = p[i-1];
        for(int j=0; j<ncol; j++)
        {
            partialsum += (double)(curp[j]);
            curp[j] = prep[j] + partialsum;
        }
    }
} //!< A zero column and a zero row is padded, so 24*24 image will be 25*25 in size
//!< if the input image is not padded, the results on 1st row will be problematic

template<class T>
void IntImage<T>::swap(IntImage<T>& image2) {
    Array2d<T>::swap(image2);
    std::swap(variance, image2.variance);
    std::swap(label, image2.label);
}

template<class T>
void IntImage<T>::Sobel(IntImage<double>& result,const bool useSqrt,const bool normalize) {
    result.create(nrow,ncol);
    for(int i=0; i<nrow; i++) result.p[i][0] = result.p[i][ncol-1] = 0;
    std::fill(result.p[0],result.p[0]+ncol,0.0);
    std::fill(result.p[nrow-1],result.p[nrow-1],0.0);
    for(int i=1; i<nrow-1; i++)
    {
        T* p1 = p[i-1];
        T* p2 = p[i];
        T* p3 = p[i+1];
        double* pr = result.p[i];
        for(int j=1; j<ncol-1; j++)
        {
            double gx =     p1[j-1] - p1[j+1]
                          + 2*(p2[j-1]   - p2[j+1])
                          +    p3[j-1] - p3[j+1];
            double gy =     p1[j-1] - p3[j-1]
                          + 2*(p1[j]   - p3[j])
                          +    p1[j+1] - p3[j+1];
            pr[j] = gx*gx+gy*gy;
        }
    }
    if(useSqrt || normalize ) // if we want to normalize the result image, we'd better use the true Sobel gradient
        for(int i=1; i<nrow-1; i++)
            for(int j=1; j<ncol-1; j++)
                result.p[i][j] = sqrt(result.p[i][j]);

    if(normalize)
    {
        double minv = 1e20, maxv = -minv;
        for(int i=1; i<nrow-1; i++)
        {
            for(int j=1; j<ncol-1; j++)
            {
                if(result.p[i][j]<minv)
                    minv = result.p[i][j];
                else if(result.p[i][j]>maxv)
                    maxv = result.p[i][j];
            }
        }
        for(int i=0; i<nrow; i++) result.p[i][0] = result.p[i][ncol-1] = minv;
        for(int i=0; i<ncol; i++) result.p[0][i] = result.p[nrow-1][i] = minv;
        double s = 255.0/(maxv-minv);
        for(int i=0; i<nrow*ncol; i++) result.buf[i] = (result.buf[i]-minv)*s;
    }
} 
//!< compute the Sobel gradient. For now, we just use the very inefficient way. Optimization can be done later
//!< if useSqrt = true, we compute the double Sobel gradient; otherwise, the square of it
//!< if normalize = true, the numbers are normalized to be in 0..255

/*********************
*   CRect
**********************/

bool CRect::empty() const {
    return (left >= right) || (top >= bottom);
}

void CRect::clear() {
    left = right = top = bottom = 0;
}

double CRect::size() const {
    if(empty())
        return 0;
    else
        return (bottom-top)*(right-left);
}

bool CRect::intersect(CRect& result,const CRect& rect2) const {
    if(this->empty() || rect2.empty() || 
       left >= rect2.right || rect2.left >= right ||
       top >= rect2.bottom || rect2.top >= bottom ) {
        result.clear();
        return false;
    }
    result.left   = std::max( left, rect2.left );
    result.right  = std::min( right, rect2.right );
    result.top    = std::max( top, rect2.top );
    result.bottom = std::min( bottom, rect2.bottom );
    return true;
}

bool CRect::Union(CRect& result,const CRect& rect2) const {
    if(this->empty()) {
        if (rect2.empty()) {
            result.clear();
            return false;
        }
    else
        result = rect2;
    }
    else {
        if (rect2.empty())
            result = *this;
        else {
            result.left   = std::min( left, rect2.left );
            result.right  = std::max( right, rect2.right );
            result.top    = std::min( top, rect2.top );
            result.bottom = std::max( bottom, rect2.bottom );
        }
    }
    return true;
}

/*********************
*   Node
**********************/

void Node::load(const NodeType _type,const int _featurelength,const int _upperBound,const int _index,const char* _filename)
{
    type = _type;
    index = _index;
    fileName = _filename;
    featureLength = _featurelength;
    upperBound = _upperBound;
    if(type==CD_LIN)
        thresh = useSVM_CD_FastEvaluationStructure(_filename,_featurelength,classifier);
    else if(type==CD_HIK)
        thresh = useSVM_CD_FastEvaluationStructure(_filename,_featurelength,upperBound,classifier);

    if(type==CD_LIN) type = LINEAR;
    if(type==CD_HIK) type = HISTOGRAM;
}

/*********************
*   Cascade
**********************/

Cascade::Cascade() : size(20), length(0) {
    nodes = new Node*[size];
}

Cascade::~Cascade() {
    for (int i=0; i<length; i++)
        delete nodes[i];
    delete nodes;
}

void Cascade::add_node(const Node::NodeType _type,\
                       const int featureLength, \
                       const int upperBound, \
                       std::string fileName) {
    if (length == size) {
        int newSize = size * 2;
        Node **newNodes = new Node*[newSize];
        assert(newNodes != NULL);
        std::copy(nodes, nodes+size, newNodes);
        size = newSize;
        nodes = newNodes;
    }
}

/*********************
*   Detector
**********************/

void Detector::loadDetector(std::vector<Node::NodeType>& types,std::vector<int>& upperBounds,std::vector<std::string>& filenames)
{
    unsigned int depth = types.size();
    assert(depth>0 && depth==upperBounds.size() && depth==filenames.size());
    if(cascade)
        delete cascade;
    cascade = new Cascade;
    assert(xdiv>0 && ydiv>0);
    for(unsigned int i=0; i<depth; i++)
        cascade->add_node(types[i],(xdiv-EXT)*(ydiv-EXT)*baseflength,upperBounds[i],filenames[i].c_str());

    hist.create(1,baseflength*(xdiv-EXT)*(ydiv-EXT));
}

void Detector::initImage(IntImage<double>& original)
{
    image = original;
    image.Sobel(sobelImage, false, false);
    computeCT(sobelImage, ct);
}
//!< initialization -- compute the Census Tranform image for CENTRIST

void Detector::initIntegralImages(const int stepsize)
{
    if(cascade->nodes[0]->type!=Node::LINEAR)
        return; // No need to prepare integral images

    const int hd = height/xdiv*2-2;
    const int wd = width/ydiv*2-2;
    scores.create(ct.nrow,ct.ncol);
    scores.zero(cascade->nodes[0]->thresh/hd/wd);
    double* linearweights = cascade->nodes[0]->classifier.buf;
    for(int i=0; i<xdiv-EXT; i++)
    {
        const int xoffset = height/xdiv*i;
        for(int j=0; j<ydiv-EXT; j++)
        {
            const int yoffset = width/ydiv*j;
            for(int x=2; x<ct.nrow-2-xoffset; x++)
            {
                int* ctp = ct.p[x+xoffset]+yoffset;
                double* tempp = scores.p[x];
                for(int y=2; y<ct.ncol-2-yoffset; y++)
                    tempp[y] += linearweights[ctp[y]];
            }
            linearweights += baseflength;
        }
    }
    scores.calcIntegralImageInPlace();
    for(int i=2; i<ct.nrow-2-height; i+=stepsize)
    {
        double* p1 = scores.p[i];
        double* p2 = scores.p[i+hd];
        for(int j=2; j<ct.ncol-2-width; j+=stepsize)
            p1[j] += (p2[j+wd] - p2[j] - p1[j+wd]);
    }
}

void Detector::resizeImage() {
    image.resize(sobelImage,ratio);
    image.swap(sobelImage);
    image.Sobel(sobelImage,false,false);
    computeCT(sobelImage,ct);
}

int Detector::fastScan(IntImage<double>& original,std::vector<CRect>& results,const int stepsize) {
    if(original.nrow<height+5 || original.ncol<width+5) return 0;
    const int hd = height/xdiv;
    const int wd = width/ydiv;
    initImage(original);
    results.clear();

    hist.create(1,baseflength*(xdiv-EXT)*(ydiv-EXT));

    Node* node = cascade->nodes[1];
    double** pc = node->classifier.p;
    int oheight = original.nrow, owidth = original.ncol;
    CRect rect;
    while(image.nrow>=height && image.ncol>=width)
    {
        initIntegralImages(stepsize);
        for(int i=2; i+height<image.nrow-2; i+=stepsize)
        {
            const double* sp = scores.p[i];
            for(int j=2; j+width<image.ncol-2; j+=stepsize)
            {
                if(sp[j]<=0) continue;
                int* p = hist.buf;
                hist.zero();
                for(int k=0; k<xdiv-EXT; k++)
                {
                    for(int t=0; t<ydiv-EXT; t++)
                    {
                        for(int x=i+k*hd+1; x<i+(k+1+EXT)*hd-1; x++)
                        {
                            int* ctp = ct.p[x];
                            for(int y=j+t*wd+1; y<j+(t+1+EXT)*wd-1; y++)
                                p[ctp[y]]++;
                        }
                        p += baseflength;
                    }
                }
                double score = node->thresh;
                for(int k=0; k<node->classifier.nrow; k++) score += pc[k][hist.buf[k]];
                if(score>0)
                {
                    rect.top = i*oheight/image.nrow;
                    rect.bottom = (i+height)*oheight/image.nrow;
                    rect.left = j*owidth/image.ncol;
                    rect.right = (j+width)*owidth/image.ncol;
                    results.push_back(rect);
                }
            }
        }
        resizeImage();
    }
    return 0;
}

void computeCT(IntImage<double>& original,IntImage<int>& ct)
{
    ct.create(original.nrow,original.ncol);
    for(int i=2; i<original.nrow-2; i++)
    {
        double* p1 = original.p[i-1];
        double* p2 = original.p[i];
        double* p3 = original.p[i+1];
        int* ctp = ct.p[i];
        for(int j=2; j<original.ncol-2; j++)
        {
            int index = 0;
            if(p2[j]<=p1[j-1]) index += 0x80;
            if(p2[j]<=p1[j]) index += 0x40;
            if(p2[j]<=p1[j+1]) index += 0x20;
            if(p2[j]<=p2[j-1]) index += 0x10;
            if(p2[j]<=p2[j+1]) index += 0x08;
            if(p2[j]<=p3[j-1]) index += 0x04;
            if(p2[j]<=p3[j]) index += 0x02;
            if(p2[j]<=p3[j+1]) index ++;
            ctp[j] = index;
        }
    }
}

double useSVM_CD_FastEvaluationStructure(const char* modelFile,const int m,Array2d<double>& result)
{
    std::ifstream in(modelFile);
    if(in.good()==false)
    {
        std::cout<<"SVM model "<<modelFile<<" can not be loaded."<<std::endl;
        exit(-1);
    }
    std::string buffer;
    std::getline(in,buffer); // first line
    std::getline(in,buffer); // second line
    std::getline(in,buffer); // third line
    in>>buffer;
    assert(buffer=="nr_feature");
    int num_dim;
    in>>num_dim;
    assert(num_dim>0 && num_dim==m);
    std::getline(in,buffer); // end of line 4
    in>>buffer;
    assert(buffer=="bias");
    int bias;
    in>>bias;
    std::getline(in,buffer); //end of line 5;
    in>>buffer;
    assert(buffer=="w");
    std::getline(in,buffer); //end of line 6
    result.create(1,num_dim);
    for(int i=0; i<num_dim; i++) in>>result.buf[i];
    double rho = 0;
    if(bias>=0) in>>rho;
    in.close();
    return rho;
}

double useSVM_CD_FastEvaluationStructure(const char* modelFile, const int m, const int upperBound, Array2d<double>& result)
{

    std::ifstream fs(modelFile, std::fstream::binary);
    if( !fs.is_open() )
    {
        std::cout << "SVM model " << modelFile << " can not be loaded." << std::endl;
        exit(-1);
    }
    // Header
    int rows, cols, type, channels;
    fs.read((char*)&rows, sizeof(int));         // rows
    fs.read((char*)&cols, sizeof(int));         // cols
    fs.read((char*)&type, sizeof(int));         // type
    fs.read((char*)&channels, sizeof(int));     // channels

    // Data
    cv::Mat mat(rows, cols, type);
    fs.read((char*)mat.data, CV_ELEM_SIZE(type) * rows * cols);

    int num_dim = m;

    result.create(num_dim, upperBound);
    for(int i=0; i<num_dim; i++)
        for (int j = 0; j < upperBound; j++)
        {
            result.p[i][j]= mat.at<double>(i, j);
        }

    return -0.00455891;
}

void postProcess(std::vector<CRect>& result,const int combine_min) {
    std::vector<CRect> res1;
    std::vector<CRect> resmax;
    std::vector<int> res2;
    bool yet;
    CRect rectInter;

    for(unsigned int i=0,size_i=result.size(); i<size_i; i++)
    {
        yet = false;
        CRect& result_i = result[i];
        for(unsigned int j=0,size_r=res1.size(); j<size_r; j++)
        {
            CRect& resmax_j = resmax[j];
            if(result_i.intersect(rectInter,resmax_j))
            {
                if(  rectInter.size()>0.6*result_i.size()
                        && rectInter.size()>0.6*resmax_j.size()
                  )
                {
                    CRect& res1_j = res1[j];
                    resmax_j.Union(resmax_j,result_i);
                    res1_j.bottom += result_i.bottom;
                    res1_j.top += result_i.top;
                    res1_j.left += result_i.left;
                    res1_j.right += result_i.right;
                    res2[j]++;
                    yet = true;
                    break;
                }
            }
        }
        if(yet==false)
        {
            res1.push_back(result_i);
            resmax.push_back(result_i);
            res2.push_back(1);
        }
    }

    for(unsigned int i=0,size=res1.size(); i<size; i++)
    {
        const int count = res2[i];
        CRect& res1_i = res1[i];
        res1_i.top /= count;
        res1_i.bottom /= count;
        res1_i.left /= count;
        res1_i.right /= count;
    }

    result.clear();
    for(unsigned int i=0,size=res1.size(); i<size; i++)
        if(res2[i]>combine_min)
            result.push_back(res1[i]);
}

void removeCoveredRectangles(std::vector<CRect>& result) {
    std::vector<bool> covered;
    covered.resize(result.size());
    std::fill(covered.begin(),covered.end(),false);
    CRect inter;
    for(unsigned int i=0; i<result.size(); i++)
    {
        for(unsigned int j=i+1; j<result.size(); j++)
        {
            result[i].intersect(inter,result[j]);
            double isize = inter.size();
            if(isize>result[i].size()*0.65)
                covered[i] = true;
            if(isize>result[j].size()*0.65)
                covered[j] = true;
        }
    }
    std::vector<CRect> newresult;
    for(unsigned int i=0; i<result.size(); i++)
        if(covered[i]==false)
            newresult.push_back(result[i]);
    result.clear();
    result.insert(result.begin(),newresult.begin(),newresult.end());
    newresult.clear();
}

void loadCascade(Detector& ds)
{
    std::vector<Node::NodeType> types;
    std::vector<int> upperBounds;
    std::vector<std::string> fileNames;

    types.push_back(Node::CD_LIN); // first node
    upperBounds.push_back(100);
    fileNames.push_back("combined.txt.model");
    types.push_back(Node::CD_HIK); // second node
    upperBounds.push_back(353);
    fileNames.push_back("combined.txt.model_");

    ds.loadDetector(types,upperBounds,fileNames);
    // You can adjust these parameters for different speed, accuracy etc
    ds.cascade->nodes[0]->thresh += 0.8;
    ds.cascade->nodes[1]->thresh -= 0.095;
}


template class Array2d<double>;
template class Array2d<int>;

template class IntImage<double>;
template class IntImage<int>;
