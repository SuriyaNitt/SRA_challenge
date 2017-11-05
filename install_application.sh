a=$(pwd)/opencv/opencv_install
mkdir build
cd build
rm -r ./*
cmake -DOPENCV_PATH:STRING=$a ..
make -j4

