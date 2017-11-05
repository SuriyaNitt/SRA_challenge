a=$(pwd)/opencv/opencv_install
cd caff*
mkdir build
cd build
rm -r ./*
cmake -DOPENCV_PATH:STRING=$a ..
make -j4
