a = $(pwd)/opencv
cd caff*
mkdir build
cd build
rm -r ./*
cmake -DOPENCV_PATH:STRING=$a ..
make -j4
