cd caff*
mkdir build
cd build
rm -r ./*
cmake -DOPENCV_PATH:STRING=$(pwd) ..
make -j4
