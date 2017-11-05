a=$(pwd)/opencv/opencv_install
mkdir opencv
cd opencv
mkdir opencv_install
if [ ! -f ./2.4.13.4.zip ]; then
    wget https://github.com/opencv/opencv/archive/2.4.13.4.zip
    unzip *.zip
fi
cp ../opencv_CMakeLists.txt ./opencv*/CMakeLists.txt
cd opencv-2*
mkdir build
cd build
rm -r ./*
cmake -DOPENCV_INSTALL_PATH:STRING=$a ..
make -j4
make install
