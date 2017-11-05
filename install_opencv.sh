mkdir opencv
cd opencv
if [ ! -f ./2.4.13.4.zip ]; then
    wget https://github.com/opencv/opencv/archive/2.4.13.4.zip
    unzip *.zip
fi
cp ../opencv_CMakeLists.txt ./opencv*/
cd opencv*
mkdir build
cd build
rm -r ./*
cmake ..
make -j4
